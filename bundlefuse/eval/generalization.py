"""Learned baseline, fixed-selection sparse-input tests, and warm-cache efficiency measurements."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from bundlefuse.artifacts import write_json, flat_metrics
from bundlefuse.baselines.metadata_ranker import GRID, train_ranker
from bundlefuse.eval.components import apply_method, select_controls
from bundlefuse.eval.provided import DiskSignals
from bundlefuse.fusion_selection import ValidationSignals, fit_fusion
from bundlefuse.metrics import ranking_metrics, as_numpy, TIE_POLICY
from bundlefuse.profiling import profile_scorers, peak_rss_bytes
from bundlefuse.provided_splits import file_hash
from bundlefuse.robust_protocol import CASES, sparse_pairs


METHODS = ('BundleFuse', 'SimpleHybrid', 'HardSwitchHybrid', 'MetadataRanker_s0', 'MetadataRanker_s1', 'MetadataRanker_s2')


def validation_ndcg(model, valid, chunk):
    catalogue = model.catalogue()
    total = 0.0
    for start in range(0, len(valid), chunk):
        batch = valid[start:start + chunk]
        score = model.score([s for s, _ in batch], catalogue)
        total += ranking_metrics(score, batch, Ks=(20,))[20][1] * len(batch)
    return total / len(valid)


def fit_learners(content, categories, frequency, train, valid, output, chunk, auxiliary, grid=GRID):
    output = Path(output)
    records = []
    best_model, best_record, best_index = None, None, None
    validate = lambda model: validation_ndcg(model, valid, chunk)
    for index, hp in enumerate(grid):
        print(f'TRAIN MetadataRanker grid={index} seed=0 hp={hp}', flush=True)
        model, record = train_ranker(content, categories, frequency, train, validate, hp, 0,
                                     output / f'grid_{index}', auxiliary)
        record['checkpoint'] = f'grid_{index}/best.pt'
        records.append(record)
        if best_record is None or record['val_ndcg20'] > best_record['val_ndcg20']:
            best_model, best_record, best_index = model, record, index
        write_json(output / 'grid.json', records)
    models = {'MetadataRanker_s0': best_model}
    chosen = {'MetadataRanker_s0': best_record}
    for seed in (1, 2):
        print(f'TRAIN MetadataRanker confirmation seed={seed} hp selected on seed-0 validation', flush=True)
        model, record = train_ranker(content, categories, frequency, train, validate, grid[best_index], seed,
                                     output / f'seed_{seed}', auxiliary)
        record['checkpoint'] = f'seed_{seed}/best.pt'
        models[f'MetadataRanker_s{seed}'] = model
        chosen[f'MetadataRanker_s{seed}'] = record
    for record in chosen.values():
        record['checkpoint_sha256'] = file_hash(output / record['checkpoint'])
    selection = {'grid': records, 'selected_grid': best_index, 'selected': chosen,
                 'selection': 'Seed-0 hyperparameters and each seed checkpoint selected by full supplied validation NDCG@20; no test selection'}
    write_json(output / 'selection.json', selection)
    return models, selection


def retune_retail(scorer, hybrid, signals, frequency, valid, output, chunk):
    def combined(seeds):
        raw, z = scorer(seeds)
        z = {key: torch.as_tensor(value) for key, value in z.items()}
        z.update({'raw_' + key: torch.as_tensor(value) for key, value in hybrid.signals(seeds).items()})
        return raw, z
    raw_keys = ['raw_' + key for key in hybrid.signals([valid[0][0]])]
    cache = DiskSignals(Path(output) / 'validation_cache', valid, combined,
                        signals + ['pair', 'diff2_pair', 'rao'] + raw_keys, len(frequency), chunk)
    try:
        validation = ValidationSignals(cache.blocks, valid, frequency,
                                       progress=lambda line: print(line, flush=True))
        weights, prior, eb, ho, variance = fit_fusion(validation, signals, ('cat', 'content', 'bridge'), rao=(0., .15, .25))
        selected = {'BundleFuse': {'kind': 'fusion', 'weights': weights, 'prior_keys': prior,
                                  'kappa': eb[0], 'prior_scale': eb[1], 'alpha_inc': ho[0],
                                  'gamma_rao': ho[1], 'variance_components': variance}}
        selected.update(select_controls(validation, cache, torch.as_tensor(frequency)[:, None], raw_keys))
        return selected
    finally:
        cache.close()


def run_generalization(dataset, train, valid, test, test_ids, groups, protocol, scorer, signals,
                       frequency, hybrid, content, categories, output, *, reference=None,
                       preprocessing_seconds=0., auxiliary=None, chunk=64, grid=GRID):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(output / 'protocol.json', protocol)
    selection_started = time.perf_counter()
    if reference is not None:
        original = json.loads((Path(reference) / 'result.json').read_text())
        if original['protocol']['sha256'] != protocol['sha256'] or original['test_query_ids'] != test_ids:
            raise RuntimeError('Frozen reference settings belong to different split files or query IDs')
        selected = {name: original['selected'][name] for name in METHODS[:3]}
    else:
        selected = retune_retail(scorer, hybrid, signals, frequency, valid, output, chunk)
    selection_seconds = time.perf_counter() - selection_started
    learning_started = time.perf_counter()
    models, learner_selection = fit_learners(content, categories, frequency, train, valid,
                                             output / 'learned', chunk, auxiliary, grid)
    learning_seconds = time.perf_counter() - learning_started
    frozen = {'baselines': selected, 'learned': learner_selection,
              'stress_test_policy': 'All settings selected on the original validation masks, then frozen for all test cases'}
    write_json(output / 'selection.json', frozen)
    selection_hash = file_hash(output / 'selection.json')
    catalogues, encoding_seconds = {}, {}
    for name, model in models.items():
        begin = time.perf_counter(); catalogues[name] = model.catalogue()
        encoding_seconds[name] = time.perf_counter() - begin
    freq = torch.as_tensor(frequency)[:, None]
    cases = {}
    for case in CASES:
        pairs = sparse_pairs(test, test_ids, case)
        records = {name: [] for name in METHODS}
        case_started = time.perf_counter()
        for start in range(0, len(pairs), chunk):
            batch = pairs[start:start + chunk]; seeds = [s for s, _ in batch]
            raw, z = scorer(seeds); del raw
            z = {key: torch.as_tensor(value) for key, value in z.items()}
            z.update({'raw_' + key: torch.as_tensor(value) for key, value in hybrid.signals(seeds).items()})
            for name in METHODS:
                score = apply_method(z, selected[name], freq) if name in selected else models[name].score(seeds, catalogues[name])
                array = as_numpy(score)
                for j, pair in enumerate(batch):
                    records[name].append(flat_metrics(ranking_metrics(array[:, j:j + 1], [pair], Ks=(10, 20, 50))))
            print(f'TEST {dataset} {case}: {start + len(batch)}/{len(pairs)}', flush=True)
        metrics = {name: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} for name, rows in records.items()}
        cases[case] = {'methods': metrics, 'per_query': records,
                       'pairs_sha256': hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
                       'queries_changed': sum(set(a[0]) != set(b[0]) for a, b in zip(test, pairs)),
                       'mean_revealed': float(np.mean([len(s) for s, _ in pairs])),
                       'mean_targets': float(np.mean([len(g) for _, g in pairs])),
                       'seconds': time.perf_counter() - case_started}
        write_json(output / f'case_{case}.json', cases[case], compact=True)
        print(f'COMPLETE CASE {dataset} {case}: ' + json.dumps({name: value['R@20'] for name, value in metrics.items()}), flush=True)
    if reference is not None:
        for name in METHODS[:3]:
            difference = max(abs(a[key] - b[key]) for a, b in
                             zip(original['per_query'][name], cases['supplied']['per_query'][name]) for key in a)
            if difference > 1e-9:
                raise RuntimeError(f'Frozen {name} no longer reproduces the preceding experiment: {difference}')
    if file_hash(output / 'selection.json') != selection_hash:
        raise RuntimeError('Selections changed during test evaluation')

    def bundle_score(seeds):
        raw, z = scorer(seeds); del raw
        return apply_method({key: torch.as_tensor(value) for key, value in z.items()}, selected['BundleFuse'], freq)

    def control_score(name, seeds):
        z = {'raw_' + key: torch.as_tensor(value) for key, value in hybrid.signals(seeds).items()}
        return apply_method(z, selected[name], freq)

    profiles = {'BundleFuse': bundle_score,
                'SimpleHybrid': lambda seeds: control_score('SimpleHybrid', seeds),
                'HardSwitchHybrid': lambda seeds: control_score('HardSwitchHybrid', seeds),
                'MetadataRanker_s0': lambda seeds: models['MetadataRanker_s0'].score(seeds, catalogues['MetadataRanker_s0'])}
    timing = {'preprocessing_seconds': preprocessing_seconds, 'baseline_selection_seconds': selection_seconds,
              'learned_training_and_validation_seconds': learning_seconds, 'catalogue_encoding_seconds': encoding_seconds,
              'warm_cache_inference': profile_scorers(profiles, test), 'worker_peak_rss_bytes': peak_rss_bytes(),
              'scope': 'CPU; fixed precomputed source embeddings/metadata. External embedding pretraining/extraction excluded. Peak RSS is whole-worker, not per-method.'}
    write_json(output / 'timing.json', timing)
    result = {'dataset': dataset, 'protocol': protocol, 'test_query_ids': test_ids, 'query_groups': groups,
              'tie_policy': TIE_POLICY, 'selection_sha256': selection_hash, 'selected': frozen,
              'cases': {case: dict({key: value for key, value in record.items() if key != 'per_query'},
                                   per_query_file=f'case_{case}.json') for case, record in cases.items()},
              'timing': timing, 'seconds': time.perf_counter() - started}
    write_json(output / 'result.json', result)
    print(f'COMPLETE DATASET {dataset}', flush=True)
