"""Original-input raw/engineered compact-model pilot, after exhaustive scorer parity."""
import json
from pathlib import Path
import time

import numpy as np

from bundlefuse.artifacts import write_json, flat_metrics
from bundlefuse.baselines.compact_ranker import CompactRanker, GRID
from bundlefuse.baselines.metadata_ranker import train_ranker
from bundlefuse.compact_features import build_features
from bundlefuse.eval.generalization import validation_ndcg
from bundlefuse.metrics import ranking_metrics, as_numpy, TIE_POLICY
from bundlefuse.paths import PROJECT_ROOT
from bundlefuse.profiling import profile_scorers, peak_rss_bytes
from bundlefuse.provided_splits import file_hash
from bundlefuse.scoring_audit import audit_selected_scores, frozen_scorers


def model_metrics(model, pairs, catalogue, chunk=64, per_query=False):
    rows = []
    totals = {k: np.zeros(2) for k in (10, 20, 50)}
    for start in range(0, len(pairs), chunk):
        batch = pairs[start:start + chunk]
        scores = model.score([s for s, _ in batch], catalogue)
        if per_query:
            scores = as_numpy(scores)
            rows.extend(flat_metrics(ranking_metrics(scores[:, j:j + 1], [pair], Ks=(10, 20, 50)))
                        for j, pair in enumerate(batch))
        else:
            metrics = ranking_metrics(scores, batch, Ks=(10, 20, 50))
            for k in totals:
                totals[k] += np.asarray(metrics[k]) * len(batch)
    if per_query:
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}, rows
    return flat_metrics({k: value / len(pairs) for k, value in totals.items()}), rows


def fit_compact(packs, frequency, train, valid, output, grid=GRID, chunk=64):
    models, chosen, validation = {}, {}, {}
    for variant, features in packs.items():
        records = []
        best_model, best_record, best_index = None, None, None
        for index, hp in enumerate(grid):
            print(f'TRAIN {variant} grid={index} seed=0 hp={hp}', flush=True)
            model, record = train_ranker(None, [], frequency, train,
                                         lambda model: validation_ndcg(model, valid, chunk),
                                         hp, 0, output / variant / f'grid_{index}',
                                         model_factory=lambda: CompactRanker(features, frequency, hp['dim']))
            record['checkpoint'] = f'{variant}/grid_{index}/best.pt'
            records.append(record)
            if best_record is None or record['val_ndcg20'] > best_record['val_ndcg20']:
                best_model, best_record, best_index = model, record, index
            write_json(output / variant / 'grid.json', records)
        models[f'{variant}_s0'], chosen[f'{variant}_s0'] = best_model, best_record
        for seed in (1, 2):
            hp = grid[best_index]
            print(f'TRAIN {variant} seed={seed} validation-selected grid={best_index}', flush=True)
            model, record = train_ranker(None, [], frequency, train,
                                         lambda model: validation_ndcg(model, valid, chunk),
                                         hp, seed, output / variant / f'seed_{seed}',
                                         model_factory=lambda: CompactRanker(features, frequency, hp['dim']))
            record['checkpoint'] = f'{variant}/seed_{seed}/best.pt'
            models[f'{variant}_s{seed}'], chosen[f'{variant}_s{seed}'] = model, record
        for seed in (0, 1, 2):
            name = f'{variant}_s{seed}'
            chosen[name]['checkpoint_sha256'] = file_hash(output / chosen[name]['checkpoint'])
            model = models[name]
            validation[name], _ = model_metrics(model, valid, model.catalogue(), chunk)
        write_json(output / 'selection_in_progress.json', {'selected': chosen, 'validation': validation})
    return models, chosen, validation


def run_compact(dataset, splits, scorer, graph, ontology, features, content, categories, albums,
                music, output, *, preprocessing_seconds=0., grid=GRID, chunk=64, reference_path=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    reference_path = (Path(reference_path) if reference_path else
                      Path(PROJECT_ROOT) / 'results' / 'component_study' / dataset / 'result.json')
    original = json.loads(reference_path.read_text())
    protocol = splits.manifest()
    test_ids = sorted(splits.splits['test'][0])
    if original['protocol']['sha256'] != protocol['sha256'] or original['test_query_ids'] != test_ids:
        raise RuntimeError('Frozen BundleFuse reference uses different inputs/query IDs')
    write_json(output / 'protocol.json', protocol)
    valid, test = splits.pairs('valid'), splits.pairs('test')
    selected = original['selected']['BundleFuse']
    records, parity = audit_selected_scores(scorer, selected, graph.freq, test,
                                            original['per_query']['BundleFuse'], output / 'parity.json', chunk)
    write_json(output / 'status.json', {'stage': 'feature_construction'})
    category_inputs = [categories] + ([albums] if albums is not None else [])
    packs, feature_record = build_features(content, category_inputs, graph.freq, graph, ontology, features, music)
    write_json(output / 'features.json', feature_record)
    write_json(output / 'status.json', {'stage': 'direct_training'})
    training_started = time.perf_counter()
    models, chosen, validation = fit_compact(packs, graph.freq, list(splits.train.values()), valid,
                                            output / 'learned', grid, chunk)
    training_seconds = time.perf_counter() - training_started
    # A feasibility target, not a claim of statistical equivalence or a test-set decision.
    full_validation = original['validation']['BundleFuse']
    retention = {key: validation['Engineered_s0'][key] / full_validation[key] for key in ('R@20', 'N@20')}
    gate = {'engineered_beats_raw_validation_ndcg20': validation['Engineered_s0']['N@20'] > validation['Raw_s0']['N@20'],
            'validation_retention': retention, 'retention_target': .975,
            'policy': 'Proceed only if seed-0 engineered beats raw validation NDCG@20 and retains >=97.5% of frozen BundleFuse validation R@20 and NDCG@20 on BOTH datasets. Feasibility threshold, not equivalence.'}
    gate['pass_this_dataset'] = gate['engineered_beats_raw_validation_ndcg20'] and min(retention.values()) >= .975
    frozen = {'selected': chosen, 'validation': validation, 'BundleFuse_validation': full_validation, 'gate': gate,
              'policy': 'Per-arm seed-0 full-validation NDCG@20 selects learning rate; seeds 1/2 repeat that rate with validation checkpoints. No test selection; no teacher or sparse-input augmentation.'}
    write_json(output / 'selection.json', frozen)
    selection_hash = file_hash(output / 'selection.json')
    write_json(output / 'status.json', {'stage': 'frozen_test_evaluation'})
    per_query = {'BundleFuse': records}
    methods = {'BundleFuse': original['methods']['BundleFuse']}
    catalogues, encoding = {}, {}
    for name, model in models.items():
        started = time.perf_counter(); catalogues[name] = model.catalogue()
        encoding[name] = time.perf_counter() - started
        methods[name], per_query[name] = model_metrics(model, test, catalogues[name], chunk, True)
        print(f'TEST {dataset} {name}: {methods[name]}', flush=True)
        write_json(output / 'test_in_progress.json', {'methods': methods})
    if file_hash(output / 'selection.json') != selection_hash:
        raise RuntimeError('Selections changed during test scoring')
    reference_score, optimized_score, _ = frozen_scorers(scorer, selected, graph.freq)
    profiles = {'BundleFuse_reference': reference_score, 'BundleFuse_optimized': optimized_score}
    for name in ('Raw_s0', 'Engineered_s0'):
        profiles[name] = lambda seeds, name=name: models[name].score(seeds, catalogues[name])
    timing = {'warm_cache': profile_scorers(profiles, test), 'preprocessing_seconds': preprocessing_seconds,
              'feature_construction_seconds': feature_record['seconds'], 'training_and_validation_seconds': training_seconds,
              'catalogue_encoding_seconds': encoding,
              'serving_catalogue_vector_bytes': {name: cat.numel() * cat.element_size() for name, cat in catalogues.items()},
              'worker_peak_rss_bytes': peak_rss_bytes(),
              'scope': 'One-thread CPU; 32-query warm batches, three repeats; full catalogue scoring. Ranking/metric cost separate; external embedding extraction excluded. RSS is whole-worker.'}
    write_json(output / 'timing.json', timing)
    result = {'dataset': dataset, 'protocol': protocol, 'test_query_ids': test_ids,
              'query_groups': original['query_groups'], 'tie_policy': TIE_POLICY, 'parity': parity,
              'selection_sha256': selection_hash, 'selected': frozen,
              'methods': methods, 'per_query': per_query, 'features': feature_record, 'timing': timing}
    write_json(output / 'result.json', result, compact=True)
    write_json(output / 'status.json', {'stage': 'complete'})
    print(f'COMPLETE DATASET {dataset}; validation feasibility gate={gate["pass_this_dataset"]}', flush=True)
