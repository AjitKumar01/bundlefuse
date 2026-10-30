"""Retuned component ablations and raw hybrid controls on supplied splits."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from bundlefuse.artifacts import flat_metrics, write_json
from bundlefuse.eval.provided import DiskSignals
from bundlefuse.fusion_selection import ValidationSignals, fit_fusion, apply_fusion
from bundlefuse.metrics import as_numpy, ranking_metrics, TIE_POLICY


ABLATIONS = {
    'BundleFuse': {},
    'NoIncidence': {'incidence': (0.0,)},
    'NoRao': {'rao': (0.0,)},
    'NoHigherOrder': {'incidence': (0.0,), 'rao': (0.0,)},
    'NoShrinkage': {'shrinkage': False},
    'NoBridge': {},
}
CONTROL_KEYS = ['raw_cooc', 'raw_content', 'raw_category', 'raw_popularity']
METHODS = tuple(ABLATIONS) + ('SimpleHybrid', 'HardSwitchHybrid')


def apply_method(z, selected, freq):
    if selected['kind'] == 'fusion':
        return apply_fusion(z, selected['weights'], selected['prior_keys'],
                            (selected['kappa'], selected['prior_scale']), freq,
                            (selected['alpha_inc'], selected['gamma_rao']))
    weights = selected['weights']
    base = sum(value * z[key] for key, value in weights.items())
    if selected['gate'] is None:
        return base
    threshold, scale = selected['gate']
    prior = sum((weights.get(key, 0) or 0.5) * z[key] for key in ('raw_content', 'raw_category'))
    return torch.where(freq <= threshold, scale * prior, base)


def select_controls(validation, cache, freq):
    weights, _, _, _, _ = fit_fusion(validation, CONTROL_KEYS, ('raw_content', 'raw_category'),
                                    incidence=(0.0,), rao=(0.0,), shrinkage=False)
    baseline = {'kind': 'hybrid', 'weights': weights, 'gate': None}
    gates = [None] + [(threshold, scale) for threshold in (0, 1, 5) for scale in (0.5, 1.0, 2.0)]
    totals = np.zeros(len(gates))
    for start, z in cache.blocks():
        count = next(iter(z.values())).shape[1]
        for i, gate in enumerate(gates):
            score = apply_method(z, dict(baseline, gate=gate), freq)
            totals[i] += ranking_metrics(score, validation.pairs[start:start + count], Ks=(20,))[20][1] * count
    values = totals / len(validation.pairs)
    best, index = values[0], 0
    for i in range(1, len(gates)):
        if values[i] > best + 2e-4:
            best, index = values[i], i
    return {'SimpleHybrid': dict(baseline, val_ndcg20=float(values[0])),
            'HardSwitchHybrid': dict(baseline, gate=gates[index], val_ndcg20=float(best))}


def selected_metrics(blocks, pairs, selected, freq, per_query=False):
    """Score identical fitted configurations once; retain every original query's multiplicity."""
    totals = {name: {k: np.zeros(2) for k in (10, 20, 50)} for name in selected}
    records = {name: [] for name in selected}
    groups = {}
    for name, config in selected.items():
        signature = {k: v for k, v in config.items() if k not in ('variance_components', 'val_ndcg20', 'constraints')}
        groups.setdefault(json.dumps(signature, sort_keys=True), []).append(name)
    for start, z in blocks():
        count = next(iter(z.values())).shape[1]
        batch = pairs[start:start + count]
        for names in groups.values():
            scores = apply_method(z, selected[names[0]], freq)
            if per_query:
                array = as_numpy(scores)
                query_results = [ranking_metrics(array[:, j:j + 1], [pair], Ks=(10, 20, 50))
                                 for j, pair in enumerate(batch)]
                for name in names:
                    records[name].extend(flat_metrics(result) for result in query_results)
                    for k in totals[name]:
                        totals[name][k] += np.sum([result[k] for result in query_results], axis=0)
            else:
                result = ranking_metrics(scores, batch, Ks=(10, 20, 50))
                for name in names:
                    for k in totals[name]:
                        totals[name][k] += np.asarray(result[k]) * count
        print(f'{"test" if per_query else "validation"} evaluation {start + count}/{len(pairs)}', flush=True)
    return ({name: flat_metrics({k: value / len(pairs) for k, value in result.items()})
             for name, result in totals.items()}, records)


def query_groups(pairs):
    """Group repeated complete bundles for cluster bootstrap / paired sign flips."""
    groups = {}
    return [groups.setdefault(tuple(sorted(set(seeds) | set(truth))), len(groups)) for seeds, truth in pairs]


def run_components(dataset, splits, scorer, signals, freq, music, objective, output, hybrid):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    chunk = int(os.environ.get('PROVIDED_CHUNK', '64'))
    if chunk < 1:
        raise ValueError('PROVIDED_CHUNK must be positive')
    valid, test = splits.pairs('valid'), splits.pairs('test')
    protocol = splits.manifest()
    write_json(output / 'protocol.json', protocol)

    def combined(seeds):
        raw, normalized = scorer(seeds)
        normalized.update({'raw_' + key: torch.as_tensor(value) for key, value in hybrid.signals(seeds).items()})
        return raw, normalized

    keys = signals + ['pair', 'diff2_pair', 'rao'] + CONTROL_KEYS
    cache = DiskSignals(output / 'validation_cache', valid, combined, keys, len(freq), chunk)
    frequency = torch.as_tensor(freq)[:, None]
    selected = {}
    try:
        validation = ValidationSignals(cache.blocks, valid, freq, objective,
                                       progress=lambda text: print(text, flush=True))
        prior_candidates = (('cat', 'homo', 'audiostyle', 'bridge', 'contrast') if music else
                            ('cat', 'softcat', 'style', 'imgstyle', 'bridge', 'contrast'))
        for name, constraints in ABLATIONS.items():
            available = [key for key in signals if name != 'NoBridge' or key != 'bridge']
            print(f'SELECT {dataset} {name}: validation queries={len(valid)} constraints={constraints}', flush=True)
            weights, prior, eb, ho, variance = fit_fusion(validation, available, prior_candidates, **constraints)
            selected[name] = {'kind': 'fusion', 'weights': weights, 'prior_keys': prior,
                              'kappa': eb[0], 'prior_scale': eb[1], 'alpha_inc': ho[0],
                              'gamma_rao': ho[1], 'variance_components': variance,
                              'constraints': dict(constraints, remove_bridge=name == 'NoBridge')}
            write_json(output / 'selection.json', selected)
        print(f'SELECT {dataset} SimpleHybrid / HardSwitchHybrid', flush=True)
        selected.update(select_controls(validation, cache, frequency))
        # Freeze every method before evaluating any test query.
        write_json(output / 'selection.json', selected)
        val_metrics, _ = selected_metrics(cache.blocks, valid, selected, frequency)
    finally:
        cache.close()

    def test_blocks():
        for start in range(0, len(test), chunk):
            raw, z = combined([s for s, _ in test[start:start + chunk]])
            del raw
            yield start, z

    metrics, records = selected_metrics(test_blocks, test, selected, frequency, per_query=True)
    result = {'dataset': dataset, 'protocol': protocol, 'selection_objective': objective,
              'tie_policy': TIE_POLICY, 'selected': selected, 'validation': val_metrics,
              'methods': metrics, 'per_query': records, 'test_query_ids': sorted(splits.splits['test'][0]),
              'query_groups': query_groups(test), 'grouping': 'identical complete held-out bundle item sets',
              'seconds': time.time() - started}
    write_json(output / 'result.json', result)
    print(f'COMPLETE {dataset}: ' + json.dumps({name: value['R@20'] for name, value in metrics.items()}), flush=True)
