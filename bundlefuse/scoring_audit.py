"""Exhaustive CPU score-bit and historical-metric parity for frozen BundleFuse settings."""
import hashlib
import time

import numpy as np
import torch

from .artifacts import write_json, flat_metrics
from .eval.components import apply_method
from .fusion_selection import required_fusion_signals
from .metrics import as_numpy, ranking_metrics


def frozen_scorers(scorer, selected, frequency):
    required = required_fusion_signals(selected['weights'], selected['prior_keys'],
                                      (selected['kappa'], selected['prior_scale']),
                                      (selected['alpha_inc'], selected['gamma_rao']))
    freq = torch.as_tensor(frequency)[:, None]
    def score(seeds, optimized):
        raw, z = scorer(seeds, required=required) if optimized else scorer(seeds)
        del raw
        return apply_method(z, selected, freq)
    return (lambda seeds: score(seeds, False)), (lambda seeds: score(seeds, True)), required


def audit_selected_scores(scorer, selected, frequency, pairs, reference_rows, output, chunk=64):
    if len(pairs) != len(reference_rows):
        raise RuntimeError('Historical per-query record count differs')
    reference_score, selected_score, required = frozen_scorers(scorer, selected, frequency)
    records, original_hash, selected_hash = [], hashlib.sha256(), hashlib.sha256()
    started = time.perf_counter()
    for start in range(0, len(pairs), chunk):
        batch = pairs[start:start + chunk]
        seeds = [s for s, _ in batch]
        old = as_numpy(reference_score(seeds)).copy()
        new = as_numpy(selected_score(seeds))
        if not np.isfinite(old).all() or not np.isfinite(new).all():
            raise RuntimeError('Nonfinite fused scores')
        if not np.array_equal(old.view(np.uint32), new.view(np.uint32)):
            raise RuntimeError(f'Optimized score-bit parity failed at query {start}; max error={np.max(np.abs(old - new))}')
        original_hash.update(old.tobytes()); selected_hash.update(new.tobytes())
        for j, pair in enumerate(batch):
            row = flat_metrics(ranking_metrics(new[:, j:j + 1], [pair], Ks=(10, 20, 50)))
            historical = reference_rows[start + j]
            if set(row) != set(historical) or any(abs(row[k] - historical[k]) > 1e-12 for k in row):
                raise RuntimeError(f'Historical BundleFuse per-query metrics changed at query {start + j}')
            records.append(row)
        print(f'PARITY {start + len(batch)}/{len(pairs)} bit-identical fused scores and historical metrics', flush=True)
        write_json(output, {'status': 'running', 'queries_verified': len(records), 'queries_total': len(pairs),
                            'required_signals': sorted(required)})
    result = {'status': 'complete', 'queries_verified': len(records), 'queries_total': len(pairs),
              'required_signals': sorted(required), 'fused_score_sha256_reference': original_hash.hexdigest(),
              'fused_score_sha256_optimized': selected_hash.hexdigest(),
              'score_bit_identity': True, 'historical_metrics_match': True,
              'seconds': time.perf_counter() - started,
              'scope': 'Every candidate score on every original supplied test query, same CPU/chunking. Equal score bits guarantee identical deterministic rankings; no approximation.'}
    write_json(output, result)
    return records, result
