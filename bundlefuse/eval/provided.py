"""Full-population evaluation on provided neural benchmark splits, with bounded RAM."""
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from bundlefuse.artifacts import write_json, flat_metrics
from bundlefuse.fusion_selection import ValidationSignals, fit_fusion, apply_fusion
from bundlefuse.metrics import ranking_metrics, as_numpy, TIE_POLICY


class DiskSignals:
    """Query-major float32 cache; no quantization, query sampling, or score approximation."""
    def __init__(self, directory, pairs, scorer, keys, n_items, chunk):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.pairs, self.chunk = pairs, chunk
        self.arrays = {key: np.lib.format.open_memmap(self.directory / f'{key}.npy', mode='w+',
                       dtype=np.float32, shape=(len(pairs), n_items)) for key in dict.fromkeys(keys)}
        for start in range(0, len(pairs), chunk):
            raw, normalized = scorer([s for s, _ in pairs[start:start + chunk]])
            for key, array in self.arrays.items():
                array[start:start + chunk] = as_numpy(normalized[key]).T
            del raw, normalized
            print(f'validation cache {min(start + chunk, len(pairs))}/{len(pairs)}', flush=True)
        for array in self.arrays.values():
            array.flush()

    def blocks(self):
        for start in range(0, len(self.pairs), self.chunk):
            yield start, {key: torch.from_numpy(array[start:start + self.chunk].T)
                          for key, array in self.arrays.items()}

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()
        shutil.rmtree(self.directory)


def run_provided(dataset, splits, scorer, signals, freq, music, objective, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    chunk = int(os.environ.get('PROVIDED_CHUNK', '64'))
    if chunk < 1:
        raise ValueError('PROVIDED_CHUNK must be positive')
    started = time.time()
    valid, test = splits.pairs('valid'), splits.pairs('test')
    protocol = splits.manifest()
    write_json(output / 'protocol.json', protocol)
    cache = DiskSignals(output / 'validation_cache', valid, scorer,
                        signals + ['pair', 'diff2_pair', 'rao'], len(freq), chunk)
    try:
        print(f'{dataset}: selecting fusion on ALL {len(valid)} validation queries', flush=True)
        validation = ValidationSignals(cache.blocks, valid, freq, objective,
                                       progress=lambda line: print(line, flush=True))
        candidates = (('cat', 'homo', 'audiostyle', 'bridge', 'contrast') if music else
                      ('cat', 'softcat', 'style', 'imgstyle', 'bridge', 'contrast'))
        w, priors, backoff, ho, variance = fit_fusion(validation, signals, candidates)
        selected = {'weights': w, 'prior_keys': priors, 'kappa': backoff[0],
                    'prior_scale': backoff[1], 'alpha_inc': ho[0], 'gamma_rao': ho[1],
                    'variance_components': variance}
        write_json(output / 'selection.json', selected)
        print(f'{dataset}: selected {selected}', flush=True)
        totals = {k: np.zeros(2) for k in (10, 20, 50)}
        for start, z in cache.blocks():
            count = next(iter(z.values())).shape[1]
            fused = apply_fusion(z, w, priors, backoff, torch.as_tensor(freq)[:, None], ho)
            result = ranking_metrics(fused, valid[start:start + count], Ks=(10, 20, 50))
            for k in totals:
                totals[k] += np.asarray(result[k]) * count
        valid_metrics = flat_metrics({k: v / len(valid) for k, v in totals.items()})
    finally:
        cache.close()
    totals = {k: np.zeros(2) for k in (10, 20, 50)}
    per_query = []
    for start in range(0, len(test), chunk):
        batch = test[start:start + chunk]
        raw, z = scorer([s for s, _ in batch])
        first = next(iter(z.values()))
        fused = apply_fusion(z, w, priors, backoff, torch.as_tensor(freq, device=first.device)[:, None], ho)
        scores = as_numpy(fused)
        for j, pair in enumerate(batch):
            result = ranking_metrics(scores[:, j:j + 1], [pair], Ks=(10, 20, 50))
            per_query.append(flat_metrics(result))
            for k in totals:
                totals[k] += result[k]
        del raw, z, fused, scores
        print(f'test {min(start + chunk, len(test))}/{len(test)}', flush=True)
    result = {'dataset': dataset, 'protocol': protocol, 'selection_objective': objective,
              'tie_policy': TIE_POLICY, 'selected': selected, 'validation': valid_metrics,
              'methods': {'BundleFuse': flat_metrics({k: v / len(test) for k, v in totals.items()})},
              'test_query_ids': sorted(splits.splits['test'][0]), 'per_query': per_query,
              'seconds': time.time() - started}
    write_json(output / 'result.json', result)
    print(f'{dataset}: COMPLETE {result["methods"]}', flush=True)
