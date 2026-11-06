"""Summarize paired test-query outcomes from the completed mechanism studies."""
import json
from pathlib import Path

import numpy as np


def summarize(dataset):
    root = Path('results/publication_mechanisms') / dataset
    data = json.loads((root / 'queries.json').read_text())['metrics']
    metrics = {key: np.asarray(data[key], dtype=float)
               for key in ('history', 'shared', 'combined', 'full')}
    history, shared, combined = (metrics[key][:, 0] > 0
                                 for key in ('history', 'shared', 'combined'))
    output = {'queries': len(history), 'recall_and_ndcg': {
        key: value.mean(axis=0).tolist() for key, value in metrics.items()},
        'any_hit': {key: int((value[:, 0] > 0).sum())
                    for key, value in metrics.items()},
        'history_only_hit': int((history & ~shared).sum()),
        'shared_only_hit': int((shared & ~history).sum()),
        'both_hit': int((history & shared).sum()),
        'neither_hit': int((~history & ~shared).sum()),
        'combined_hit_when_neither_hit': int((combined & ~history & ~shared).sum())}
    for key in ('history', 'shared'):
        delta = metrics['combined'][:, 0] - metrics[key][:, 0]
        output['combined_vs_' + key] = {
            'better': int((delta > 1e-9).sum()),
            'worse': int((delta < -1e-9).sum()),
            'equal': int((np.abs(delta) <= 1e-9).sum())}
    return output


if __name__ == '__main__':
    result = {dataset: summarize(dataset) for dataset in ('pog', 'pog_dense')}
    destination = Path('results/algebraic_diagnostics/evidence_overlap.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
