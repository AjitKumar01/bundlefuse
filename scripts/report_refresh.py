"""Verify paired artifacts and render the corrected selection study after a full refresh."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.experiments import code_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='results/evaluation_refresh')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = Path(args.run)
    manifest = json.loads((run / 'manifest.json').read_text())
    if any(job['status'] != 'complete' for job in manifest['jobs']):
        raise RuntimeError('Cannot verify an incomplete run')
    if code_digest() != manifest['code_sha256']:
        raise RuntimeError('Current experiment source differs from the recorded source')
    summary = json.loads((run / 'summary.json').read_text())
    comparisons = 0
    for ds, matrix in summary['matrix'].items():
        baseline = summary['baselines'].get(ds)
        if not baseline:
            continue
        lookup = {record['seed']: record for record in baseline['per_seed']}
        for record in matrix['per_seed']:
            other = lookup[record['seed']]
            for name in ('SimpleHybrid', 'HardSwitchHybrid'):
                for key in ('R@10', 'N@10', 'R@20', 'N@20'):
                    if abs(record['methods'][name][key] - other['methods'][name][key]) > 1e-10:
                        raise RuntimeError(f'Unpaired hybrid result: {ds}, {record["seed"]}, {name}, {key}')
                    comparisons += 1
    canonical = json.loads((root / 'results/matrix_zeroparam.json').read_text())
    for row in csv.DictReader((root / 'results/matrix_zeroparam.csv').open()):
        values = canonical[row['dataset']][row['method']]
        for index, key in enumerate(('R@10', 'N@10', 'R@20', 'N@20')):
            if abs(float(row[key]) - values[index][0]) > 1e-7:
                raise RuntimeError('Canonical CSV/JSON mean mismatch')
        if abs(float(row['R@20_std']) - values[2][1]) > 1e-7:
            raise RuntimeError('Canonical CSV/JSON dispersion mismatch')
    datasets = ('pog', 'electronic', 'dunnhumby')
    lines = ['# Corrected selection-objective study', '',
             'Single split seed 0 for each dataset; POG uses 1,500 test queries. These are sensitivity',
             'probes, not multi-seed estimates or evidence that selection cannot overfit. MRR uses the',
             'first relevant item; full NDCG assigns distinct positions to tied items. All objectives',
             'share the same deterministic item-ID tie rule. Values below are test Recall@20.', '',
             '| Validation objective | POG | Electronic | Dunnhumby |', '|---|---:|---:|---:|']
    for objective in ('ndcg10', 'ndcg20', 'ndcg50', 'recall20', 'mrr', 'ndcgfull', 'nll'):
        values = [summary['selection'].get(ds, {}).get(objective, {}).get('R@20') for ds in datasets]
        lines.append('| ' + objective + ' | ' + ' | '.join('not run' if x is None else f'{x:.4f}' for x in values) + ' |')
    lines += ['', 'The default remains NDCG@20 on held-out validation. Objective sensitivity is reported',
              'directly; no blanket plateau or no-overfitting conclusion is inferred. Older MRR/full-NDCG',
              'tables were generated with incorrect definitions and are superseded.', '']
    (run / 'SELECTION.md').write_text('\n'.join(lines))
    print(f'Verified {len(manifest["jobs"])} completed jobs, matching source fingerprint,')
    print(f'{comparisons} paired hybrid metric comparisons, and canonical CSV/JSON consistency.')
    print('Wrote', run / 'SELECTION.md')


if __name__ == '__main__':
    main()
