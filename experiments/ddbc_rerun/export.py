"""Inputs for rerunning DDBC (Tu et al., ICLR 2026) with its released checkpoints, on our DDBC-protocol queries.

DDBC's rec_eval reads <dataset>/test.txt, keeps bundles with at least k items, truncates each to its first k items and
splits them in half (seeds, targets). Our DDBC-protocol queries (experiments/ddbc_protocol) are exactly those windows,
taken from DDBC's released test_input_<k>/test_gt_<k> files. This writes one test.txt per case holding those queries
(bundle id, seeds, targets), in our query order (sorted bundle id), so DDBC's outputs line up with ours row by row.

PYTHONPATH=. .venv/bin/python -m experiments.ddbc_rerun.export --out <dir>
"""
import argparse
from pathlib import Path
from experiments.ddbc_protocol.run import DDBC, CASES, rows


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--out', required=True);out = Path(p.parse_args().out)
    for name, folder, k in CASES:
        seeds, targets = rows(DDBC/folder/f'test_input_{k}.txt'), rows(DDBC/folder/f'test_gt_{k}.txt')
        lines = [', '.join(map(str, [b] + seeds[b] + targets[b])) for b in sorted(seeds)]
        assert all(len(seeds[b]) + len(targets[b]) == k and len(seeds[b]) == k//2 for b in seeds)
        (out/f'{name}_k{k}').mkdir(parents=True, exist_ok=True);(out/f'{name}_k{k}'/'test.txt').write_text('\n'.join(lines) + '\n')
        print(name, k, len(lines), 'queries')


if __name__ == '__main__':
    main()
