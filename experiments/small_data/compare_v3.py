"""Compare the three parameter-suggested changes (experiments.small_data.triad_v3) with the current full-rank TRIAD (v2)
on full training data: test NDCG@20 on clean test queries (mean over seeds 0-2, per query), mean validation NDCG@20,
and paired tests (bootstrap + sign flips, one cluster per query) of each variant minus v2, Holm over the datasets
within each variant. Writes results/small_data/comparison_v3.json.

cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python -m experiments.small_data.compare_v3
"""
import json
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust
from experiments.small_data.readings import CACHE

R = Path('results/small_data');NAMES = ('electronic', 'food', 'clothing', 'pog_dense', 'pog')
VARIANTS = ('v3_prior', 'v3_taste', 'v3_oneexp', 'v3_all')


def load(kind, name):
    p = R/kind/f'{name}_full.json'
    if not p.exists():return None
    r = json.loads(p.read_text());return np.array(r['queries'])[:, 1], float(np.mean([x['validation'] for x in r['runs']])), [x['best_epoch'] for x in r['runs']]


def main():
    rows, tests = {}, {v: [] for v in VARIANTS}
    for name in NAMES:
        base = load('v2', name)
        if base is None:continue
        clean = np.array(data.load(name, CACHE).clean_test);b = base[0][clean]
        rows[name] = dict(v2=dict(test=float(b.mean()), validation=base[1], epochs=base[2]))
        for v in VARIANTS:
            r = load(v, name)
            if r is None:continue
            x = r[0][clean];rows[name][v] = dict(test=float(x.mean()), validation=r[1], epochs=r[2])
            tests[v].append(dict(paired_comparison(x, b, [str(i) for i in range(len(b))], bootstrap=2000, permutations=10000, seed=20261121), dataset=name))
    for v in tests:holm_adjust(tests[v], len(tests[v]))
    write_json(R/'comparison_v3.json', dict(note=__doc__, rows=rows, tests=tests))
    by = {v: {t['dataset']: t for t in ts} for v, ts in tests.items()}
    print(f"{'dataset':11s} {'v2':>7s} " + ' '.join(f'{v:>18s}' for v in VARIANTS))
    for name, r in rows.items():
        cells = []
        for v in VARIANTS:
            if v not in r:cells.append(f"{'-':>18s}");continue
            t = by[v][name];d = r[v]['test'] - r['v2']['test']
            cells.append(f"{r[v]['test']:.4f} {d:+.4f}{'*' if t['p_holm'] < .05 else ' '}".rjust(18))
        print(f"{name:11s} {r['v2']['test']:.4f} " + ' '.join(cells))
    print('test NDCG@20, clean queries; * : variant - v2 significant after Holm over datasets')
    print('validation NDCG@20:');[print(f"{n:11s} " + ' '.join(f"{k} {x['validation']:.4f} ep{x['epochs']}" for k, x in r.items())) for n, r in rows.items()]


if __name__ == '__main__':
    main()
