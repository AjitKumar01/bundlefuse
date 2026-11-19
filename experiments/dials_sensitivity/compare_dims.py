"""Feature-size check on Electronic, Food and Clothing (BLIP title features): d = 64, 128, 256 or 768 (no SVD), each
at content rank 64 (frozen) and 128, seeds 0-2. d = 128 comes from the sensitivity sweep (results/dials_sensitivity
<dataset>/{ref,rank128}); the others from results/dials_feature_size/<dataset>_d<d>/{ref,rank128}.
Validation NDCG@20 (the decision) and test NDCG@20, seed means; test deltas against the frozen d = 128, rank 64, with
paired cluster bootstrap and sign flips (clusters: identical complete bundles), Holm over all 21 tests.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_sensitivity.compare_dims
"""
import triad  # pins BLAS threads before numpy loads
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust
from .compare import finished, summarize

SWEEP = Path('results/dials_sensitivity');DIMS = Path('results/dials_feature_size');CACHE = Path(DATA_ROOT)/'dials'/'cache'


def root(name, d, setting):return (SWEEP/name if d == 128 else DIMS/f'{name}_d{d}')/setting


def main():
    report = dict(note=__doc__, datasets={});tests = []
    for name in ('electronic', 'food', 'clothing'):
        ds = data.load(name, CACHE);clusters = [str(tuple(sorted(set(s) | set(g)))) for s, g in ds.test_pairs]
        rv, rq = summarize(finished(root(name, 128, 'ref')));rows = report['datasets'][name] = {}
        for d in (64, 128, 256, 768):
            for setting, rank in (('ref', 64), ('rank128', 128)):
                runs = finished(root(name, d, setting))
                if len(runs) != 3:rows[f'd{d}_r{rank}'] = f'{len(runs)} of 3 seeds';continue
                v, q = summarize(runs);row = rows[f'd{d}_r{rank}'] = dict(validation_ndcg20=v, validation_delta=v - rv, test_ndcg20=q[:, 1].mean())
                if (d, rank) != (128, 64):
                    tests.append(dict(paired_comparison(q[:, 1], rq[:, 1], clusters, bootstrap=2000, permutations=10000, seed=20261117), dataset=name, cell=f'd{d}_r{rank}'))
    holm_adjust(tests, len(tests))
    for t in tests:report['datasets'][t['dataset']][t['cell']].update(test_delta=t['delta'], ci95=t['ci95'], p_holm=t['p_holm'])
    report['tests'] = tests;write_json(DIMS/'comparison.json', report)
    for name, rows in report['datasets'].items():
        print(name)
        for cell, r in rows.items():
            if isinstance(r, str):print(f'  {cell:10s} {r}');continue
            extra = f"  test delta {r['test_delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p_holm {r['p_holm']:.3f}" if 'test_delta' in r else ''
            print(f"  {cell:10s} val {r['validation_ndcg20']:.4f} ({r['validation_delta']:+.4f})  test {r['test_ndcg20']:.4f}{extra}")


if __name__ == '__main__':
    main()
