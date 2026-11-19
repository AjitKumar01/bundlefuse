"""Sensitivity sweep: each setting against the frozen configuration, on validation (the decision) and test (reported).

Runs: results/dials_sensitivity/<dataset>/<setting>/seed<s>/ (sweep.py; Kaggle outputs copied here).
Reference: the sweep's own 'ref' runs where present (retail, seeds 0-2); otherwise the frozen runs with the same seeds
(../dials/runs). The low-rank ablation (results/dials_ablations/lowrank) is shown alongside, as "label rank 0".
Per setting: validation NDCG@20 and test NDCG@20 / Recall@20 averaged over the seeds present; test deltas against
the reference with paired cluster bootstrap and sign flips on clean test queries (clusters: identical complete
bundles), Holm over every (dataset, setting) test of NDCG@20.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_sensitivity.compare
"""
import triad  # pins BLAS threads before numpy loads
import json
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust

OUT = Path('results/dials_sensitivity');FROZEN = Path(DATA_ROOT)/'dials'/'runs';ABLATION = Path('results/dials_ablations/lowrank')
CACHE = Path(DATA_ROOT)/'dials'/'cache'
SETTINGS = ['label_rank4', 'label_rank64', 'label_rank128', 'width4', 'width64', 'rank32', 'rank128', 'hidden16', 'hidden64', 'fastlr3e-3', 'fastlr3e-2']


def finished(root):
    return sorted(r for r in Path(root).glob('seed*') if (r/'selected.json').exists() and (r/'test.json').exists())


def summarize(runs):
    val = np.mean([json.loads((r/'selected.json').read_text())['validation']['ndcg20'] for r in runs])
    q = np.mean([[[x['recall20'], x['ndcg20']] for x in json.loads((r/'test.json').read_text())['queries']] for r in runs], 0)
    return val, q


def main():
    report = dict(note=__doc__, datasets={});tests = []
    for name in ('electronic', 'food', 'clothing', 'pog_dense', 'spotify_sparse'):
        if not (OUT/name).exists():continue
        ref_runs = finished(OUT/name/'ref')
        seeds = [r.name for r in ref_runs] if ref_runs else None
        if not ref_runs:  # big datasets: the frozen run with the seeds the sweep used
            seeds = sorted({r.name for s in SETTINGS for r in finished(OUT/name/s)});ref_runs = [FROZEN/name/s for s in seeds]
        if not seeds:continue
        ds = data.load(name, CACHE);keep = np.asarray(ds.clean_test)
        clusters = [str(tuple(sorted(set(s) | set(g)))) for s, g in (ds.test_pairs[i] for i in keep)]
        rv, rq = summarize(ref_runs);rows = {'frozen': dict(seeds=seeds, validation_ndcg20=rv, test_ndcg20=rq[keep, 1].mean(), test_recall20=rq[keep, 0].mean())}
        extra = {'label_rank0 (low-rank off)': [ABLATION/name/s for s in seeds]}
        for s in SETTINGS:extra[s] = finished(OUT/name/s)
        for s, runs in extra.items():
            runs = [r for r in runs if (r/'selected.json').exists() and (r/'test.json').exists()]
            if [r.name for r in runs] != seeds:
                if runs:rows[s] = f'seeds {[r.name for r in runs]} of {seeds}'
                continue
            v, q = summarize(runs);t = paired_comparison(q[keep, 1], rq[keep, 1], clusters, bootstrap=2000, permutations=10000, seed=20261117)
            rows[s] = dict(validation_ndcg20=v, validation_delta=v - rv, test_ndcg20=q[keep, 1].mean(), test_recall20=q[keep, 0].mean())
            tests.append(dict(t, dataset=name, setting=s))
        report['datasets'][name] = rows
    holm_adjust(tests, len(tests));report['tests'] = tests
    for t in tests:report['datasets'][t['dataset']][t['setting']].update(test_delta=t['delta'], ci95=t['ci95'], p_holm=t['p_holm'])
    write_json(OUT/'comparison.json', report)
    for name, rows in report['datasets'].items():
        f = rows['frozen'];print(f"\n{name} (seeds {f['seeds']}): frozen validation {f['validation_ndcg20']:.4f}, test NDCG {f['test_ndcg20']:.4f}")
        for s, r in rows.items():
            if s == 'frozen':continue
            if isinstance(r, str):print(f'  {s:28s} {r}');continue
            print(f"  {s:28s} val {r['validation_ndcg20']:.4f} ({r['validation_delta']:+.4f})  test {r['test_ndcg20']:.4f} ({r['test_delta']:+.4f} "
                  f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}], p_holm {r['p_holm']:.3f})")


if __name__ == '__main__':
    main()
