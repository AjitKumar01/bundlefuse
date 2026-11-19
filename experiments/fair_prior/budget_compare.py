"""Training-set size with fair baselines: TRIAD against the best simple baseline of each condition, where the five
main simple methods may also use the frequency prior (experiments.fair_prior.baselines, weight chosen on validation).

Conditions and TRIAD runs as in experiments.bundle_budget_retail.compare (e-commerce, Spotify) and
experiments.bundle_budget.compare (POG, POG_DENSE: 10%, 1%, users' histories as pseudo-bundles). The best simple
baseline is chosen on validation NDCG@20 over the condition's published pool and the prior reruns. Clean test queries,
paired cluster bootstrap and sign flips, Holm within each family over all its tests (2 metrics x datasets x
conditions). Writes results/fair_prior/budget_comparison.json.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.fair_prior.budget_compare
"""
import triad  # pins BLAS threads before numpy loads
import json
from pathlib import Path
import numpy as np
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS, groups
from experiments.mechanisms.statistics import paired_comparison, holm_adjust
from experiments.bundle_budget_retail import conditions as RC
from experiments.bundle_budget_retail.train import RUNS as RETAIL_RUNS
from experiments.bundle_budget import conditions as PC
from experiments.bundle_budget.train import RUNS as POG_RUNS
from .baselines import OUT

FAMILIES = dict(ecommerce=[(n, c) for n in ('electronic', 'food', 'clothing') for c in ('full', 'n600', 'n300', 'n100', 'n50')],
                music=[(n, c) for n in ('spotify_sparse', 'spotify') for c in ('full', 'n3000', 'n1000', 'n600', 'n300', 'n100', 'n50')],
                fashion=[(n, c) for n in ('pog_dense', 'pog') for c in ('full', 'frac10', 'frac1', 'pseudo', 'pseudo_clean')])


def triad_runs(name, cond):
    root = RUNS/name if cond == 'full' else (POG_RUNS if name.startswith('pog') else RETAIL_RUNS)/name/cond
    return np.mean([[[q['recall20'], q['ndcg20']] for q in json.loads((root/f'seed{s}'/'test.json').read_text())['queries']] for s in (0, 1, 2)], 0)


def pool(name, cond, n):
    """method -> (validation NDCG@20, clean per-query [R, N]), published runs and prior reruns."""
    if cond == 'full':
        c = json.loads((OUT/'comparison.json').read_text())['datasets'][name]['methods']
        full = json.loads((OUT/f'{name}.json').read_text())['methods']
        out = {k + '+prior': (v['validation_ndcg20'], np.asarray(v['queries'])) for k, v in full.items()}
        from .compare import published
        ds = (PC if name.startswith('pog') else RC).dataset(name)
        out.update({k: v for k, v in published(name, sorted(ds.clean_test)).items()});return out
    root = Path('results/bundle_budget' if name.startswith('pog') else 'results/bundle_budget_retail')/'baselines'/name/f'{cond}.json'
    out = {k: (v['validation_ndcg20'], np.asarray(v['queries'])) for k, v in json.loads(root.read_text())['methods'].items()}
    fair = OUT/f'{name}_{cond}.json'
    if fair.exists():out.update({k + '+prior': (v['validation_ndcg20'], np.asarray(v['queries'])) for k, v in json.loads(fair.read_text())['methods'].items()})
    else:print('missing', fair)
    return out


def main():
    report = dict(note=__doc__, families={});missing = []
    for fam, jobs in FAMILIES.items():
        recs = [];rows = {}
        for name, cond in jobs:
            ds = (PC if name.startswith('pog') else RC).dataset(name);keep = sorted(ds.clean_test);cl = [groups(ds.test_pairs)[i] for i in keep]
            t = triad_runs(name, cond)[keep];p = pool(name, cond, len(keep))
            if cond != 'full' and not (OUT/f'{name}_{cond}.json').exists():missing.append((name, cond))
            best = max(p, key=lambda k: p[k][0]);b = p[best][1];assert len(b) == len(t), (name, cond)
            rows[f'{name}:{cond}'] = dict(triad=t.mean(0).tolist(), best_simple=best, best_simple_test=b.mean(0).tolist())
            for j, metric in ((1, 'ndcg20'), (0, 'recall20')):
                recs.append(dict(paired_comparison(t[:, j], b[:, j], cl, bootstrap=2000, permutations=10000, seed=20261118), dataset=name, condition=cond, metric=metric, baseline=best))
        holm_adjust(recs, len(recs));report['families'][fam] = dict(rows=rows, tests=recs)
        for r in recs:
            if r['metric'] == 'ndcg20':print(f"{fam:9s} {r['dataset']:15s} {r['condition']:12s} {r['baseline']:24s} {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p_holm {r['p_holm']:.4f}")
    report['missing_prior_reruns'] = missing;write_json(OUT/'budget_comparison.json', report)


if __name__ == '__main__':
    main()
