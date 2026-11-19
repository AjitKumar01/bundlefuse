"""TRIAD against simple baselines that get the same frequency prior (experiments.fair_prior.baselines).

For each dataset the best simple baseline is chosen on validation NDCG@20 over the whole published pool (all
methods, without the prior) and the five methods rerun with the prior (lambda = 0 included). Two families, each with
Holm over its 14 tests (7 datasets x NDCG@20, Recall@20), paired cluster bootstrap and sign flips as in the paper
(clusters: identical complete bundles), clean test queries:
  triad_vs_best_simple_prior            TRIAD (seeds 0-2) minus the best simple baseline with the prior
  triad_no_bonus_vs_best_simple_prior   TRIAD without the unseen-item bonus (dials/paper/figdata/unseen_bonus.py)
Also reports every method's clean test R / N with and without the prior. Writes results/fair_prior/comparison.json.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.fair_prior.compare
"""
import triad  # pins BLAS threads before numpy loads
import json
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust
from .baselines import OUT, ECOM, METHODS

DIALS = Path(__file__).resolve().parents[3]/'dials';NAMES = ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify')


def published(name, keep):
    """method -> (validation NDCG@20, clean per-query [R, N]) for every published method with per-query results."""
    out = {};content = ('hybrid', 'label_knn_content')
    def add(k, val, q):
        if k not in out or val > out[k][0]:out[k] = (val, np.asarray(q, float))
    for path, sel in ((f'results/simple_baselines/{name}/summary.json', lambda k: name not in ECOM or k not in content),
                      (f'results/bundle_budget/baselines/{name}/full.json', lambda k: True),
                      (f'results/bundle_budget_retail/baselines/{name}/full.json', lambda k: name not in ECOM or k not in content),
                      (f'results/dials_blip/baselines/{name}/full.json', lambda k: True)):
        p = Path(path)
        if not p.exists():continue
        d = json.loads(p.read_text());methods = d.get('methods') or d['test']
        for k, v in methods.items():
            if sel(k) and isinstance(v.get('queries'), list) and len(v['queries']) == len(keep):add(k, v['validation_ndcg20'], v['queries'])
    lab = Path(f'results/label_baselines/{name}/summary.json')
    if lab.exists():
        for k, v in json.loads(lab.read_text())['test'].items():add(k, v['validation_ndcg20'], [v['queries'][i] for i in keep])
    # Methods without stored per-query results (GF-CF, iALS, SGFCF on some datasets) must not be the validation best.
    ss = Path(f'results/simple_baselines/{name}/summary.json')
    if ss.exists():
        best = max(v[0] for v in out.values())
        for k, v in json.loads(ss.read_text())['test'].items():
            assert k in out or v['validation_ndcg20'] < best, (name, k)
    return out


def main():
    bonus = json.loads((DIALS/'paper'/'figdata'/'unseen_bonus.json').read_text())
    report = dict(note=__doc__, datasets={});families = {'triad_vs_best_simple_prior': [], 'triad_no_bonus_vs_best_simple_prior': []}
    for name in NAMES:
        ds = data.load(name, DIALS/'cache');keep = sorted(ds.clean_test)
        clusters = [str(tuple(sorted(set(ds.test_pairs[i][0]) | set(ds.test_pairs[i][1])))) for i in keep]
        triad_q = np.mean([[[r['queries'][i]['recall20'], r['queries'][i]['ndcg20']] for i in keep]
                           for r in (json.loads((DIALS/'runs'/name/f'seed{s}'/'test.json').read_text()) for s in (0, 1, 2))], 0)
        nb = np.asarray(bonus[name]['queries'])
        pool = published(name, keep);fair = json.loads((OUT/f'{name}.json').read_text())['methods']
        methods = {}
        for k, (val, q) in pool.items():methods[k] = dict(validation_ndcg20=val, prior=False, test=q.mean(0).tolist(), queries=q)
        for k, v in fair.items():
            q = np.asarray(v['queries'], float)
            methods[k + '+prior'] = dict(validation_ndcg20=v['validation_ndcg20'], prior=True, lam=v['lam'], params=v['params'], test=q.mean(0).tolist(), queries=q)
        best = max(methods, key=lambda k: methods[k]['validation_ndcg20']);bq = methods[best]['queries']
        assert len({len(x) for x in (triad_q, nb, bq, clusters)}) == 1, name
        report['datasets'][name] = dict(triad=triad_q.mean(0).tolist(), triad_no_bonus=nb.mean(0).tolist(), best_simple=best,
                                        best_simple_test=bq.mean(0).tolist(), unseen_target_share=bonus[name]['unseen_target_share'],
                                        methods={k: {a: b for a, b in v.items() if a != 'queries'} for k, v in methods.items()})
        for j, metric in ((1, 'ndcg20'), (0, 'recall20')):
            for fam, a in (('triad_vs_best_simple_prior', triad_q), ('triad_no_bonus_vs_best_simple_prior', nb)):
                families[fam].append(dict(paired_comparison(a[:, j], bq[:, j], clusters, bootstrap=2000, permutations=10000, seed=20261118),
                                          dataset=name, metric=metric, baseline=best))
    for records in families.values():holm_adjust(records, len(records))
    report['tests'] = families;write_json(OUT/'comparison.json', report)
    for name, r in report['datasets'].items():
        print(f"{name:15s} TRIAD {np.round(r['triad'], 4)}  no bonus {np.round(r['triad_no_bonus'], 4)}  best simple {r['best_simple']} {np.round(r['best_simple_test'], 4)}")
    for fam, recs in families.items():
        for t in recs:print(f"{fam:38s} {t['dataset']:15s} {t['metric']:8s} {t['delta']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.4f}")


if __name__ == '__main__':
    main()
