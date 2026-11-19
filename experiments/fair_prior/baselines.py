"""Fairness check: give the simple baselines the frequency prior that TRIAD's rho learns.

TRIAD's popularity prior rho(log(1 + f)) follows the base rate of targets by training frequency, including the fact
that an item with f = 0 is in the catalogue only because a validation or test bundle holds it
(dials/ARCHITECTURE.md section 14). The simple baselines rank such items last. Here every method of the main table
gets the same kind of prior, estimated on validation:

    b(f) = log( (targets in f's bin + 1) / (validation targets + bins) ) - log( items in f's bin / n )
    score' = maxabs(score) + lambda * b(f_c),   lambda in LAMBDAS chosen on validation NDCG@20

bins as in paper/figdata/parameters.py (0, 1, 2, 3-4, 5-9, 10-19, 20-49, 50+). Each method keeps the setting chosen
on validation in the published runs (the pool of experiments.dials_blip.compare.simple_q on e-commerce; simple_baselines,
label_baselines and bundle_budget(_retail) elsewhere); lambda = 0 reproduces them (asserted). Test scored once.
Writes results/fair_prior/<dataset>.json.

PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.fair_prior.baselines electronic food:n300 pog:frac10 ...
(dataset[:condition]; conditions as in experiments.bundle_budget(_retail); results/fair_prior/<dataset>[_<condition>].json)
"""
import triad  # pins BLAS threads before numpy loads
import json, sys
from pathlib import Path
import numpy as np
from triad import data
from bundlefuse.provenance import write_json
from experiments.label_baselines.run import LabelBaselines, maxabs
from experiments.simple_baselines.run import evaluate, mean
from experiments.dials_blip import run as blip
from experiments.bundle_budget_retail.conditions import CACHE

METHODS = ('itemknn', 'rp3beta', 'ease', 'hybrid', 'label_knn_content')
LAMBDAS = (0., .01, .03, .1, .3, 1., 3.)
EDGES = [0, 1, 2, 3, 5, 10, 20, 50, 1e9]
ECOM = ('electronic', 'food', 'clothing');OUT = Path('results/fair_prior')


def pool(name):
    """method -> (validation NDCG@20, params, clean test [R, N] per query) from the published runs, best on validation."""
    out = {};content = ('hybrid', 'label_knn_content')
    srcs = [(f'results/simple_baselines/{name}/summary.json', 'test', lambda k: name not in ECOM or k not in content),
            (f'results/label_baselines/{name}/summary.json', 'test', lambda k: True),
            (f'results/bundle_budget/baselines/{name}/full.json', 'methods', lambda k: True),
            (f'results/bundle_budget_retail/baselines/{name}/full.json', 'methods', lambda k: name not in ECOM or k not in content),
            (f'results/dials_blip/baselines/{name}/full.json', 'methods', lambda k: True)]
    for path, key, keep in srcs:
        p = Path(path)
        if not p.exists():continue
        for k, v in json.loads(p.read_text())[key].items():
            if keep(k) and (k not in out or v['validation_ndcg20'] > out[k][0]):
                out[k] = (v['validation_ndcg20'], v['params'], v.get('filtered', v.get('test')))
    return out


class WithPrior(LabelBaselines):
    lam = 0.;prior = None
    def scores(self, name, p, seeds):
        s = super().scores(name, p, seeds)
        if self.lam == 0:return s
        return (maxabs(s - self.tie) + self.lam*self.prior[None, :] + self.tie).astype(np.float32)


def base_rate(f, pairs):
    t = np.array([f[g] for _, G in pairs for g in set(G)]);n = len(f);b = np.zeros(n, np.float32)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        items = (f >= lo) & (f < hi)
        if items.any():b[items] = np.log((((t >= lo) & (t < hi)).sum() + 1)/(len(t) + len(EDGES) - 1)) - np.log(items.sum()/n)
    return b


def setup(name, cond):
    """(dataset, training bundles of the condition, published pool) for one dataset and training-set-size condition."""
    if cond == 'full':
        ds = blip.load(name) if name in ECOM else data.load(name, CACHE);return ds, ds.train, pool(name)
    if name in ('pog', 'pog_dense'):
        from experiments.bundle_budget import conditions as C;ds = C.dataset(name);train = C.condition(name, cond)[0];root = 'results/bundle_budget'
    else:
        from experiments.bundle_budget_retail import conditions as C;ds = C.dataset(name);train = C.condition(name, cond);root = 'results/bundle_budget_retail'
    methods = json.loads((Path(root)/'baselines'/name/f'{cond}.json').read_text())['methods']
    return ds, train, {k: (v['validation_ndcg20'], v['params'], v['test']) for k, v in methods.items()}


def main(jobs):
    OUT.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        name, cond = (job.split(':') + ['full'])[:2];ds, train, published = setup(name, cond);keep = sorted(ds.clean_test)
        model = WithPrior(train, ds.n_items, ds.levels, ds.features);model.prior = base_rate(model.f, ds.val_pairs)
        report = dict(note=__doc__, dataset=name, condition=cond, prior_bins=sorted(set(model.prior.round(3).tolist())), methods={})
        for m in METHODS:
            if m not in published:continue
            val, params, test = published[m];rows = []
            # The blends' mixing weight is retuned together with lambda (the published grid), the rest stays fixed.
            mix = dict(hybrid=('content', (0., .1, .2, .3, .5, .7, .9, 1.)), label_knn_content=('v', (0., .05, .1, .2, .3, .5, .7, 1., 1.5, 2., 3.))).get(m)
            settings = [params] + ([dict(params, **{mix[0]: x}) for x in mix[1] if x != params[mix[0]]] if mix else [])
            for lam in LAMBDAS:
                for q in settings:
                    model.lam = lam;v = mean(evaluate(model, m, q, ds.val_pairs))['ndcg20'];rows.append((v, lam, q))
                    print(name, cond, m, 'lambda', lam, q.get(mix[0]) if mix else '', 'VAL N20', round(v, 5), flush=True)
            assert abs(rows[0][0] - val) < 1e-6, (name, m, rows[0][0], val)
            best = max(rows, key=lambda r: r[0]);model.lam = best[1];chosen = best[2];r = evaluate(model, m, chosen, ds.test_pairs)
            model.lam = 0.;r0 = evaluate(model, m, params, ds.test_pairs);t0 = mean(r0, keep)
            assert abs(t0['ndcg20'] - test['ndcg20']) < 1e-6, (name, m, t0, test)
            report['methods'][m] = dict(params=chosen, published_params=params, validation=[[v, l, q] for v, l, q in rows], lam=best[1], validation_ndcg20=best[0],
                                        test=mean(r, keep), test_without_prior=t0,
                                        queries=[[r[i]['recall20'], r[i]['ndcg20']] for i in keep],
                                        queries_without_prior=[[r0[i]['recall20'], r0[i]['ndcg20']] for i in keep])
            print(name, cond, m, 'lambda', best[1], chosen.get(mix[0]) if mix else '', 'TEST', report['methods'][m]['test'], 'without', t0, flush=True)
        write_json(OUT/(f'{name}.json' if cond == 'full' else f'{name}_{cond}.json'), report)


if __name__ == '__main__':
    main(sys.argv[1:])
