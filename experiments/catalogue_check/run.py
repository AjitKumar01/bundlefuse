"""Does the unseen-item bonus depend on the split, or on how the catalogue is built?

Two checks on Electronic and Food (dials/ARCHITECTURE.md section 14, paper section "Equal information"):

  split<k>  a fresh seeded 70/10/20 split (triad.data.load_resplit(seed=k)); split0 is the published one, retrained
            here with the same code so every condition is comparable.
  pad<m>    the published split with P = m x n0 extra items that are in no bundle (n0 = items with f = 0 in the
            published catalogue). Each extra item copies the features and category of a uniformly drawn catalogue
            item, so features and labels cannot tell it apart from a real one. It is a negative everywhere.

The bonus is learned from training data alone, so a fresh split should leave it unchanged. Padding adds f = 0 items
that are never targets: the base-rate argument predicts the bonus falls by about log(1 + P/n0), and its test gain
falls with it.

Per condition: TRIAD seeds 0-2 (the one configuration, early stopping on validation), scored with and without the
bonus (rho of f = 0 items set to rho(log 2), experiments.fair_prior.no_bonus); the learned bonus
w_G (rho(0) - rho(log 2)) averaged over test queries; the bonus the test base rates call for,
log(share of f=0 targets / share of f=0 items) - log(share of f=1 targets / share of f=1 items); and the two content
blends of the main table with the frequency prior (experiments.fair_prior.baselines: published setting kept, mixing
weight and lambda retuned on validation; also lambda = 0). Tests (paired cluster bootstrap and sign flips, Holm over
all tests of this run): TRIAD - TRIAD w/o bonus, and TRIAD - the better blend with the prior chosen on validation.
All test queries are clean (duplicate bundles are removed before splitting). Writes results/catalogue_check/summary.json.

PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.catalogue_check.run [name:cond ...]
"""
import triad  # pins BLAS threads before numpy loads
import json, math, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
from triad import data, train as T
from triad.evaluate import evaluate as triad_eval
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison, holm_adjust
from experiments.simple_baselines.run import evaluate, mean
from experiments.fair_prior.baselines import WithPrior, base_rate, LAMBDAS

DIALS = Path(__file__).resolve().parents[3]/'dials';CACHE = DIALS/'cache';OUT = Path('results/catalogue_check')
JOBS = [f'{n}:{c}' for n in ('electronic', 'food') for c in ('split0', 'split1', 'split2', 'pad1', 'pad4', 'pad16')]
MIX = dict(hybrid=('content', (0., .1, .2, .3, .5, .7, .9, 1.)), label_knn_content=('v', (0., .05, .1, .2, .3, .5, .7, 1., 1.5, 2., 3.)))


def frequency(ds):
    f = np.zeros(ds.n_items, int)
    for b in ds.train:
        for i in set(b):f[i] += 1
    return f


def dataset(name, cond):
    if cond.startswith('split'):return data.load_resplit(name, seed=int(cond[5:]), cache=CACHE)
    ds = data.load_resplit(name, seed=0, cache=CACHE);n = ds.n_items;n0 = int((frequency(ds) == 0).sum())
    P = int(cond[3:])*n0;src = np.random.default_rng(20261119).integers(0, n, P)
    return replace(ds, n_items=n + P, features=np.concatenate([ds.features, ds.features[src]]),
                   levels=[np.concatenate([g, g[src]]) for g in ds.levels])


def ideal_bonus(f, pairs):
    """The bonus the base rates call for: log-odds of an f=0 item being a target, relative to an f=1 item."""
    t = np.array([f[g] for _, G in pairs for g in set(G)]);share = lambda x, v: max((x == v).mean(), 1e-12)
    return float(np.log(share(t, 0)/share(f, 0)) - np.log(share(t, 1)/share(f, 1)))


def triad_runs(ds, root):
    full, nb, bonus = [], [], []
    for s in (0, 1, 2):
        run = root/f'seed{s}';T.fit(ds, s, run, CACHE);m = T.load(ds, run, CACHE)
        torch.set_grad_enabled(False)
        r = triad_eval(m, ds.test_pairs);full.append([[q['recall20'], q['ndcg20']] for q in r['queries']])
        rho = lambda x: m.prior(torch.tensor([[x]], dtype=torch.float32)).item()
        S = [torch.as_tensor(sorted(set(a))) for a, _ in ds.test_pairs]
        bonus.append(dict(rho_gap=rho(0.) - rho(math.log(2.)), weighted=float(m.gate_weights(S, None)[:, -1].mean())*(rho(0.) - rho(math.log(2.)))))
        whole = m.cache;once = torch.tensor(rho(math.log(2.)))
        def no_bonus(m=m, whole=whole, once=once):
            c = whole();c['prior'] = torch.where(m.frequency == 0, once, c['prior']);return c
        m.cache = no_bonus;r = triad_eval(m, ds.test_pairs);nb.append([[q['recall20'], q['ndcg20']] for q in r['queries']])
        torch.set_grad_enabled(True)
    return np.mean(full, 0), np.mean(nb, 0), bonus


def blends(name, ds):
    published = json.loads(Path(f'results/fair_prior/{name}.json').read_text())['methods']
    model = WithPrior(ds.train, ds.n_items, ds.levels, ds.features);model.prior = base_rate(model.f, ds.val_pairs);out = {}
    for m, (key, grid) in MIX.items():
        p0 = published[m]['published_params'];rows = []
        for lam in LAMBDAS:
            for x in grid:
                model.lam = lam;q = dict(p0, **{key: x});rows.append((mean(evaluate(model, m, q, ds.val_pairs))['ndcg20'], lam, q))
        for tag, cand in (('+prior', rows), ('', [r for r in rows if r[1] == 0])):
            v, lam, q = max(cand, key=lambda r: r[0]);model.lam = lam;r = evaluate(model, m, q, ds.test_pairs)
            out[m + tag] = dict(validation_ndcg20=v, lam=lam, params=q, queries=np.array([[x['recall20'], x['ndcg20']] for x in r]))
            print(name, m + tag, 'lambda', lam, key, q[key], 'VAL', round(v, 4), 'TEST', mean(r), flush=True)
    return out


def main(jobs):
    report = dict(note=__doc__, conditions={});recs = []
    for job in jobs:
        name, cond = job.split(':');ds = dataset(name, cond);f = frequency(ds)
        clusters = [str(tuple(sorted(set(s) | set(g)))) for s, g in ds.test_pairs]
        t, nb, bonus = triad_runs(ds, OUT/name/cond);b = blends(name, ds)
        best = max([k for k in b if k.endswith('+prior')], key=lambda k: b[k]['validation_ndcg20'])
        row = dict(n_items=ds.n_items, n0=int((f == 0).sum()),
                   unseen_target_share=float(np.mean([f[g] == 0 for _, G in ds.test_pairs for g in set(G)])),
                   ideal_bonus_test=ideal_bonus(f, ds.test_pairs), ideal_bonus_val=ideal_bonus(f, ds.val_pairs),
                   learned_bonus=dict(per_seed=bonus, weighted=float(np.mean([x['weighted'] for x in bonus])), rho_gap=float(np.mean([x['rho_gap'] for x in bonus]))),
                   triad=t.mean(0).tolist(), triad_no_bonus=nb.mean(0).tolist(), best_simple=best,
                   simple={k: dict(test=v['queries'].mean(0).tolist(), **{a: v[a] for a in ('validation_ndcg20', 'lam', 'params')}) for k, v in b.items()})
        report['conditions'][job] = row
        for j, metric in ((1, 'ndcg20'), (0, 'recall20')):
            for test, other, base in (('triad_minus_no_bonus', nb, 'TRIAD w/o bonus'), ('triad_minus_best_simple', b[best]['queries'], best)):
                recs.append(dict(paired_comparison(t[:, j], other[:, j], clusters, bootstrap=2000, permutations=10000, seed=20261119),
                                 condition=job, test=test, metric=metric, baseline=base))
        print(f"{job:18s} n={ds.n_items} n0={row['n0']} unseen-target {row['unseen_target_share']:.2f} bonus learned {row['learned_bonus']['weighted']:+.2f} "
              f"ideal(test) {row['ideal_bonus_test']:+.2f}  TRIAD {np.round(t.mean(0), 4)} w/o bonus {np.round(nb.mean(0), 4)} {best} {np.round(b[best]['queries'].mean(0), 4)}", flush=True)
        holm_adjust(recs, len(recs));report['tests'] = recs;write_json(OUT/'summary.json', report)
    for r in recs:
        if r['metric'] == 'ndcg20':print(f"{r['condition']:18s} {r['test']:24s} {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p_holm {r['p_holm']:.4f}")


if __name__ == '__main__':
    main(sys.argv[1:] or JOBS)
