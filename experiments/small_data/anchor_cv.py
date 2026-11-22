"""Tune the residual model's anchor (a simple scorer) on the training bundles only: no validation query is used.

5-fold cross-validation inside the training set: the bundles of fold j are split in half (seeds / targets, fixed
generator), and scored by the scorer fitted on the other four folds, with its frequency prior from those four folds
(leave-one-out base rate, experiments.small_data.residual.base_rate_train). The criterion is the mean NDCG@20 over all
held-out halves. Search, in the order the paper's simple baselines are tuned (same grids):
  ItemKNN, RP3beta, EASE; the content+ItemKNN blend (ItemKNN at its chosen setting); the better of same-label counts
  and label transitions, then label+ItemKNN and label+ItemKNN+content (mixing weights as in experiments.fair_prior);
  then, for each of the five main methods at its chosen setting, lambda in LAMBDAS.
Returns (method, params, lambda, cross-validated NDCG@20).
"""
import numpy as np
from experiments.neural.evaluation import summarize
from experiments.fair_prior.baselines import WithPrior, LAMBDAS

FOLDS = 5
KNN = [dict(k=k, shrink=h) for k in (50, 200, 1000) for h in (0, 10, 100)]
RP3 = [dict(alpha=a, beta=b) for a in (.5, 1.) for b in (0., .25, .5, .75)]
EASE = [dict(l2=l) for l in (50., 200., 500., 1000., 3000.)]
CONTENT = (0., .1, .2, .3, .5, .7, .9, 1.);LABEL_W = (0., .1, .2, .3, .5, .7, .9, 1.)
V = (0., .05, .1, .2, .3, .5, .7, 1., 1.5, 2., 3.)


def tune_anchor(ds, train, base_rate_train):
    rng = np.random.default_rng(20261121);train = [sorted(set(b)) for b in train if len(set(b)) >= 2]
    fold = rng.permutation(len(train)) % FOLDS;models, queries = [], []
    for j in range(FOLDS):
        fit = [b for b, k in zip(train, fold) if k != j];m = WithPrior(fit, ds.n_items, ds.levels, ds.features)
        m.prior = base_rate_train(m.f, fit);models.append(m)
        pairs = []
        for b, k in zip(train, fold):
            if k == j:
                row = list(rng.permutation(b));cut = max(1, len(row)//2);pairs.append((row[:cut], row[cut:]))
        queries.append(pairs)

    def cv(method, params, lam=0.):
        rows = []
        for m, pairs in zip(models, queries):
            if not pairs:continue
            m.lam = lam;rows.extend(summarize(m.scores(method, params, [s for s, _ in pairs]), pairs)['queries'])
        return float(np.mean([r['ndcg20'] for r in rows]))

    best = {}
    def pick(method, grid):
        v, p = max(((cv(method, p), p) for p in grid), key=lambda r: r[0]);best[method] = (p, v);return p
    knn = pick('itemknn', KNN);pick('rp3beta', RP3)
    if ds.n_items <= 40000:pick('ease', EASE)
    pick('hybrid', [dict(knn, content=w) for w in CONTENT])
    labels = models[0].label_grid()
    lab = {n: max(((cv(n, p), p) for q, p in labels if q == n), key=lambda r: r[0]) for n in ('samelabel', 'transition')}
    label = max(lab, key=lambda n: lab[n][0])
    lk = pick('label_knn', [dict(label=label, label_params=lab[label][1], itemknn=knn, w=w) for w in LABEL_W])
    pick('label_knn_content', [dict(lk, v=v) for v in V])
    del best['label_knn']
    v, method, params, lam = max(((cv(m, p, lam), m, p, lam) for m, (p, _) in best.items() for lam in LAMBDAS), key=lambda r: r[0])
    return method, params, lam, v
