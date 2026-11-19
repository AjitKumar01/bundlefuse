"""The simple and label-aware baselines on Electronic, Food and Clothing, given the same purchase log as TRIAD + purchase.

Every method of experiments.bundle_budget_retail.baselines (condition 'full', BLIP content as in
experiments.dials_blip.baselines) is rerun with the purchase histories (experiments.triad_purchase.run.
purchase_histories: users with at least two catalogue items in user_item_pretrain.csv) stacked under the bundle-item
matrix as extra rows weighted sqrt(lambda):

    X_aug = [X; sqrt(lambda) X_log],   co-occurrence X_aug^T X_aug = C + lambda P,   frequency f + lambda f_log

so co-occurrence, ItemKNN, RP3beta, EASE, the content + ItemKNN blend and the label methods (whose transition
statistics are built from C and f) all see the log. lambda in (.003, .01, .03, .1, .3, 1, 3) is one more hyperparameter: the best
simple baseline is chosen on validation NDCG@20 over these runs and the published runs without the log
(lambda = 0, results/dials_blip/baselines). Writes results/triad_purchase/baselines/lam<lambda>/<dataset>/full.json.

PYTHONPATH=.:../dials nice -n 10 .venv/bin/python -m experiments.triad_purchase.baselines
"""
import triad  # pins BLAS threads before numpy loads
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from experiments.simple_baselines.methods import Baselines, incidence, without_diagonal
from experiments.label_baselines.run import LabelBaselines
from experiments.bundle_budget_retail import baselines, conditions
from experiments.dials_blip import run as blip
from .run import purchase_histories

LAMBDAS = (.003, .01, .03, .1, .3, 1., 3.);OUT = Path('results/triad_purchase/baselines')


def with_log(lam, histories):
    class Purchase(LabelBaselines):
        def __init__(self, train, n, levels, features):
            Baselines.__init__(self, train, n, features)
            X = sp.vstack([self.X, np.sqrt(lam)*incidence(histories, n)]).tocsr()
            self.X = X;self.f = np.asarray(X.multiply(X).sum(0)).ravel()
            self.C = without_diagonal(X.T@X).astype(np.float32);self.tie = 1e-6*np.log1p(self.f)[None, :].astype(np.float32)
            # The rest is LabelBaselines.__init__, on the augmented C and f.
            self.levels = [np.asarray(g) for g in levels];self.onehot = [];self.transition = []
            for g in self.levels:
                k = int(g.max()) + 1;G = sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), g)), shape=(n, k))
                T = (G.T@self.C@G).tocsr().astype(np.float64);row = np.asarray(T.sum(1)).ravel()
                self.onehot.append(G);self.transition.append((sp.diags(1/np.maximum(row, 1e-12))@T).tocsr().astype(np.float32))
            fine = self.levels[-1] if len(self.levels) == 1 else max(self.levels, key=lambda g: g.max())
            F = np.bincount(fine, weights=self.f);size = np.bincount(fine)
            self.within = np.log((self.f + 1)/(F[fine] + size[fine])).astype(np.float32)
    return Purchase


def main():
    loaded = {}
    def dataset(name):
        if name not in loaded:loaded[name] = blip.load(name)
        return loaded[name]
    conditions.dataset = baselines.dataset = dataset
    for name in ('electronic', 'food', 'clothing'):
        histories = purchase_histories(name)
        for lam in LAMBDAS:
            if (OUT/f'lam{lam}'/'baselines'/name/'full.json').exists():continue
            baselines.LabelBaselines = with_log(lam, histories);baselines.OUT = OUT/f'lam{lam}'
            conditions.condition.cache_clear();baselines.main(argparse.Namespace(dataset=name, condition='full'))


if __name__ == '__main__':
    main()
