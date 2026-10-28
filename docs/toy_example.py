"""Reproduces the worked example in docs/HOW_IT_WORKS.md with the real BundleFuse operators.

Weights and kappa are fixed by hand (the doc says so); every signal is computed by the library.
Run from the repo root: python3 docs/toy_example.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import (
    SimplicialGraphOperator, DualSpectrumOntology, BilateralContentBridge, EmpiricalBayesRouter, zcols,
)

names = ["white sneakers", "blue jeans", "grey hoodie", "white t-shirt", "black cap",
         "denim jacket", "running shoes", "leather belt", "track pants", "canvas sneakers"]
cats = ["shoes", "bottoms", "tops", "tops", "hats", "outerwear", "shoes", "accessories", "bottoms", "shoes"]
cmap = {}
cat_lbl = np.array([cmap.setdefault(c, len(cmap)) for c in cats])
content = np.array([[.9, .3, .2], [.9, .1, .3], [.8, .5, 0], [.9, .2, .2], [.7, .6, 0],
                    [.8, .1, .4], [.2, .95, 0], [.4, 0, .9], [.2, .9, .1], [.9, .2, .2]], np.float32)
content /= np.linalg.norm(content, axis=1, keepdims=True)
train = [[0, 1, 2], [0, 1, 3, 4], [1, 3, 7], [6, 8, 2], [6, 8, 4], [0, 5, 3], [1, 5, 2]]
n, S = len(names), [1, 2]  # seeds: blue jeans, grey hoodie
W = {"pair": 1.0, "diff2": 0.5, "cat": 1.0, "style": 0.5, "bridge": 0.5}
PRIOR = ("cat", "style", "bridge")
KAPPA = 1.0

g = SimplicialGraphOperator(train, n_items=n, svd_dim=128)
onto = DualSpectrumOntology(train, cat_lbl=cat_lbl, n_items=n)
style = BilateralContentBridge.disentangle(content, cat_lbl, len(cmap))
bridge = BilateralContentBridge.build_dense_bridge(style, g, alpha_bridge=0.5)

Ind = g.build_indicator([S])
pair, diff2 = g.score_pairwise_and_diff2([S])
R = {
    "cooc": np.asarray((g.C @ Ind).todense()).ravel(),
    "pair": pair.ravel(),
    "diff2": diff2.ravel(),
    "ppmi": np.asarray((g.PPMI @ Ind).todense()).ravel(),
    "pop": np.log1p(g.freq),
    "cat": np.asarray(onto.Pcat[cat_lbl[S]].sum(0)).ravel()[cat_lbl] + 0.5 * onto.score_slot_saturation([S]).ravel(),
    "style": style @ style[S].mean(0),
    "bridge": bridge @ bridge[S].mean(0),
}
Z = {k: zcols(v[:, None]).ravel() for k, v in R.items()}
Y = sum(W[k] * Z[k] for k in W)
raw_prior = sum(W[k] * Z[k] for k in PRIOR)
mu0 = raw_prior * (Y.std() / raw_prior.std())
final = EmpiricalBayesRouter.conjugate_posterior(Y[:, None], mu0[:, None], g.freq, KAPPA).ravel()

cols = list(R) + ["Y", "mu0", "u", "final"]
u = KAPPA / (g.freq + KAPPA)
print(f"{'item':18s}" + "".join(f"{c:>9s}" for c in cols))
for i in np.argsort(-np.where(np.isin(np.arange(n), S), -1e9, final)):
    vals = [R[k][i] for k in R] + [Y[i], mu0[i], u[i], final[i]]
    tag = " (seed)" if i in S else ""
    print(f"{names[i] + tag:18s}" + "".join(f"{v:9.3f}" for v in vals))
