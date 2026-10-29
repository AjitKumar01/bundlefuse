"""LinearCombiner: a validation-tuned non-negative blend of the neighbourhood W's.

This is the deliberate foil for BundleFuse's regime-aware fusion -- a naive linear blend of the
three random-walk/neighbourhood item--item matrices (max-normalised so their scales are comparable),
found by greedy non-negative coordinate ascent on validation NDCG@20. It reuses the ItemKNN / P3alpha
/ RP3beta weight matrices already selected for this seed, so it adds no new hyperparameter search.
"""
import numpy as np
from .base import _seeds

COMPONENTS = ("ItemKNN", "P3alpha", "RP3beta")


def run_linear_combiner(comp_W, graph_op, eval_mat, vset, tset):
    """``comp_W`` maps component name -> its selected sparse ``W`` (or None). Returns test metrics."""
    cn = [k for k in COMPONENTS if comp_W.get(k) is not None]
    nW = {}
    for k in cn:
        Wm = comp_W[k].tocsr().copy().astype(np.float32)
        mx = float(Wm.data.max()) if Wm.nnz else 1.0
        Wm.data = Wm.data / (mx + 1e-12)
        nW[k] = Wm

    def _score(Wc, pairs):
        Ind = graph_op.build_indicator(_seeds(pairs))
        return np.asarray((Wc @ Ind).todense()).astype(np.float32)

    cw, cnd = {k: 0.0 for k in cn}, -1.0
    for _ in range(3):
        improved = False
        for k in cn:
            for v in (0.0, 0.5, 1.0, 2.0):
                cand = dict(cw); cand[k] = v
                if sum(cand.values()) == 0:
                    continue
                Wc = sum(cand[k2] * nW[k2] for k2 in cn if cand[k2])
                nd = eval_mat(_score(Wc, vset), vset, Ks=(10, 20))[20][1]
                if nd > cnd + 1e-5:
                    cnd, cw, improved = nd, cand, True
        if not improved:
            break
    Wc = sum(cw[k2] * nW[k2] for k2 in cn if cw[k2])
    return eval_mat(_score(Wc, tset), tset, Ks=(10, 20))
