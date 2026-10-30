"""BundleKNN: user(=bundle)-based KNN in bundle space (query-dependent)."""
import numpy as np
from .base import Baseline
from ..metrics import top_items

BUNDLEKNN_GRID = [(tn,) for tn in (25, 50, 100, 200)]


def bundleknn_scores(B, Ind, topn=100):
    """``score(c | S) = sum`` over the top-N train bundles most similar to the seed set ``S`` of
    ``cosine(S, u) * B[u, c]``. Query-dependent, so scored per chunk. Returns dense (n x nb)."""
    B = B.tocsr().astype(np.float32)
    bnorm = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-9    # ||u||
    Q = Ind.T.tocsr().astype(np.float32)                               # (nb x n) seed sets
    qnorm = np.sqrt(np.asarray(Q.multiply(Q).sum(1)).ravel()) + 1e-9    # ||S||
    sim = np.asarray((Q @ B.T).todense()) / (np.outer(qnorm, bnorm))   # (nb x n_bundles) cosine
    if topn and topn < sim.shape[1]:
        for j in range(sim.shape[0]):
            keep = top_items(sim[j], [], topn)
            selected = sim[j, keep].copy()
            sim[j] = 0.0
            sim[j, keep] = selected
    scores = sim @ B                                                   # (nb x n)
    return np.asarray(scores).T                                        # (n x nb)


class BundleKNN(Baseline):
    name = "BundleKNN"
    grid = BUNDLEKNN_GRID

    def make_scorer(self, graph_op, freq, hp):
        (tn,) = hp
        def scorer(seed_lists):
            return bundleknn_scores(graph_op.B, graph_op.build_indicator(seed_lists), topn=tn)
        return scorer
