"""SLIM: sparse linear item--item model (Ning & Karypis, ICDM'11)."""
import numpy as np
import scipy.sparse as sp
from .base import ItemItemBaseline, topk_per_column

SLIM_GRID = [(a, l) for a in (1e-4, 1e-3, 1e-2) for l in (0.01, 0.1, 0.5)]


def slim_W(B, alpha=1e-3, l1_ratio=0.1, topk=200, max_iter=100, n_max=8000):
    """Per-item non-negative elastic-net regression ``B[:,c] ~ B[:,!=c]``. O(n) ElasticNet solves,
    so gated to small catalogues (``n <= n_max``); at larger scale the closed-form EASE in the harness
    represents the sparse-linear family. Returns ``W`` (score = ``W @ Ind``), or ``None`` if gated out."""
    import warnings
    from sklearn.linear_model import ElasticNet
    from sklearn.exceptions import ConvergenceWarning
    B = B.tocsc().astype(np.float32)
    n = B.shape[1]
    if n > n_max:
        return None
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True, fit_intercept=False,
                       copy_X=False, max_iter=max_iter, tol=1e-3, selection="random")
    rows, cols, vals = [], [], []
    allcols = np.arange(n)
    warnings.simplefilter("ignore", ConvergenceWarning)  # SLIM coordinate descent rarely fully converges
    for c in range(n):
        y = B[:, c].toarray().ravel()
        if y.sum() == 0:
            continue
        others = allcols[allcols != c]
        X = B[:, others]
        model.fit(X, y)
        w = model.coef_
        nz = np.nonzero(w)[0]
        if nz.size == 0:
            continue
        src = others[nz]                        # seed items i
        rows.append(np.full(nz.size, c)); cols.append(src); vals.append(w[nz].astype(np.float32))
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n), dtype=np.float32)
    return topk_per_column(W, topk)


class SLIM(ItemItemBaseline):
    name = "SLIM"
    grid = SLIM_GRID

    def build_W(self, graph_op, freq, hp):
        a, l = hp
        return slim_W(graph_op.B, alpha=a, l1_ratio=l, topk=200)
