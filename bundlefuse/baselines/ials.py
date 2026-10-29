"""iALS: implicit-feedback ALS (Hu, Koren & Volinsky, ICDM'08), dependency-free (query-dependent)."""
import numpy as np
from .base import Baseline

IALS_GRID = [(reg, al) for reg in (1.0, 10.0, 100.0) for al in (40.0,)]


def ials_item_factors(B, d=48, reg=10.0, alpha=40.0, iters=8, seed=0):
    """Factorize the binary bundle x item matrix ``B`` (bundles as "users") with confidence
    ``C = 1 + alpha*B``. Returns item factors ``Y`` (n x d) and ``YtY`` for closed-form fold-in."""
    B = B.tocsr().astype(np.float32)
    m, n = B.shape
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((m, d)) * 0.01).astype(np.float32)   # bundle factors
    Y = (rng.standard_normal((n, d)) * 0.01).astype(np.float32)   # item factors
    Bt = B.T.tocsr()
    I = (reg * np.eye(d)).astype(np.float32)

    def _solve_side(F, rows_of):
        GtG = F.T @ F
        out = np.zeros((rows_of.shape[0], d), dtype=np.float32)
        for r in range(rows_of.shape[0]):
            s, e = rows_of.indptr[r], rows_of.indptr[r + 1]
            idx = rows_of.indices[s:e]
            if idx.size == 0:
                continue
            Fi = F[idx]                                   # (k x d)
            A = GtG + alpha * (Fi.T @ Fi) + I
            b = (1.0 + alpha) * Fi.sum(0)
            out[r] = np.linalg.solve(A, b)
        return out

    for _ in range(iters):
        X = _solve_side(Y, B)
        Y = _solve_side(X, Bt)
    return Y, (Y.T @ Y).astype(np.float32)


def ials_scores(Y, YtY, seed_lists, reg=10.0, alpha=40.0):
    """Closed-form fold-in: ``x_S = (YtY + alpha sum_{i in S} y_i y_i^T + reg I)^{-1} (1+alpha) sum y_i``;
    ``score(c | S) = x_S . y_c``. Returns dense (n x nb)."""
    d = Y.shape[1]
    I = (reg * np.eye(d)).astype(np.float32)
    X = np.zeros((len(seed_lists), d), dtype=np.float32)
    for j, s in enumerate(seed_lists):
        if not len(s):
            continue
        Ys = Y[s]
        A = YtY + alpha * (Ys.T @ Ys) + I
        X[j] = np.linalg.solve(A, (1.0 + alpha) * Ys.sum(0))
    return (Y @ X.T).astype(np.float32)                   # (n x nb)


class IALS(Baseline):
    name = "iALS"
    grid = IALS_GRID

    def make_scorer(self, graph_op, freq, hp):
        reg, al = hp
        Y, YtY = ials_item_factors(graph_op.B, reg=reg, alpha=al)
        def scorer(seed_lists):
            return ials_scores(Y, YtY, seed_lists, reg=reg, alpha=al)
        return scorer
