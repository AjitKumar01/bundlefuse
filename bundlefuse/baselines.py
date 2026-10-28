"""Tuned neighbourhood baselines for bundle completion, on the SAME splits/candidates/metrics
as BundleFuse (reviewer request). Each model is a sparse item-item matrix W (n x n) with
    score(c | S) = sum_{i in S} W[c, i] = (W @ Ind)[c],
where Ind is the (n x nb) seed-indicator the main evaluator already builds. Hyperparameters are
selected on the validation bundles by the caller; nothing here touches the test set.

Pure scipy/numpy: W inherits the co-occurrence sparsity (only co-seen items are nonzero) and is
further top-k sparsified per seed column, so these run at Spotify scale (254K items) on one CPU core.

References: Deshpande & Karypis (ItemKNN, TOIS'04); Cooper et al. / Paudel et al. (P3alpha);
Paudel et al. (RP3beta, re-ranking by target popularity).
"""
import numpy as np
import scipy.sparse as sp


def _topk_per_column(W, k):
    """Keep the k largest entries in each column of W (csc); drop the rest. Returns csr."""
    if k is None or k <= 0:
        return W.tocsr()
    W = W.tocsc()
    data, indices, indptr = W.data, W.indices, W.indptr
    keep_d, keep_r, keep_c = [], [], []
    for j in range(W.shape[1]):
        s, e = indptr[j], indptr[j + 1]
        if e - s == 0:
            continue
        col_d = data[s:e]; col_r = indices[s:e]
        if e - s > k:
            top = np.argpartition(col_d, -k)[-k:]
            col_d = col_d[top]; col_r = col_r[top]
        keep_d.append(col_d); keep_r.append(col_r)
        keep_c.append(np.full(col_r.shape[0], j))
    if not keep_d:
        return sp.csr_matrix(W.shape, dtype=np.float32)
    return sp.csr_matrix((np.concatenate(keep_d), (np.concatenate(keep_r), np.concatenate(keep_c))),
                         shape=W.shape, dtype=np.float32)


def itemknn_W(C, freq, topk=200, shrink=0.0):
    """Cosine item-item kNN with shrinkage, from the co-occurrence counts C (diag already 0):
        sim[c, i] = C[c, i] / (sqrt(f_c * f_i) + shrink).
    Symmetric; top-k neighbours per seed column."""
    Cc = C.tocoo()
    denom = np.sqrt(freq[Cc.row] * freq[Cc.col]) + float(shrink)
    data = Cc.data / np.maximum(denom, 1e-9)
    W = sp.csr_matrix((data.astype(np.float32), (Cc.row, Cc.col)), shape=C.shape)
    return _topk_per_column(W, topk)


def _bipartite_item_item(B, freq):
    """W0[c, i] = sum_u P(u | i) P(c | u): the length-2 item->bundle->item random-walk matrix.
    P(u|i) = B[u,i]/f_i (item->bundle, column-normalized B); P(c|u) = B[u,c]/|u| (bundle->item,
    row-normalized B)."""
    B = B.tocsr().astype(np.float32)
    size = np.asarray(B.sum(1)).ravel()                      # bundle sizes
    inv_f = np.divide(1.0, freq, out=np.zeros_like(freq, dtype=np.float64), where=freq > 0).astype(np.float32)
    inv_s = np.divide(1.0, size, out=np.zeros_like(size, dtype=np.float64), where=size > 0).astype(np.float32)
    Pib = B.multiply(inv_f[None, :]).tocsr()                 # P(u|i): (bundles x items)
    Pbi = B.multiply(inv_s[:, None]).tocsr()                 # P(c|u): (bundles x items)
    return (Pbi.T @ Pib).tocsr()                             # (items x items), W[c, i]


def p3alpha_W(B, freq, alpha=1.0, topk=200):
    """P3alpha: the length-3 bipartite random walk, transition weights raised to alpha."""
    W = _bipartite_item_item(B, freq)
    if alpha != 1.0:
        W.data = np.power(W.data, alpha, dtype=np.float32)
    return _topk_per_column(W, topk)


def rp3beta_W(B, freq, alpha=1.0, beta=0.0, topk=200):
    """RP3beta: P3alpha re-weighted by the target item's popularity, W[c,i] /= f_c^beta."""
    W = _bipartite_item_item(B, freq)
    if alpha != 1.0:
        W.data = np.power(W.data, alpha, dtype=np.float32)
    if beta != 0.0:
        pen = np.divide(1.0, np.power(freq, beta), out=np.zeros_like(freq, dtype=np.float64),
                        where=freq > 0).astype(np.float32)   # 1 / f_c^beta
        W = W.multiply(pen[:, None]).tocsr()                 # scale rows (candidate c)
    return _topk_per_column(W, topk)


# grids the caller selects over on validation
ITEMKNN_GRID = [(k, h) for k in (50, 100, 200, 400) for h in (0.0, 10.0, 100.0)]
P3ALPHA_GRID = [(a,) for a in (0.5, 1.0, 1.5)]
RP3BETA_GRID = [(a, b) for a in (0.5, 1.0, 1.5) for b in (0.0, 0.3, 0.6, 1.0)]
