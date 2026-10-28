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


def slim_W(B, alpha=1e-3, l1_ratio=0.1, topk=200, max_iter=100, n_max=8000):
    """SLIM (Ning & Karypis, ICDM'11): sparse item-item via per-item non-negative elastic-net
    regression B[:,c] ~ B[:,!=c]. O(n) ElasticNet solves, so gated to small catalogues (n<=n_max);
    at larger scale the closed-form EASE already in the harness represents the sparse-linear family.
    Returns W with W[c,i] = weight of seed i for candidate c (score = W @ Ind), or None if gated out."""
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
    return _topk_per_column(W, topk)


def bundleknn_scores(B, Ind, topn=100):
    """User(=bundle)-based KNN in bundle space: score(c | S) = sum over the top-N train bundles most
    similar to the seed set S of cosine(S, u) * B[u, c]. Query-dependent, so scored per chunk rather
    than as a fixed item-item W. Returns a dense (n x nb) numpy array."""
    B = B.tocsr().astype(np.float32)
    bnorm = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-9    # ||u||
    Q = Ind.T.tocsr().astype(np.float32)                               # (nb x n) seed sets
    qnorm = np.sqrt(np.asarray(Q.multiply(Q).sum(1)).ravel()) + 1e-9    # ||S||
    sim = np.asarray((Q @ B.T).todense()) / (np.outer(qnorm, bnorm))   # (nb x n_bundles) cosine
    nb = sim.shape[0]
    if topn and topn < sim.shape[1]:
        cut = np.partition(sim, -topn, axis=1)[:, -topn][:, None]
        sim = np.where(sim >= cut, sim, 0.0)
    scores = sim @ B                                                   # (nb x n)
    return np.asarray(scores).T                                        # (n x nb)


# grids the caller selects over on validation
ITEMKNN_GRID = [(k, h) for k in (50, 100, 200, 400) for h in (0.0, 10.0, 100.0)]
P3ALPHA_GRID = [(a,) for a in (0.5, 1.0, 1.5)]
RP3BETA_GRID = [(a, b) for a in (0.5, 1.0, 1.5) for b in (0.0, 0.3, 0.6, 1.0)]
SLIM_GRID = [(a, l) for a in (1e-4, 1e-3, 1e-2) for l in (0.01, 0.1, 0.5)]
BUNDLEKNN_GRID = [(tn,) for tn in (25, 50, 100, 200)]


def run_matched_seed(graph_op, freq, eval_mat, vset, tset):
    """One seed of the matched baseline suite for the (seed, gt)-pair evaluators (bundlerec, dunnhumby).
    `eval_mat(score_np, pairs, Ks)` is the caller's own metric. Hyperparameters are selected on `vset`
    and scored on `tset`. Returns {method: {K: (R, N)}} for this seed."""
    import numpy as _np
    Cm, Bm = graph_op.C, graph_op.B

    def _ind(pairs):
        return graph_op.build_indicator([s for (s, _) in pairs])

    def _wfn(Wm):
        return lambda pairs: _np.asarray((Wm @ _ind(pairs)).todense()).astype(_np.float32)

    def _ev(fn, pairs):
        return eval_mat(fn(pairs), pairs, Ks=(10, 20))

    def _selW(build, grid):
        bW, bhp, bnd = None, None, -1.0
        for hp in grid:
            Wm = build(*hp)
            if Wm is None:
                return None, None, None
            nd = _ev(_wfn(Wm), vset)[20][1]
            if nd > bnd:
                bnd, bhp, bW = nd, hp, Wm
        return bW, bhp, bnd

    res, comp = {}, {}
    for nm, bd, gr in [("ItemKNN", lambda k, h: itemknn_W(Cm, freq, topk=int(k), shrink=h), ITEMKNN_GRID),
                       ("P3alpha", lambda a: p3alpha_W(Bm, freq, alpha=a, topk=200), P3ALPHA_GRID),
                       ("RP3beta", lambda a, b: rp3beta_W(Bm, freq, alpha=a, beta=b, topk=200), RP3BETA_GRID)]:
        Wm, _, _ = _selW(bd, gr); comp[nm] = Wm; res[nm] = _ev(_wfn(Wm), tset)
    Ws, _, _ = _selW(lambda a, l: slim_W(Bm, alpha=a, l1_ratio=l, topk=200), SLIM_GRID)
    if Ws is not None:
        res["SLIM"] = _ev(_wfn(Ws), tset)

    def _bk(tn):
        return lambda pairs: bundleknn_scores(Bm, _ind(pairs), topn=tn)
    bbk, bnd = None, -1.0
    for (tn,) in BUNDLEKNN_GRID:
        nd = _ev(_bk(tn), vset)[20][1]
        if nd > bnd:
            bnd, bbk = nd, tn
    res["BundleKNN"] = _ev(_bk(bbk), tset)

    cn = [k for k in ("ItemKNN", "P3alpha", "RP3beta") if comp.get(k) is not None]
    nW = {}
    for k in cn:
        Wm = comp[k].tocsr().copy().astype(_np.float32); mx = float(Wm.data.max()) if Wm.nnz else 1.0
        Wm.data = Wm.data / (mx + 1e-12); nW[k] = Wm
    cw, cnd = {k: 0.0 for k in cn}, -1.0
    for _ in range(3):
        imp = False
        for k in cn:
            for v in (0.0, 0.5, 1.0, 2.0):
                cand = dict(cw); cand[k] = v
                if sum(cand.values()) == 0:
                    continue
                Wc = sum(cand[k2] * nW[k2] for k2 in cn if cand[k2])
                nd = _ev(_wfn(Wc), vset)[20][1]
                if nd > cnd + 1e-5:
                    cnd, cw, imp = nd, cand, True
        if not imp:
            break
    Wc = sum(cw[k2] * nW[k2] for k2 in cn if cw[k2])
    res["LinearCombiner"] = _ev(_wfn(Wc), tset)
    return res
