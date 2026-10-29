"""Base classes and shared sparsification for the matched baseline suite.

Every baseline scores a candidate for a seed set ``S`` and is evaluated on the SAME
splits / candidates / metrics as BundleFuse. The common interface is:

    bl = ItemKNN()                      # a Baseline
    scorer = bl.make_scorer(graph_op, freq, hp)   # hp is one tuple from bl.grid
    M = scorer(seed_lists)              # dense (n_items x n_queries) score matrix

Most baselines are *item--item* models: a sparse ``W`` (n x n) with
``score(c | S) = sum_{i in S} W[c, i] = (W @ Ind)[c]``, where ``Ind`` is the
(n x nb) seed indicator the evaluator already builds. Those subclass
:class:`ItemItemBaseline` and only implement ``build_W``. Query-dependent models
(bundle-KNN, iALS) subclass :class:`Baseline` and implement ``make_scorer`` directly.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def topk_per_column(W, k):
    """Keep the ``k`` largest entries in each column of ``W`` (drop the rest); return csr.

    This caps memory/time so an item--item ``W`` runs at Spotify scale (254K items) on one core.
    """
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


class Baseline:
    """Abstract matched baseline.

    Subclasses set :attr:`name` and :attr:`grid` (a list of hyperparameter tuples selected over
    on validation) and implement :meth:`make_scorer`.
    """
    name: str = ""
    grid: list = []

    def make_scorer(self, graph_op, freq, hp):
        """Return ``scorer(seed_lists) -> np.ndarray`` of shape (n_items, n_queries) for one ``hp``.

        Return ``None`` to signal the baseline is gated out for this dataset (e.g. SLIM at large n).
        """
        raise NotImplementedError

    def select(self, graph_op, freq, eval_mat, vset, tset):
        """Grid-select on ``vset`` (validation NDCG@20), evaluate the winner on ``tset``.

        Returns ``(best_hp, test_metrics)`` or ``(None, None)`` if gated out. ``eval_mat`` is the
        caller's own metric: ``eval_mat(score_matrix, pairs, Ks) -> {K: (recall, ndcg)}``. The
        selection rule is strict-greater (first maximiser wins), matching the original harness.
        """
        best_hp, best_nd, best_scorer = None, -1.0, None
        for hp in self.grid:
            scorer = self.make_scorer(graph_op, freq, hp)
            if scorer is None:
                return None, None
            nd = eval_mat(scorer(_seeds(vset)), vset, Ks=(10, 20))[20][1]
            if nd > best_nd:
                best_nd, best_hp, best_scorer = nd, hp, scorer
        return best_hp, eval_mat(best_scorer(_seeds(tset)), tset, Ks=(10, 20))


class ItemItemBaseline(Baseline):
    """A baseline defined by a sparse item--item ``W``; scores by ``W @ Ind``."""

    def build_W(self, graph_op, freq, hp):
        """Return the sparse (n x n) weight matrix for one ``hp`` (or ``None`` if gated out)."""
        raise NotImplementedError

    def make_scorer(self, graph_op, freq, hp):
        W = self.build_W(graph_op, freq, hp)
        if W is None:
            return None
        def scorer(seed_lists):
            Ind = graph_op.build_indicator(seed_lists)
            return np.asarray((W @ Ind).todense()).astype(np.float32)
        scorer.W = W                      # exposed so the combiner can reuse selected W's
        return scorer


def _seeds(pairs):
    """Seed lists from (seeds, ground_truth) pairs."""
    return [s for (s, _) in pairs]
