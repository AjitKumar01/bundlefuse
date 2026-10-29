"""Bipartite random-walk item--item baselines: P3alpha and RP3beta (Cooper/Paudel et al.)."""
import numpy as np
from .base import ItemItemBaseline, topk_per_column

P3ALPHA_GRID = [(a,) for a in (0.5, 1.0, 1.5)]
RP3BETA_GRID = [(a, b) for a in (0.5, 1.0, 1.5) for b in (0.0, 0.3, 0.6, 1.0)]


def bipartite_item_item(B, freq):
    """``W0[c, i] = sum_u P(u | i) P(c | u)``: the length-2 item->bundle->item random-walk matrix.
    ``P(u|i) = B[u,i]/f_i`` (item->bundle, column-normalized B); ``P(c|u) = B[u,c]/|u|`` (bundle->item)."""
    B = B.tocsr().astype(np.float32)
    size = np.asarray(B.sum(1)).ravel()                      # bundle sizes
    inv_f = np.divide(1.0, freq, out=np.zeros_like(freq, dtype=np.float64), where=freq > 0).astype(np.float32)
    inv_s = np.divide(1.0, size, out=np.zeros_like(size, dtype=np.float64), where=size > 0).astype(np.float32)
    Pib = B.multiply(inv_f[None, :]).tocsr()                 # P(u|i): (bundles x items)
    Pbi = B.multiply(inv_s[:, None]).tocsr()                 # P(c|u): (bundles x items)
    return (Pbi.T @ Pib).tocsr()                             # (items x items), W[c, i]


def p3alpha_W(B, freq, alpha=1.0, topk=200):
    """P3alpha: the length-3 bipartite random walk, transition weights raised to ``alpha``."""
    W = bipartite_item_item(B, freq)
    if alpha != 1.0:
        W.data = np.power(W.data, alpha, dtype=np.float32)
    return topk_per_column(W, topk)


def rp3beta_W(B, freq, alpha=1.0, beta=0.0, topk=200):
    """RP3beta: P3alpha re-weighted by the target item's popularity, ``W[c,i] /= f_c^beta``."""
    W = bipartite_item_item(B, freq)
    if alpha != 1.0:
        W.data = np.power(W.data, alpha, dtype=np.float32)
    if beta != 0.0:
        pen = np.divide(1.0, np.power(freq, beta), out=np.zeros_like(freq, dtype=np.float64),
                        where=freq > 0).astype(np.float32)   # 1 / f_c^beta
        W = W.multiply(pen[:, None]).tocsr()                 # scale rows (candidate c)
    return topk_per_column(W, topk)


class P3alpha(ItemItemBaseline):
    name = "P3alpha"
    grid = P3ALPHA_GRID

    def build_W(self, graph_op, freq, hp):
        (a,) = hp
        return p3alpha_W(graph_op.B, freq, alpha=a, topk=200)


class RP3beta(ItemItemBaseline):
    name = "RP3beta"
    grid = RP3BETA_GRID

    def build_W(self, graph_op, freq, hp):
        a, b = hp
        return rp3beta_W(graph_op.B, freq, alpha=a, beta=b, topk=200)
