"""ItemKNN: cosine item--item neighbourhood with shrinkage (Deshpande & Karypis, TOIS'04)."""
import numpy as np
import scipy.sparse as sp
from .base import ItemItemBaseline, topk_per_column

ITEMKNN_GRID = [(k, h) for k in (50, 100, 200, 400) for h in (0.0, 10.0, 100.0)]


def itemknn_W(C, freq, topk=200, shrink=0.0):
    """Cosine item--item kNN with shrinkage, from the co-occurrence counts ``C`` (diag already 0):
    ``sim[c, i] = C[c, i] / (sqrt(f_c * f_i) + shrink)``. Symmetric; top-k neighbours per seed column."""
    Cc = C.tocoo()
    denom = np.sqrt(freq[Cc.row] * freq[Cc.col]) + float(shrink)
    data = Cc.data / np.maximum(denom, 1e-9)
    W = sp.csr_matrix((data.astype(np.float32), (Cc.row, Cc.col)), shape=C.shape)
    return topk_per_column(W, topk)


class ItemKNN(ItemItemBaseline):
    name = "ItemKNN"
    grid = ITEMKNN_GRID

    def build_W(self, graph_op, freq, hp):
        k, h = hp
        return itemknn_W(graph_op.C, freq, topk=int(k), shrink=h)
