"""Restore omitted graph matrices with the same formulas as SimplicialGraphOperator.

Checkpoint storage may omit derived matrices. Rebuilding them is exact, not a different
incidence-based approximation; in particular, the co-occurrence diagonal remains zero.
"""
import numpy as np
import scipy.sparse as sp


def ensure_graph_matrices(graph, required):
    required = set(required)
    need_ppmi = 'ppmi' in required and getattr(graph, 'PPMI', None) is None
    if ({'cooc', 'user_cooc'} & required or need_ppmi) and getattr(graph, 'C', None) is None:
        graph.C = (graph.B.T @ graph.B).tocsr()
        graph.C.setdiag(0)
        graph.C.eliminate_zeros()
    if need_ppmi:
        freq = np.asarray(graph.C.sum(1)).ravel() + 1e-9
        total = float(graph.C.sum())
        values = graph.C.tocoo()
        ppmi = np.maximum(0., np.log((values.data * total) /
                          (freq[values.row] * freq[values.col]) + 1e-12)).astype(np.float32)
        graph.PPMI = sp.csr_matrix((ppmi, (values.row, values.col)), shape=graph.C.shape)
