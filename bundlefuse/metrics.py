"""Shared ranking metrics, with exact ties resolved by ascending catalogue item ID.

No random perturbations: a query's ranking is independent of batching, method order,
and the random state used by training. Seeds are excluded before ranks are computed.
"""
import math
import numpy as np

TIE_POLICY = "score_desc_item_id_asc"


def as_numpy(scores):
    if hasattr(scores, "detach"):
        return scores.detach().cpu().numpy()
    return np.asarray(scores)


def top_items(column, seeds, k):
    """Exact top-k (including boundary ties), excluding seeds. Never mutates input."""
    values = np.asarray(column).copy()
    values[list(seeds)] = -np.inf
    k = min(int(k), len(values) - len(set(seeds)))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    # Partition keeps large catalogues linear-time; only the selected items are sorted.
    threshold = np.partition(values, len(values) - k)[len(values) - k]
    above = np.flatnonzero(values > threshold)
    tied = np.flatnonzero(values == threshold)[:k - len(above)]
    candidates = np.concatenate((above, tied))
    return candidates[np.lexsort((candidates, -values[candidates]))]


def relevant_ranks(column, seeds, ground_truth):
    """One-based ranks of distinct positives, under the same total order as top_items."""
    values = np.asarray(column)
    available = np.ones(len(values), dtype=bool)
    available[list(seeds)] = False
    ids = np.arange(len(values))
    return np.asarray([
        1 + np.count_nonzero(available & ((values > values[item]) |
            ((values == values[item]) & (ids < item))))
        for item in sorted(set(ground_truth))
    ], dtype=np.int64)


def ranking_metrics(scores, pairs, Ks=(10, 20)):
    """Macro Recall/NDCG at each cutoff, for (seeds, ground_truth) pairs."""
    scores = as_numpy(scores)
    if scores.ndim != 2 or scores.shape[1] != len(pairs):
        raise ValueError("scores must have shape (catalogue items, queries)")
    if not pairs or not Ks or min(Ks) < 1:
        raise ValueError("non-empty queries and positive cutoffs are required")
    totals = {k: np.zeros(2, dtype=np.float64) for k in Ks}
    for j, (seeds, gt) in enumerate(pairs):
        truth = set(gt)
        if not truth or truth.intersection(seeds):
            raise ValueError("ground truth must be non-empty and disjoint from seeds")
        predicted = top_items(scores[:, j], seeds, max(Ks))
        hits = np.isin(predicted, list(truth))
        for k in Ks:
            h = hits[:k]
            dcg = float((h / np.log2(np.arange(len(h)) + 2)).sum())
            ideal = float((1 / np.log2(np.arange(min(len(truth), k)) + 2)).sum())
            totals[k] += (float(h.sum()) / len(truth), dcg / ideal)
    return {k: tuple(totals[k] / len(pairs)) for k in Ks}


def full_ndcg(scores, pairs):
    scores = as_numpy(scores)
    values = []
    for j, (seeds, gt) in enumerate(pairs):
        if gt:
            ranks = relevant_ranks(scores[:, j], seeds, gt)
            dcg = float((1 / np.log2(ranks + 1)).sum())
            ideal = sum(1 / math.log2(i + 2) for i in range(len(ranks)))
            values.append(dcg / ideal)
    return float(np.mean(values)) if values else 0.0


def mean_reciprocal_rank(scores, pairs):
    """MRR uses the FIRST relevant result for each query, including multi-positive queries."""
    scores = as_numpy(scores)
    values = [1.0 / relevant_ranks(scores[:, j], seeds, gt).min()
              for j, (seeds, gt) in enumerate(pairs) if gt]
    return float(np.mean(values)) if values else 0.0
