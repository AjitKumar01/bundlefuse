"""Validation selection objectives, shared by the e-commerce and retail evaluators.

The fusion weights, the higher-order mixing, and the shrinkage are all chosen to *maximise a
selection objective on the validation split*. That objective is deliberately decoupled from the
reported cutoff so the comparison is not "we tuned to our own metric":

======== ===================================================================================
name      meaning
======== ===================================================================================
ndcg<K>   NDCG at cutoff K (``ndcg20`` is the default; self-referential with the headline K=20)
recall<K> Recall at cutoff K
ndcgfull  untruncated NDCG over the whole ranking (no cutoff)
mrr       mean reciprocal rank (cutoff-free, top-weighted)
nll       standardised listwise softmax negative log-likelihood (a proper scoring rule, no cutoff)
======== ===================================================================================

All objectives are returned *higher-is-better* by :func:`sel_score` (``nll`` is negated), so the
greedy coordinate-ascent search is written once against a single sign convention. These numpy
implementations are shared verbatim by ``eval/bundlerec.py`` and ``eval/dunnhumby.py``; the
``eval/fullrank.py`` full-ranking evaluator keeps torch-tensor counterparts for its GPU path.
See ``notes/methodology.tex`` §"Why validate at NDCG@20" for why ~20 is the operating point.
"""
import math
import numpy as np


def softmax_nll_np(F, bundles):
    """Standardised listwise softmax NLL of the held-out items (lower=better).

    Per query: z-standardise the score column so the overall weight scale is irrelevant, mask the
    seeds out of the denominator, average ``-log softmax`` over the held-out items.
    ``bundles`` is a list of ``(seeds, held_out)`` index pairs.
    """
    mu = F.mean(0, keepdims=True); sd = F.std(0, keepdims=True) + 1e-9
    Z = ((F - mu) / sd).astype(np.float64)
    tot, cnt = 0.0, 0
    for j, (s, gt) in enumerate(bundles):
        if not gt:
            continue
        col = Z[:, j].copy(); col[list(s)] = -1e9
        m = float(col.max()); logden = m + math.log(float(np.exp(col - m).sum()))
        tot += float(-np.mean(col[list(gt)] - logden)); cnt += 1
    return tot / max(cnt, 1)


def full_ndcg_np(F, bundles):
    """Untruncated NDCG (no cutoff): discounted gain over the WHOLE ranking (higher=better)."""
    tot, cnt = 0.0, 0
    for j, (s, gt) in enumerate(bundles):
        if not gt:
            continue
        col = F[:, j].copy(); col[list(s)] = -1e9
        dcg = sum(1.0 / math.log2(int(np.count_nonzero(col > col[h])) + 2) for h in gt)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(len(gt)))
        tot += dcg / idcg; cnt += 1
    return tot / max(cnt, 1)


def mrr_np(F, bundles):
    """Mean reciprocal rank (cutoff-free, top-weighted by 1/rank; higher=better)."""
    tot, cnt = 0.0, 0
    for j, (s, gt) in enumerate(bundles):
        if not gt:
            continue
        col = F[:, j].copy(); col[list(s)] = -1e9
        rr = sum(1.0 / (int(np.count_nonzero(col > col[h])) + 1) for h in gt)
        tot += rr / len(gt); cnt += 1
    return tot / max(cnt, 1)


def sel_score(F, bundles, sel_obj, eval_mat):
    """Selection objective (higher=better) named by ``sel_obj``.

    ``ndcg<K>``/``recall<K>`` select at cutoff K and delegate to the evaluator's own ``eval_mat``
    (so the cutoff metric is computed identically to how it is reported); ``ndcgfull``/``mrr``/``nll``
    are cutoff-free and computed here.
    """
    if sel_obj == "nll":
        return -softmax_nll_np(F, bundles)
    if sel_obj == "ndcgfull":
        return full_ndcg_np(F, bundles)
    if sel_obj == "mrr":
        return mrr_np(F, bundles)
    if sel_obj.startswith("recall") and sel_obj[6:].isdigit():
        K = int(sel_obj[6:]); return eval_mat(F, bundles, Ks=(K,))[K][0]
    if sel_obj.startswith("ndcg") and sel_obj[4:].isdigit():
        K = int(sel_obj[4:]); return eval_mat(F, bundles, Ks=(K,))[K][1]
    return eval_mat(F, bundles, Ks=(20,))[20][1]
