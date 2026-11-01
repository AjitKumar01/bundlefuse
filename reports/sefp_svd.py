"""SEFP-P via SVD: the rate-distortion curve of compressing bundle completion into d parameters/item.

A parameter-only item embedding is the rank-d SVD of a pairwise kernel M (co-occurrence C, PPMI, or
W_pair). By Eckart-Young the rank-d SVD is THE optimal rank-d approximation of M (Frobenius), computed
from the FULL statistics (all pairs) -- no data-starvation, unlike gradient-trained embeddings fit from
14k resampled bundles. For M ~= A B^T (A = U S = svd.fit_transform, B = V = components_.T):

    score(c|S) = sum_{s in S} Mhat[c,s] = A_c . (sum_{s in S} B_s) = <A_c, q_S>   (MIPS-retrievable)

As d -> full rank, Mhat -> M and the score -> the EXACT kernel-sum (computed directly as an anchor).
ONE SVD at D_MAX gives the whole curve (nested singular vectors). Inference is parameter-only: store
A (and B for asymmetric M), one dense matvec, NO co-occurrence graph at query time.

Smoke finding that set the kernel list: exact cooc-sum (0.106 supplied) BEATS PPMI-sum (0.067) and all
SVD-of-PPMI ranks -- this data rewards popularity, which PPMI removes -- and PPMI-sum collapses at
one_seed (0.0008, the rare-item pathology). So C is the leading target; PPMI/W_pair are compared.

d = rate (~32d bits/item), Recall@20 = 1 - distortion. Where each kernel's curve saturates, and how
close it gets to the feature-bound references (BundleFuse 0.156, SEFP-feat 0.120, two-tower 0.105), is
the answer to "to what level can we compress the task into per-item parameters". Honesty: Eckart-Young
is optimal for RECONSTRUCTING M, not for RECALL; a ranking-tuned code may sit above this per dim.

Env: KERNELS (cooc,ppmi,wpair), DIMS (8,16,...,512), POP (w*log-freq added, default 0), TEST_N,
CHUNK, DATASET. Writes results/sefp/svd_<dataset>.json.  Run: python3 reports/sefp_svd.py
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import ranking_metrics
from bundlefuse.robust_protocol import sparse_pairs

DATASET = os.environ.get("DATASET", "spotify")
KERNELS = os.environ.get("KERNELS", "cooc,ppmi,wpair").split(",")
DIMS = [int(x) for x in os.environ.get("DIMS", "8,16,32,64,128,256,512").split(",")]
D_MAX = max(DIMS)
POP = float(os.environ.get("POP", "0"))
TEST_N = int(os.environ.get("TEST_N", "0"))
CHUNK = int(os.environ.get("CHUNK", "200"))
ANCHORS = {"BundleFuse": 0.1556, "SEFP_feat": 0.1203, "TwoTower_G1": 0.1053}


def log(*a):
    print(*a, flush=True)


def recall_curve(A, B, pairs, dims, n, logf, pop, Ks=(10, 20, 50)):
    """R@k vs rank d for score(c|S) = A[:, :d] @ (sum_{s} B[s, :d]). Batched: one ranking_metrics call
    per (d, chunk) over the (N, chunk) score matrix."""
    out = {d: {k: np.zeros(2) for k in Ks} for d in dims}
    nq = len(pairs)
    for c0 in range(0, nq, CHUNK):
        chunk = pairs[c0:c0 + CHUNK]
        qB = np.zeros((len(chunk), B.shape[1]), np.float32)
        for i, (seeds, _) in enumerate(chunk):
            qB[i] = B[list(seeds)].sum(0)
        for d in dims:
            sc = A[:, :d] @ qB[:, :d].T                     # (N, chunk)
            if pop:
                sc = sc + pop * logf[:, None]
            mm = ranking_metrics(sc, chunk, Ks=Ks)
            for k in Ks:
                out[d][k] += np.asarray(mm[k]) * len(chunk)
    return {d: {k: tuple(out[d][k] / nq) for k in Ks} for d in dims}


def exact_sum_recall(M, pairs, n, Ks=(10, 20, 50)):
    """Exact Sum_s M[:,s] ranking (d=inf anchor), batched by chunk."""
    rec = {k: np.zeros(2) for k in Ks}
    nq = len(pairs)
    for c0 in range(0, nq, CHUNK):
        chunk = pairs[c0:c0 + CHUNK]
        X = np.zeros((n, len(chunk)), np.float32)
        for i, (seeds, _) in enumerate(chunk):
            X[list(seeds), i] = 1.0
        SC = np.asarray(M @ X)                              # (N, chunk)
        mm = ranking_metrics(SC, chunk, Ks=Ks)
        for k in Ks:
            rec[k] += np.asarray(mm[k]) * len(chunk)
    return {k: tuple(rec[k] / nq) for k in Ks}


def main():
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DATASET)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    provided = ProvidedSplits(dd, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    freq = graph.freq.astype(np.float32)
    logf = np.log1p(freq).astype(np.float32)
    mats = {"cooc": graph.C.tocsr(), "ppmi": graph.PPMI.tocsr(), "wpair": graph.W_pair.tocsr()}
    symmetric = {"cooc": True, "ppmi": True, "wpair": False}
    log(f"{DATASET}: n={n} | KERNELS={KERNELS} DIMS={DIMS} POP={POP}")

    test_pairs = provided.pairs("test")
    test_ids = sorted(provided.splits["test"][0])
    oneseed = sparse_pairs(provided.pairs("test"), test_ids, "one_seed")
    if TEST_N:
        test_pairs, oneseed = test_pairs[:TEST_N], oneseed[:TEST_N]
    conds = [("supplied", test_pairs), ("one_seed", oneseed)]
    log(f"supplied={len(test_pairs)} one_seed={len(oneseed)}")

    results = {"dataset": DATASET, "dims": DIMS, "pop": POP, "anchors": ANCHORS, "kernels": {}}
    for kname in KERNELS:
        M = mats[kname]
        ts = time.time()
        svd = TruncatedSVD(n_components=D_MAX, random_state=0)
        A = svd.fit_transform(M).astype(np.float32)         # U S  (n, D_MAX)
        B = svd.components_.T.astype(np.float32)             # V    (n, D_MAX)
        S = svd.singular_values_.astype(np.float32)
        evr = float(svd.explained_variance_ratio_.sum())
        store = "A" if symmetric[kname] else "A+B"
        log(f"\n### kernel={kname} ({'sym' if symmetric[kname] else 'asym'}, store {store}) "
            f"nnz={M.nnz:,} SVD(d={D_MAX}) in {time.time()-ts:.0f}s | explained_var={evr:.3f} "
            f"| S/S0 at dims: " + " ".join(f"{dm}:{S[min(dm,len(S))-1]/S[0]:.3f}" for dm in DIMS))
        kres = {"symmetric": symmetric[kname], "explained_var": evr,
                "spectrum_head": S[:16].tolist(), "conditions": {}}
        for cond, pairs in conds:
            tc = time.time()
            curve = recall_curve(A, B, pairs, DIMS, n, logf, POP)
            ex = exact_sum_recall(M, pairs, n)
            kres["conditions"][cond] = {
                "curve": {str(dm): {"R@20": curve[dm][20][0], "N@20": curve[dm][20][1],
                                    "R@10": curve[dm][10][0], "R@50": curve[dm][50][0]} for dm in DIMS},
                "exact_sum": {"R@20": ex[20][0], "N@20": ex[20][1]}}
            log(f"  [{kname}/{cond}] ({time.time()-tc:.0f}s) R@20 by rank:  "
                + "  ".join(f"d{dm}={curve[dm][20][0]:.4f}" for dm in DIMS)
                + f"  | d=inf(exact)={ex[20][0]:.4f}")
        results["kernels"][kname] = kres

    out_dir = os.path.join(PROJECT_ROOT, "results", "sefp")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"svd_{DATASET}.json")
    json.dump(results, open(outp, "w"), indent=2)

    log(f"\n=== SEFP-P RATE-DISTORTION SUMMARY ({time.time()-t0:.0f}s) ===")
    log(f"references: BundleFuse(feat)=0.1556  SEFP(feat)=0.1203  two-tower(G1)=0.1053")
    for cond in ("supplied", "one_seed"):
        log(f"\n[{cond}] R@20   " + "".join(f"{('d'+str(dm)):>9}" for dm in DIMS) + f"{'exact':>9}")
        for kname in KERNELS:
            c = results["kernels"][kname]["conditions"][cond]
            row = "".join(f"{c['curve'][str(dm)]['R@20']:9.4f}" for dm in DIMS)
            log(f"  {kname:10s}" + row + f"{c['exact_sum']['R@20']:9.4f}")
    log(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
