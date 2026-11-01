"""Can we shrink the sparse co-occurrence memory by PRUNING (keep each item's top-k strongest partners,
drop the weak long tail) instead of low-rank? Pruning keeps exact strong edges + all of a rare item's
few edges, so it should preserve rare while cutting head's noise tail. Measures memory (nnz) and
per-bucket Hit@20 of the pruned exact signal vs full. Env: DS, KS. Run: DS=pog_dense python3 reports/prune_probe.py
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import top_items

DS = os.environ.get("DS", "pog_dense")
KS = [int(k) for k in os.environ.get("KS", "10,20,50,100").split(",")]
EDGES = [(0, 0, "cold"), (1, 2, "rare"), (3, 10, "mid"), (11, 10**12, "head")]
ORDER = ["cold", "rare", "mid", "head"]


def log(*a):
    print(*a, flush=True)


def bkof(f):
    for i, (lo, hi, nm) in enumerate(EDGES):
        if lo <= f <= hi:
            return i
    return 3


def prune_rows_topk(M, k):
    """Keep the top-k largest entries per row of a CSR matrix."""
    M = M.tocsr(); data, ind, indptr = [], [], [0]
    for r in range(M.shape[0]):
        s, e = M.indptr[r], M.indptr[r + 1]
        d = M.data[s:e]; c = M.indices[s:e]
        if len(d) > k:
            top = np.argpartition(d, -k)[-k:]
            d, c = d[top], c[top]
        data.append(d); ind.append(c); indptr.append(indptr[-1] + len(d))
    return sp.csr_matrix((np.concatenate(data) if data else np.array([]),
                          np.concatenate(ind) if ind else np.array([], int),
                          np.array(indptr)), shape=M.shape)


def smb(m):
    return (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1e6


def bucket_hits(M, pairs, freq, n, k=20):
    buck = {nm: [0, 0] for _, _, nm in EDGES}
    for seeds, gt in pairs:
        v = np.zeros(n, np.float32); v[list(seeds)] = 1.0
        sc = np.asarray(M @ v).ravel()
        top = set(top_items(sc, list(seeds), k).tolist())
        for t in set(gt):
            b = buck[EDGES[bkof(int(freq[t]))][2]]; b[1] += 1; b[0] += int(t in top)
    return {nm: (v[0] / v[1] if v[1] else float("nan")) for nm, v in buck.items()}


def main():
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DS)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    prov = ProvidedSplits(dd, n)
    g = SimplicialGraphOperator(prov.train, n_items=n, svd_dim=2)
    freq = g.freq.astype(np.float64); test = prov.pairs("test")
    for mname, M in [("W_pair", g.W_pair.tocsr()), ("PPMI", g.PPMI.tocsr())]:
        log(f"\n  === {mname} ({DS}) ===  full: {M.nnz:,} nnz = {smb(M):.1f}MB")
        log(f"    {'variant':10s} {'nnz':>12} {'MB':>7} {'cold':>8} {'rare':>8} {'mid':>8} {'head':>8}")
        bk = bucket_hits(M, test, freq, n)
        log(f"    {'full':10s} {M.nnz:>12,} {smb(M):>7.1f} " + " ".join(f"{bk[b]:8.4f}" for b in ORDER))
        for k in KS:
            Mp = prune_rows_topk(M, k)
            bk = bucket_hits(Mp, test, freq, n)
            log(f"    {'top-' + str(k):10s} {Mp.nnz:>12,} {smb(Mp):>7.1f} " + " ".join(f"{bk[b]:8.4f}" for b in ORDER) + f"   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
