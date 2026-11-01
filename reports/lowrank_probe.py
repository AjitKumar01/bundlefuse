"""Does low-rank factorisation capture RARE items, or destroy them? Measures each embedding's per-bucket
Hit@20 ALONE (no fusion, no training -- just the dot-product ranker), across ranks, vs the EXACT signal.
Resolves the open contradiction: sefp_svd found low-rank PPMI-SVD can BEAT exact PPMI (denoising), which
would refute the claim that compression kills rare. Pure measurement, no conclusions baked in.

For matrix M (W_pair or PPMI):
  exact signal : score(c|S) = (M @ 1_S)[c]
  rank-r SVD   : score(c|S) = <U_c, sum_s V_s>,  U=svd.fit_transform(M), V=svd.components_.T
Buckets by TRAIN frequency. Env: DS, RANKS. Run: DS=pog_dense python3 reports/lowrank_probe.py
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import top_items

DS = os.environ.get("DS", "pog_dense")
RANKS = [int(r) for r in os.environ.get("RANKS", "16,64,128,256,512").split(",")]
EDGES = [(0, 0, "cold"), (1, 2, "rare"), (3, 10, "mid"), (11, 10**12, "head")]
ORDER = ["cold", "rare", "mid", "head"]


def log(*a):
    print(*a, flush=True)


def bkof(f):
    for i, (lo, hi, nm) in enumerate(EDGES):
        if lo <= f <= hi:
            return i
    return 3


def bucket_hits(score_fn, pairs, freq, k=20):
    buck = {nm: [0, 0] for _, _, nm in EDGES}
    for seeds, gt in pairs:
        sc = score_fn(list(seeds))
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
    freq = g.freq.astype(np.float64)
    test = prov.pairs("test")
    mats = {"W_pair": g.W_pair.tocsr(), "PPMI": g.PPMI.tocsr()}
    bcount = {nm: 0 for _, _, nm in EDGES}
    for _, gt in test:
        for t in set(gt):
            bcount[EDGES[bkof(int(freq[t]))][2]] += 1
    log(f"{DS}: n={n:,} test queries={len(test)} | bucket sizes {bcount} (setup {time.time()-t0:.0f}s)")

    for mname, M in mats.items():
        log(f"\n  === {mname} ===")
        log(f"    {'variant':12s} " + " ".join(f"{b:>8}" for b in ORDER))
        # exact signal
        def exact(seeds, _M=M):
            v = np.zeros(n, np.float32); v[seeds] = 1.0
            return np.asarray(_M @ v).ravel()
        bk = bucket_hits(exact, test, freq)
        log(f"    {'exact':12s} " + " ".join(f"{bk[b]:8.4f}" for b in ORDER))
        for r in RANKS:
            if r >= n - 1:
                continue
            svd = TruncatedSVD(n_components=r, random_state=0)
            U = svd.fit_transform(M).astype(np.float32); V = svd.components_.T.astype(np.float32)
            def emb(seeds, _U=U, _V=V):
                return _U @ _V[seeds].sum(0)
            bk = bucket_hits(emb, test, freq)
            log(f"    {'rank-' + str(r):12s} " + " ".join(f"{bk[b]:8.4f}" for b in ORDER) + f"   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
