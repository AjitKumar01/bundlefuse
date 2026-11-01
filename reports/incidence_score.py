"""VERIFY the general memory algorithm: compute the collaborative signals from the sparse incidence B
(bundles x items) instead of the materialized item-item matrices C / PPMI, and check the scores are
IDENTICAL (lossless). Footprint = B + a few per-item vectors, vs C+W_pair+PPMI materialized.

  cooc(c|S) = (C @ 1_S)[c],  C = B^T B  =>  cooc_B = B^T (B 1_S)            (exact for non-seed c)
  ppmi(c|S) = sum_s PPMI[c,s],  PPMI[c,s]=max(log(C[c,s]*T/(f_c f_s)),0)
            => from B:  C[:,S] = B^T B[:,S] (n x |S|), transform per entry, sum over s   (exact)

Env: DS, NQ. Run: DS=pog_dense python3 reports/incidence_score.py
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

DS = os.environ.get("DS", "pog_dense")
NQ = int(os.environ.get("NQ", "200"))


def log(*a):
    print(*a, flush=True)


def smb(m):
    m = m.tocsr(); return (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1e6


def main():
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DS)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    prov = ProvidedSplits(dd, n)
    g = SimplicialGraphOperator(prov.train, n_items=n, svd_dim=2)
    B = g.B.tocsr(); BT = B.T.tocsr()                 # B: (n_bundles, n_items)
    C = g.C.tocsr(); PPMI = g.PPMI.tocsr()
    # PPMI normaliser = co-occurrence DEGREE (C.sum(1)), a per-item vector derivable from B as
    # B^T(B 1) - diag(B^T B); stored with B (tiny). totc = C.sum() is one scalar.
    ones = np.ones(n, np.float64)
    diagC = np.asarray(B.multiply(B).sum(0)).ravel()
    freq_c = np.asarray(BT @ (B @ ones)).ravel() - diagC + 1e-9          # == C.sum(1), from B
    totc = float(freq_c.sum() + diagC.sum() - diagC.sum())               # = C.sum() (setdiag already excluded)
    totc = float(np.asarray(BT @ (B @ ones)).ravel().sum() - diagC.sum())
    test = prov.pairs("test")[:NQ]
    log(f"{DS}: n={n:,} bundles={B.shape[0]:,} | verifying B-based == materialized on {len(test)} queries (setup {time.time()-t0:.0f}s)")

    # footprint
    mat_mb = smb(C) + smb(g.W_pair) + smb(PPMI)
    inc_mb = smb(B) + (freq_c.nbytes + g.inv_sqrt_f.nbytes + g.inv_vol_pow.nbytes) / 1e6
    log(f"  footprint: materialized C+W_pair+PPMI = {mat_mb:.1f}MB  vs  incidence B + vectors = {inc_mb:.1f}MB  ({mat_mb/max(inc_mb,1e-9):.0f}x)")

    max_cooc, max_ppmi = 0.0, 0.0
    for seeds, gt in test:
        s = list(seeds); ind = np.zeros(n, np.float32); ind[s] = 1.0
        # exact (materialized)
        cooc_mat = np.asarray(C @ ind).ravel()
        ppmi_mat = np.asarray(PPMI @ ind).ravel()
        # from incidence B
        cooc_B = np.asarray(BT @ (B @ ind)).ravel()                 # = (B^T B) 1_S
        Cs = np.asarray(BT @ B[:, s].toarray())                     # (n,|S|) = C[:, S]  (incl diagonal for c in s)
        with np.errstate(divide="ignore", invalid="ignore"):
            pp = np.log(Cs * totc / (freq_c[:, None] * freq_c[s][None, :] + 1e-12))
        pp = np.maximum(pp, 0.0); pp[Cs <= 0] = 0.0
        ppmi_B = pp.sum(1).astype(np.float32)
        # compare on NON-seed items (seeds are excluded from ranking; diagonal differs only there)
        mask = np.ones(n, bool); mask[s] = False
        max_cooc = max(max_cooc, float(np.abs(cooc_mat[mask] - cooc_B[mask]).max()))
        max_ppmi = max(max_ppmi, float(np.abs(ppmi_mat[mask] - ppmi_B[mask]).max()))
    log(f"  max|Δ| over non-seed candidates:  cooc = {max_cooc:.2e}   ppmi = {max_ppmi:.2e}")
    log(f"  => {'LOSSLESS (identical scores)' if max(max_cooc, max_ppmi) < 1e-3 else 'MISMATCH'} ; "
        f"collaborative signals reproduced from a {inc_mb:.1f}MB incidence instead of {mat_mb:.1f}MB of matrices ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
