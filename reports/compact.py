"""DiffBF-compact: the honest compact model. Keep the SPARSE collaborative signals EXACT (co-occurrence
and PPMI are ~1.3MB each as sparse matrices and carry rare/mid/head exactly -- low-rank would make them
BIGGER and destroy rare), and apply low-rank ONLY to the DENSE content (which is the real footprint).
Learnable fusion weights + empirical-Bayes shrinkage; fit on validation, test reported once.

  score(c|S) = (1-u) [ w_c * z_cooc + w_p * z_ppmi + w_t * z_content ] + u * w_prior * z_content
  z_* standardised per query; z_cooc=(C@1_S), z_ppmi=(PPMI@1_S) exact sparse; z_content=<U_c, sum V_s>
  (U,V = rank-r SVD of title content); u = kappa/(f_c+kappa)

Footprint = sparse C + sparse PPMI + rank-r content table. Keeps rare because the rare-carrying signal
(sparse co-occurrence) is exact. Env: DS, CONTENT_RANK, EPOCHS, LR. Run: DS=pog_dense python3 reports/compact.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import ranking_metrics, top_items

DS = os.environ.get("DS", "pog_dense")
CONTENT_RANK = int(os.environ.get("CONTENT_RANK", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
LR = float(os.environ.get("LR", "0.05"))
SEED = int(os.environ.get("SEED", "0"))
EDGES = [(0, 0, "cold"), (1, 2, "rare"), (3, 10, "mid"), (11, 10**12, "head")]
REF = {"pog_dense": {"CLHE": (0.3815, 0.2771), "BundleFuse": (0.3969, 0.2914)},
       "spotify": {"CLHE": (0.1082, 0.1806), "RaMen": (0.1091, 0.1882), "BundleFuse": (0.1486, 0.2532)},
       "spotify_sparse": {"CLHE": (0.0969, 0.1150), "BundleFuse": (0.1620, 0.2265)}}


def log(*a):
    print(*a, flush=True)


def bkof(f):
    for i, (lo, hi, nm) in enumerate(EDGES):
        if lo <= f <= hi:
            return i
    return 3


def zstd(v):
    return (v - v.mean()) / (v.std() + 1e-8)


class Compact(nn.Module):
    def __init__(self, freq_t, nsig):
        super().__init__()
        self.w = nn.Parameter(torch.full((nsig,), 0.4))        # fusion weights (head regime, u->0)
        self.wp = nn.Parameter(torch.full((nsig,), 0.1))       # LEARNED prior over ALL signals (rare regime,
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(np.expm1(5.0)))))   # u->1 -> can pick PPMI)
        self.freq_t = freq_t

    def forward(self, Z):
        """Z: (nsig,n) standardised signals. score = (1-u) Y + u mu0 ; u=kappa/(f+kappa) routes rare->prior."""
        Y = (Fn.softplus(self.w)[:, None] * Z).sum(0)
        mu0 = (Fn.softplus(self.wp)[:, None] * Z).sum(0)
        kappa = Fn.softplus(self.log_kappa); u = kappa / (self.freq_t + kappa)
        return (1 - u) * Y + u * mu0


def main():
    t0 = time.time()
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DS)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    prov = ProvidedSplits(dd, n)
    g = SimplicialGraphOperator(prov.train, n_items=n, svd_dim=2)
    freq = g.freq.astype(np.float64); freq_t = torch.tensor(freq, dtype=torch.float32)
    C = g.C.tocsr().astype(np.float32); PP = g.PPMI.tocsr().astype(np.float32)
    # low-rank content (dense -> compact)
    tpath = os.path.join(dd, "title_emb.npy")
    if os.path.exists(tpath):
        T = np.load(tpath).astype(np.float32)
    else:
        T = torch.load(os.path.join(dd, "content_feature.pt"), map_location="cpu", weights_only=False).detach().float().numpy()
    svd = TruncatedSVD(n_components=min(CONTENT_RANK, T.shape[1] - 1), random_state=0)
    Ut = svd.fit_transform(T).astype(np.float32)               # (n,r) item content vectors
    zpop = zstd(np.log1p(freq).astype(np.float32))             # popularity signal (per-item, seed-independent)
    # category co-occurrence (sparse/tiny): CatC[a,b] = # train co-occurrences of cat a with cat b
    raw = json.load(open(os.path.join(dd, "item_info.json")))
    def catof(i):
        v = raw.get(str(i), {}); return v.get("cate_id", v.get("cate", v.get("artist_name", -1)))
    labs = [catof(i) for i in range(n)]
    uniq = {c: k for k, c in enumerate(sorted(set(labs), key=str))}
    cat_lbl = np.array([uniq[c] for c in labs], dtype=np.int64); ncat = len(uniq)
    CatC = np.zeros((ncat, ncat), np.float32)
    for b in prov.train.values():
        cs = [cat_lbl[i] for i in b]
        for a in cs:
            for d in cs:
                if a != d:
                    CatC[a, d] += 1.0
    SIG = ["cooc", "ppmi", "simp", "diff2", "pop", "cat", "content"]
    def signals(seeds):
        ind = np.zeros(n, np.float32); ind[seeds] = 1.0
        zc = zstd(np.asarray(C @ ind).ravel())
        zp = zstd(np.asarray(PP @ ind).ravel())
        sc_pair, sc_d2 = g.score_pairwise_and_diff2([list(seeds)])
        delta_inc, _ = g.score_simplicial_incidence_only([list(seeds)], beta_clique=1.5)
        zs = zstd(np.asarray(sc_pair).ravel() + np.asarray(delta_inc).ravel())
        zd = zstd(np.asarray(sc_d2).ravel())
        seedcat = np.bincount(cat_lbl[seeds], minlength=ncat).astype(np.float32)
        zcat = zstd((CatC @ seedcat)[cat_lbl])
        zt = zstd(Ut @ Ut[seeds].sum(0))
        return torch.from_numpy(np.stack([zc, zp, zs, zd, zpop, zcat, zt]))   # (7,n), content last

    sparse_mb = lambda m: (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1e6
    B_mb = (g.B.data.nbytes + g.B.indices.nbytes + g.B.indptr.nbytes) / 1e6 if hasattr(g.B, "data") else 1.0
    foot = sparse_mb(C) + sparse_mb(PP) + B_mb + Ut.nbytes / 1e6
    model = Compact(freq_t, len(SIG))
    log(f"{DS}: n={n:,} content_rank={CONTENT_RANK} signals={SIG} | footprint: C {sparse_mb(C):.1f} + PPMI {sparse_mb(PP):.1f} + graphB {B_mb:.1f} + content {Ut.nbytes/1e6:.1f} = {foot:.1f}MB (setup {time.time()-t0:.0f}s)")

    val = prov.pairs("valid"); test = prov.pairs("test")
    nv = len(val); perm = rng.permutation(nv); nfit = int(0.8 * nv)
    fit_idx, es_idx = perm[:nfit], perm[nfit:]
    vpos = [([x for x in gt if x not in set(s)] or list(gt) or [0])[0] for s, gt in val]
    # precompute signals for all val queries once (fixed; only weights train)
    log(f"  precomputing val signals ({nv} queries)...")
    valsig = [signals(list(val[j][0])) for j in range(nv)]

    def es_recall(k=20):
        acc = np.zeros(2)
        with torch.no_grad():
            for j in es_idx:
                sc = model(valsig[j])
                acc += np.asarray(ranking_metrics(sc[:, None], [(list(val[j][0]), list(val[j][1]))], (k,))[k])
        return acc[0] / len(es_idx)

    if os.environ.get("FIT", "ce") == "metric":
        # BundleFuse's own regime: optimise validation Recall@20 DIRECTLY over the 15 weights (black-box),
        # instead of the CE proxy that over-sacrifices rare.
        from scipy.optimize import minimize
        keys = ["w", "wp", "log_kappa"]
        def set_vec(x):
            i = 0
            for k in keys:
                p = getattr(model, k); num = p.numel()
                p.data = torch.tensor(x[i:i + num], dtype=torch.float32).reshape(p.shape); i += num
        def obj(x):
            set_vec(x); return -es_recall(20)
        x0 = np.concatenate([getattr(model, k).detach().numpy().ravel() for k in keys])
        log(f"  FIT=metric: direct val-R@20 optimisation over {len(x0)} weights...")
        res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 1200, "xatol": 1e-3, "fatol": 1e-4})
        set_vec(res.x); log(f"    best val-es R@20={-res.fun:.4f} ({time.time()-t0:.0f}s)")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt = None
    else:
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, stale = (-1, best_state, 0) if opt is None else (-1, None, 0)
    for ep in (range(0) if opt is None else range(1, EPOCHS + 1)):
        o = fit_idx.copy(); rng.shuffle(o)
        for j in o:
            Z = valsig[j]; seeds = list(val[j][0])
            sc = model(Z); sc[seeds] = -1e9
            loss = Fn.cross_entropy(sc[None, :], torch.tensor([vpos[j]]))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0 or ep == EPOCHS:
            acc = np.zeros(2)
            with torch.no_grad():
                for j in es_idx:
                    sc = model(valsig[j])
                    m = ranking_metrics(sc[:, None], [(list(val[j][0]), list(val[j][1]))], (20,))
                    acc += np.asarray(m[20])
            r = acc[0] / len(es_idx)
            if r > best + 1e-5:
                best, best_state, stale = r, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"    [ep={ep}] es_R@20={r:.4f} best={best:.4f} w={[round(float(x),2) for x in Fn.softplus(model.w)]} ({time.time()-t0:.0f}s)")
            if stale >= 4:
                break
    if best_state:
        model.load_state_dict(best_state)

    acc = {k: np.zeros(2) for k in (10, 20)}; buck = {nm: [0, 0] for _, _, nm in EDGES}
    with torch.no_grad():
        for s, gt in test:
            sc = model(signals(list(s)))
            m = ranking_metrics(sc[:, None], [(list(s), list(gt))], (10, 20))
            for k in (10, 20):
                acc[k] += np.asarray(m[k])
            top = set(top_items(sc.numpy(), list(s), 20).tolist())
            for t in set(gt):
                b = buck[EDGES[bkof(int(freq[t]))][2]]; b[1] += 1; b[0] += int(t in top)
    N = len(test)
    bk = {nm: (v[0] / v[1] if v[1] else float("nan")) for nm, v in buck.items()}
    ref = REF.get(DS, {})
    log(f"\n  {DS}  DiffBF-compact (sparse-exact collab + rank-{CONTENT_RANK} content)  ({time.time()-t0:.0f}s)")
    log(f"    {'method':14s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}")
    for nm, v in ref.items():
        log(f"    {nm:14s} {'':8} {'':8} {v[0]:8.4f} {v[1]:8.4f}")
    log(f"    {'Compact':14s} {acc[10][0]/N:8.4f} {acc[10][1]/N:8.4f} {acc[20][0]/N:8.4f} {acc[20][1]/N:8.4f}")
    log(f"    buckets: cold {bk['cold']:.4f}  rare {bk['rare']:.4f}  mid {bk['mid']:.4f}  head {bk['head']:.4f}")
    log(f"    INFERENCE footprint: {foot:.1f}MB (sparse collab + low-rank content), {sum(p.numel() for p in model.parameters())} trainable scalars")


if __name__ == "__main__":
    main()
