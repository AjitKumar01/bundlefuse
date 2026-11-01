"""Differentiable BundleFuse: BundleFuse's exact closed form (signal fusion + empirical-Bayes shrinkage)
with ALL constants turned into trainable parameters, fit by gradient on the ranking objective instead of
BundleFuse's grid/greedy selection. The trainable parameters are the FUSION + SHRINKAGE coefficients over
the EXACT computed signals -- not an embedding table -- so there is no data-starvation (the failure mode of
learned embeddings) and no per-bucket discontinuity (the failure mode of the routers).

  theta(c|S) = (1-u) * sum_k w_k z_k   +   u * s * sum_k wp_k z_k ,   u = kappa/(f_c + kappa)
  (optional) per-signal frequency modulation:  w_k -> w_k * sigmoid(a_k * zlogf + b_k)   [MODULATE=1]

BundleFuse's weights are a feasible point of this model, so a continuous optimizer should match or beat its
grid search. CORRECT PROTOCOL (fixes the earlier test-peeking): fit on VALIDATION (BundleFuse's own regime)
with an internal val hold-out for early-stopping; report TEST exactly once. Uses the cached signal tensors
(no score_all_signals). Env: DS, SIG_CACHE, EPOCHS, LR, MODULATE, USE_SIG, SEED.
Run: DS=pog_dense python3 reports/diffbf.py
"""
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.paths import RESULTS_DIR, PROJECT_ROOT
from bundlefuse.metrics import ranking_metrics

DS = os.environ.get("DS", "pog_dense")
CACHE = os.environ.get("SIG_CACHE", os.path.join(RESULTS_DIR, "sefp", "sigcache"))
EPOCHS = int(os.environ.get("EPOCHS", "120"))
LR = float(os.environ.get("LR", "0.05"))
LOSS = os.environ.get("LOSS", "ce")                 # ce (softmax) | rank (smoothed-rank / RankNet surrogate)
TAU = float(os.environ.get("TAU", "1.0"))
MODULATE = os.environ.get("MODULATE", "0") != "0"
PRIOR_SIG = os.environ.get("PRIOR_SIG", "cat,bridge,softcat,style,cooc,ppmi").split(",")
SEED = int(os.environ.get("SEED", "0"))
EDGES = [(0, 0, "cold"), (1, 2, "rare"), (3, 10, "mid"), (11, 10**12, "head")]
# reference bars from results/matrix.csv
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


class DiffBF(nn.Module):
    def __init__(self, K, prior_mask, modulate):
        super().__init__()
        self.w = nn.Parameter(torch.full((K,), 0.3))
        self.wp = nn.Parameter(torch.full((K,), 0.1))
        self.prior_mask = prior_mask                        # (K,) 1 for prior-eligible signals
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(np.expm1(5.0)))))
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.modulate = modulate
        if modulate:
            self.a = nn.Parameter(torch.zeros(K)); self.b = nn.Parameter(torch.zeros(K))

    def weights(self, zlogf):
        w = Fn.softplus(self.w)
        if self.modulate:
            # per-signal smooth frequency gate; zlogf:(n,) standardized log-freq -> (n,K)
            g = torch.sigmoid(self.a[None, :] * zlogf[:, None] + self.b[None, :])
            w = w[None, :] * g                              # (n,K)
        return w

    def forward(self, Z, f, zlogf):
        """Z:(n,K) signals for one query; f:(n,) freq; zlogf:(n,). -> (n,) scores."""
        w = self.weights(zlogf)
        Y = (Z * w).sum(-1) if w.dim() == 2 else (Z * w).sum(-1)
        wp = Fn.softplus(self.wp) * self.prior_mask
        mu0 = Fn.softplus(self.log_scale) * (Z * wp).sum(-1)
        kappa = Fn.softplus(self.log_kappa)
        u = kappa / (f + kappa)
        return (1.0 - u) * Y + u * mu0


def load():
    meta = pickle.load(open(os.path.join(CACHE, f"{DS}_meta.pkl"), "rb"))
    valZ = np.load(os.path.join(CACHE, f"{DS}_validZ.npy"))              # (K,n,nv) fp16 -> RAM
    testZ = np.load(os.path.join(CACHE, f"{DS}_testZ.npy"), mmap_mode="r")
    return meta, valZ, testZ


def recall_at(model, Zsrc, seeds, gt, freq_t, zlogf, Ks=(10, 20), subset=None):
    acc = {k: np.zeros(2) for k in Ks}; cnt = 0
    idx = range(Zsrc.shape[2]) if subset is None else subset
    with torch.no_grad():
        for j in idx:
            Z = torch.from_numpy(np.ascontiguousarray(Zsrc[:, :, j]).astype(np.float32)).T  # (n,K)
            sc = model(Z, freq_t, zlogf)
            m = ranking_metrics(sc[:, None], [(list(seeds[j]), list(gt[j]))], Ks)
            for k in Ks:
                acc[k] += np.asarray(m[k])
            cnt += 1
    return {k: tuple(acc[k] / cnt) for k in Ks}


def buckets_at(model, Zsrc, seeds, gt, freq, freq_t, zlogf, k=20):
    from bundlefuse.metrics import top_items
    buck = {nm: [0, 0] for _, _, nm in EDGES}
    with torch.no_grad():
        for j in range(Zsrc.shape[2]):
            Z = torch.from_numpy(np.ascontiguousarray(Zsrc[:, :, j]).astype(np.float32)).T
            sc = model(Z, freq_t, zlogf).numpy()
            top = set(top_items(sc, list(seeds[j]), k).tolist())
            for t in set(gt[j]):
                b = buck[EDGES[bkof(int(freq[t]))][2]]; b[1] += 1; b[0] += int(t in top)
    return {nm: (v[0] / v[1] if v[1] else float("nan")) for nm, v in buck.items()}


def main():
    t0 = time.time()
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    meta, valZ, testZ = load()
    rsig = meta["rsig"]; freq = meta["freq"]; n = meta["n"]
    use = [s for s in os.environ.get("USE_SIG", ",".join(rsig)).split(",") if s in rsig]
    sidx = [rsig.index(s) for s in use]
    valZ = valZ[sidx]; freq_t = torch.tensor(freq, dtype=torch.float32)
    logf = np.log1p(freq); zlogf = torch.tensor((logf - logf.mean()) / (logf.std() + 1e-8), dtype=torch.float32)
    prior_mask = torch.tensor([1.0 if s in PRIOR_SIG else 0.0 for s in use])
    vseeds, vgt = meta["val_seeds"], meta["val_gt"]; nv = valZ.shape[2]
    # one pos per val query
    vpos = []
    for g, s in zip(vgt, vseeds):
        c = [x for x in g if x not in set(s)]; vpos.append(c[0] if c else (g[0] if g else 0))
    # split val: fit (80%) / earlystop (20%)
    perm = rng.permutation(nv); nfit = int(0.8 * nv)
    fit_idx, es_idx = perm[:nfit], perm[nfit:]
    log(f"{DS} DiffBF modulate={MODULATE} signals={use} | val fit={len(fit_idx)} es={len(es_idx)} test={testZ.shape[2]} (load {time.time()-t0:.0f}s)")

    model = DiffBF(len(use), prior_mask, MODULATE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    valZ_t = torch.from_numpy(valZ)                                      # (K,n,nv) fp16 in RAM
    best, best_state, stale = -1, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train(); o = fit_idx.copy(); rng.shuffle(o)
        for off in range(0, len(o), 128):
            bi = o[off:off + 128]
            Zc = valZ_t[:, :, bi].to(torch.float32).permute(2, 1, 0)     # (b,n,K)
            w = model.weights(zlogf)
            Y = (Zc * (w if w.dim() == 2 else w[None, None])).sum(-1)    # (b,n)
            wp = Fn.softplus(model.wp) * model.prior_mask
            mu0 = Fn.softplus(model.log_scale) * (Zc * wp).sum(-1)
            kappa = Fn.softplus(model.log_kappa); u = kappa / (freq_t + kappa)
            fused = (1 - u) * Y + u * mu0
            for r, j in enumerate(bi):
                fused[r, vseeds[j]] = -1e9
            tgt = torch.tensor([vpos[j] for j in bi])
            if LOSS == "rank":
                # smoothed rank of the positive: sum_j sigmoid((s_j - s_pos)/tau). Minimizing it directly
                # pushes the positive above every candidate (ranking objective, unlike softmax CE).
                spos = fused[torch.arange(len(bi)), tgt]                 # (b,)
                diff = (fused - spos[:, None]) / TAU                     # (b,n)
                diff[torch.arange(len(bi)), tgt] = -1e9                  # exclude the positive itself
                loss = torch.sigmoid(diff).sum(1).mean()
            else:
                loss = Fn.cross_entropy(fused, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == EPOCHS:
            model.eval()
            r = recall_at(model, valZ, vseeds, vgt, freq_t, zlogf, Ks=(20,), subset=es_idx)[20][0]
            if r > best + 1e-5:
                best, best_state, stale = r, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"    [ep={ep}] es_R@20={r:.4f} best={best:.4f} kappa={float(Fn.softplus(model.log_kappa)):.2f} ({time.time()-t0:.0f}s)")
            if stale >= 4:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    tseeds, tgt = meta["test_seeds"], meta["test_gt"]
    rr = recall_at(model, testZ, tseeds, tgt, freq_t, zlogf, Ks=(10, 20))
    bk = buckets_at(model, testZ, tseeds, tgt, freq, freq_t, zlogf)
    ref = REF.get(DS, {})
    log(f"\n  {DS}  Differentiable BundleFuse  (test, fit-on-val)  ({time.time()-t0:.0f}s)")
    log(f"    {'method':14s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}")
    for nm, v in ref.items():
        log(f"    {nm:14s} {'':8} {'':8} {v[0]:8.4f} {v[1]:8.4f}")
    log(f"    {'DiffBF':14s} {rr[10][0]:8.4f} {rr[10][1]:8.4f} {rr[20][0]:8.4f} {rr[20][1]:8.4f}")
    log(f"    buckets: cold {bk['cold']:.4f}  rare {bk['rare']:.4f}  mid {bk['mid']:.4f}  head {bk['head']:.4f}")
    np_params = sum(p.numel() for p in model.parameters())
    log(f"    trainable params: {np_params}")


if __name__ == "__main__":
    main()
