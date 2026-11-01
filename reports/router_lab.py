"""Fast router iteration over CACHED exact signals (built once by `DUMP_SIGNALS` in eval/fullrank.py).
No score_all_signals here -- loads the cached train pos/neg signal tensors and the full test signal
tensor, so each router idea trains+evaluates in seconds. Reports per-frequency-bucket Hit@20 + R@/N@20
vs BundleFuse. Pick the router family with ROUTER=... ; add families here, not new fullrank runs.

Env: DS, SIG_CACHE, ROUTER (linear_std|linear_bal|mlp|...), EPOCHS, USE_SIG (comma subset of cached).
Run: DS=pog_dense ROUTER=linear_bal python3 reports/router_lab.py
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
from bundlefuse.paths import RESULTS_DIR
from bundlefuse.metrics import ranking_metrics

DS = os.environ.get("DS", "pog_dense")
CACHE = os.environ.get("SIG_CACHE", os.path.join(RESULTS_DIR, "sefp", "sigcache"))
ROUTER = os.environ.get("ROUTER", "linear_bal")
EPOCHS = int(os.environ.get("EPOCHS", "400"))
EDGES = [(0, 0, "cold"), (1, 2, "rare"), (3, 10, "mid"), (11, 10**12, "head")]
BF = {"pog_dense": {"cold": 0.0, "rare": 0.0935, "mid": 0.1578, "head": 0.7503, "R@20": 0.3969, "N@20": 0.2914}}


def log(*a):
    print(*a, flush=True)


def bkof(f):
    for i, (lo, hi, nm) in enumerate(EDGES):
        if lo <= f <= hi:
            return i
    return 3


def load():
    meta = pickle.load(open(os.path.join(CACHE, f"{DS}_meta.pkl"), "rb"))
    testZ = np.load(os.path.join(CACHE, f"{DS}_testZ.npy"), mmap_mode="r")   # (nsig,n,nb) fp16
    return meta, testZ


# BundleFuse selected pog_dense fusion weights (results/component_study/pog_dense/selection.json)
BF_W = {"cat": 0.5, "bridge": 0.25, "simp": 0.5, "pop": 0.25, "csvd": 0.25}


def make_router(name, nsig, use=None):
    """Returns (params, forward(Z, logf, bidx)) where Z:(...,nsig), logf:(...), bidx:(...) long."""
    if name == "fixed_bf":
        w = torch.tensor([BF_W.get(s, 0.0) for s in use])                # (nsig,)
        def fwd(Z, logf, bidx):
            return (Z * w).sum(-1)
        return [], fwd                                                   # no params -> no training
    if name.startswith("linear"):
        W = nn.Parameter(torch.full((len(EDGES), nsig), 0.1))
        def fwd(Z, logf, bidx):
            return (W[bidx] * Z).sum(-1)
        return [W], fwd
    if name == "mlp":
        net = nn.Sequential(nn.Linear(nsig + 1, 32), nn.GELU(), nn.Linear(32, 1))
        def fwd(Z, logf, bidx):
            return net(torch.cat([Z, logf.unsqueeze(-1)], -1)).squeeze(-1)
        return list(net.parameters()), fwd
    raise ValueError(name)


def main():
    t0 = time.time()
    meta, testZ = load()
    rsig = meta["rsig"]; freq = meta["freq"]; n = meta["n"]
    use = [s for s in os.environ.get("USE_SIG", ",".join(rsig)).split(",") if s in rsig]
    sidx = [rsig.index(s) for s in use]
    tr_Z = torch.tensor(meta["train_Z"][:, :, sidx]); tr_F = torch.tensor(meta["train_F"])
    tr_bidx = torch.zeros_like(tr_F, dtype=torch.long)
    for i, (lo, hi, nm) in enumerate(EDGES):
        tr_bidx[(tr_F >= lo) & (tr_F <= hi)] = i
    tr_logf = torch.log1p(tr_F)
    posbk = tr_bidx[:, 0]
    bkcount = torch.tensor([max(1, int((posbk == i).sum())) for i in range(len(EDGES))], dtype=torch.float32)
    reweight = ROUTER.endswith("_bal")
    wq = (1.0 / bkcount[posbk]) if reweight else torch.ones(len(tr_Z))
    wq = wq / wq.mean()
    log(f"{DS} router_lab[{ROUTER}] signals={use} train={tuple(tr_Z.shape)} (load {time.time()-t0:.0f}s)")

    if ROUTER.startswith("valfit"):
        # per-bucket weights fit by FULL-CATALOGUE listwise CE on cached VAL signals (every item a true
        # negative -- BundleFuse's objective, not sampled negatives). _bal => bucket-balanced query weights.
        valZ_t = torch.from_numpy(np.load(os.path.join(CACHE, f"{DS}_validZ.npy")))[sidx]  # (nsig,n,nv) fp16 in RAM
        vseeds, vgt = meta["val_seeds"], meta["val_gt"]; nv = valZ_t.shape[2]
        log(f"  valfit: loaded val signals {tuple(valZ_t.shape)} into RAM ({time.time()-t0:.0f}s)")
        fb = torch.zeros(n, dtype=torch.long); ft = torch.tensor(freq)
        for i, (lo, hi, nm) in enumerate(EDGES):
            fb[(ft >= lo) & (ft <= hi)] = i
        # pick one pos per val query (prefer a held-out item; its bucket sets the query weight)
        vpos = []
        for g, s in zip(vgt, vseeds):
            cand = [x for x in g if x not in set(s)]
            vpos.append(cand[0] if cand else (g[0] if g else 0))
        vposbk = torch.tensor([bkof(int(freq[p])) for p in vpos])
        bkc = torch.tensor([max(1, int((vposbk == i).sum())) for i in range(len(EDGES))], dtype=torch.float32)
        reweight = ROUTER.endswith("_bal")
        W = nn.Parameter(torch.full((len(EDGES), len(use)), 0.1))
        opt = torch.optim.Adam([W], lr=0.05)
        sep = ROUTER.startswith("valfit_sep")
        def fit_on(qidx, w_init):
            """Fit one global weight vector by full-catalogue CE on the given val queries."""
            w = nn.Parameter(w_init.clone()); op = torch.optim.Adam([w], lr=0.05)
            for ep in range(1, EPOCHS + 1):
                o = np.array(qidx); np.random.default_rng(ep).shuffle(o)
                for off in range(0, len(o), 256):
                    bi = o[off:off + 256]
                    Zc = valZ_t[:, :, bi].to(torch.float32).permute(2, 1, 0)  # (b,n,nsig)
                    fused = (w * Zc).sum(-1)                              # (b,n)
                    for r, j in enumerate(bi):
                        fused[r, vseeds[j]] = -1e9
                    loss = Fn.cross_entropy(fused, torch.tensor([vpos[j] for j in bi]))
                    op.zero_grad(); loss.backward(); op.step()
            return w.detach()
        if ROUTER.startswith("valfit_combo"):
            # rare/mid from JOINT balanced fit (needs cross-query data); head/cold from INDEPENDENT head fit
            Wj = nn.Parameter(torch.full((len(EDGES), len(use)), 0.1)); opj = torch.optim.Adam([Wj], lr=0.05)
            for ep in range(1, EPOCHS + 1):
                o = np.arange(nv); np.random.default_rng(ep).shuffle(o)
                for off in range(0, nv, 256):
                    bi = o[off:off + 256]
                    Zc = valZ_t[:, :, bi].to(torch.float32).permute(2, 1, 0)
                    fused = (Wj[fb][None] * Zc).sum(-1)
                    for r, j in enumerate(bi):
                        fused[r, vseeds[j]] = -1e9
                    wqq = (1.0 / bkc[vposbk[bi]]); wqq = wqq / wqq.mean()
                    loss = (Fn.cross_entropy(fused, torch.tensor([vpos[j] for j in bi]), reduction="none") * wqq).mean()
                    opj.zero_grad(); loss.backward(); opj.step()
            qhead = [j for j in range(nv) if int(vposbk[j]) == 3]
            whead = fit_on(qhead, torch.full((len(use),), 0.1))
            Wd = Wj.detach().clone(); Wd[3] = whead; Wd[0] = whead
            log(f"    [valfit_combo] joint-balanced rare/mid + independent head ({len(qhead)} head q) ({time.time()-t0:.0f}s)")
        elif sep:
            # independent per-bucket fit: each bucket's weights trained only on its own positives
            Wd = torch.full((len(EDGES), len(use)), 0.1)
            for bkt in range(len(EDGES)):
                qi = [j for j in range(nv) if int(vposbk[j]) == bkt]
                if len(qi) >= 10:
                    Wd[bkt] = fit_on(qi, torch.full((len(use),), 0.1))
                    log(f"    [valfit_sep bucket={EDGES[bkt][2]}] fit on {len(qi)} queries ({time.time()-t0:.0f}s)")
                elif bkt > 0:
                    Wd[bkt] = Wd[bkt - 1]                                 # too few -> borrow neighbour
        else:
            Wfb_idx = fb
            for ep in range(1, EPOCHS + 1):
                order = np.arange(nv); np.random.default_rng(ep).shuffle(order); losses = []
                for off in range(0, nv, 256):
                    bi = order[off:off + 256]
                    Zc = valZ_t[:, :, bi].to(torch.float32).permute(2, 1, 0)
                    fused = (W[Wfb_idx][None] * Zc).sum(-1)
                    for r, j in enumerate(bi):
                        fused[r, vseeds[j]] = -1e9
                    wqq = (1.0 / bkc[vposbk[bi]]) if reweight else torch.ones(len(bi)); wqq = wqq / wqq.mean()
                    loss = (Fn.cross_entropy(fused, torch.tensor([vpos[j] for j in bi]), reduction="none") * wqq).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
                if ep % 5 == 0 or ep == EPOCHS:
                    log(f"    [valfit ep={ep}] loss={np.mean(losses):.4f} ({time.time()-t0:.0f}s)")
            Wd = W.detach()
        def fwd(Z, logf, bidx):
            return (Wd[bidx] * Z).sum(-1)
        params = []
    else:
        params, fwd = make_router(ROUTER, len(use), use)
    if params:
        opt = torch.optim.Adam(params, lr=0.05 if ROUTER.startswith("linear") else 0.01, weight_decay=0 if ROUTER.startswith("linear") else 1e-4)
        for ep in range(1, EPOCHS + 1):
            logits = fwd(tr_Z, tr_logf, tr_bidx)
            loss = (Fn.cross_entropy(logits, torch.zeros(len(tr_Z), dtype=torch.long), reduction="none") * wq).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # eval on cached full test signals
    fb = torch.zeros(n, dtype=torch.long); ft = torch.tensor(freq)
    for i, (lo, hi, nm) in enumerate(EDGES):
        fb[(ft >= lo) & (ft <= hi)] = i
    logf_all = torch.log1p(ft)
    buck = {nm: [0, 0] for _, _, nm in EDGES}; Rk = {10: np.zeros(2), 20: np.zeros(2)}
    seeds_all, gt_all = meta["test_seeds"], meta["test_gt"]; nb = testZ.shape[2]
    with torch.no_grad():
        for off in range(0, nb, 200):
            sl = slice(off, min(off + 200, nb))
            Zc = torch.from_numpy(np.ascontiguousarray(testZ[:, :, sl]).astype(np.float32))  # (nsig,n,b)
            Zc = Zc[sidx].permute(1, 2, 0)                                                    # (n,b,nsig)
            b = Zc.shape[1]
            logf = logf_all[:, None].expand(n, b)
            bidx = fb[:, None].expand(n, b)
            fused = fwd(Zc, logf, bidx)                                                       # (n,b)
            cb = [seeds_all[off + j] for j in range(b)]; cg = [gt_all[off + j] for j in range(b)]
            for j, s in enumerate(cb):
                fused[s, j] = -1e9
            m = ranking_metrics(fused, list(zip(cb, cg)), (10, 20))
            for K in (10, 20):
                Rk[K] += np.array(m[K]) * b
            top = fused.topk(20, dim=0).indices.cpu().numpy()
            for j, gts in enumerate(cg):
                t20 = set(top[:, j].tolist())
                for t in set(gts):
                    bb = buck[EDGES[bkof(int(freq[t]))][2]]; bb[1] += 1; bb[0] += int(t in t20)
    for K in (10, 20):
        Rk[K] /= nb
    bk = {nm: (v[0] / v[1] if v[1] else float("nan")) for nm, v in buck.items()}
    ref = BF.get(DS, {})
    log(f"\n  {DS}  [{ROUTER}]  ({time.time()-t0:.0f}s)")
    log(f"    {'':12s} {'cold':>8} {'rare':>8} {'mid':>8} {'head':>8} {'R@20':>8} {'N@20':>8}")
    if ref:
        log(f"    {'BundleFuse':12s} {ref['cold']:8.4f} {ref['rare']:8.4f} {ref['mid']:8.4f} {ref['head']:8.4f} {ref['R@20']:8.4f} {ref['N@20']:8.4f}")
    log(f"    {ROUTER:12s} {bk['cold']:8.4f} {bk['rare']:8.4f} {bk['mid']:8.4f} {bk['head']:8.4f} {Rk[20][0]:8.4f} {Rk[20][1]:8.4f}")


if __name__ == "__main__":
    main()
