"""Mixture-of-Experts for bundle completion: one expert per signal FAMILY, a learned gate routing per
candidate. Tests the one genuinely-untried idea -- a dedicated SET-INTERACTION expert (DeepSets) that can
represent higher-order "c completes the combination {s1,s2}", which a pairwise dot product structurally
cannot (the `simp` signal that carried BundleFuse's head/mid). Experts over FROZEN SVD collab embedding
(the right estimator for sparse data) + content; only the experts + gate are learned.

  E_collab(c|S) = <A u_c, sum_s A u_s>                          (pairwise co-occurrence -> head)
  E_set(c|S)    = MLP([h_c, p, h_c*p, sum_s h_c*h_s])           (higher-order set interaction -> simp-like)
  E_content(c|S)= <B t_c, sum_s B t_s>                          (content/style)
  gate g(c)     = softmax(MLP(log f_c))                         (route per candidate frequency)
  score(c|S)    = sum_e g_e(c) * E_e(c|S)

Honest expectation: head/mid learnable (as before); the open question is whether E_set captures the
higher-order signal and whether any learnable expert lifts RARE (which prior work says needs the exact
sparse lookup, not learnable params). Metric-selected on val (CE over-sacrifices rare), test once.
Env: DS, DIM, EPOCHS. Run: DS=pog_dense python3 reports/moe.py
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
import reports.mmfuse as M
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import ranking_metrics, top_items

DS = os.environ.get("DS", "pog_dense")
DIM = int(os.environ.get("DIM", "128"))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
LR = float(os.environ.get("LR", "0.002"))
SEED = int(os.environ.get("SEED", "0"))
EDGES = [(0, 0, "cold"), (1, 2, "rare"), (3, 10, "mid"), (11, 10**12, "head")]
REF = {"pog_dense": {"CLHE": (0.3815, 0.2771), "BundleFuse": (0.3969, 0.2914)}}


def log(*a):
    print(*a, flush=True)


def bkof(f):
    for i, (lo, hi, nm) in enumerate(EDGES):
        if lo <= f <= hi:
            return i
    return 3


class MoE(nn.Module):
    EXPERTS = ["collab", "set", "content"]

    def __init__(self, du, dc, d):
        super().__init__()
        self.Au = nn.Linear(du, d, bias=False); self.Ac = nn.Linear(dc, d, bias=False)
        nn.init.orthogonal_(self.Au.weight); nn.init.orthogonal_(self.Ac.weight)
        # set expert: candidate interacts with the 2nd-ORDER seed pooling P2 (seed-pair co-structure),
        # the term a mean/sum pool discards -> genuinely higher-order, no attention.
        self.set_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.gate = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 3))

    def expert_scores(self, u_c, c_c, u_s, c_s, smask):
        """-> (B,nc,3) raw expert scores [collab, set, content]."""
        hc = self.Au(u_c)                                        # (B,nc,d)
        hs = self.Au(u_s) * smask[..., None]                     # (B,ns,d)
        sum1 = hs.sum(1)                                         # (B,d)  1st-order seed pool
        sq = (hs * hs).sum(1)                                    # (B,d)
        P2 = 0.5 * (sum1 * sum1 - sq)                            # (B,d)  2nd-order: sum over seed PAIRS
        E_collab = (hc * sum1[:, None]).sum(-1)                  # pairwise <hc, sum_s hs>
        # set expert: PURE higher-order -- only the 2nd-order seed-pair term P2, no pairwise feature,
        # so it cannot collapse into the collab expert and must capture higher-order or fail visibly.
        E_set = self.set_mlp(hc * P2[:, None]).squeeze(-1)
        cc = self.Ac(c_c); cs_sum = (self.Ac(c_s) * smask[..., None]).sum(1)
        E_content = (cc * cs_sum[:, None]).sum(-1)
        return torch.stack([E_collab, E_set, E_content], dim=-1)  # (B,nc,3)

    def gate_w(self, logf_c):
        return Fn.softmax(self.gate(logf_c.unsqueeze(-1)), dim=-1)  # (...,3)

    def forward(self, u_c, c_c, u_s, c_s, smask, logf_c):
        E = self.expert_scores(u_c, c_c, u_s, c_s, smask)         # (B,nc,3)
        return (self.gate_w(logf_c) * E).sum(-1)


def main():
    t0 = time.time()
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DS)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    prov = ProvidedSplits(dd, n)
    g = SimplicialGraphOperator(prov.train, n_items=n, svd_dim=2)
    freq = g.freq.astype(np.float64); C = g.C.tocsr(); freq_t = torch.tensor(freq, dtype=torch.float32)
    logf = torch.tensor(((np.log1p(freq) - np.log1p(freq).mean()) / (np.log1p(freq).std() + 1e-8)), dtype=torch.float32)
    svd = TruncatedSVD(n_components=min(DIM, n - 2), random_state=0)
    A = svd.fit_transform(g.W_pair.tocsr()).astype(np.float32); B = svd.components_.T.astype(np.float32)
    U = torch.from_numpy(A); Bside = torch.from_numpy(B)
    tp = os.path.join(dd, "title_emb.npy")
    T = np.load(tp).astype(np.float32) if os.path.exists(tp) else torch.load(os.path.join(dd, "content_feature.pt"), map_location="cpu", weights_only=False).detach().float().numpy()
    T = (T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)).astype(np.float32); E = torch.from_numpy(T)
    du, dc = A.shape[1], T.shape[1]
    model = MoE(du, dc, DIM)
    log(f"{DS}: n={n:,} du={du} dc={dc} DIM={DIM} | experts=[collab,set,content] gate=freq | params={sum(p.numel() for p in model.parameters()):,} (setup {time.time()-t0:.0f}s)")

    bundles = [list(b) for b in prov.train.values() if len(b) >= 2]
    val = prov.pairs("valid"); test = prov.pairs("test")

    @torch.no_grad()
    def score_parts(seeds):
        """-> fused (n,), expert_scores (n,3), gate (n,3)."""
        s = list(seeds); su = Bside[s][None]; sc = E[s][None]; sm = torch.ones(1, len(s))
        Eout = np.empty((n, 3), np.float32); Gout = np.empty((n, 3), np.float32)
        for c0 in range(0, n, 20000):
            ci = slice(c0, min(c0 + 20000, n))
            es = model.expert_scores(U[ci][None], E[ci][None], su, sc, sm).squeeze(0)   # (nc,3)
            g = model.gate_w(logf[ci][None]).squeeze(0)                                 # (nc,3)
            Eout[ci] = es.numpy(); Gout[ci] = g.numpy()
        fused = (Gout * Eout).sum(1)
        return fused, Eout, Gout

    @torch.no_grad()
    def score_all(seeds):
        return score_parts(seeds)[0]

    def val_recall(subset):
        acc = np.zeros(2)
        with torch.no_grad():
            for seeds, gt in subset:
                sc = score_all(seeds)
                acc += np.asarray(ranking_metrics(sc[:, None], [(list(seeds), list(gt))], (20,))[20])
        return acc[0] / len(subset)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    vsub = val[:300]; best, best_state, stale = -1, None, 0
    steps = max(1, len(bundles) // 128)
    for ep in range(1, EPOCHS + 1):
        model.train(); order = rng.permutation(len(bundles)); losses = []
        for st in range(steps):
            idx = order[(st * 128) % len(order): (st * 128) % len(order) + 128]
            if len(idx) < 2:
                continue
            sidx, smask, cidx = M.make_batch(C, [bundles[i] for i in idx], rng, n, 32, 20, 0.5)
            u_c, _, c_c = U[cidx], None, E[cidx]; _, su, sc = U[sidx], Bside[sidx], E[sidx]
            logits = model(U[cidx], E[cidx], Bside[sidx], E[sidx], smask, logf[cidx])
            loss = Fn.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach()))
        if ep % 3 == 0 or ep == EPOCHS:
            model.eval(); r = val_recall(vsub)
            if r > best + 1e-5:
                best, best_state, stale = r, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"    [ep={ep}] loss={np.mean(losses):.4f} val_R@20={r:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
            if stale >= 4:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    acc = {k: np.zeros(2) for k in (10, 20)}; buck = {nm: [0, 0] for _, _, nm in EDGES}
    # instrumentation: each expert ALONE per bucket, gate distribution per bucket, expert correlation
    exp_buck = {e: {nm: [0, 0] for _, _, nm in EDGES} for e in MoE.EXPERTS}
    gate_by_bucket = {nm: [np.zeros(3), 0] for _, _, nm in EDGES}
    corr_acc = []
    with torch.no_grad():
        for seeds, gt in test:
            fused, Esc, G = score_parts(seeds)
            m = ranking_metrics(fused[:, None], [(list(seeds), list(gt))], (10, 20))
            for k in (10, 20):
                acc[k] += np.asarray(m[k])
            top = set(top_items(fused, list(seeds), 20).tolist())
            tops_e = {e: set(top_items(Esc[:, i], list(seeds), 20).tolist()) for i, e in enumerate(MoE.EXPERTS)}
            for t in set(gt):
                nm = EDGES[bkof(int(freq[t]))][2]
                buck[nm][1] += 1; buck[nm][0] += int(t in top)
                gate_by_bucket[nm][0] += G[t]; gate_by_bucket[nm][1] += 1
                for e in MoE.EXPERTS:
                    exp_buck[e][nm][1] += 1; exp_buck[e][nm][0] += int(t in tops_e[e])
            if len(corr_acc) < 400:
                corr_acc.append(np.corrcoef(Esc.T))
    N = len(test); bk = {nm: (v[0] / v[1] if v[1] else float("nan")) for nm, v in buck.items()}
    ref = REF.get(DS, {})
    log(f"\n  {DS}  MoE (collab + SET + content experts, freq gate)  ({time.time()-t0:.0f}s)")
    log(f"    {'method':12s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}")
    for nm, v in ref.items():
        log(f"    {nm:12s} {'':8} {'':8} {v[0]:8.4f} {v[1]:8.4f}")
    log(f"    {'MoE':12s} {acc[10][0]/N:8.4f} {acc[10][1]/N:8.4f} {acc[20][0]/N:8.4f} {acc[20][1]/N:8.4f}")
    log(f"    fused buckets: cold {bk['cold']:.4f}  rare {bk['rare']:.4f}  mid {bk['mid']:.4f}  head {bk['head']:.4f}")
    log("\n  == INSTRUMENTATION ==")
    log(f"    each expert ALONE, Hit@20 by bucket:")
    log(f"      {'expert':10s} {'cold':>8} {'rare':>8} {'mid':>8} {'head':>8}")
    for e in MoE.EXPERTS:
        eb = {nm: (v[0] / v[1] if v[1] else float('nan')) for nm, v in exp_buck[e].items()}
        log(f"      {e:10s} {eb['cold']:8.4f} {eb['rare']:8.4f} {eb['mid']:8.4f} {eb['head']:8.4f}")
    log(f"    gate weight [collab set content] by bucket (does it route?):")
    for nm in ("cold", "rare", "mid", "head"):
        gv = gate_by_bucket[nm][0] / max(1, gate_by_bucket[nm][1])
        log(f"      {nm:10s} {gv[0]:.3f} {gv[1]:.3f} {gv[2]:.3f}")
    cm = np.nanmean(corr_acc, axis=0)
    log(f"    expert score correlation (are they distinct?): collab-set {cm[0,1]:.2f}  collab-content {cm[0,2]:.2f}  set-content {cm[1,2]:.2f}")


if __name__ == "__main__":
    main()
