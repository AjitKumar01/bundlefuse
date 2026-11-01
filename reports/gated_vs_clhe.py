"""Benchmark v1 GatedBundle (parameter-only: learned SGNS association + learned frequency-gated
popularity bias) on the CANONICAL full-ranking metric (Recall@/NDCG@{10,20}), vs the reproduced CLHE
and BundleFuse rows from results/matrix.csv. Same splits, supplied protocol, candidate universe and
tie policy as Table 1 (reports/mmfuse_vs_clhe.py); ranking_metrics called per-query (memory-safe).

v1 model:  s(c|S) = <u_c, sum_s v_s> + (1 - g(f_c)) * b_c ,  g(f)=kappa/(f+kappa)   (all learned)
Inference is one mat-vec + a gated bias -- graph-free, embeddings only. Env: DATASETS, DIM, SGNS_EPOCHS,
GATE_EPOCHS, INIT_KAPPA, LR. Run: DATASETS=pog_dense,spotify_sparse python3 reports/gated_vs_clhe.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reports.mmfuse as M
from reports.gated_bundle import build_pairs, train_sgns, SGNS_EPOCHS
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import ranking_metrics

DATASETS = os.environ.get("DATASETS", "pog_dense,spotify_sparse").split(",")
DIM = int(os.environ.get("DIM", "128"))
GATE_EPOCHS = int(os.environ.get("GATE_EPOCHS", "30"))
INIT_KAPPA = float(os.environ.get("INIT_KAPPA", "5.0"))
NEG = int(os.environ.get("NEG", "32")); HARD_FRAC = float(os.environ.get("HARD_FRAC", "0.5"))
BATCH = int(os.environ.get("BATCH", "128")); SEED_CAP = int(os.environ.get("SEED_CAP", "20"))
LR = float(os.environ.get("LR", "0.01")); VAL_N = int(os.environ.get("VAL_N", "400"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "10"))

# results/matrix.csv  (R@10, N@10, R@20, N@20); MMFuse_nd from reports/sefp/mmfuse_vs_clhe.json
BARS = {
    "pog_dense":      {"CLHE": (0.3438, 0.2662, 0.3815, 0.2771), "BundleFuse": (0.3539, 0.2793, 0.3969, 0.2914), "MMFuse_nd": (0.3093, 0.2351, 0.3476, 0.2460)},
    "spotify_sparse": {"CLHE": (0.0608, 0.1164, 0.0969, 0.1150), "BundleFuse": (0.1063, 0.2411, 0.1620, 0.2265), "MMFuse_nd": (0.0502, 0.1031, 0.0797, 0.0996)},
}


def log(*a):
    print(*a, flush=True)


class GatedV1(nn.Module):
    def __init__(self, n, d, init_kappa):
        super().__init__()
        self.U = nn.Embedding(n, d); self.V = nn.Embedding(n, d); self.b = nn.Embedding(n, 1)
        nn.init.normal_(self.U.weight, std=0.1); nn.init.normal_(self.V.weight, std=0.1); nn.init.zeros_(self.b.weight)
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(np.expm1(init_kappa)))))

    def kappa(self):
        return Fn.softplus(self.log_kappa)

    def gate(self, f):
        k = self.kappa(); return k / (f + k)

    def score_batch(self, cidx, sidx, smask, fcand):
        vs = (self.V(sidx) * smask[..., None]).sum(1)
        assoc = (self.U(cidx) * vs.unsqueeze(1)).sum(-1)
        return assoc + (1.0 - self.gate(fcand)) * self.b(cidx).squeeze(-1)

    @torch.no_grad()
    def score_all(self, seeds, freq_t):
        vs = self.V.weight[list(seeds)].sum(0)
        return (self.U.weight @ vs + (1.0 - self.gate(freq_t)) * self.b.weight.squeeze(-1)).cpu().numpy()


def fullrank(model, pairs, freq_t, Ks=(10, 20)):
    acc = {k: np.zeros(2, np.float64) for k in Ks}
    for seeds, gt in pairs:
        sc = model.score_all(seeds, freq_t)[:, None].astype(np.float32)
        m = ranking_metrics(sc, [(list(seeds), list(gt))], Ks)
        for k in Ks:
            acc[k] += np.asarray(m[k])
    return {k: tuple(acc[k] / len(pairs)) for k in Ks}


@torch.no_grad()
def warm_hit(model, pairs, freq_t, freq, k=20):
    hit, tot = 0, 0
    from bundlefuse.metrics import top_items
    for seeds, gt in pairs:
        top = set(top_items(model.score_all(seeds, freq_t), list(seeds), k).tolist())
        for t in gt:
            if freq[t] > 2:
                tot += 1; hit += (t in top)
    return hit / max(1, tot)


def run(dataset):
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    provided = ProvidedSplits(dd, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    freq = graph.freq.astype(np.float64); C = graph.C.tocsr(); freq_t = torch.tensor(freq, dtype=torch.float32)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    bundles = list(provided.train.values())
    log(f"\n### {dataset}: n={n:,} DIM={DIM} | setup {time.time()-t0:.0f}s")

    model = GatedV1(n, DIM, INIT_KAPPA)
    pr, pc = build_pairs(bundles, rng)
    pn = (freq + 1e-9) ** 0.75; pn = pn / pn.sum()
    log(f"  SGNS pmi pathway (pairs={len(pr):,})...")
    train_sgns(model.U, model.V, pr, pc, pn, n, rng, "pmi")

    opt = torch.optim.Adam([model.b.weight, model.log_kappa], lr=LR)
    train_b = [list(b) for b in bundles if len(b) >= 2]; steps = max(1, len(train_b) // BATCH)
    val_pairs = provided.pairs("valid")[:VAL_N]; best, best_state = -1e9, None
    log("  gate+bias on bundle CE...")
    for ep in range(1, GATE_EPOCHS + 1):
        order = rng.permutation(len(train_b)); losses = []
        for st in range(steps):
            idx = order[(st * BATCH) % len(order): (st * BATCH) % len(order) + BATCH]
            if len(idx) < 2:
                continue
            sidx, smask, cidx = M.make_batch(C, [train_b[i] for i in idx], rng, n, NEG, SEED_CAP, HARD_FRAC)
            logits = model.score_batch(cidx, sidx, smask, freq_t[cidx])
            loss = Fn.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        if ep % EVAL_EVERY == 0 or ep == GATE_EPOCHS:
            v = warm_hit(model, val_pairs, freq_t, freq)
            if v > best + 1e-5:
                best = v; best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            log(f"    [gate ep={ep}] loss={np.mean(losses):.4f} kappa={float(model.kappa().detach()):.2f} val_warm={v:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
    if best_state is not None:
        model.load_state_dict(best_state)

    test_pairs = provided.pairs("test")
    log(f"  scoring {len(test_pairs)} test queries (graph-free mat-vec)...")
    r = fullrank(model, test_pairs, freq_t)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"\n  {dataset}  full-ranking (supplied protocol, {len(test_pairs)} queries)")
    log(f"    {'method':14s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}")
    for name in ("CLHE", "BundleFuse", "MMFuse_nd"):
        b = BARS.get(dataset, {}).get(name)
        if b:
            log(f"    {name:14s} {b[0]:8.4f} {b[1]:8.4f} {b[2]:8.4f} {b[3]:8.4f}   (ref)")
    log(f"    {'GatedBundle':14s} {r[10][0]:8.4f} {r[10][1]:8.4f} {r[20][0]:8.4f} {r[20][1]:8.4f}   (params={nparam:,}, graph-free)")
    log(f"  ({dataset} done in {time.time()-t0:.0f}s, kappa={float(model.kappa().detach()):.2f})")
    return {"dataset": dataset, "params": int(nparam), "kappa": float(model.kappa().detach()),
            "R@10": r[10][0], "N@10": r[10][1], "R@20": r[20][0], "N@20": r[20][1], "bars": BARS.get(dataset, {})}


def main():
    out = [run(ds) for ds in DATASETS]
    d = os.path.join(PROJECT_ROOT, "results", "sefp"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "gated_vs_clhe.json"); json.dump(out, open(p, "w"), indent=2)
    log(f"\nwrote {p}")


if __name__ == "__main__":
    main()
