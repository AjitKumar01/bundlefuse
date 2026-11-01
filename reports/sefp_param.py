"""SEFP-P - the PARAMETER-ONLY bundle-completion model, and its rate-distortion sweep.

Unlike SEFP (feature-bound: 2.1 GB of co-occurrence/PPMI matrices sliced per query, learned params
vestigial), SEFP-P scores using ONLY learned tensors. At inference there is NO co-occurrence graph:

    score(c | S) = <v_c, W q_S> + b_c,     q_S = pool_{s in S} v_s

  V in R^{N x d}  : learned item embedding table (the signals ARE these parameters)
  b in R^N        : learned per-item base measure (popularity bias; init from log freq, then learned)
  W in R^{d x d}  : learned metric M (= the prior-precision tilt from the SEFP derivation), init I
  q_S             : pooled seed embeddings (mean)

This is MIPS-retrievable (one dense matvec V @ (W q_S)) -- no sparse slicing, no 92 ms graph matvec.
Storage is N*d floats (~65 MB at d=64) vs 2.1 GB. The co-occurrence graph is used ONLY to mine hard
negatives during TRAINING; inference is pure parameters.

THE RATE-DISTORTION QUESTION (the Shannon frame): the embedding dim d is the RATE (bits per item ~
d * 32). Recall@20 is 1 - distortion. Sweeping d traces R(D): how much of the task's conditional
probability p(c|S) compresses into d learned parameters per item. We locate where the curve saturates
relative to (i) the feature-bound plateau SEFP/SCACS reach (~0.12), (ii) the Gate-1 two-tower (0.105),
(iii) BundleFuse (0.156). Levy-Goldberg says PPMI IS the collaborative-kernel MLE, so a rank-d V is a
lossy compression of PPMI -- the recall gap at each d is the distortion of that compression.

Arms: dims x init. init=content projects CLAP (prior knowledge) into R^d as the starting V (handles
cold items); init=random is the pure-collaborative control (how much co-occurrence alone compresses).

Env: QUICK, EPOCHS, STEPS, NEG, HARD_FRAC, SEED_CAP, LR, EVAL_EVERY, PATIENCE, TEST_N, VAL_N,
DIMS (e.g. 32,64,128), INITS (content,random), MIX_SEEDS, DATASET. Writes results/sefp/param_<dataset>.json.
Run: python3 reports/sefp_param.py
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
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import ranking_metrics
from bundlefuse.robust_protocol import sparse_pairs

DATASET = os.environ.get("DATASET", "spotify")
ANCHORS = {"supplied": {"BundleFuse": (0.1556, 0.2579), "SEFP_feat": (0.1203, 0.2016),
                        "TwoTower_G1": (0.1053, 0.0)},
           "one_seed": {"BundleFuse": (0.0616, 0.1548), "SEFP_feat": (0.0563, 0.1492)}}

QUICK = os.environ.get("QUICK", "0") != "0"
EPOCHS = int(os.environ.get("EPOCHS", "3" if QUICK else "40"))
STEPS = int(os.environ.get("STEPS", "20" if QUICK else "0"))
NEG = int(os.environ.get("NEG", "32"))
HARD_FRAC = float(os.environ.get("HARD_FRAC", "0.5"))
BATCH = int(os.environ.get("BATCH", "256"))
SEED_CAP = int(os.environ.get("SEED_CAP", "20"))
LR = float(os.environ.get("LR", "0.01"))
WD = float(os.environ.get("WD", "1e-6"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "2"))
PATIENCE = int(os.environ.get("PATIENCE", "4"))
MIX_SEEDS = os.environ.get("MIX_SEEDS", "1") != "0"
TEST_N = int(os.environ.get("TEST_N", "200" if QUICK else "0"))
VAL_N = int(os.environ.get("VAL_N", "200" if QUICK else "1000"))
DIMS = [int(x) for x in os.environ.get("DIMS", "32,64,128").split(",")]
INITS = os.environ.get("INITS", "content,random").split(",")
DEVICE = torch.device("cpu")


def log(*a):
    print(*a, flush=True)


def load_everything():
    d = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DATASET)
    n = json.load(open(os.path.join(d, "count.json")))["#I"]
    provided = ProvidedSplits(d, n)
    t0 = time.time()
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    C = graph.C.tocsr()                      # used ONLY for training-time hard-negative mining
    freq = graph.freq.astype(np.float32)
    x = torch.load(os.path.join(d, "content_feature.pt"), map_location="cpu",
                   weights_only=False).detach().float().numpy()
    content = (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    logf = np.log1p(freq).astype(np.float32)
    logf_norm = ((logf - logf.mean()) / (logf.std() + 1e-8)).astype(np.float32)
    log(f"{DATASET}: n={n} train={len(provided.train)} built in {time.time()-t0:.0f}s "
        f"| content_dim={content.shape[1]} DIMS={DIMS} INITS={INITS}")
    return dict(n=n, provided=provided, C=C, freq=freq, content=content, logf=logf_norm)


class SEFP_P(nn.Module):
    """Parameter-only scorer. Inference uses V, b, W only -- no co-occurrence graph."""
    def __init__(self, n, d, content=None, logf=None, init="content"):
        super().__init__()
        self.V = nn.Embedding(n, d)
        self.b = nn.Embedding(n, 1)
        self.W = nn.Parameter(torch.eye(d))
        with torch.no_grad():
            if init == "content" and content is not None:
                # project CLAP (prior knowledge) into R^d: SVD of content gives the best rank-d start
                cc = torch.from_numpy(content)
                U, Sv, Vt = torch.linalg.svd(cc - cc.mean(0), full_matrices=False)
                proj = (U[:, :d] * Sv[:d])                     # (n,d) top-d content coords
                proj = proj / (proj.std() + 1e-6) * 0.1
                self.V.weight.copy_(proj)
            else:
                nn.init.normal_(self.V.weight, std=0.1)
            self.b.weight.copy_(torch.from_numpy(logf)[:, None] if logf is not None
                                else torch.zeros(n, 1))

    def pool(self, seed_idx, smask):
        vs = self.V(seed_idx)                                  # (B,Ks,d)
        return (vs * smask[:, :, None]).sum(1) / smask.sum(1, keepdim=True).clamp_min(1)

    def score_cands(self, q, cand_idx):
        vc = self.V(cand_idx)                                  # (B,Kc,d)
        qw = q @ self.W.T                                      # (B,d)
        return torch.einsum("bcd,bd->bc", vc, qw) + self.b(cand_idx).squeeze(-1)

    def score_all(self, q):
        qw = q @ self.W.T                                      # (B,d)
        return qw @ self.V.weight.T + self.b.weight.T          # (B,N)


def make_batch(D, bundles, rng, n, neg, seed_cap, hard_frac):
    C = D["C"]
    seeds, smask_len, pos, negs = [], [], [], []
    for b in bundles:
        it = np.asarray(b, np.int64).copy()
        rng.shuffle(it)
        split = int(rng.integers(1, len(it))) if MIX_SEEDS else max(1, min(len(it) - 1, round(len(it) / 2)))
        s = it[:split][:seed_cap].tolist()
        p = int(rng.choice(it[split:]))
        itset = set(it.tolist())
        hard = []
        n_hard = int(round(neg * hard_frac))
        if n_hard > 0:
            pool = set()
            for ss in s:
                pool.update(C.indices[C.indptr[ss]:C.indptr[ss + 1]].tolist())
            pool.difference_update(itset)
            if pool:
                pool = np.fromiter(pool, np.int64)
                hard = rng.choice(pool, size=min(n_hard, len(pool)), replace=len(pool) < n_hard).tolist()
        need = neg - len(hard)
        forbidden = itset.union(hard)
        ng = rng.integers(n, size=need)
        bad = np.array([x in forbidden for x in ng])
        while bad.any():
            ng[bad] = rng.integers(n, size=int(bad.sum()))
            bad = np.array([x in forbidden for x in ng])
        seeds.append(s); pos.append(p); negs.append(hard + ng.tolist())
    Ks = max(len(s) for s in seeds)
    B = len(seeds)
    sidx = torch.zeros(B, Ks, dtype=torch.long)
    smask = torch.zeros(B, Ks)
    for j, s in enumerate(seeds):
        sidx[j, :len(s)] = torch.tensor(s); smask[j, :len(s)] = 1.0
    cidx = torch.tensor([[pos[j]] + negs[j] for j in range(B)], dtype=torch.long)
    return sidx, smask, cidx


def train_model(model, D, train_bundles, val_pairs, rng_seed, tag):
    rng = np.random.default_rng(rng_seed)
    torch.manual_seed(rng_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    train = [list(b) for b in train_bundles if len(b) >= 2]
    steps = STEPS if STEPS > 0 else max(1, len(train) // BATCH)
    best, best_state, stale = -1e9, None, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for st in range(steps):
            idx = order[(st * BATCH) % len(order): (st * BATCH) % len(order) + BATCH]
            if len(idx) < 2:
                continue
            sidx, smask, cidx = make_batch(D, [train[i] for i in idx], rng, D["n"], NEG, SEED_CAP, HARD_FRAC)
            q = model.pool(sidx, smask)
            logits = model.score_cands(q, cidx)
            loss = Fn.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            val = eval_full(model, D, val_pairs, Ks=(20,))[20][1]
            if val > best + 1e-5:
                best, best_state, stale = val, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"  [{tag}] ep={ep} loss={np.mean(losses):.4f} val_N20={val:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


@torch.no_grad()
def eval_full(model, D, pairs, Ks=(10, 20, 50)):
    """Full-catalogue, parameter-only ranking: score every item by <v_c, W q_S> + b_c. Per-query
    accumulation (no N x Q matrix)."""
    model.eval()
    rec = {k: np.zeros(2) for k in Ks}
    for seeds, gt in pairs:
        s = list(seeds)[:SEED_CAP]
        sidx = torch.tensor(s, dtype=torch.long)[None, :]
        smask = torch.ones(1, len(s))
        q = model.pool(sidx, smask)
        sc = model.score_all(q).squeeze(0).numpy()            # (N,)
        mm = ranking_metrics(sc[:, None], [(list(seeds), list(gt))], Ks=Ks)
        for k in Ks:
            rec[k] += np.asarray(mm[k])
    return {k: tuple(rec[k] / len(pairs)) for k in Ks}


def main():
    t0 = time.time()
    D = load_everything()
    provided = D["provided"]
    train_bundles = list(provided.train.values())
    val_pairs = provided.pairs("valid")
    test_pairs = provided.pairs("test")
    if VAL_N and VAL_N < len(val_pairs):
        val_pairs = val_pairs[:VAL_N]
    if TEST_N and TEST_N < len(test_pairs):
        test_pairs = test_pairs[:TEST_N]
    test_ids = sorted(provided.splits["test"][0])
    oneseed_full = sparse_pairs(provided.pairs("test"), test_ids, "one_seed")
    oneseed_pairs = oneseed_full[:TEST_N] if (TEST_N and TEST_N < len(oneseed_full)) else oneseed_full
    log(f"train={len(train_bundles)} val={len(val_pairs)} supplied={len(test_pairs)} "
        f"one_seed={len(oneseed_pairs)} | EPOCHS={EPOCHS} STEPS={STEPS or 'full'} NEG={NEG} "
        f"HARD_FRAC={HARD_FRAC} MIX_SEEDS={MIX_SEEDS} QUICK={QUICK}")

    results = {"dataset": DATASET, "anchors": ANCHORS, "curve": [], "models": {}}
    for init in INITS:
        for d in DIMS:
            tag = f"d{d}_{init}"
            log(f"\n=== training SEFP-P {tag} ===")
            model = SEFP_P(D["n"], d, content=D["content"], logf=D["logf"], init=init)
            nparam = sum(p.numel() for p in model.parameters())
            model, best_val = train_model(model, D, train_bundles, val_pairs, rng_seed=0, tag=tag)
            ms = eval_full(model, D, test_pairs, Ks=(10, 20, 50))
            mo = eval_full(model, D, oneseed_pairs, Ks=(10, 20, 50))
            rec = {"dim": d, "init": init, "params": int(nparam), "bits_per_item": d * 32,
                   "best_val_ndcg20": float(best_val),
                   "supplied": {f"{M}": {"R": float(ms[M][0]), "N": float(ms[M][1])} for M in (10, 20, 50)},
                   "one_seed": {f"{M}": {"R": float(mo[M][0]), "N": float(mo[M][1])} for M in (10, 20, 50)}}
            results["models"][tag] = rec
            results["curve"].append({"dim": d, "init": init, "supplied_R20": float(ms[20][0]),
                                     "one_seed_R20": float(mo[20][0])})
            log(f"{tag}: params={nparam:,} ({nparam*4/1e6:.0f} MB) val_N20={best_val:.4f} | "
                f"supplied R@20={ms[20][0]:.4f} | one_seed R@20={mo[20][0]:.4f}")

    out_dir = os.path.join(PROJECT_ROOT, "results", "sefp")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"param_{DATASET}.json")
    json.dump(results, open(outp, "w"), indent=2)

    bf = ANCHORS["supplied"]["BundleFuse"]; sf = ANCHORS["supplied"]["SEFP_feat"]
    log(f"\n=== SEFP-P RATE-DISTORTION CURVE ({time.time()-t0:.0f}s) | supplied R@20 ===")
    log(f"  reference: BundleFuse(feat)={bf[0]:.4f}  SEFP(feat)={sf[0]:.4f}  TwoTower(G1)=0.1053")
    log(f"  {'config':16s} {'dim':>4} {'MB':>6} {'supp R@20':>10} {'1seed R@20':>11}")
    for init in INITS:
        for d in DIMS:
            r = results["models"][f"d{d}_{init}"]
            log(f"  {f'd{d}_{init}':16s} {d:4d} {r['params']*4/1e6:6.0f} "
                f"{r['supplied']['20']['R']:10.4f} {r['one_seed']['20']['R']:11.4f}")
    log(f"wrote {outp}")


if __name__ == "__main__":
    main()
