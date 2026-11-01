"""STBR -- Set Transformer for Bundle Recommendation (Kumar, Swami, Sarkar, TCS Pune, ESM 2026),
adapted to BundleFuse's bundle-COMPLETION task for a head-to-head comparison.

The paper's model is user-personalized bundle SCORING: X=[e_u;e_1;..;e_k] -> SAB x2 -> PMA -> h_{u,b}
-> FC -> scalar preference for a (user, candidate bundle) pair, trained with BPR. Our task has no user
and ranks ITEMS given seed items, so we keep the paper's ENCODER faithfully (learnable item embeddings,
two Set Attention Blocks, Pooling-by-Multihead-Attention) and adapt only the head: the pooled bundle
representation h_S is dotted with every catalogue item embedding (like CLHE's bundle_feature @
item_features.T), so full-catalogue scoring is one GEMM. No content/metadata (the paper uses none).

Fairness choices (to give the set-encoder its BEST shot, never a strawman):
  - learnable item embeddings, trained end-to-end on the completion objective (the paper pretrains with
    BPR on user-item; we have no users, so the completion loss subsumes it);
  - listwise softmax cross-entropy with hard (co-occurrence) negatives -- stronger than the paper's
    pairwise BPR; set STBR_LOSS=bpr for the faithful pairwise loss;
  - early stopping on validation Recall@20; d=64 (>= the paper's 24), 4 heads, as the paper.
  - STBR_INIT=content initialises the item embeddings from frozen content (best-shot warm start); default
    random (faithful: embeddings are the only learnable signal).

Eval is IDENTICAL to the BundleFuse SEEDTYPE / CLHE tables: the provided test split (byte-identical
bi_test files), seed-cuts supplied/3/2/1, Recall@20 / NDCG@20 via bundlefuse.metrics.ranking_metrics.
Run: python3 reports/stbr.py     Env: DATASETS, DIM, HEADS, EPOCHS, LR, WD, NEG, HARD_FRAC, BATCH,
SEED_CAP, VAL_N, TEST_N, PATIENCE, EVAL_EVERY, STBR_LOSS, STBR_INIT, SEED_CUTS.
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

DATASETS = os.environ.get("DATASETS", "pog_dense").split(",")
DIM = int(os.environ.get("DIM", "64"))
HEADS = int(os.environ.get("HEADS", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "80"))
LR = float(os.environ.get("LR", "1e-3"))
WD = float(os.environ.get("WD", "1e-4"))
NEG = int(os.environ.get("NEG", "64"))
HARD_FRAC = float(os.environ.get("HARD_FRAC", "0.5"))
BATCH = int(os.environ.get("BATCH", "128"))
SEED_CAP = int(os.environ.get("SEED_CAP", "20"))
VAL_N = int(os.environ.get("VAL_N", "400"))
TEST_N = int(os.environ.get("TEST_N", "2000"))
PATIENCE = int(os.environ.get("PATIENCE", "5"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "4"))
LOSS = os.environ.get("STBR_LOSS", "ce")           # ce | bpr
INIT = os.environ.get("STBR_INIT", "random")       # random | content
SEED_CUTS = [int(x) for x in os.environ.get("SEED_CUTS", "0,3,2,1").split(",")]   # 0 = all supplied
EVAL_CH = int(os.environ.get("EVAL_CH", "200"))


def log(*a):
    print(*a, flush=True)


class MAB(nn.Module):
    """Multihead Attention Block (Set Transformer): attends queries X to keys Y with masking,
    residual + LayerNorm + row-wise FFN + LayerNorm."""
    def __init__(self, d, heads):
        super().__init__()
        self.d, self.h, self.dh = d, heads, d // heads
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, X, Y, ymask):
        # X:(B,nq,d)  Y:(B,nk,d)  ymask:(B,nk) 1=valid
        B, nq, _ = X.shape; nk = Y.shape[1]
        Q = self.Wq(X).view(B, nq, self.h, self.dh).transpose(1, 2)   # (B,h,nq,dh)
        K = self.Wk(Y).view(B, nk, self.h, self.dh).transpose(1, 2)
        V = self.Wv(Y).view(B, nk, self.h, self.dh).transpose(1, 2)
        att = (Q @ K.transpose(-1, -2)) / (self.dh ** 0.5)           # (B,h,nq,nk)
        att = att.masked_fill(ymask[:, None, None, :] == 0, float("-inf"))
        att = Fn.softmax(att, dim=-1)
        O = (att @ V).transpose(1, 2).reshape(B, nq, self.d)         # (B,nq,d)
        H = self.ln1(X + O)
        return self.ln2(H + self.ff(H))


class SAB(nn.Module):
    def __init__(self, d, heads):
        super().__init__(); self.mab = MAB(d, heads)

    def forward(self, X, mask):
        return self.mab(X, X, mask)


class PMA(nn.Module):
    def __init__(self, d, heads, seeds=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, seeds, d) * 0.1)
        self.mab = MAB(d, heads)

    def forward(self, X, mask):
        B = X.shape[0]
        return self.mab(self.S.expand(B, -1, -1), X, mask)           # (B,seeds,d)


class STBR(nn.Module):
    def __init__(self, n, d, heads):
        super().__init__()
        self.n = n
        self.emb = nn.Embedding(n + 1, d, padding_idx=n)
        nn.init.normal_(self.emb.weight, std=0.1)
        with torch.no_grad():
            self.emb.weight[n].zero_()
        self.sab1 = SAB(d, heads); self.sab2 = SAB(d, heads)
        self.pma = PMA(d, heads, seeds=1)
        self.out = nn.Linear(d, d)

    def encode(self, sidx, smask):
        x = self.emb(sidx)                      # (B,k,d)
        x = self.sab1(x, smask)
        x = self.sab2(x, smask)
        h = self.pma(x, smask).squeeze(1)       # (B,d)
        return self.out(h)

    def forward(self, sidx, smask, cidx):
        q = self.encode(sidx, smask)            # (B,d)
        ce = self.emb(cidx)                     # (B,C,d)
        return torch.einsum("bd,bcd->bc", q, ce)

    @torch.no_grad()
    def score_all(self, sidx, smask):
        q = self.encode(sidx, smask)            # (B,d)
        return q @ self.emb.weight[:self.n].T   # (B,n)


def make_batch(C, bundles, rng, n, neg, seed_cap, hard_frac, pad):
    seeds, pos, negs = [], [], []
    for b in bundles:
        it = np.asarray(b, np.int64).copy(); rng.shuffle(it)
        split = int(rng.integers(1, len(it)))
        s = it[:split][:seed_cap].tolist(); p = int(rng.choice(it[split:]))
        itset = set(it.tolist()); hard = []
        n_hard = int(round(neg * hard_frac))
        if n_hard > 0:
            pool = set()
            for ss in s:
                pool.update(C.indices[C.indptr[ss]:C.indptr[ss + 1]].tolist())
            pool.difference_update(itset)
            if pool:
                pool = np.fromiter(pool, np.int64)
                hard = rng.choice(pool, size=min(n_hard, len(pool)), replace=len(pool) < n_hard).tolist()
        need = neg - len(hard); forbidden = itset.union(hard)
        ng = rng.integers(n, size=need); bad = np.array([x in forbidden for x in ng])
        while bad.any():
            ng[bad] = rng.integers(n, size=int(bad.sum())); bad = np.array([x in forbidden for x in ng])
        seeds.append(s); pos.append(p); negs.append(hard + ng.tolist())
    Ks = max(len(s) for s in seeds); B = len(seeds)
    sidx = torch.full((B, Ks), pad, dtype=torch.long); smask = torch.zeros(B, Ks)
    for j, s in enumerate(seeds):
        sidx[j, :len(s)] = torch.tensor(s); smask[j, :len(s)] = 1.0
    cidx = torch.tensor([[pos[j]] + negs[j] for j in range(B)], dtype=torch.long)
    return sidx, smask, cidx


def pad_seeds(seed_lists, pad):
    Ks = max(len(s) for s in seed_lists)
    B = len(seed_lists)
    sidx = torch.full((B, Ks), pad, dtype=torch.long); smask = torch.zeros(B, Ks)
    for j, s in enumerate(seed_lists):
        s = list(s)[:SEED_CAP] if len(s) > SEED_CAP else list(s)
        sidx[j, :len(s)] = torch.tensor(s, dtype=torch.long); smask[j, :len(s)] = 1.0
    return sidx, smask


@torch.no_grad()
def eval_cut(model, pairs, n, pad, cut):
    """Recall@20 / NDCG@20 on the provided split, seeds truncated to sorted first-`cut` (0=all)."""
    agg = np.zeros(2); tot = 0
    for off in range(0, len(pairs), EVAL_CH):
        chunk = pairs[off:off + EVAL_CH]
        seed_lists, cpairs = [], []
        for seeds, gt in chunk:
            sub = sorted(seeds)[:cut] if cut > 0 else list(seeds)
            seed_lists.append(sub); cpairs.append((sub, gt))
        sidx, smask = pad_seeds(seed_lists, pad)
        sc = model.score_all(sidx, smask).T.cpu().numpy()      # (n, chunk)
        r = ranking_metrics(sc, cpairs, (20,))[20]
        agg += np.asarray(r) * len(chunk); tot += len(chunk)
    return agg / tot


def train(model, C, bundles, val_pairs, n, pad, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    train_b = [list(b) for b in bundles if len(b) >= 2]
    steps = max(1, len(train_b) // BATCH)
    best, best_state, stale = -1e9, None, 0; t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); order = rng.permutation(len(train_b)); losses = []
        for st in range(steps):
            idx = order[(st * BATCH) % len(order): (st * BATCH) % len(order) + BATCH]
            if len(idx) < 2:
                continue
            sidx, smask, cidx = make_batch(C, [train_b[i] for i in idx], rng, n, NEG, SEED_CAP, HARD_FRAC, pad)
            logits = model(sidx, smask, cidx)                  # (B, 1+neg), col 0 = positive
            if LOSS == "bpr":
                loss = -Fn.logsigmoid(logits[:, :1] - logits[:, 1:]).mean()
            else:
                loss = Fn.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach()))
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            model.eval(); v = eval_cut(model, val_pairs, n, pad, 0)[0]
            if v > best + 1e-5:
                best, best_state, stale = v, {k: vv.detach().clone() for k, vv in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"    [stbr ep={ep}] loss={np.mean(losses):.4f} val_R@20={v:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); return model


# BundleFuse / CLHE reference numbers from this session's matched SEEDTYPE tables (provided split).
REF = {
    "pog_dense": {"BundleFuse": {0: (0.3952, 0.2879), 1: (0.3512, None)},
                  "CLHE": {0: (0.3779, None), 1: (0.3354, None)}},
    "spotify_sparse": {"BundleFuse": {0: (0.1627, 0.2267), 3: (0.0962, 0.1170), 2: (0.0813, 0.1006), 1: (0.0639, 0.0804)},
                       "CLHE": {0: (0.0960, 0.1142), 3: (0.0744, 0.0823), 2: (0.0666, 0.0719), 1: (0.0507, 0.0542)}},
}


def run(dataset):
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    provided = ProvidedSplits(dd, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    C = graph.C.tocsr()
    pad = n
    model = STBR(n, DIM, HEADS)
    if INIT == "content" and os.path.exists(os.path.join(dd, "content_feature.pt")):
        x = torch.load(os.path.join(dd, "content_feature.pt"), map_location="cpu", weights_only=False).detach().float()
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        with torch.no_grad():
            if x.shape[1] >= DIM:
                model.emb.weight[:n] = x[:, :DIM] * 0.1
    nparams = sum(p.numel() for p in model.parameters())
    log(f"\n### {dataset}: n={n} DIM={DIM} heads={HEADS} loss={LOSS} init={INIT} params={nparams:,} | setup {time.time()-t0:.0f}s")

    val_pairs = provided.pairs("valid")[:VAL_N]
    test_pairs = provided.pairs("test")
    if TEST_N:
        test_pairs = test_pairs[:TEST_N]
    bundles = list(provided.train.values())
    log(f"  training STBR ({len(bundles):,} train bundles, {len(val_pairs)} val, {len(test_pairs)} test)...")
    model = train(model, C, bundles, val_pairs, n, pad)

    log(f"\n  STBR {dataset}, Recall@20 / NDCG@20 by seed-set type ({len(test_pairs)} test bundles):")
    names = {0: "supplied(all)", 1: "1-seed", 2: "2-seed", 3: "3-seed"}
    out = {"dataset": dataset, "params": nparams, "loss": LOSS, "init": INIT, "cuts": {}}
    for k in SEED_CUTS:
        r, nd = eval_cut(model, test_pairs, n, pad, k)
        out["cuts"][k] = [float(r), float(nd)]
        ref = REF.get(dataset, {})
        bf = ref.get("BundleFuse", {}).get(k); ch = ref.get("CLHE", {}).get(k)
        extra = ""
        if bf:
            extra += f"   | BundleFuse R@20={bf[0]:.4f}" + (f" N@20={bf[1]:.4f}" if bf[1] else "")
        if ch:
            extra += f"   | CLHE R@20={ch[0]:.4f}" + (f" N@20={ch[1]:.4f}" if ch[1] else "")
        log(f"    {names[k]:14s} R@20={r:.4f}  N@20={nd:.4f}{extra}")
    log(f"  ({dataset} done in {time.time()-t0:.0f}s)")
    return out


def main():
    results = [run(ds) for ds in DATASETS]
    out_dir = os.path.join(PROJECT_ROOT, "results", "sefp"); os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "stbr.json"); json.dump(results, open(outp, "w"), indent=2)
    log(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
