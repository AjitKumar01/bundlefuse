"""Cold-start via a LEARNED content->complementarity map, tested on cold slices.

Collaborative embeddings are 0 for unseen items (no co-occurrence row). Content-cosine fails too: it
scores SIMILARITY, but bundles need COMPLEMENTS (shoes with a dress, not another dress) -- measured
~0.00-0.01 Hit@20 on cold targets. The fix learns the GOES-WITH rule from warm items, written in the
language of content, so it transfers to unseen items:

    score(c|S) = < P e_c ,  mean_{s in S} Q e_s >        P (candidate), Q (seed): learned d x dc maps

P, Q are trained on warm bundles (which have both content and co-occurrence) by listwise softmax, so
they capture "content like e_c complements content like e_s", NOT "e_c ~= e_s". At test a COLD item
needs no history -- only its content e_c -- so the map applies. This is the SEFP content tilt e^T M e
with M = P^T Q learned (M != I), the asymmetry letting it encode complementarity rather than similarity.
Parameter-only at inference: precompute P e for all items, one dense matvec; graph-free.

We report Hit@20 split by target coldness (train freq) for four scorers on BOTH datasets:
  collab        : SVD-wpair rank-d (pure collaborative; 0 on cold, our baseline blind spot)
  content_cos   : sum of content cosines to seeds (similarity; the operator that fails)
  content_learn : the learned complementarity map above
  fused         : per-query-standardized collab + content_learn (collab carries warm, content the cold)

The decisive test: does content_learn lift COLD Hit@20 above content_cos (~0.01)? Honest ceiling: it
only recovers the part of goes-with that content can predict; cold will stay below warm. Env:
DATASETS, DIM (collab+content dim), EPOCHS, STEPS, NEG, HARD_FRAC, LR, TEST_N, VAL_N.
Run: python3 reports/sefp_cold.py
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
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import top_items
from bundlefuse.robust_protocol import sparse_pairs

DATASETS = os.environ.get("DATASETS", "spotify,pog_dense").split(",")
DIM = int(os.environ.get("DIM", "128"))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
STEPS = int(os.environ.get("STEPS", "0"))
NEG = int(os.environ.get("NEG", "32"))
HARD_FRAC = float(os.environ.get("HARD_FRAC", "0.5"))
BATCH = int(os.environ.get("BATCH", "256"))
SEED_CAP = int(os.environ.get("SEED_CAP", "20"))
LR = float(os.environ.get("LR", "0.003"))
WD = float(os.environ.get("WD", "1e-5"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "3"))
PATIENCE = int(os.environ.get("PATIENCE", "3"))
TEST_N = int(os.environ.get("TEST_N", "0"))
VAL_N = int(os.environ.get("VAL_N", "500"))


def log(*a):
    print(*a, flush=True)


class ContentComp(nn.Module):
    """Learned content->complementarity: candidate map P, seed map Q. M = P^T Q (asymmetric => can
    encode 'goes-with' rather than 'similar-to')."""
    def __init__(self, dc, d):
        super().__init__()
        self.P = nn.Linear(dc, d, bias=False)
        self.Q = nn.Linear(dc, d, bias=False)
        nn.init.orthogonal_(self.P.weight)
        nn.init.orthogonal_(self.Q.weight)

    def forward(self, Ec, Es, smask):          # Ec:(B,Kc,dc) Es:(B,Ks,dc) smask:(B,Ks)
        fc = self.P(Ec)                         # (B,Kc,d)
        gs = self.Q(Es)                         # (B,Ks,d)
        q = (gs * smask[..., None]).sum(1) / smask.sum(1, keepdim=True).clamp_min(1)
        return torch.einsum("bcd,bd->bc", fc, q)


def make_batch(C, bundles, rng, n, neg, seed_cap, hard_frac, neg_pool=None):
    """neg_pool: array to draw RANDOM negatives from (default whole catalogue). Passing the warm items
    only => cold items are never used as negatives, so the model is not trained to downrank cold-looking
    content (the suppression test)."""
    seeds, pos, negs = [], [], []
    for b in bundles:
        it = np.asarray(b, np.int64).copy()
        rng.shuffle(it)
        split = int(rng.integers(1, len(it)))
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
        draw = (lambda m: rng.integers(n, size=m)) if neg_pool is None else \
               (lambda m: neg_pool[rng.integers(len(neg_pool), size=m)])
        ng = draw(need)
        bad = np.array([x in forbidden for x in ng])
        while bad.any():
            ng[bad] = draw(int(bad.sum()))
            bad = np.array([x in forbidden for x in ng])
        seeds.append(s); pos.append(p); negs.append(hard + ng.tolist())
    Ks = max(len(s) for s in seeds)
    B = len(seeds)
    sidx = torch.zeros(B, Ks, dtype=torch.long); smask = torch.zeros(B, Ks)
    for j, s in enumerate(seeds):
        sidx[j, :len(s)] = torch.tensor(s); smask[j, :len(s)] = 1.0
    cidx = torch.tensor([[pos[j]] + negs[j] for j in range(B)], dtype=torch.long)
    return sidx, smask, cidx


def train_content(model, content_t, C, train_bundles, val_pairs, Fall_fn, n, seed, neg_pool=None):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    train = [list(b) for b in train_bundles if len(b) >= 2]
    steps = STEPS if STEPS > 0 else max(1, len(train) // BATCH)
    best, best_state, stale = -1e9, None, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train)); losses = []
        for st in range(steps):
            idx = order[(st * BATCH) % len(order): (st * BATCH) % len(order) + BATCH]
            if len(idx) < 2:
                continue
            sidx, smask, cidx = make_batch(C, [train[i] for i in idx], rng, n, NEG, SEED_CAP, HARD_FRAC, neg_pool)
            logits = model(content_t[cidx], content_t[sidx], smask)
            loss = Fn.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach()))
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            val = val_hit(model, content_t, val_pairs, Fall_fn)
            if val > best + 1e-5:
                best, best_state, stale = val, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"    [content ep={ep}] loss={np.mean(losses):.4f} val_hit20={val:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def val_hit(model, content_t, pairs, Fall_fn, k=20):
    Fall = Fall_fn(model)                         # (N,d) = P(content) for all items
    hit, tot = 0, 0
    for seeds, gt in pairs:
        gs = model.Q(content_t[list(seeds)]).mean(0)
        sc = (Fall @ gs).numpy()
        top = set(top_items(sc, list(seeds), k).tolist())
        for t in gt:
            tot += 1; hit += (t in top)
    return hit / max(1, tot)


def hits_by_bucket(score_fn, pairs, freq, k=20):
    buckets = {"cold(f=0)": [0, 0], "rare(1-2)": [0, 0], "warm(>2)": [0, 0]}
    for seeds, gt in pairs:
        sc = score_fn(list(seeds))
        top = set(top_items(sc, list(seeds), k).tolist())
        for t in gt:
            f = freq[t]
            b = "cold(f=0)" if f == 0 else ("rare(1-2)" if f <= 2 else "warm(>2)")
            buckets[b][1] += 1
            if t in top:
                buckets[b][0] += 1
    return {kk: (v[0] / v[1] if v[1] else float("nan"), v[1]) for kk, v in buckets.items()}


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def run(dataset):
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    provided = ProvidedSplits(dd, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    freq = graph.freq.astype(np.float64)
    C = graph.C.tocsr(); W = graph.W_pair.tocsr()
    import torch as _t
    x = _t.load(os.path.join(dd, "content_feature.pt"), map_location="cpu", weights_only=False).detach().float().numpy()
    content = (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    content_t = torch.from_numpy(content)
    dc = content.shape[1]
    # collaborative SVD-wpair (baseline)
    svd = TruncatedSVD(n_components=min(DIM, n - 2), random_state=0)
    A = svd.fit_transform(W).astype(np.float32); B = svd.components_.T.astype(np.float32)
    log(f"\n### {dataset}: n={n} dc={dc} DIM={DIM} | cold={float((freq==0).mean())*100:.1f}% | setup {time.time()-t0:.0f}s")

    test_pairs = provided.pairs("test")
    test_ids = sorted(provided.splits["test"][0])
    oneseed = sparse_pairs(provided.pairs("test"), test_ids, "one_seed")
    val_pairs = provided.pairs("valid")[:VAL_N]
    if TEST_N:
        test_pairs, oneseed = test_pairs[:TEST_N], oneseed[:TEST_N]

    train_bundles = list(provided.train.values())
    warm_ids = np.where(freq > 0)[0].astype(np.int64)
    Fall_fn = lambda m: m.P(content_t)
    log(f"  training content->complementarity map (P,Q: {dc}->{DIM}), ALL-neg...")
    model = ContentComp(dc, DIM)
    model = train_content(model, content_t, C, train_bundles, val_pairs, Fall_fn, n, seed=0)
    log(f"  training content map, WARM-only negatives (cold-suppression test)...")
    model_wn = ContentComp(dc, DIM)
    model_wn = train_content(model_wn, content_t, C, train_bundles, val_pairs, Fall_fn, n, seed=0, neg_pool=warm_ids)
    with torch.no_grad():
        Fall = model.P(content_t).numpy()          # (N,d) candidate embeddings, precomputed
        Fall_wn = model_wn.P(content_t).numpy()

    def s_collab(seeds):
        return A @ B[seeds].sum(0)

    def s_contcos(seeds):
        return content @ content[seeds].sum(0)

    def s_contlearn(seeds):
        with torch.no_grad():
            gs = model.Q(content_t[seeds]).mean(0).numpy()
        return Fall @ gs

    def s_contlearn_wn(seeds):
        with torch.no_grad():
            gs = model_wn.Q(content_t[seeds]).mean(0).numpy()
        return Fall_wn @ gs

    def s_fused(seeds):
        return zscore(s_collab(seeds)) + zscore(s_contlearn(seeds))

    def s_fused_wn(seeds):
        return zscore(s_collab(seeds)) + zscore(s_contlearn_wn(seeds))

    out = {"dataset": dataset, "n": int(n), "dim": DIM, "conditions": {}}
    for cond, pairs in [("supplied", test_pairs), ("one_seed", oneseed)]:
        res = {name: hits_by_bucket(fn, pairs, freq) for name, fn in
               [("collab", s_collab), ("content_cos", s_contcos),
                ("content_learn", s_contlearn), ("content_learn_wn", s_contlearn_wn),
                ("fused", s_fused), ("fused_wn", s_fused_wn)]}
        out["conditions"][cond] = {name: {b: v for b, v in r.items()} for name, r in res.items()}
        log(f"  [{cond}] {len(pairs)} queries  Hit@20 (cold / rare / warm):")
        for name, r in res.items():
            log(f"      {name:14s} "
                f"cold={r['cold(f=0)'][0]:.4f}(n={r['cold(f=0)'][1]})  "
                f"rare={r['rare(1-2)'][0]:.4f}  warm={r['warm(>2)'][0]:.4f}")
    log(f"  ({dataset} done in {time.time()-t0:.0f}s)")
    return out


def main():
    results = [run(ds) for ds in DATASETS]
    out_dir = os.path.join(PROJECT_ROOT, "results", "sefp")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "cold_learned.json")
    json.dump(results, open(outp, "w"), indent=2)
    log(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
