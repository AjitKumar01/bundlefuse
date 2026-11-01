"""MMFuse: multimodal (content) + collaborative (co-occurrence) fusion by CROSS-ATTENTION, trained
end-to-end on the ranking task.

Fixes the weak earlier attention test (which was distillation-only, single-linear, borrowed from random
anchors, and never fused content+collab at scoring time). Here the candidate cross-attends over the
SEED set using BOTH modalities, and a collaborative residual preserves warm accuracy:

  per item i:  u_i = frozen SVD-W_pair collab embedding (0 for cold)   e_i = multimodal content
  token       t_i = A_u u_i + A_e e_i            (fused; cold items fall back to content automatically)
  query       q_c = A_u u_c + A_e e_c            (candidate; content-driven when cold)
  context     ctx_c = MultiHeadAttn(query=q_c, keys/vals = {t_s : s in S})      (H heads, masked)
  score(c|S)  = scale_r * <u_c, sum_s u_s>   +   head([q_c ; ctx_c])
                \_____ collaborative residual (warm) ___/   \__ learned multimodal fusion (helps cold) __/

Trained listwise (softmax CE, hard negatives from co-occurrence at TRAIN time only). The head learns a
content->complementarity map GROUNDED in the collaborative ranking objective -- content and co-occurrence
used jointly, which is the point. Collab embeddings are FROZEN inputs (the training-data signal); only
the fusion (A_u, A_e, attention, head, scale_r) is learned.

Eval: full-catalogue ranking. We report COLD-target median rank + Hit@{20,100,500,2000} (tie-aware) and
warm Hit@20, vs baselines collab-only and ridge-borrow -- the fair test of whether a PROPER attention
fusion beats the crude borrow at the top, not just the mid-ranks. Env: DATASETS, DIM, HEADS, EPOCHS,
STEPS, NEG, HARD_FRAC, LR, TEST_N, CAND_CHUNK, SHORTLIST_N. Run: python3 reports/mmfuse.py

SHORTLIST_N>0 switches scoring to a two-stage path: a full collab-residual base (one cheap mat-vec)
plus attention rerank restricted to the collab top-N candidates UNIONED WITH ALL COLD items.
WARNING -- this is APPROXIMATE and does NOT reproduce the full-catalogue ranks. Verified 2026-11-01
(scratchpad/verify_shortlist.py, SAME model): 0/41 cold ranks matched on pog, 0/427 on spotify,
max|Drank| ~2e5, and only ~1.1-1.6x faster (cold is 20-38% of the catalogue, so the shortlist is
still huge). Root cause: the fused MLP term has a non-zero, unbounded offset, so scoring off-shortlist
items with fused=0 puts them on a different scale and corrupts the global ordering. There is no exact
shortlist for this head. Kept off by default (SHORTLIST_N=0). Real levers: batch/GPU the attention
(exact, faster) for diagnostics; a content-ANN cold candidate set (approximate, serving-only, measured
as a Recall@K drop not a rank) for latency. Do not use this path for reported numbers.
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
from bundlefuse.metrics import relevant_ranks
from bundlefuse.robust_protocol import sparse_pairs

DATASETS = os.environ.get("DATASETS", "pog_dense,spotify").split(",")
DIM = int(os.environ.get("DIM", "128"))
HEADS = int(os.environ.get("HEADS", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
STEPS = int(os.environ.get("STEPS", "0"))
NEG = int(os.environ.get("NEG", "32"))
HARD_FRAC = float(os.environ.get("HARD_FRAC", "0.5"))
BATCH = int(os.environ.get("BATCH", "128"))
SEED_CAP = int(os.environ.get("SEED_CAP", "20"))
LR = float(os.environ.get("LR", "0.002"))
WD = float(os.environ.get("WD", "1e-5"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "3"))
PATIENCE = int(os.environ.get("PATIENCE", "3"))
RIDGE = float(os.environ.get("RIDGE", "10.0"))
DROP_COLLAB = float(os.environ.get("DROP_COLLAB", "0.5"))   # simulate cold: zero positive's collab (DropoutNet)
TEST_N = int(os.environ.get("TEST_N", "2000"))
VAL_N = int(os.environ.get("VAL_N", "400"))
CAND_CHUNK = int(os.environ.get("CAND_CHUNK", "20000"))
SHORTLIST_N = int(os.environ.get("SHORTLIST_N", "0"))   # 0 = full catalogue; >0 = collab top-N (∪ cold) rerank


def log(*a):
    print(*a, flush=True)


class MMFuse(nn.Module):
    def __init__(self, du, de, d, heads):
        super().__init__()
        self.d, self.h, self.dh = d, heads, d // heads
        self.A_u = nn.Linear(du, d, bias=False)
        self.A_e = nn.Linear(de, d, bias=False)
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.scale_r = nn.Parameter(torch.tensor(1.0))
        nn.init.orthogonal_(self.A_u.weight); nn.init.orthogonal_(self.A_e.weight)

    def _mha(self, q, t, smask):
        # q:(B,Kc,d)  t:(B,Ks,d)  smask:(B,Ks)
        B, Kc, _ = q.shape; Ks = t.shape[1]
        Q = self.Wq(q).view(B, Kc, self.h, self.dh).transpose(1, 2)      # (B,h,Kc,dh)
        K = self.Wk(t).view(B, Ks, self.h, self.dh).transpose(1, 2)      # (B,h,Ks,dh)
        V = self.Wv(t).view(B, Ks, self.h, self.dh).transpose(1, 2)
        att = (Q @ K.transpose(-1, -2)) / (self.dh ** 0.5)               # (B,h,Kc,Ks)
        att = att.masked_fill(smask[:, None, None, :] == 0, float("-inf"))
        att = Fn.softmax(att, dim=-1)
        ctx = att @ V                                                    # (B,h,Kc,dh)
        return ctx.transpose(1, 2).reshape(B, Kc, self.d)

    def forward(self, cand_u, cand_e, seed_u, seed_e, smask):
        q = self.A_u(cand_u) + self.A_e(cand_e)                          # (B,Kc,d)
        t = self.A_u(seed_u) + self.A_e(seed_e)                          # (B,Ks,d)
        ctx = self._mha(q, t, smask)
        fused = self.head(torch.cat([q, ctx], dim=-1)).squeeze(-1)       # (B,Kc)
        r = torch.einsum("bcd,bd->bc", cand_u, (seed_u * smask[..., None]).sum(1))  # collab residual
        return self.scale_r * r + fused


def make_batch(C, bundles, rng, n, neg, seed_cap, hard_frac):
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
    sidx = torch.zeros(B, Ks, dtype=torch.long); smask = torch.zeros(B, Ks)
    for j, s in enumerate(seeds):
        sidx[j, :len(s)] = torch.tensor(s); smask[j, :len(s)] = 1.0
    cidx = torch.tensor([[pos[j]] + negs[j] for j in range(B)], dtype=torch.long)
    return sidx, smask, cidx


def gather(U, Bside, E, idx):
    return U[idx], Bside[idx], E[idx]


def train(model, U, Bside, E, C, bundles, val_pairs, n, freq, seed=0, drop_collab=0.0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    train_b = [list(b) for b in bundles if len(b) >= 2]
    steps = STEPS if STEPS > 0 else max(1, len(train_b) // BATCH)
    best, best_state, stale = -1e9, None, 0; t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); order = rng.permutation(len(train_b)); losses = []
        for st in range(steps):
            idx = order[(st * BATCH) % len(order): (st * BATCH) % len(order) + BATCH]
            if len(idx) < 2:
                continue
            sidx, smask, cidx = make_batch(C, [train_b[i] for i in idx], rng, n, NEG, SEED_CAP, HARD_FRAC)
            cu, _, ce = gather(U, Bside, E, cidx)
            _, su, se = gather(U, Bside, E, sidx)
            if drop_collab > 0:
                # DropoutNet-style cold simulation: zero the POSITIVE's collab (index 0) so the model
                # must rank it from content, and drop a subset of seeds' collab too.
                cu = cu.clone(); su = su.clone()
                dm = torch.rand(len(idx)) < drop_collab
                cu[dm, 0] = 0.0
                sd = torch.rand_like(su[..., 0]) < (drop_collab * 0.5)
                su[sd] = 0.0
            logits = model(cu, ce, su, se, smask)
            loss = Fn.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach()))
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            v = quick_warm_hit(model, U, Bside, E, val_pairs, n)
            if v > best + 1e-5:
                best, best_state, stale = v, {k: vv.detach().clone() for k, vv in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"    [mmfuse ep={ep}] loss={np.mean(losses):.4f} val_warmhit20={v:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); return model


@torch.no_grad()
def score_all(model, U, Bside, E, seeds, n):
    """Full-catalogue score vector for one query (chunked over candidates)."""
    s = list(seeds)
    su = Bside[s][None]; se = E[s][None]; smask = torch.ones(1, len(s))
    out = np.empty(n, np.float32)
    for c0 in range(0, n, CAND_CHUNK):
        ci = slice(c0, min(c0 + CAND_CHUNK, n))
        cu = U[ci][None]; ce = E[ci][None]
        out[ci] = model(cu, ce, su, se, smask).squeeze(0).numpy()
    return out


@torch.no_grad()
def score_shortlist(model, A, Bs, U, Bside, E, seeds, n, topn, cold_ids=None):
    """APPROXIMATE two-stage scorer -- does NOT reproduce full-catalogue ranks; see module docstring.
    Base = scale_r * collab-residual over the whole catalogue (cheap mat-vec). Attention rerank only on
    (collab top-N) u (all cold). Off-shortlist items keep the base (fused=0), which is WRONG because
    fused has a non-zero offset -- this corrupts the global ordering. Kept for serving experiments that
    measure a Recall@K drop, never for reported rank metrics."""
    s = list(seeds)
    coll = A @ Bs[s].sum(0)                             # full collab residual (== model's r)
    out = (float(model.scale_r) * coll).astype(np.float32)   # base: fused = 0 off-shortlist
    k = min(topn, n)
    short = np.argpartition(-coll, k - 1)[:k]
    if cold_ids is not None and len(cold_ids):
        short = np.union1d(short, cold_ids)
    su = Bside[s][None]; se = E[s][None]; smask = torch.ones(1, len(s))
    for c0 in range(0, len(short), CAND_CHUNK):
        sl = short[c0:c0 + CAND_CHUNK]
        cu = U[sl][None]; ce = E[sl][None]
        out[sl] = model(cu, ce, su, se, smask).squeeze(0).numpy()
    return out


@torch.no_grad()
def quick_warm_hit(model, U, Bside, E, pairs, n, k=20):
    hit, tot = 0, 0
    for seeds, gt in pairs:
        sc = score_all(model, U, Bside, E, seeds, n)
        from bundlefuse.metrics import top_items
        top = set(top_items(sc, list(seeds), k).tolist())
        for t in gt:
            tot += 1; hit += (t in top)
    return hit / max(1, tot)


def ridge_borrow(content, A, warm, cold_ids, lam):
    Ew, Vw = content[warm].astype(np.float64), A[warm].astype(np.float64)
    G = Ew.T @ Ew + lam * np.eye(content.shape[1])
    Wm = np.linalg.solve(G, Ew.T @ Vw).astype(np.float32)
    V = A.copy(); V[cold_ids] = content[cold_ids] @ Wm
    return V


def cold_stats(score_fn, pairs, freq, coldset, Ks=(20, 100, 500, 2000)):
    ranks, wh = [], [0, 0]
    for seeds, gt in pairs:
        sc = score_fn(seeds)
        gts = sorted(set(gt)); rr = relevant_ranks(sc, list(seeds), gts)
        for t, r in zip(gts, rr):
            if t in coldset:
                ranks.append(int(r))
            elif freq[t] > 2:
                wh[1] += 1; wh[0] += (r <= 20)
    ranks = np.array(ranks)
    return ({"median_rank": float(np.median(ranks)), "n": len(ranks),
             **{f"hit@{k}": float((ranks <= k).mean()) for k in Ks}}, wh[0] / max(1, wh[1]))


def run(dataset):
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    provided = ProvidedSplits(dd, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    freq = graph.freq.astype(np.float64); W = graph.W_pair.tocsr(); C = graph.C.tocsr()
    x = torch.load(os.path.join(dd, "content_feature.pt"), map_location="cpu", weights_only=False).detach().float().numpy()
    content = (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    de = content.shape[1]
    svd = TruncatedSVD(n_components=min(DIM, n - 2), random_state=0)
    A = svd.fit_transform(W).astype(np.float32); Bs = svd.components_.T.astype(np.float32)
    du = A.shape[1]
    U = torch.from_numpy(A); Bside = torch.from_numpy(Bs); E = torch.from_numpy(content)
    warm = np.where(freq > 0)[0]; cold_ids = np.where(freq == 0)[0]; coldset = set(cold_ids.tolist())
    log(f"\n### {dataset}: n={n} du={du} de={de} DIM={DIM} heads={HEADS} | warm={len(warm):,} cold={len(cold_ids):,} | setup {time.time()-t0:.0f}s")

    test_pairs = provided.pairs("test"); test_ids = sorted(provided.splits["test"][0])
    oneseed = sparse_pairs(provided.pairs("test"), test_ids, "one_seed")
    val_pairs = provided.pairs("valid")[:VAL_N]
    if TEST_N:
        test_pairs, oneseed = test_pairs[:TEST_N], oneseed[:TEST_N]

    bundles = list(provided.train.values())
    log(f"  training MMFuse no-dropout (params={sum(p.numel() for p in MMFuse(du, de, DIM, HEADS).parameters()):,})...")
    model_nd = train(MMFuse(du, de, DIM, HEADS), U, Bside, E, C, bundles, val_pairs, n, freq, drop_collab=0.0)
    log(f"  training MMFuse dropout={DROP_COLLAB} (cold simulation)...")
    model_dp = train(MMFuse(du, de, DIM, HEADS), U, Bside, E, C, bundles, val_pairs, n, freq, drop_collab=DROP_COLLAB)

    Vridge = ridge_borrow(content, A, warm, cold_ids, RIDGE)

    def s_collab(seeds):
        return A @ Bs[list(seeds)].sum(0)

    def s_ridge(seeds):
        return Vridge @ Bs[list(seeds)].sum(0)

    def s_mmfuse_nd(seeds):
        if SHORTLIST_N:
            return score_shortlist(model_nd, A, Bs, U, Bside, E, seeds, n, SHORTLIST_N, cold_ids)
        return score_all(model_nd, U, Bside, E, seeds, n)

    def s_mmfuse_dp(seeds):
        if SHORTLIST_N:
            return score_shortlist(model_dp, A, Bs, U, Bside, E, seeds, n, SHORTLIST_N, cold_ids)
        return score_all(model_dp, U, Bside, E, seeds, n)

    out = {"dataset": dataset, "conditions": {}}
    for cond, pairs in [("supplied", test_pairs), ("one_seed", oneseed)]:
        res, warm20 = {}, {}
        for name, fn in [("collab", s_collab), ("ridge", s_ridge),
                         ("mmfuse_nd", s_mmfuse_nd), ("mmfuse_dp", s_mmfuse_dp)]:
            res[name], warm20[name] = cold_stats(fn, pairs, freq, coldset)
        out["conditions"][cond] = {"cold": res, "warm_hit20": warm20}
        log(f"  [{cond}] {len(pairs)} q | COLD rank (n={res['collab']['n']}) | warm Hit@20: "
            + " ".join(f"{k}={warm20[k]:.3f}" for k in warm20))
        log(f"      {'method':8s} {'med_rank':>9} {'H@20':>8} {'H@100':>8} {'H@500':>8} {'H@2000':>8}")
        for name in res:
            r = res[name]
            log(f"      {name:8s} {r['median_rank']:9.0f} {r['hit@20']:8.4f} {r['hit@100']:8.4f} {r['hit@500']:8.4f} {r['hit@2000']:8.4f}")
    log(f"  ({dataset} done in {time.time()-t0:.0f}s)")
    return out


def main():
    results = [run(ds) for ds in DATASETS]
    out_dir = os.path.join(PROJECT_ROOT, "results", "sefp")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "mmfuse.json")
    json.dump(results, open(outp, "w"), indent=2)
    log(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
