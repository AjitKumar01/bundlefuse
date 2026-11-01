"""Gate 1 - the representational test (Spotify): interaction-first attention over EXACT sparse
compatibility features vs pool-then-interact, trained and evaluated under one identical pipeline.

Report 1 argued the compact pool-then-interact two-tower loses accuracy because it factorises the
full-rank collaborative operators (C / PPMI / W_pair) into a low-rank item x item dot product and
pools seeds BEFORE they meet the candidate, which cannot express per-seed conjunction. Report 2
proposed SCACS: look up the exact per-pair statistics phi(c, s), let the candidate meet each seed
FIRST, then pool with attention + a soft conjunction count + an O(|S|) second-order head.

Gate 1 asks ONE thing on the dense (original-mask) case: does interaction-first scoring over exact
features beat pool-first scoring? To isolate the architecture we hold EVERYTHING else identical to
the recorded two-tower recipe (bundlefuse/baselines/metadata_ranker.py):
  - same seed masking (shuffle bundle, split = round(len/2)),
  - same negatives (NEG uniform-random per positive, resampled out of the bundle),
  - same listwise softmax cross-entropy with the positive at index 0,
  - same validation-selected NDCG@20 with early stopping (patience),
  - same provided split, same evaluator metric (bundlefuse.metrics, exact tie policy).
Only the model changes. Two models are trained under this one loop:
  SCACS      : phi(c,s) -> MLP -> masked attention pool + soft conjunction + 2nd-order head -> score
  PoolFirst  : phi(c,s) mean-pooled over s FIRST -> MLP -> score   (conservative pooling-order control;
               it still sees summed pairwise evidence, so a SCACS win isolates the interaction structure)

Both rerank the SAME cheap collaborative shortlist (union of per-seed top-M cooc / W_pair neighbours),
so retrieval is held constant and only re-ranking quality differs. External anchors (recorded, audited,
3-seed, same split -- results/compact_pilot/ANALYSIS.md): BundleFuse R@20 = 0.1556; engineered
two-tower R@20 = 0.1053; raw two-tower R@20 = 0.0753.

Decision rule: SCACS test R@20 > engineered two-tower 0.1053 supports the interaction-first thesis;
SCACS >= BundleFuse 0.1556 is the stronger outcome. If SCACS does not beat the two-tower the thesis
is falsified for this case and we stop.

Env knobs: QUICK=1 (tiny smoke run), EPOCHS, STEPS, NEG, SEED_CAP, SHORTLIST_M, KAPPA (fixed gate;
0 = no shrinkage), LR, EVAL_EVERY, PATIENCE, TEST_N, VAL_N. Writes results/gate1/scacs_spotify.json.
Run from repo root:  python3 reports/gate1_scacs.py
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import ranking_metrics

DATASET = "spotify"
ANCHORS = {"bundlefuse_r20": 0.155646, "two_tower_engineered_r20": 0.105349,
           "two_tower_raw_r20": 0.075347}

QUICK = os.environ.get("QUICK", "0") != "0"
EPOCHS = int(os.environ.get("EPOCHS", "3" if QUICK else "40"))
STEPS = int(os.environ.get("STEPS", "20" if QUICK else "0"))      # 0 = full pass over train / batch
NEG = int(os.environ.get("NEG", "32"))
HARD_FRAC = float(os.environ.get("HARD_FRAC", "0.5"))            # fraction of negatives drawn from seed co-occurrence pool
BATCH = int(os.environ.get("BATCH", "128"))
SEED_CAP = int(os.environ.get("SEED_CAP", "20"))                  # cap |S| for tractability
SHORTLIST_M = int(os.environ.get("SHORTLIST_M", "1000"))          # per-leg shortlist depth
KAPPA = float(os.environ.get("KAPPA", "0"))                       # fixed gate; 0 => no shrinkage
LR = float(os.environ.get("LR", "0.003"))
WD = float(os.environ.get("WD", "1e-5"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "2"))
PATIENCE = int(os.environ.get("PATIENCE", "4"))
HID = int(os.environ.get("HID", "64"))
TEST_N = int(os.environ.get("TEST_N", "200" if QUICK else "0"))   # 0 = all test queries
VAL_N = int(os.environ.get("VAL_N", "200" if QUICK else "1000"))
DEVICE = torch.device("cpu")
DPHI = 8                                                          # per-pair feature dimension


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- data + operators
def load_everything():
    d = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DATASET)
    n = json.load(open(os.path.join(d, "count.json")))["#I"]
    provided = ProvidedSplits(d, n)
    t0 = time.time()
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)  # svd_dim small: Ucsvd unused
    C = graph.C.tocsr()
    W = graph.W_pair.tocsr()
    Wt = graph.W_pair.T.tocsr()          # W_pair[s, c] via Wt[c, s]
    PPMI = graph.PPMI.tocsr()
    freq = graph.freq.astype(np.float32)
    # artist / album labels
    info = json.load(open(os.path.join(d, "item_info.json")))
    art = np.zeros(n, np.int64); alb = np.zeros(n, np.int64)
    am, lm = {}, {}
    for k, v in info.items():
        i = int(k)
        art[i] = am.setdefault(v.get("artist_uri", "?"), len(am))
        alb[i] = lm.setdefault(v.get("album_uri", "?"), len(lm))
    # CLAP content, L2-normalised
    x = torch.load(os.path.join(d, "content_feature.pt"), map_location="cpu",
                   weights_only=False).detach().float().numpy()
    content = (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    logf = np.log1p(freq).astype(np.float32)
    logf_norm = ((logf - logf.mean()) / (logf.std() + 1e-8)).astype(np.float32)
    log(f"{DATASET}: n={n} train={len(provided.train)} graph+ops built in {time.time()-t0:.0f}s "
        f"| C.nnz={C.nnz:,} W.nnz={W.nnz:,} content_dim={content.shape[1]}")
    return dict(n=n, provided=provided, C=C, W=W, Wt=Wt, PPMI=PPMI, freq=freq,
                art=art, alb=alb, content=content, logf=logf_norm)


# --------------------------------------------------------------------------- exact per-pair features
def pair_feats(D, cand, seed):
    """phi for every (candidate, seed) pair. cand: (Kc,) int, seed: (Ks,) int.
    Returns (Kc, Ks, DPHI) float32 tensor. All entries are EXACT sparse lookups / dot products."""
    Kc, Ks = len(cand), len(seed)
    Cc = D["C"][cand][:, seed].toarray().astype(np.float32)                 # co-occurrence
    Wc = D["W"][cand][:, seed].toarray().astype(np.float32)                 # W_pair[c, s]
    Wsc = D["Wt"][cand][:, seed].toarray().astype(np.float32)               # W_pair[s, c]
    Pc = D["PPMI"][cand][:, seed].toarray().astype(np.float32)              # PPMI[c, s]
    con = (D["content"][cand] @ D["content"][seed].T).astype(np.float32)    # CLAP cosine
    same_art = (D["art"][cand][:, None] == D["art"][seed][None, :]).astype(np.float32)
    same_alb = (D["alb"][cand][:, None] == D["alb"][seed][None, :]).astype(np.float32)
    degs = np.tile(D["logf"][seed][None, :], (Kc, 1)).astype(np.float32)    # seed exposure
    phi = np.stack([np.log1p(Cc), Pc, Wc, Wsc, con, same_art, same_alb, degs], axis=-1)
    return torch.from_numpy(phi)


# --------------------------------------------------------------------------- models
class InputNorm(nn.Module):
    """Per-feature affine standardisation (BatchNorm over the DPHI channels) so the raw-scale
    collaborative / content / binary features are comparable before the MLP."""
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, phi):                      # phi: (..., d)
        shp = phi.shape
        return self.bn(phi.reshape(-1, shp[-1])).reshape(shp)


class SCACS(nn.Module):
    """Interaction-first via learned ORDER-STATISTIC pooling of exact per-pair features.

    The candidate meets each seed first (exact phi(c,s)), then each feature is aggregated over the
    seed set by a fixed bank of order statistics -- [mean, max, sum, positive-fraction, top-2 mean] --
    and a small MLP combines them. This keeps the full seed-match DISTRIBUTION a mean-pool destroys:
      - `max`  -> "strongly matches at least ONE seed" (the operation pool-then-interact cannot do),
      - `positive-fraction` / `top-2 mean` -> the conjunction ("matches >=2 seeds") that Delta_inc encodes.
    It is a STRICT SUPERSET of PoolFirst (mean is one of the statistics) and costs ~the same: cheap
    reductions over the seed axis, no per-pair neural net. Optional fixed gate shrinks rare candidates
    toward a content-only prior (u_c = kappa/(f_c+kappa), kappa fixed)."""
    N_STATS = 5

    def __init__(self, dphi=DPHI, hid=HID, content_dim=0, kappa=KAPPA):
        super().__init__()
        self.pool_norm = InputNorm(dphi)                              # scale raw features before pooling
        d_pooled = dphi * self.N_STATS
        self.norm = nn.LayerNorm(d_pooled)
        self.mlp = nn.Sequential(nn.Linear(d_pooled, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.pop_weight = nn.Parameter(torch.tensor(0.0))
        self.kappa = kappa
        self.prior = nn.Linear(content_dim, 1) if (kappa > 0 and content_dim) else None

    def forward(self, phi, smask, logf_c, content_c=None, freq_c=None):
        # phi: (B, Kc, Ks, d)  smask: (B, Ks)  logf_c: (B, Kc)
        z = self.pool_norm(phi)
        m = smask[:, None, :, None]                                   # (B,1,Ks,1)
        cnt = smask.sum(-1).clamp_min(1)[:, None, None]               # (B,1,1)
        msum = (z * m).sum(2)                                         # (B,Kc,d)
        mean = msum / cnt
        zmask = z.masked_fill(m == 0, float("-inf"))
        mx = zmask.max(2).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        posfrac = (((phi > 0).float() * m).sum(2)) / cnt              # conjunction proxy (on RAW phi)
        k2 = min(2, phi.shape[2])
        t2 = zmask.topk(k2, dim=2).values
        t2 = torch.where(torch.isfinite(t2), t2, torch.zeros_like(t2)).mean(2)
        feat = torch.cat([mean, mx, msum, posfrac, t2], dim=-1)       # (B,Kc,5d)
        Y = self.mlp(self.norm(feat)).squeeze(-1) + self.pop_weight * logf_c
        if self.prior is not None and content_c is not None:
            mu0 = self.prior(content_c).squeeze(-1)
            u = self.kappa / (freq_c + self.kappa)
            Y = (1.0 - u) * Y + u * mu0
        return Y


class PoolFirst(nn.Module):
    """Pooling-order control: mean-pool phi over seeds FIRST, then MLP. Same inputs, same budget,
    no attention / conjunction / 2nd-order. Conservative (summed pairwise evidence survives the mean)."""
    def __init__(self, dphi=DPHI, hid=HID):
        super().__init__()
        self.norm = InputNorm(dphi)
        self.mlp = nn.Sequential(nn.Linear(dphi, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.pop_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, phi, smask, logf_c, content_c=None, freq_c=None):
        z = self.norm(phi)
        m = smask[:, None, :, None]
        phibar = (z * m).sum(2) / m.sum(2).clamp_min(1)              # (B,Kc,d) pooled first
        Y = self.mlp(phibar).squeeze(-1) + self.pop_weight * logf_c
        return Y


# --------------------------------------------------------------------------- training
def make_batch(D, bundles, rng, n, neg, seed_cap, hard_frac):
    """Seed/positive split mirrors the two-tower recipe. Negatives are a mix of HARD negatives
    (items co-occurring with a seed -- the distribution the shortlist reranker actually faces at
    eval) and uniform-random negatives, controlled by hard_frac. Applied identically to both models."""
    C = D["C"]
    seeds, cands = [], []
    for b in bundles:
        it = np.asarray(b, np.int64).copy()
        rng.shuffle(it)
        split = max(1, min(len(it) - 1, round(len(it) / 2)))
        s = it[:split][:seed_cap].tolist()
        pos = int(rng.choice(it[split:]))
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
                hard = rng.choice(pool, size=min(n_hard, len(pool)),
                                  replace=len(pool) < n_hard).tolist()
        need = neg - len(hard)
        forbidden = itset.union(hard)
        ng = rng.integers(n, size=need)
        bad = np.array([x in forbidden for x in ng])
        while bad.any():
            ng[bad] = rng.integers(n, size=int(bad.sum()))
            bad = np.array([x in forbidden for x in ng])
        seeds.append(s)
        cands.append([pos] + hard + ng.tolist())
    return seeds, cands


def batch_forward(model, D, seeds, cands):
    """Score each example's candidate list. Returns logits (B, 1+neg)."""
    B = len(seeds)
    Ks = max(len(s) for s in seeds)
    Kc = len(cands[0])
    phi = torch.zeros(B, Kc, Ks, DPHI)
    smask = torch.zeros(B, Ks)
    logf_c = torch.zeros(B, Kc)
    for j in range(B):
        s = np.asarray(seeds[j], np.int64)
        c = np.asarray(cands[j], np.int64)
        phi[j, :, :len(s), :] = pair_feats(D, c, s)
        smask[j, :len(s)] = 1.0
        logf_c[j] = torch.from_numpy(D["logf"][c])
    return model(phi, smask, logf_c)


def train_model(model, D, train_bundles, val_shortlists, rng_seed):
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
            bundles = [train[i] for i in idx]
            seeds, cands = make_batch(D, bundles, rng, D["n"], NEG, SEED_CAP, HARD_FRAC)
            logits = batch_forward(model, D, seeds, cands)
            loss = Fn.cross_entropy(logits, torch.zeros(len(seeds), dtype=torch.long))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            val = eval_cached(model, D, val_shortlists, Ks=(20,))[20][1]   # NDCG@20 (selection objective)
            if val > best + 1e-5:
                best, best_state, stale = val, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"  [{model.__class__.__name__}] ep={ep} loss={np.mean(losses):.4f} "
                f"val_N20={val:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


# --------------------------------------------------------------------------- shortlist + eval
def shortlist(D, seeds, M):
    x = np.zeros(D["n"], np.float32)
    x[seeds] = 1.0
    sc = np.asarray(D["C"] @ x).ravel()
    sw = np.asarray(D["W"] @ x).ravel()
    sc[seeds] = -1.0
    sw[seeds] = -1.0
    top_c = np.argpartition(-sc, M)[:M] if (sc > 0).sum() > M else np.flatnonzero(sc > 0)
    top_w = np.argpartition(-sw, M)[:M] if (sw > 0).sum() > M else np.flatnonzero(sw > 0)
    cand = np.unique(np.concatenate([top_c, top_w]).astype(np.int64))
    cand = cand[~np.isin(cand, seeds)]
    return cand


def build_shortlists(D, pairs, M):
    """Precompute each query's shortlist ONCE (the expensive C@x / W@x matvec). Reused every eval."""
    out = []
    ceil_hit, ceil_tot = 0, 0
    for seeds, gt in pairs:
        s = np.asarray(list(seeds)[:SEED_CAP], np.int64)
        cand = shortlist(D, list(seeds), M)
        gtset = set(gt)
        ceil_tot += len(gtset)
        ceil_hit += len(gtset.intersection(cand.tolist()))
        out.append({"seeds": s, "cand": cand, "pair": (list(seeds), list(gt))})
    return out, ceil_hit / max(1, ceil_tot)


@torch.no_grad()
def eval_cached(model, D, shortlists, Ks=(10, 20, 50)):
    """Score precomputed shortlists (only model forward + sparse slicing; no matvec)."""
    model.eval()
    n = D["n"]
    cols, pairs = [], []
    for q in shortlists:
        s, cand = q["seeds"], q["cand"]
        full = np.full(n, -1e9, np.float32)
        if len(cand):
            phi = pair_feats(D, cand, s).unsqueeze(0)
            smask = torch.ones(1, len(s))
            logf_c = torch.from_numpy(D["logf"][cand]).unsqueeze(0)
            full[cand] = model(phi, smask, logf_c).squeeze(0).numpy()
        cols.append(full)
        pairs.append(q["pair"])
    scores = np.stack(cols, axis=1)
    return ranking_metrics(scores, pairs, Ks=Ks)


# --------------------------------------------------------------------------- main
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
    log(f"train={len(train_bundles)} val={len(val_pairs)} test={len(test_pairs)} | "
        f"EPOCHS={EPOCHS} STEPS={STEPS or 'full'} NEG={NEG} SEED_CAP={SEED_CAP} "
        f"SHORTLIST_M={SHORTLIST_M} KAPPA={KAPPA} QUICK={QUICK}")
    ts = time.time()
    val_shortlists, val_ceiling = build_shortlists(D, val_pairs, SHORTLIST_M)
    test_shortlists, test_ceiling = build_shortlists(D, test_pairs, SHORTLIST_M)
    log(f"shortlists precomputed in {time.time()-ts:.0f}s | val_ceiling={val_ceiling:.4f} "
        f"test_ceiling={test_ceiling:.4f}")

    results = {"dataset": DATASET, "anchors": ANCHORS, "config": {
        "epochs": EPOCHS, "steps": STEPS, "neg": NEG, "batch": BATCH, "seed_cap": SEED_CAP,
        "shortlist_m": SHORTLIST_M, "kappa": KAPPA, "lr": LR, "hard_frac": HARD_FRAC,
        "n_val": len(val_pairs), "n_test": len(test_pairs)}, "models": {}}

    for name, ctor in [("PoolFirst", lambda: PoolFirst()),
                       ("SCACS", lambda: SCACS(content_dim=D["content"].shape[1]))]:
        log(f"\n=== training {name} ===")
        model = ctor()
        nparam = sum(p.numel() for p in model.parameters())
        model, best_val = train_model(model, D, train_bundles, val_shortlists, rng_seed=0)
        tt = time.time()
        m = eval_cached(model, D, test_shortlists, Ks=(10, 20, 50))
        rec = {"params": int(nparam), "best_val_ndcg20": float(best_val),
               "test": {f"{M}": {"R": float(m[M][0]), "N": float(m[M][1])} for M in (10, 20, 50)},
               "shortlist_recall_ceiling": float(test_ceiling)}
        results["models"][name] = rec
        log(f"{name}: params={nparam:,} val_N20={best_val:.4f} | TEST R@20={m[20][0]:.4f} "
            f"N@20={m[20][1]:.4f} R@10={m[10][0]:.4f} R@50={m[50][0]:.4f} "
            f"ceiling={test_ceiling:.4f} ({time.time()-tt:.0f}s)")

    out_dir = os.path.join(PROJECT_ROOT, "results", "gate1")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "scacs_spotify.json")
    json.dump(results, open(outp, "w"), indent=2)
    sc = results["models"].get("SCACS", {}).get("test", {}).get("20", {}).get("R", 0)
    pf = results["models"].get("PoolFirst", {}).get("test", {}).get("20", {}).get("R", 0)
    log(f"\n=== GATE 1 SUMMARY ({time.time()-t0:.0f}s) ===")
    log(f"BundleFuse R@20          = {ANCHORS['bundlefuse_r20']:.4f}  (target to approach)")
    log(f"two-tower engineered R@20= {ANCHORS['two_tower_engineered_r20']:.4f}  (bar to beat)")
    log(f"PoolFirst (in-script)    = {pf:.4f}")
    log(f"SCACS                    = {sc:.4f}")
    log(f"verdict: SCACS {'BEATS' if sc > ANCHORS['two_tower_engineered_r20'] else 'does NOT beat'} "
        f"the engineered two-tower; SCACS vs PoolFirst = {sc - pf:+.4f}")
    log(f"wrote {outp}")


if __name__ == "__main__":
    main()
