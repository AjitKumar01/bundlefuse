"""Gate 2 - the sparse-input test (Spotify): does a LEARNED context gate beat BundleFuse's FIXED
closed form in the single-seed regime, where BundleFuse is structurally weakest?

Report 1 showed BundleFuse's conjunction term Delta_inc(c|S) is identically 0 when |S|=1, and its
shrinkage gate u_c = kappa/(f_c+kappa) uses ONE kappa frozen on validation. The generalization study
confirms the consequence empirically on this exact split: at one_seed BundleFuse collapses from
R@20 0.1556 -> 0.0616, and simple hybrids MATCH or beat it there (HardSwitchHybrid 0.0625 R@20,
0.1669 N@20 vs BundleFuse 0.1548). The closed form's hand-tuned edge evaporates where the fixed
rules bite. Gate 2 asks: can a learned, context-dependent gate win in that regime?

Three models are trained under one identical pipeline (mixed seed-count masking so the gate can learn
|S|-dependence; same NEG/hard negatives, listwise softmax, NDCG@20 val-selection, patience):
  PoolFirst    : phi(c,s) mean-pooled over s FIRST -> MLP            (pooling-order control, no gate)
  SCACS_nogate : order-statistic pooling of exact phi(c,s) -> MLP    (architecture only, no gate)
  SCACS_gate   : SCACS_nogate + learned gate  u_c = sigma(MLP(log f_c, log|S|, coverage))
                 and content-only cold-start prior mu0; score = (1-u_c)*Y + u_c*mu0

All rerank the SAME cheap collaborative shortlist. Each is evaluated on BOTH the supplied masks and
the EXACT one_seed pairs used by the generalization study (robust_protocol.sparse_pairs), so numbers
are directly comparable. Anchors (results/generalization_study/spotify, same split):
  supplied : BundleFuse 0.1556 / N 0.2579 ; one_seed : BundleFuse 0.0616 / N 0.1548,
             HardSwitchHybrid 0.0625 / N 0.1669, MetadataRanker 0.0530 / N 0.1446.

Decision rule: SCACS_gate one_seed R@20 > BundleFuse 0.0616 (ideally > HardSwitch 0.0625) WITHOUT
collapsing on supplied => the learned-gate thesis holds and the architecture has a point. If the
learned gate gives no single-seed advantage, BundleFuse's fixed shrinkage was already adequate and
we stop (clean negative result).

Env knobs: QUICK=1, EPOCHS, STEPS, NEG, HARD_FRAC, SEED_CAP, SHORTLIST_M, MIX_SEEDS (default 1:
variable seed counts in training), GATE (unused toggle; the three arms are fixed), LR, EVAL_EVERY,
PATIENCE, TEST_N, VAL_N. Writes results/gate2/scacs_spotify.json.  Run: python3 reports/gate2_scacs.py
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
from bundlefuse.robust_protocol import sparse_pairs

DATASET = "spotify"
# From results/generalization_study/spotify (same provided split), R@20 / N@20:
ANCHORS = {"supplied": {"BundleFuse": (0.1556, 0.2579), "HardSwitchHybrid": (0.1000, 0.1662),
                        "MetadataRanker": (0.1206, 0.1969)},
           "one_seed": {"BundleFuse": (0.0616, 0.1548), "HardSwitchHybrid": (0.0625, 0.1669),
                        "MetadataRanker": (0.0530, 0.1446)}}

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
MIX_SEEDS = os.environ.get("MIX_SEEDS", "1") != "0"               # train over variable |S| (1..len-1) so the gate learns |S|-dependence
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
    reductions over the seed axis, no per-pair neural net.

    LEARNED GATE (gate=True): generalises BundleFuse's fixed u_c = kappa/(f_c+kappa) to a learned,
    CONTEXT-dependent reliability  u_c = sigma(MLP(log f_c, log|S|, cooc-coverage, max-cooc)), which
    blends the interaction score Y with a content-only cold-start prior mu0:  (1-u_c)*Y + u_c*mu0.
    The gate can shrink harder exactly when seed evidence is thin (small |S|, low coverage) -- the
    single-seed regime where BundleFuse's single frozen kappa cannot adapt."""
    N_STATS = 5

    def __init__(self, dphi=DPHI, hid=HID, content_dim=0, gate=False):
        super().__init__()
        self.pool_norm = InputNorm(dphi)                              # scale raw features before pooling
        d_pooled = dphi * self.N_STATS
        self.norm = nn.LayerNorm(d_pooled)
        self.mlp = nn.Sequential(nn.Linear(d_pooled, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.pop_weight = nn.Parameter(torch.tensor(0.0))
        self.gate_on = gate
        if gate:
            self.gate_mlp = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
            # Query-DEPENDENT cold-start prior mu0 from the NON-collaborative channels only
            # (content cosine, same-artist, same-album) pooled over seeds -- mean+max = 6 inputs.
            # Mirrors BundleFuse's metadata prior; it carries candidates with no co-occurrence evidence.
            self.prior = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, phi, smask, logf_c, content_c=None, freq_c=None):
        # phi: (B, Kc, Ks, d)  smask: (B, Ks)  logf_c: (B, Kc)  content_c: (B, Kc, dc)
        B, Kc = phi.shape[0], phi.shape[1]
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
        if self.gate_on:
            logS = torch.log(smask.sum(-1).clamp_min(1))[:, None].expand(B, Kc)  # (B,Kc)
            cov = posfrac[..., 0]                                     # fraction of seeds candidate co-occurs with
            mxc = mx[..., 0]                                          # strongest single-seed co-occurrence
            u = torch.sigmoid(self.gate_mlp(torch.stack([logf_c, logS, cov, mxc], -1))).squeeze(-1)
            # cold-start prior: content/taxonomy affinity to the seeds (channels 4,5,6), mean+max pooled
            mu0_in = torch.cat([mean[..., 4:7], mx[..., 4:7]], dim=-1)          # (B,Kc,6)
            mu0 = self.prior(mu0_in).squeeze(-1)
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
        if MIX_SEEDS:
            split = int(rng.integers(1, len(it)))                 # |S| in 1..len-1 so the gate sees single-seed
        else:
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
    dc = D["content"].shape[1]
    phi = torch.zeros(B, Kc, Ks, DPHI)
    smask = torch.zeros(B, Ks)
    logf_c = torch.zeros(B, Kc)
    content_c = torch.zeros(B, Kc, dc)
    for j in range(B):
        s = np.asarray(seeds[j], np.int64)
        c = np.asarray(cands[j], np.int64)
        phi[j, :, :len(s), :] = pair_feats(D, c, s)
        smask[j, :len(s)] = 1.0
        logf_c[j] = torch.from_numpy(D["logf"][c])
        content_c[j] = torch.from_numpy(D["content"][c])
    return model(phi, smask, logf_c, content_c=content_c)


def train_model(model, D, train_bundles, val_shortlists, rng_seed, tag=None):
    tag = tag or model.__class__.__name__
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
            log(f"  [{tag}] ep={ep} loss={np.mean(losses):.4f} "
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
            content_c = torch.from_numpy(D["content"][cand]).unsqueeze(0)
            full[cand] = model(phi, smask, logf_c, content_c=content_c).squeeze(0).numpy()
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
    # EXACT one_seed pairs used by the generalization study (so our numbers match BundleFuse's 0.0616).
    test_ids = sorted(provided.splits["test"][0])
    full_test = provided.pairs("test")
    oneseed_full = sparse_pairs(full_test, test_ids, "one_seed")
    if TEST_N and TEST_N < len(oneseed_full):
        oneseed_pairs = oneseed_full[:TEST_N]
    else:
        oneseed_pairs = oneseed_full
    log(f"train={len(train_bundles)} val={len(val_pairs)} test(supplied)={len(test_pairs)} "
        f"test(one_seed)={len(oneseed_pairs)} | EPOCHS={EPOCHS} STEPS={STEPS or 'full'} NEG={NEG} "
        f"HARD_FRAC={HARD_FRAC} SEED_CAP={SEED_CAP} SHORTLIST_M={SHORTLIST_M} MIX_SEEDS={MIX_SEEDS} QUICK={QUICK}")
    ts = time.time()
    val_shortlists, val_ceiling = build_shortlists(D, val_pairs, SHORTLIST_M)             # supplied val (selection)
    test_supplied, ceil_sup = build_shortlists(D, test_pairs, SHORTLIST_M)
    test_oneseed, ceil_one = build_shortlists(D, oneseed_pairs, SHORTLIST_M)
    log(f"shortlists precomputed in {time.time()-ts:.0f}s | val_ceiling={val_ceiling:.4f} "
        f"supplied_ceiling={ceil_sup:.4f} one_seed_ceiling={ceil_one:.4f}")

    results = {"dataset": DATASET, "anchors": ANCHORS, "config": {
        "epochs": EPOCHS, "steps": STEPS, "neg": NEG, "batch": BATCH, "seed_cap": SEED_CAP,
        "shortlist_m": SHORTLIST_M, "lr": LR, "hard_frac": HARD_FRAC, "mix_seeds": MIX_SEEDS,
        "n_val": len(val_pairs), "n_test": len(test_pairs), "n_one_seed": len(oneseed_pairs)},
        "ceilings": {"supplied": ceil_sup, "one_seed": ceil_one}, "models": {}}

    arms = [("PoolFirst", lambda: PoolFirst()),
            ("SCACS_nogate", lambda: SCACS(content_dim=D["content"].shape[1], gate=False)),
            ("SCACS_gate", lambda: SCACS(content_dim=D["content"].shape[1], gate=True))]
    for name, ctor in arms:
        log(f"\n=== training {name} (mixed seed counts) ===")
        model = ctor()
        nparam = sum(p.numel() for p in model.parameters())
        model, best_val = train_model(model, D, train_bundles, val_shortlists, rng_seed=0, tag=name)
        tt = time.time()
        ms = eval_cached(model, D, test_supplied, Ks=(10, 20, 50))
        mo = eval_cached(model, D, test_oneseed, Ks=(10, 20, 50))
        rec = {"params": int(nparam), "best_val_ndcg20": float(best_val),
               "supplied": {f"{M}": {"R": float(ms[M][0]), "N": float(ms[M][1])} for M in (10, 20, 50)},
               "one_seed": {f"{M}": {"R": float(mo[M][0]), "N": float(mo[M][1])} for M in (10, 20, 50)}}
        results["models"][name] = rec
        log(f"{name}: params={nparam:,} val_N20={best_val:.4f} ({time.time()-tt:.0f}s)")
        log(f"   supplied R@20={ms[20][0]:.4f} N@20={ms[20][1]:.4f} | "
            f"one_seed R@20={mo[20][0]:.4f} N@20={mo[20][1]:.4f}")

    out_dir = os.path.join(PROJECT_ROOT, "results", "gate2")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "scacs_spotify.json")
    json.dump(results, open(outp, "w"), indent=2)

    def g(name, case):
        return results["models"].get(name, {}).get(case, {}).get("20", {})
    bf_one = ANCHORS["one_seed"]["BundleFuse"]
    hs_one = ANCHORS["one_seed"]["HardSwitchHybrid"]
    sg = g("SCACS_gate", "one_seed"); sn = g("SCACS_nogate", "one_seed")
    log(f"\n=== GATE 2 SUMMARY ({time.time()-t0:.0f}s) ===")
    log(f"{'model':16s} {'supplied R@20':>14} {'one_seed R@20':>14} {'one_seed N@20':>14}")
    log(f"{'BundleFuse':16s} {ANCHORS['supplied']['BundleFuse'][0]:14.4f} {bf_one[0]:14.4f} {bf_one[1]:14.4f}")
    log(f"{'HardSwitch':16s} {ANCHORS['supplied']['HardSwitchHybrid'][0]:14.4f} {hs_one[0]:14.4f} {hs_one[1]:14.4f}")
    for name in ("PoolFirst", "SCACS_nogate", "SCACS_gate"):
        s, o = g(name, "supplied"), g(name, "one_seed")
        log(f"{name:16s} {s.get('R',0):14.4f} {o.get('R',0):14.4f} {o.get('N',0):14.4f}")
    win_r = sg.get("R", 0) > bf_one[0]
    win_n = sg.get("N", 0) > bf_one[1]
    log(f"\nverdict: SCACS_gate one_seed R@20={sg.get('R',0):.4f} vs BundleFuse {bf_one[0]:.4f} "
        f"=> {'BEATS' if win_r else 'does NOT beat'}; N@20={sg.get('N',0):.4f} vs {bf_one[1]:.4f} "
        f"=> {'BEATS' if win_n else 'does NOT beat'}")
    log(f"gate effect (one_seed R@20): SCACS_gate - SCACS_nogate = {sg.get('R',0)-sn.get('R',0):+.4f}")
    log(f"wrote {outp}")


if __name__ == "__main__":
    main()
