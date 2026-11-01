"""SEFP - Set Exponential-Family Predictor: the general bundle-completion architecture derived from
the Laplace approximation of the exchangeable-source posterior-predictive  p*(c|S) = E_{eta|S}[theta_eta(c)].

This is NOT "a model to beat BundleFuse". It is the FUNCTION CLASS that contains BundleFuse, the
two-tower, SCACS, topic models and DeepSets as special cases; the reference is the rate-distortion
ceiling, not any one algorithm. Every learned parameter has a closed-form target from the derivation:

    log p*(c|S) ~=  w_base * log(f_c + alpha)                      (a) KT base measure, alpha* = 1/2
                 +  g(S) * Y_collab(c,S)                           (b) collaborative tilt, EXACT suff. stats
                 +  w_con * sum_{s in S} <P e_c, P e_s>            (b') content tilt, learned PSD metric M=P^T P
                 +  beta * (1/2)(P e_c - pbar)^T Lam_S^{-1}(P e_c - pbar)   (c) uncertainty bonus  [NEW]

  (a) base measure  : log(f_c + alpha), learned alpha (Krichevsky-Trofimov target 1/2).
  (b) collaborative : order-statistic pooling of the EXACT per-pair features phi(c,s) -> MLP. Keeping
      phi exact (co-occurrence, PPMI, W_pair) is forced by Gate-1: a low-rank re-factorisation LOSES
      the sufficient statistic (Levy-Goldberg: shifted PPMI IS the collaborative-kernel MLE).
  (b') content tilt : <P e_c, P e_s> summed over seeds. P is a learned projection; M = P^T P is the
      PSD metric that the derivation identifies with the prior precision Lambda_0^{-1}. Sum-pooling is
      the de Finetti sufficient statistic, not a design choice.
  (b) gate g(S)     : the STRUCTURED precision gate  g = L v / (L v + c0),  L = |S|. Tied to seed
      count (posterior precision Lambda_S = Sigma0^{-1} + L Sigma), NOT a free MLP -- which is exactly
      why Gate-2's free gate learned nothing (-0.0008): there is no residual once g is set by precision.
  (c) uncertainty bonus : Mahalanobis term under the posterior covariance Lam_S^{-1} = (Sigma0^{-1}+L Sigma)^{-1}
      in the learned content-projection space. NO baseline has this. It is large when the theme is
      uncertain (few/spread seeds) and vanishes as L -> infinity. FALSIFIABLE PREDICTION: turning it on
      (SEFP_full vs SEFP_core) should lift Recall specifically in the cold one_seed regime and do ~nothing
      dense. If it doesn't, the Laplace derivation is wrong.

Arms (all under one identical pipeline; mixed seed counts so the gate sees single-seed):
  PoolFirst   : mean-pool phi first -> MLP (pooling-order control, lineage)
  SEFP_core   : (a)+(b)+(b')+gate, NO bonus
  SEFP_full   : SEFP_core + uncertainty bonus (c)
  SEFP_nogate : SEFP_full with g(S)=1 (tests the structured gate)

Evaluated on supplied masks AND the exact one_seed pairs (robust_protocol), comparable to the
generalization-study anchors. Env: QUICK, EPOCHS, STEPS, NEG, HARD_FRAC, SEED_CAP, SHORTLIST_M,
MIX_SEEDS, RANK_R (content projection dim), LR, EVAL_EVERY, PATIENCE, TEST_N, VAL_N, ARMS, DATASET.
Writes results/sefp/<dataset>.json.  Run: python3 reports/sefp.py
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
# generalization_study anchors (same provided split), R@20 / N@20:
ANCHORS = {"supplied": {"BundleFuse": (0.1556, 0.2579), "HardSwitchHybrid": (0.1000, 0.1662),
                        "MetadataRanker": (0.1206, 0.1969)},
           "one_seed": {"BundleFuse": (0.0616, 0.1548), "HardSwitchHybrid": (0.0625, 0.1669),
                        "MetadataRanker": (0.0530, 0.1446)}}

QUICK = os.environ.get("QUICK", "0") != "0"
EPOCHS = int(os.environ.get("EPOCHS", "3" if QUICK else "40"))
STEPS = int(os.environ.get("STEPS", "20" if QUICK else "0"))
NEG = int(os.environ.get("NEG", "32"))
HARD_FRAC = float(os.environ.get("HARD_FRAC", "0.5"))
BATCH = int(os.environ.get("BATCH", "128"))
SEED_CAP = int(os.environ.get("SEED_CAP", "20"))
SHORTLIST_M = int(os.environ.get("SHORTLIST_M", "1000"))
RANK_R = int(os.environ.get("RANK_R", "16"))                      # content projection dim (metric rank)
LR = float(os.environ.get("LR", "0.003"))
WD = float(os.environ.get("WD", "1e-5"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "2"))
PATIENCE = int(os.environ.get("PATIENCE", "4"))
HID = int(os.environ.get("HID", "64"))
MIX_SEEDS = os.environ.get("MIX_SEEDS", "1") != "0"
TEST_N = int(os.environ.get("TEST_N", "200" if QUICK else "0"))
VAL_N = int(os.environ.get("VAL_N", "200" if QUICK else "1000"))
ARMS = os.environ.get("ARMS", "PoolFirst,SEFP_core,SEFP_full,SEFP_nogate").split(",")
DEVICE = torch.device("cpu")
DPHI = 8


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- data + operators
def load_everything():
    d = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DATASET)
    n = json.load(open(os.path.join(d, "count.json")))["#I"]
    provided = ProvidedSplits(d, n)
    t0 = time.time()
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    C = graph.C.tocsr()
    W = graph.W_pair.tocsr()
    Wt = graph.W_pair.T.tocsr()
    PPMI = graph.PPMI.tocsr()
    freq = graph.freq.astype(np.float32)
    info = json.load(open(os.path.join(d, "item_info.json")))
    art = np.zeros(n, np.int64); alb = np.zeros(n, np.int64)
    am, lm = {}, {}
    for k, v in info.items():
        i = int(k)
        # spotify: artist_uri / album_uri ; other datasets fall back to a single taxonomy field.
        a_key = v.get("artist_uri", v.get("category", v.get("cat", "?")))
        l_key = v.get("album_uri", v.get("sub_category", v.get("category", "?")))
        art[i] = am.setdefault(a_key, len(am))
        alb[i] = lm.setdefault(l_key, len(lm))
    x = torch.load(os.path.join(d, "content_feature.pt"), map_location="cpu",
                   weights_only=False).detach().float().numpy()
    content = (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    logf = np.log1p(freq).astype(np.float32)
    logf_norm = ((logf - logf.mean()) / (logf.std() + 1e-8)).astype(np.float32)
    log(f"{DATASET}: n={n} train={len(provided.train)} graph+ops built in {time.time()-t0:.0f}s "
        f"| C.nnz={C.nnz:,} W.nnz={W.nnz:,} content_dim={content.shape[1]} rank_r={RANK_R}")
    return dict(n=n, provided=provided, C=C, W=W, Wt=Wt, PPMI=PPMI, freq=freq,
                art=art, alb=alb, content=content, logf=logf_norm)


# --------------------------------------------------------------------------- exact per-pair features
def pair_feats(D, cand, seed):
    """phi for every (candidate, seed) pair -> (Kc, Ks, DPHI). All EXACT sparse lookups / dot products."""
    Cc = D["C"][cand][:, seed].toarray().astype(np.float32)
    Wc = D["W"][cand][:, seed].toarray().astype(np.float32)
    Wsc = D["Wt"][cand][:, seed].toarray().astype(np.float32)
    Pc = D["PPMI"][cand][:, seed].toarray().astype(np.float32)
    con = (D["content"][cand] @ D["content"][seed].T).astype(np.float32)
    same_art = (D["art"][cand][:, None] == D["art"][seed][None, :]).astype(np.float32)
    same_alb = (D["alb"][cand][:, None] == D["alb"][seed][None, :]).astype(np.float32)
    degs = np.tile(D["logf"][seed][None, :], (len(cand), 1)).astype(np.float32)
    phi = np.stack([np.log1p(Cc), Pc, Wc, Wsc, con, same_art, same_alb, degs], axis=-1)
    return torch.from_numpy(phi)


# --------------------------------------------------------------------------- models
class InputNorm(nn.Module):
    """Per-feature affine standardisation (BatchNorm over the DPHI channels)."""
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, phi):
        shp = phi.shape
        return self.bn(phi.reshape(-1, shp[-1])).reshape(shp)


class SEFP(nn.Module):
    """Set Exponential-Family Predictor. The additive terms are the Laplace read-off (see module docstring).

    use_gate / use_content / use_bonus toggle (b)'s structured precision gate, the (b') learned content
    kernel, and the (c) uncertainty bonus respectively, so each derived piece is independently falsifiable.
    """
    N_STATS = 5

    def __init__(self, dphi=DPHI, hid=HID, content_dim=512, rank_r=RANK_R,
                 use_gate=True, use_content=True, use_bonus=True):
        super().__init__()
        self.use_gate, self.use_content, self.use_bonus = use_gate, use_content, use_bonus
        # (b) collaborative: order-stat pooling of exact phi -> MLP
        self.pool_norm = InputNorm(dphi)
        d_pooled = dphi * self.N_STATS
        self.norm = nn.LayerNorm(d_pooled)
        self.mlp = nn.Sequential(nn.Linear(d_pooled, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        # (a) base measure  log(f_c + alpha): learned alpha via softplus(alpha_raw), init ~ 0.5 (KT)
        self.alpha_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(0.5)))))  # softplus^{-1}(0.5)
        self.w_base = nn.Parameter(torch.tensor(0.0))     # start with no popularity bias; learn the weight
        # (b) structured precision gate g(S) = L v / (L v + c0): learned scalars v, c0.
        # Init near-OPEN (c0 small) so collaborative evidence is trusted at start; gate learns to close.
        self.gate_v = nn.Parameter(torch.tensor(0.0))     # softplus -> ~0.69
        self.gate_c0 = nn.Parameter(torch.tensor(-2.0))   # softplus -> ~0.13 => g(|S|=1)~0.84, g(20)~0.99
        # (b') learned content kernel P (M = P^T P is PSD by construction) + weight
        if use_content or use_bonus:
            self.P = nn.Linear(content_dim, rank_r, bias=False)
            nn.init.orthogonal_(self.P.weight)
        self.w_con = nn.Parameter(torch.tensor(0.0))
        # (c) uncertainty bonus: prior precision Sigma0^{-1} = (1/sigma0^2) I (learned scalar) + weight beta
        self.sig0_raw = nn.Parameter(torch.tensor(0.0))   # softplus -> sigma0^2 ~ 0.69
        self.beta_unc = nn.Parameter(torch.tensor(0.0))

    def forward(self, phi, smask, logf_c, freq_c, content_c, content_s):
        # phi:(B,Kc,Ks,d) smask:(B,Ks) logf_c,freq_c:(B,Kc) content_c:(B,Kc,dc) content_s:(B,Ks,dc)
        B, Kc, Ks = phi.shape[0], phi.shape[1], phi.shape[2]
        m = smask[:, None, :, None]
        L = smask.sum(-1).clamp_min(1.0)                                  # (B,) seed count
        z = self.pool_norm(phi)
        cnt = L[:, None, None]
        msum = (z * m).sum(2)
        mean = msum / cnt
        zmask = z.masked_fill(m == 0, float("-inf"))
        mx = zmask.max(2).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        posfrac = (((phi > 0).float() * m).sum(2)) / cnt
        k2 = min(2, Ks)
        t2 = zmask.topk(k2, dim=2).values
        t2 = torch.where(torch.isfinite(t2), t2, torch.zeros_like(t2)).mean(2)
        feat = torch.cat([mean, mx, msum, posfrac, t2], dim=-1)
        Y_collab = self.mlp(self.norm(feat)).squeeze(-1)                  # (B,Kc) collaborative tilt

        # (a) base measure
        alpha = Fn.softplus(self.alpha_raw)
        score = self.w_base * torch.log(freq_c + alpha)

        # (b) structured precision gate
        if self.use_gate:
            v, c0 = Fn.softplus(self.gate_v), Fn.softplus(self.gate_c0) + 1e-3
            g = (L * v) / (L * v + c0)                                   # (B,) increases with |S|
            score = score + g[:, None] * Y_collab
        else:
            score = score + Y_collab

        # (b') learned content kernel + (c) uncertainty bonus share the projection P
        if self.use_content or self.use_bonus:
            pc = self.P(content_c)                                       # (B,Kc,r)
            ps = self.P(content_s)                                       # (B,Ks,r)
            ms = smask[:, :, None]
            if self.use_content:
                kern = torch.einsum("bcr,bsr->bcs", pc, ps) * smask[:, None, :]
                score = score + self.w_con * kern.sum(2)                 # sum-pool = de Finetti suff. stat
            if self.use_bonus:
                pbar = (ps * ms).sum(1) / L[:, None]                     # (B,r) mean seed feature
                d_s = (ps - pbar[:, None, :]) * ms                       # (B,Ks,r)
                Sigma = torch.einsum("bsr,bsk->brk", d_s, d_s) / L[:, None, None]   # (B,r,r) seed feat cov
                r = pc.shape[-1]
                I = torch.eye(r, device=pc.device)
                sig0 = Fn.softplus(self.sig0_raw) + 1e-3                  # sigma0^2
                Lam = (1.0 / sig0) * I + L[:, None, None] * Sigma + 1e-4 * I   # (B,r,r) posterior precision
                Lam_inv = torch.linalg.inv(Lam)
                dcp = pc - pbar[:, None, :]                              # (B,Kc,r)
                bonus = 0.5 * torch.einsum("bcr,brk,bck->bc", dcp, Lam_inv, dcp)
                score = score + self.beta_unc * bonus
        return score


class PoolFirst(nn.Module):
    """Pooling-order control: mean-pool phi over seeds FIRST, then MLP (+ popularity)."""
    def __init__(self, dphi=DPHI, hid=HID, **kw):
        super().__init__()
        self.norm = InputNorm(dphi)
        self.mlp = nn.Sequential(nn.Linear(dphi, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.pop_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, phi, smask, logf_c, freq_c, content_c, content_s):
        z = self.norm(phi)
        m = smask[:, None, :, None]
        phibar = (z * m).sum(2) / m.sum(2).clamp_min(1)
        return self.mlp(phibar).squeeze(-1) + self.pop_weight * logf_c


# --------------------------------------------------------------------------- training
def make_batch(D, bundles, rng, n, neg, seed_cap, hard_frac):
    C = D["C"]
    seeds, cands = [], []
    for b in bundles:
        it = np.asarray(b, np.int64).copy()
        rng.shuffle(it)
        if MIX_SEEDS:
            split = int(rng.integers(1, len(it)))
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
    B = len(seeds)
    Ks = max(len(s) for s in seeds)
    Kc = len(cands[0])
    dc = D["content"].shape[1]
    phi = torch.zeros(B, Kc, Ks, DPHI)
    smask = torch.zeros(B, Ks)
    logf_c = torch.zeros(B, Kc)
    freq_c = torch.zeros(B, Kc)
    content_c = torch.zeros(B, Kc, dc)
    content_s = torch.zeros(B, Ks, dc)
    for j in range(B):
        s = np.asarray(seeds[j], np.int64)
        c = np.asarray(cands[j], np.int64)
        phi[j, :, :len(s), :] = pair_feats(D, c, s)
        smask[j, :len(s)] = 1.0
        logf_c[j] = torch.from_numpy(D["logf"][c])
        freq_c[j] = torch.from_numpy(D["freq"][c])
        content_c[j] = torch.from_numpy(D["content"][c])
        content_s[j, :len(s)] = torch.from_numpy(D["content"][s])
    return model(phi, smask, logf_c, freq_c, content_c, content_s)


def train_model(model, D, train_bundles, val_shortlists, rng_seed, tag):
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
            val = eval_cached(model, D, val_shortlists, Ks=(20,))[20][1]
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
            freq_c = torch.from_numpy(D["freq"][cand]).unsqueeze(0)
            content_c = torch.from_numpy(D["content"][cand]).unsqueeze(0)
            content_s = torch.from_numpy(D["content"][s]).unsqueeze(0)
            full[cand] = model(phi, smask, logf_c, freq_c, content_c, content_s).squeeze(0).numpy()
        cols.append(full)
        pairs.append(q["pair"])
    scores = np.stack(cols, axis=1)
    return ranking_metrics(scores, pairs, Ks=Ks)


def report_params(model, tag):
    """Print the learned parameters that act as signals, against their theoretical targets."""
    if isinstance(model, PoolFirst):
        return
    with torch.no_grad():
        alpha = float(Fn.softplus(model.alpha_raw))
    bits = [f"alpha={alpha:.3f}(KT~0.5)", f"w_base={float(model.w_base):.3f}"]
    if model.use_gate:
        v, c0 = float(Fn.softplus(model.gate_v)), float(Fn.softplus(model.gate_c0)) + 1e-3
        g1, g20 = v / (v + c0), 20 * v / (20 * v + c0)
        bits.append(f"gate g(|S|=1)={g1:.3f} g(20)={g20:.3f}")
    if model.use_content:
        bits.append(f"w_con={float(model.w_con):.4f}")
    if model.use_bonus:
        bits.append(f"sigma0^2={float(Fn.softplus(model.sig0_raw)):.3f} beta_unc={float(model.beta_unc):+.4f}")
    log(f"   [{tag}] learned signals: " + " | ".join(bits))


# --------------------------------------------------------------------------- main
def make_arm(name, dc):
    if name == "PoolFirst":
        return PoolFirst()
    if name == "SEFP_core":
        return SEFP(content_dim=dc, use_gate=True, use_content=True, use_bonus=False)
    if name == "SEFP_full":
        return SEFP(content_dim=dc, use_gate=True, use_content=True, use_bonus=True)
    if name == "SEFP_nogate":
        return SEFP(content_dim=dc, use_gate=False, use_content=True, use_bonus=True)
    raise ValueError(f"unknown arm {name}")


def main():
    t0 = time.time()
    D = load_everything()
    provided = D["provided"]
    dc = D["content"].shape[1]
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
    log(f"train={len(train_bundles)} val={len(val_pairs)} test(supplied)={len(test_pairs)} "
        f"test(one_seed)={len(oneseed_pairs)} | EPOCHS={EPOCHS} STEPS={STEPS or 'full'} NEG={NEG} "
        f"HARD_FRAC={HARD_FRAC} MIX_SEEDS={MIX_SEEDS} RANK_R={RANK_R} ARMS={ARMS} QUICK={QUICK}")
    ts = time.time()
    val_shortlists, val_ceiling = build_shortlists(D, val_pairs, SHORTLIST_M)
    test_supplied, ceil_sup = build_shortlists(D, test_pairs, SHORTLIST_M)
    test_oneseed, ceil_one = build_shortlists(D, oneseed_pairs, SHORTLIST_M)
    log(f"shortlists precomputed in {time.time()-ts:.0f}s | val_ceiling={val_ceiling:.4f} "
        f"supplied_ceiling={ceil_sup:.4f} one_seed_ceiling={ceil_one:.4f}")

    results = {"dataset": DATASET, "anchors": ANCHORS, "config": {
        "epochs": EPOCHS, "steps": STEPS, "neg": NEG, "batch": BATCH, "seed_cap": SEED_CAP,
        "shortlist_m": SHORTLIST_M, "rank_r": RANK_R, "lr": LR, "hard_frac": HARD_FRAC,
        "mix_seeds": MIX_SEEDS, "n_val": len(val_pairs), "n_test": len(test_pairs),
        "n_one_seed": len(oneseed_pairs)},
        "ceilings": {"supplied": ceil_sup, "one_seed": ceil_one}, "models": {}}

    for name in ARMS:
        log(f"\n=== training {name} ===")
        model = make_arm(name, dc)
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
        report_params(model, name)
        log(f"   supplied R@20={ms[20][0]:.4f} N@20={ms[20][1]:.4f} | "
            f"one_seed R@20={mo[20][0]:.4f} N@20={mo[20][1]:.4f}")

    out_dir = os.path.join(PROJECT_ROOT, "results", "sefp")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"{DATASET}.json")
    json.dump(results, open(outp, "w"), indent=2)

    def g(name, case):
        return results["models"].get(name, {}).get(case, {}).get("20", {})
    bf_s, bf_o = ANCHORS["supplied"]["BundleFuse"], ANCHORS["one_seed"]["BundleFuse"]
    log(f"\n=== SEFP SUMMARY ({time.time()-t0:.0f}s) ===")
    log(f"{'model':14s} {'supp R@20':>10} {'supp N@20':>10} {'1seed R@20':>11} {'1seed N@20':>11}")
    log(f"{'BundleFuse':14s} {bf_s[0]:10.4f} {bf_s[1]:10.4f} {bf_o[0]:11.4f} {bf_o[1]:11.4f}")
    for name in ARMS:
        s, o = g(name, "supplied"), g(name, "one_seed")
        log(f"{name:14s} {s.get('R',0):10.4f} {s.get('N',0):10.4f} {o.get('R',0):11.4f} {o.get('N',0):11.4f}")
    # falsifiable test: the uncertainty bonus should help the cold one_seed regime specifically.
    if "SEFP_full" in results["models"] and "SEFP_core" in results["models"]:
        fc, cc = g("SEFP_full", "one_seed"), g("SEFP_core", "one_seed")
        fs, cs = g("SEFP_full", "supplied"), g("SEFP_core", "supplied")
        log(f"\nuncertainty-bonus effect (SEFP_full - SEFP_core):")
        log(f"  one_seed R@20 {fc.get('R',0)-cc.get('R',0):+.4f}  (theory: POSITIVE)")
        log(f"  supplied R@20 {fs.get('R',0)-cs.get('R',0):+.4f}  (theory: ~0)")
    log(f"wrote {outp}")


if __name__ == "__main__":
    main()
