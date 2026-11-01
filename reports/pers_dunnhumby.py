"""Personalized BundleFuse -- dunnhumby prototype.

Tests whether adding USER signals (built from each household's OWN training baskets) improves bundle
completion over the user-agnostic scorer -- staying zero-parameter: personalization is just extra
signals, computed with the same operators over the user's history instead of the seeds, fused with a
few more validation-fitted weights (no trainable params).

Signals (all z-scored over the n candidates, per query):
  Base (user-agnostic, from the revealed seeds S):
    seed_cooc = C @ 1_S           seed_ppmi = PPMI @ 1_S           seed_csvd = Ucsvd (Ucsvd^T 1_S)
  User (personalized, from the household's TRAINING baskets H_u only -- leakage-safe):
    user_freq = training purchase count of c by u   (repurchase)
    user_cooc = C @ 1_{H_u}                          (items related to the user's history)
    user_csvd = Ucsvd (mean_{h in H_u} Ucsvd[h])

C / PPMI / Ucsvd are built from TRAIN baskets only. The user's history excludes the current basket
(it lives in the test/val split), so repurchase is a genuine user prior, not leakage. We fit base-only
weights and base+user weights on validation NDCG@20 (greedy), then report test Recall@20/NDCG@20 for
both -- the A/B that shows whether personalization helps and by how much. BundleFuse dunnhumby
(user-agnostic, matrix_zeroparam): R@20 0.150 / N@20 0.108.

Run: python3 reports/pers_dunnhumby.py     Env: BASKET_CAP, VAL_N, TEST_N, MASK_FRAC, SVD_DIM, SPLIT_SEED.
"""
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT
from bundlefuse.metrics import ranking_metrics
from bundlefuse import protocol
from stbr import SAB, PMA   # Set Transformer blocks (faithful STBR encoder)

ROOT = os.environ.get("DUNN_DIR") or os.path.join(DATA_ROOT, "dunnhumby")
BASKET_CAP = int(os.environ.get("BASKET_CAP", "30000"))
VAL_N = int(os.environ.get("VAL_N", "800"))
TEST_N = int(os.environ.get("TEST_N", "2000"))
MASK_FRAC = float(os.environ.get("MASK_FRAC", "0.5"))
SVD_DIM = int(os.environ.get("SVD_DIM", "128"))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "0"))
EPS = 1e-8
BASE = ["seed_cooc", "seed_ppmi", "seed_csvd"]
USER = ["user_freq", "user_cooc", "user_csvd"]


# --- personalized STBR (faithful: user token as a set element, learnable item+user embeddings) ---
S_DIM, S_HEADS, S_NEG, S_HARD = 64, 4, 64, 0.5
S_BATCH, S_CAP, S_LR, S_WD = 128, 20, 1e-3, 1e-4
S_EPOCHS, S_EVAL, S_PAT = 40, 4, 4


class STBRPers(nn.Module):
    def __init__(self, n, nusers, d, heads, use_user):
        super().__init__()
        self.n, self.use_user = n, use_user
        self.item = nn.Embedding(n + 1, d, padding_idx=n)
        nn.init.normal_(self.item.weight, std=0.1)
        with torch.no_grad():
            self.item.weight[n].zero_()
        if use_user:
            self.user = nn.Embedding(nusers, d); nn.init.normal_(self.user.weight, std=0.1)
        self.sab1 = SAB(d, heads); self.sab2 = SAB(d, heads); self.pma = PMA(d, heads, seeds=1)
        self.out = nn.Linear(d, d)

    def encode(self, uidx, sidx, smask):
        x = self.item(sidx); m = smask
        if self.use_user:
            x = torch.cat([self.user(uidx).unsqueeze(1), x], dim=1)
            m = torch.cat([torch.ones(x.shape[0], 1), smask], dim=1)
        x = self.sab1(x, m); x = self.sab2(x, m)
        return self.out(self.pma(x, m).squeeze(1))

    def forward(self, uidx, sidx, smask, cidx):
        q = self.encode(uidx, sidx, smask)
        return torch.einsum("bd,bcd->bc", q, self.item(cidx))

    @torch.no_grad()
    def score_all(self, uidx, sidx, smask):
        return self.encode(uidx, sidx, smask) @ self.item.weight[:self.n].T


def make_batch_pers(batch_su, C, rng, n):
    us, seeds, pos, negs = [], [], [], []
    for its, u in batch_su:
        it = np.asarray(its, np.int64).copy(); rng.shuffle(it)
        sp_ = int(rng.integers(1, len(it)))
        s = it[:sp_][:S_CAP].tolist(); p = int(rng.choice(it[sp_:]))
        itset = set(it.tolist()); nh = int(round(S_NEG * S_HARD)); hard = []
        if nh > 0:
            pool = set()
            for ss in s:
                pool.update(C.indices[C.indptr[ss]:C.indptr[ss + 1]].tolist())
            pool.difference_update(itset)
            if pool:
                pool = np.fromiter(pool, np.int64)
                hard = rng.choice(pool, size=min(nh, len(pool)), replace=len(pool) < nh).tolist()
        need = S_NEG - len(hard); forb = itset.union(hard)
        ng = rng.integers(n, size=need); bad = np.array([x in forb for x in ng])
        while bad.any():
            ng[bad] = rng.integers(n, size=int(bad.sum())); bad = np.array([x in forb for x in ng])
        us.append(u); seeds.append(s); pos.append(p); negs.append(hard + ng.tolist())
    Ks = max(len(s) for s in seeds); B = len(seeds)
    sidx = torch.full((B, Ks), n, dtype=torch.long); smask = torch.zeros(B, Ks)
    for j, s in enumerate(seeds):
        sidx[j, :len(s)] = torch.tensor(s); smask[j, :len(s)] = 1.0
    cidx = torch.tensor([[pos[j]] + negs[j] for j in range(B)], dtype=torch.long)
    return torch.tensor(us, dtype=torch.long), sidx, smask, cidx


def eval_stbr(model, queries, n):
    agg = np.zeros(2); tot = 0
    for off in range(0, len(queries), 200):
        ch = queries[off:off + 200]
        us = torch.tensor([u for u, _, _ in ch], dtype=torch.long)
        sl = [sorted(s)[:S_CAP] for _, s, _ in ch]
        Ks = max(len(s) for s in sl); sidx = torch.full((len(ch), Ks), n, dtype=torch.long); smask = torch.zeros(len(ch), Ks)
        for j, s in enumerate(sl):
            sidx[j, :len(s)] = torch.tensor(s); smask[j, :len(s)] = 1.0
        sc = model.score_all(us, sidx, smask).T.cpu().numpy()
        pairs = [(s, g) for _, s, g in ch]
        agg += np.asarray(ranking_metrics(sc, pairs, (20,))[20]) * len(ch); tot += len(ch)
    return agg / tot


def train_stbr(n, nusers, train_su, C, valq, use_user, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = STBRPers(n, nusers, S_DIM, S_HEADS, use_user)
    opt = torch.optim.AdamW(model.parameters(), lr=S_LR, weight_decay=S_WD)
    steps = max(1, len(train_su) // S_BATCH); best, best_state, stale = -1, None, 0
    for ep in range(1, S_EPOCHS + 1):
        model.train(); order = rng.permutation(len(train_su))
        for st in range(steps):
            bi = order[(st * S_BATCH) % len(order): (st * S_BATCH) % len(order) + S_BATCH]
            if len(bi) < 2:
                continue
            us, sidx, smask, cidx = make_batch_pers([train_su[i] for i in bi], C, rng, n)
            logits = model(us, sidx, smask, cidx)
            loss = Fn.cross_entropy(logits, torch.zeros(len(bi), dtype=torch.long))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if ep % S_EVAL == 0:
            model.eval(); v = eval_stbr(model, valq, n)[0]
            if v > best + 1e-5:
                best, best_state, stale = v, {k: vv.detach().clone() for k, vv in model.state_dict().items()}, 0
            else:
                stale += 1
            if stale >= S_PAT:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval(); return model


def log(*a):
    print(*a, flush=True)


def load():
    raw = defaultdict(set); buser = {}
    with open(os.path.join(ROOT, "transaction_data.csv")) as f:
        r = csv.reader(f); h = next(r); bi = h.index("BASKET_ID"); pi = h.index("PRODUCT_ID"); hi = h.index("household_key")
        for row in r:
            raw[row[bi]].add(int(row[pi])); buser[row[bi]] = row[hi]
    bask = [(sorted(v), buser[bid]) for bid, v in raw.items() if 3 <= len(v) <= 20]
    random.Random(0).shuffle(bask); bask = bask[:BASKET_CAP]
    fa = defaultdict(int)
    for b, _ in bask:
        for it in b:
            fa[it] += 1
    keep = {it for it, c in fa.items() if c >= 2}
    tmp = [([it for it in b if it in keep], u) for b, u in bask]
    tmp = [(b, u) for b, u in tmp if len(b) >= 3]
    items = sorted({i for b, _ in tmp for i in b}); imap = {it: k for k, it in enumerate(items)}; n = len(items)
    umap = {}; seen = set(); bundles = []
    for b, u in tmp:
        mapped = tuple(sorted(imap[i] for i in set(b))); uu = umap.setdefault(u, len(umap))
        if (uu, mapped) not in seen:
            seen.add((uu, mapped)); bundles.append((list(mapped), uu))
    return n, len(umap), bundles


def main():
    t0 = time.time()
    n, nusers, bundles = load()
    idx = list(range(len(bundles))); random.Random(SPLIT_SEED).shuffle(idx)
    ntr = int(0.7 * len(idx)); nva = int(0.1 * len(idx))
    tri, vai, tei = idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]
    trb = [bundles[i][0] for i in tri]
    log(f"dunnhumby: n={n} users={nusers} | train={len(tri)} val={len(vai)} test={len(tei)} baskets | load {time.time()-t0:.0f}s")

    graph = SimplicialGraphOperator(trb, n_items=n, svd_dim=SVD_DIM)
    C, PPMI, U = graph.C.tocsr(), graph.PPMI.tocsr(), graph.Ucsvd            # (n,n),(n,n),(n,k)

    # user history from TRAIN baskets only (leakage-safe)
    user_cnt = defaultdict(Counter)
    for i in tri:
        its, u = bundles[i]
        user_cnt[u].update(its)
    rows, cols, data = [], [], []
    for u, cnt in user_cnt.items():
        for it, c in cnt.items():
            rows.append(it); cols.append(u); data.append(float(c))
    UI = sp.csr_matrix((data, (rows, cols)), shape=(n, nusers))              # item x user counts

    def mk_queries(split_idx, seed, cap):
        bl = [bundles[i][0] for i in split_idx]; ul = [bundles[i][1] for i in split_idx]
        pairs = protocol.mask(bl, seed, MASK_FRAC)
        return list(zip(ul, [s for s, _ in pairs], [g for _, g in pairs]))[:cap]

    valq = mk_queries(vai, SPLIT_SEED + 1, VAL_N)
    testq = mk_queries(tei, SPLIT_SEED + 2, TEST_N)
    have_hist = np.mean([len(user_cnt.get(u, ())) > 0 for u, _, _ in testq])
    log(f"  built graph + user history ({time.time()-t0:.0f}s); {100*have_hist:.0f}% of test queries have user history")

    def zc(M):  # column-wise z-score of a dense (n,B)
        return (M - M.mean(0, keepdims=True)) / (M.std(0, keepdims=True) + EPS)

    def signals(queries):
        """Return dict name -> z-scored (n, B), and the (seeds, gt) pairs for ranking."""
        users = [u for u, _, _ in queries]; seeds = [s for _, s, _ in queries]
        B = len(queries)
        Ind = sp.lil_matrix((n, B))
        Hind = sp.lil_matrix((n, B))
        for j, (u, s) in enumerate(zip(users, seeds)):
            Ind[s, j] = 1.0
            hist = list(user_cnt.get(u, {}).keys())
            if hist:
                Hind[hist, j] = 1.0
        Ind = Ind.tocsr(); Hind = Hind.tocsr()
        out = {}
        out["seed_cooc"] = zc(np.asarray((C @ Ind).todense(), dtype=np.float32))
        out["seed_ppmi"] = zc(np.asarray((PPMI @ Ind).todense(), dtype=np.float32))
        out["seed_csvd"] = zc((U @ (U.T @ Ind.toarray())).astype(np.float32))
        out["user_freq"] = zc(np.asarray(UI[:, users].todense(), dtype=np.float32))
        out["user_cooc"] = zc(np.asarray((C @ Hind).todense(), dtype=np.float32))
        hsum = U.T @ Hind.toarray(); hn = np.asarray(Hind.sum(0)).ravel() + EPS
        out["user_csvd"] = zc((U @ (hsum / hn)).astype(np.float32))
        pairs = [(s, g) for _, s, g in queries]
        return out, pairs

    def ndcg(sig, w, pairs):
        F = sum(w[k] * sig[k] for k in w if w[k])
        if np.isscalar(F):
            return 0.0
        return ranking_metrics(F, pairs, (20,))[20][1]

    def metrics(sig, w, pairs):
        F = sum(w[k] * sig[k] for k in w if w[k])
        return ranking_metrics(F, pairs, (20,))[20]

    def fit(sig, pairs, names, rounds=3):
        w = {k: 0.0 for k in names}; grid = [0.25, 0.5, 1.0, 2.0, 4.0]
        best = -1.0
        for _ in range(rounds):
            improved = False
            for k in names:
                for g in grid:
                    w2 = dict(w); w2[k] = g
                    v = ndcg(sig, w2, pairs)
                    if v > best + 1e-5:
                        best, w, improved = v, w2, True
            if not improved:
                break
        return w, best

    log("  computing validation signals + fitting...")
    vsig, vpairs = signals(valq)
    w_base, nb = fit(vsig, vpairs, BASE)
    w_full, nf = fit(vsig, vpairs, BASE + USER)
    log(f"    base weights  (val N@20={nb:.4f}): " + " ".join(f"{k}={w_base[k]}" for k in BASE if w_base[k]))
    log(f"    +user weights (val N@20={nf:.4f}): " + " ".join(f"{k}={w_full[k]}" for k in BASE + USER if w_full[k]))

    # test eval, chunked (recompute signals per chunk to bound memory)
    def eval_test(w):
        agg = np.zeros(2); tot = 0
        for off in range(0, len(testq), 200):
            ch = testq[off:off + 200]
            sig, pairs = signals(ch)
            agg += np.asarray(metrics(sig, w, pairs)) * len(ch); tot += len(ch)
        return agg / tot

    rb = eval_test(w_base); rf = eval_test(w_full)
    log(f"\n  TEST ({len(testq)} queries):")
    log(f"    BundleFuse user-agnostic (base)   R@20={rb[0]:.4f}  N@20={rb[1]:.4f}")
    log(f"    BundleFuse personalized (+user)   R@20={rf[0]:.4f}  N@20={rf[1]:.4f}   "
        f"(dR@20={rf[0]-rb[0]:+.4f} {100*(rf[0]-rb[0])/max(rb[0],1e-9):+.0f}%, "
        f"dN@20={rf[1]-rb[1]:+.4f})")

    if os.environ.get("STBR", "1") != "0":
        train_su = [bundles[i] for i in tri if len(bundles[i][0]) >= 2]
        log(f"\n  training STBR on the SAME split ({len(train_su):,} train baskets, {nusers} users)...")
        t1 = time.time()
        m_ag = train_stbr(n, nusers, train_su, C, valq, use_user=False)
        s_ag = eval_stbr(m_ag, testq, n)
        m_pe = train_stbr(n, nusers, train_su, C, valq, use_user=True)
        s_pe = eval_stbr(m_pe, testq, n)
        log(f"    STBR user-agnostic (no user tok)  R@20={s_ag[0]:.4f}  N@20={s_ag[1]:.4f}")
        log(f"    STBR personalized (+user tok)     R@20={s_pe[0]:.4f}  N@20={s_pe[1]:.4f}   "
            f"(dR@20={s_pe[0]-s_ag[0]:+.4f})   [{time.time()-t1:.0f}s]")

    log(f"  (done in {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
