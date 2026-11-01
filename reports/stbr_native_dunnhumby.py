"""STBR's NATIVE task (personalized bundle ranking) on dunnhumby -- BundleFuse vs faithful STBR.

Unlike the completion comparison, this matches STBR's paper task: given a user u and a candidate BUNDLE b
(a complete item set), predict u's preference s(u,b); rank candidate bundles for u and measure HR@20 /
NDCG@20 over BUNDLES. dunnhumby: baskets = bundles, households = users (each basket has one owner).

Protocol (identical for both models):
  - Bundles = unique baskets (3<=|b|<=20). user_bundles[u] = that household's baskets.
  - Leave-one-out split per user with >=2 baskets: 1 test, (1 val if >=3), rest train. History/graph from TRAIN.
  - Eval: for each test (u, b+), candidates = b+ + 999 negatives (bundles u did not interact with);
    score all 1000, report HR@20 (=Recall@20, one positive) and NDCG@20. Random HR@20 = 0.02.

STBR-native (faithful): X=[e_u; e_i1; ..; e_ik] -> SAB x2 -> PMA -> h_{u,b} -> FC -> scalar; trained BPR over
(user, pos bundle, neg bundle). Learnable item + user embeddings. This IS the paper's model.

BundleFuse-native (zero parameters): s(u,b) fuses, over candidate bundles, z-scored components --
  aff_freq = mean_{i in b} user purchase count of i      (personalized taste / repurchase)
  aff_cooc = mean_{i in b} (C @ 1_{H_u})[i]              (relatedness of b to the user's history)
  coherence = mean_{i!=j in b} C[i,j]                     (intra-bundle co-occurrence, user-agnostic)
  pop       = mean_{i in b} log(1+freq[i])
fused with weights fit on validation NDCG@20 (greedy). No trainable parameters.

Honest scope: this RECONSTRUCTS STBR's task on dunnhumby (baskets-as-bundles, sampled-candidate ranking);
it does not reproduce the paper's exact split/candidate sampling, so it is NOT a claim about the paper's
reported numbers -- it is a fair head-to-head of the two methods on STBR's task under one shared protocol.

Run: python3 reports/stbr_native_dunnhumby.py    Env: BASKET_CAP, NEG_CAND, VAL_N, TEST_N, EPOCHS, DIM.
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
from stbr import SAB, PMA

ROOT = os.environ.get("DUNN_DIR") or os.path.join(DATA_ROOT, "dunnhumby")
BASKET_CAP = int(os.environ.get("BASKET_CAP", "30000"))
NEG_CAND = int(os.environ.get("NEG_CAND", "999"))
VAL_N = int(os.environ.get("VAL_N", "800"))
TEST_N = int(os.environ.get("TEST_N", "2000"))
CAP = int(os.environ.get("ITEM_CAP", "20"))
DIM, HEADS = int(os.environ.get("DIM", "64")), 4
EPOCHS, EVAL_EVERY, PAT = int(os.environ.get("EPOCHS", "40")), 4, 4
BATCH, LR, WD = 128, 1e-3, 1e-4
EPS = 1e-8
rng = np.random.default_rng(0)


def log(*a):
    print(*a, flush=True)


def load():
    raw = defaultdict(set); buser = {}
    with open(os.path.join(ROOT, "transaction_data.csv")) as f:
        r = csv.reader(f); h = next(r); bi = h.index("BASKET_ID"); pi = h.index("PRODUCT_ID"); hi = h.index("household_key")
        for row in r:
            raw[row[bi]].add(int(row[pi])); buser[row[bi]] = row[hi]
    bask = [(sorted(v), buser[b]) for b, v in raw.items() if 3 <= len(v) <= 20]
    random.Random(0).shuffle(bask); bask = bask[:BASKET_CAP]
    fa = defaultdict(int)
    for b, _ in bask:
        for it in b:
            fa[it] += 1
    keep = {it for it, c in fa.items() if c >= 2}
    tmp = [([it for it in b if it in keep], u) for b, u in bask]
    tmp = [(b, u) for b, u in tmp if len(b) >= 3]
    items = sorted({i for b, _ in tmp for i in b}); imap = {it: k for k, it in enumerate(items)}; n = len(items)
    umap = {}; seen = set(); bundle_items = []; bundle_user = []
    for b, u in tmp:
        mapped = tuple(sorted(imap[i] for i in set(b))); uu = umap.setdefault(u, len(umap))
        if (uu, mapped) not in seen:
            seen.add((uu, mapped)); bundle_items.append(list(mapped)); bundle_user.append(uu)
    return n, len(umap), bundle_items, bundle_user


def main():
    t0 = time.time()
    n, nusers, bundle_items, bundle_user = load()
    nb = len(bundle_items)
    user_bundles = defaultdict(list)
    for bidx, u in enumerate(bundle_user):
        user_bundles[u].append(bidx)

    # leave-one-out split per user
    train_b, val_q, test_q = [], [], []   # train bundle idxs; val/test = (user, pos_bundle_idx)
    for u, bl in user_bundles.items():
        if len(bl) < 2:
            train_b += bl; continue
        bl = bl[:]; random.Random(u).shuffle(bl)
        test_q.append((u, bl[0]))
        if len(bl) >= 3:
            val_q.append((u, bl[1])); train_b += bl[2:]
        else:
            train_b += bl[1:]
    random.Random(1).shuffle(val_q); random.Random(2).shuffle(test_q)
    val_q, test_q = val_q[:VAL_N], test_q[:TEST_N]
    train_set = set(train_b)
    log(f"dunnhumby NATIVE: n={n} users={nusers} bundles={nb} | train={len(train_b)} val={len(val_q)} test={len(test_q)} | load {time.time()-t0:.0f}s")

    trb_items = [bundle_items[i] for i in train_b]
    graph = SimplicialGraphOperator(trb_items, n_items=n, svd_dim=2)
    C = graph.C.tocsr(); freq = graph.freq.astype(np.float32); logf = np.log1p(freq)

    # user history (items) from TRAIN bundles
    ucnt = defaultdict(Counter)
    for bidx in train_b:
        ucnt[bundle_user[bidx]].update(bundle_items[bidx])
    # per-bundle precomputations for the BundleFuse scorer
    bf_pop = np.array([logf[bundle_items[b]].mean() for b in range(nb)], np.float32)
    bf_coh = np.zeros(nb, np.float32)
    for b in range(nb):
        it = bundle_items[b]
        if len(it) > 1:
            sub = C[np.ix_(it, it)].toarray()
            bf_coh[b] = (sub.sum() - np.trace(sub)) / (len(it) * (len(it) - 1))

    owned = {u: set(bl) for u, bl in user_bundles.items()}
    all_b = np.arange(nb)

    def candidates(u, pos):
        neg = []
        bad = owned.get(u, set())
        while len(neg) < NEG_CAND:
            c = int(rng.integers(nb))
            if c != pos and c not in bad:
                neg.append(c)
        return [pos] + neg

    # ---------- BundleFuse-native (zero-param) ----------
    def bf_components(u, cand):
        hist = list(ucnt.get(u, {}).keys())
        uf_vec = np.zeros(n, np.float32)
        for it, c in ucnt.get(u, {}).items():
            uf_vec[it] = c
        uc_vec = np.asarray((C @ _ind(hist)).todense()).ravel().astype(np.float32) if hist else np.zeros(n, np.float32)
        af = np.array([uf_vec[bundle_items[b]].mean() for b in cand], np.float32)
        ac = np.array([uc_vec[bundle_items[b]].mean() for b in cand], np.float32)
        return {"aff_freq": _z(af), "aff_cooc": _z(ac), "coherence": _z(bf_coh[cand]), "pop": _z(bf_pop[cand])}

    def _z(x):
        return (x - x.mean()) / (x.std() + EPS)

    def _ind(items):
        v = sp.lil_matrix((n, 1));
        if items:
            v[items, 0] = 1.0
        return v.tocsr()

    BF_SIG = ["aff_freq", "aff_cooc", "coherence", "pop"]

    def bf_eval(queries, w):
        hr = ndcg = 0
        for u, pos in queries:
            cand = candidates(u, pos)
            comp = bf_components(u, cand)
            s = sum(w[k] * comp[k] for k in w if w[k])
            rank = int((s > s[0]).sum())  # rank of positive (index 0), 0-based
            if rank < 20:
                hr += 1; ndcg += 1.0 / np.log2(rank + 2)
        return hr / len(queries), ndcg / len(queries)

    log("  fitting BundleFuse-native weights on val...")
    w = {k: 0.0 for k in BF_SIG}; best = -1
    for _ in range(3):
        imp = False
        for k in BF_SIG:
            for g in [0.25, 0.5, 1.0, 2.0, 4.0]:
                w2 = dict(w); w2[k] = g
                v = bf_eval(val_q, w2)[1]
                if v > best + 1e-5:
                    best, w, imp = v, w2, True
        if not imp:
            break
    bf_hr, bf_nd = bf_eval(test_q, w)
    log(f"    BundleFuse-native weights: " + " ".join(f"{k}={w[k]}" for k in BF_SIG if w[k]))
    log(f"    BundleFuse-native  HR@20={bf_hr:.4f}  N@20={bf_nd:.4f}")

    # ---------- STBR-native (faithful bundle scorer) ----------
    class STBRScore(nn.Module):
        def __init__(self):
            super().__init__()
            self.item = nn.Embedding(n + 1, DIM, padding_idx=n); nn.init.normal_(self.item.weight, std=0.1)
            with torch.no_grad():
                self.item.weight[n].zero_()
            self.user = nn.Embedding(nusers, DIM); nn.init.normal_(self.user.weight, std=0.1)
            self.sab1, self.sab2, self.pma = SAB(DIM, HEADS), SAB(DIM, HEADS), PMA(DIM, HEADS, seeds=1)
            self.fc = nn.Linear(DIM, 1)

        def forward(self, uidx, bidx, bmask):
            x = torch.cat([self.user(uidx).unsqueeze(1), self.item(bidx)], 1)
            m = torch.cat([torch.ones(x.shape[0], 1), bmask], 1)
            x = self.sab1(x, m); x = self.sab2(x, m)
            return self.fc(self.pma(x, m).squeeze(1)).squeeze(-1)

    def pack(bidxs):
        its = [bundle_items[b][:CAP] for b in bidxs]
        K = max(len(s) for s in its); B = len(its)
        bi = torch.full((B, K), n, dtype=torch.long); bm = torch.zeros(B, K)
        for j, s in enumerate(its):
            bi[j, :len(s)] = torch.tensor(s); bm[j, :len(s)] = 1.0
        return bi, bm

    def stbr_eval(model, queries):
        model.eval(); hr = ndcg = 0
        with torch.no_grad():
            for u, pos in queries:
                cand = candidates(u, pos)
                bi, bm = pack(cand)
                s = model(torch.full((len(cand),), u, dtype=torch.long), bi, bm).numpy()
                rank = int((s > s[0]).sum())
                if rank < 20:
                    hr += 1; ndcg += 1.0 / np.log2(rank + 2)
        return hr / len(queries), ndcg / len(queries)

    log("  training STBR-native (BPR over user,pos-bundle,neg-bundle)...")
    torch.manual_seed(0)
    model = STBRScore(); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, len(train_b) // BATCH); best_hr, best_state, stale = -1, None, 0; t1 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); order = rng.permutation(len(train_b))
        for st in range(steps):
            bb = [train_b[i] for i in order[st * BATCH:(st + 1) * BATCH]]
            if len(bb) < 2:
                continue
            us = torch.tensor([bundle_user[b] for b in bb], dtype=torch.long)
            negs = []
            for b in bb:
                u = bundle_user[b]; c = int(rng.integers(nb))
                while c in owned.get(u, set()):
                    c = int(rng.integers(nb))
                negs.append(c)
            bip, bmp = pack(bb); bin_, bmn = pack(negs)
            sp_ = model(us, bip, bmp); sn = model(us, bin_, bmn)
            loss = -Fn.logsigmoid(sp_ - sn).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if ep % EVAL_EVERY == 0:
            hr = stbr_eval(model, val_q)[0]
            if hr > best_hr + 1e-5:
                best_hr, best_state, stale = hr, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
            log(f"    [stbr-native ep={ep}] val HR@20={hr:.4f} best={best_hr:.4f} ({time.time()-t1:.0f}s)")
            if stale >= PAT:
                break
    if best_state:
        model.load_state_dict(best_state)
    st_hr, st_nd = stbr_eval(model, test_q)

    log(f"\n  NATIVE bundle-ranking TEST ({len(test_q)} queries, 1 pos + {NEG_CAND} neg; random HR@20=0.02):")
    log(f"    STBR-native (learned)   HR@20={st_hr:.4f}  N@20={st_nd:.4f}")
    log(f"    BundleFuse-native (0-p) HR@20={bf_hr:.4f}  N@20={bf_nd:.4f}   "
        f"(x{bf_hr/max(st_hr,1e-9):.1f} on HR@20)")
    log(f"  (done in {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
