"""Cold-start by INDUCING a collaborative embedding from content, three ways, tested on cold slices.

The learned content map fails on cold because a global content->complementarity map extrapolates
badly to atypical (cold) content. The fix (user's idea): don't score from content directly -- use
content to pull the cold item INTO the collaborative embedding space that already ranks well (warm
~0.45 on POG), by borrowing from content-similar WARM items. Substitutes share complements, so a cold
item that looks like warm item(s) inherits their 'goes-with' behaviour.

Induce v_c for a cold item c from its content e_c, three ways:
  linreg : ridge map  W : content -> collab,   v_c = e_c W      (closed form, global linear)
  knn    : mean collab of the k content-nearest WARM items       (non-parametric lookup)
  attn   : v_c = sum_a softmax(<Wq e_c, Wk e_a>/tau) V_a^collab   (learned attention over warm anchors)

Clean deployment eval: WARM candidates keep their REAL SVD-collab embedding; COLD candidates get the
INDUCED one. So cold Hit@20 measures purely whether induction places novel items correctly; warm is
untouched (sanity: stays at the collab baseline). Seeds use real collab (induced if a seed is cold).

Honest expectation: helps where content aligns with co-occurrence (POG), little on Spotify (content
near-orthogonal). Bounded by whether cold items have informative warm content-neighbours at all.
Env: DATASETS, DIM, K_ANCHOR, K_NN, RIDGE, ATTN_EPOCHS, TAU, TEST_N. Run: python3 reports/cold_attention.py
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
from bundlefuse.metrics import top_items, relevant_ranks
from bundlefuse.robust_protocol import sparse_pairs

DATASETS = os.environ.get("DATASETS", "pog_dense,spotify").split(",")
DIM = int(os.environ.get("DIM", "128"))
K_ANCHOR = int(os.environ.get("K_ANCHOR", "4000"))
K_NN = int(os.environ.get("K_NN", "20"))
RIDGE = float(os.environ.get("RIDGE", "10.0"))
ATTN_EPOCHS = int(os.environ.get("ATTN_EPOCHS", "30"))
ATTN_BATCH = int(os.environ.get("ATTN_BATCH", "512"))
TAU = float(os.environ.get("TAU", "0.1"))
TEST_N = int(os.environ.get("TEST_N", "0"))


def log(*a):
    print(*a, flush=True)


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


class AttnBorrow(nn.Module):
    def __init__(self, dc, d, tau):
        super().__init__()
        self.Wq = nn.Linear(dc, d, bias=False)
        self.Wk = nn.Linear(dc, d, bias=False)
        nn.init.orthogonal_(self.Wq.weight); nn.init.orthogonal_(self.Wk.weight)
        self.tau = tau

    def induce(self, E, Ek, Vk):
        """E:(B,dc) query content; Ek:(K,dc) anchor content; Vk:(K,d) anchor collab. -> (B,d)."""
        att = (self.Wq(E) @ self.Wk(Ek).T) / self.tau      # (B,K)
        att = Fn.softmax(att, dim=-1)
        return att @ Vk


def ridge_map(Ew, Vw, lam):
    """Closed-form ridge: W = (Ew^T Ew + lam I)^-1 Ew^T Vw. Ew:(m,dc) Vw:(m,d)."""
    dc = Ew.shape[1]
    G = Ew.T @ Ew + lam * np.eye(dc, dtype=np.float64)
    return np.linalg.solve(G, Ew.T @ Vw).astype(np.float32)   # (dc,d)


def run(dataset):
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    provided = ProvidedSplits(dd, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    freq = graph.freq.astype(np.float64)
    W = graph.W_pair.tocsr()
    x = torch.load(os.path.join(dd, "content_feature.pt"), map_location="cpu", weights_only=False).detach().float().numpy()
    content = (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    dc = content.shape[1]
    svd = TruncatedSVD(n_components=min(DIM, n - 2), random_state=0)
    A = svd.fit_transform(W).astype(np.float32)          # candidate collab embedding U S (n,d)
    B = svd.components_.T.astype(np.float32)             # seed-side collab (n,d)
    warm = np.where(freq > 0)[0].astype(np.int64)
    cold_mask = (freq == 0)
    log(f"\n### {dataset}: n={n} dc={dc} DIM={DIM} | warm={len(warm):,} cold={int(cold_mask.sum()):,} | setup {time.time()-t0:.0f}s")

    # anchors: a random sample of warm items (content + candidate-collab A)
    rng = np.random.default_rng(0)
    anc = warm if len(warm) <= K_ANCHOR else rng.choice(warm, K_ANCHOR, replace=False)
    Ek, Vk = content[anc], A[anc]

    # --- induce candidate embeddings for COLD items three ways (warm keep real A) ---
    cold_ids = np.where(cold_mask)[0]
    # linreg
    Wmap = ridge_map(content[warm].astype(np.float64), A[warm].astype(np.float64), RIDGE)
    ind_lin = content[cold_ids] @ Wmap
    # knn borrow (cosine content-NN among anchors)
    ind_knn = np.zeros((len(cold_ids), DIM), np.float32)
    sims_all = content[cold_ids] @ Ek.T                  # (ncold, Kanchor)
    topk = np.argpartition(-sims_all, K_NN, axis=1)[:, :K_NN]
    for i in range(len(cold_ids)):
        ind_knn[i] = Vk[topk[i]].mean(0)
    # attention (distill: induced warm embedding ~ real A[warm])
    model = AttnBorrow(dc, DIM, TAU)
    Ek_t, Vk_t = torch.from_numpy(Ek), torch.from_numpy(Vk)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    anc_set = set(anc.tolist())
    train_ids = warm
    for ep in range(1, ATTN_EPOCHS + 1):
        model.train(); order = rng.permutation(len(train_ids)); losses = []
        for st in range(max(1, len(train_ids) // ATTN_BATCH)):
            idx = train_ids[order[st * ATTN_BATCH:(st + 1) * ATTN_BATCH]]
            if len(idx) < 2:
                continue
            E = torch.from_numpy(content[idx]); tgt = torch.from_numpy(A[idx])
            v = model.induce(E, Ek_t, Vk_t)
            loss = (1 - Fn.cosine_similarity(v, tgt, dim=-1)).mean() + 0.1 * ((v - tgt) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        if ep % 10 == 0 or ep == ATTN_EPOCHS:
            log(f"    [attn ep={ep}] distill_loss={np.mean(losses):.4f} ({time.time()-t0:.0f}s)")
    model.eval()
    with torch.no_grad():
        ind_attn = model.induce(torch.from_numpy(content[cold_ids]), Ek_t, Vk_t).numpy()

    def make_eff(induced):
        V = A.copy()
        V[cold_ids] = induced
        return V

    Veff = {"collab": A, "linreg": make_eff(ind_lin), "knn": make_eff(ind_knn), "attn": make_eff(ind_attn)}

    test_pairs = provided.pairs("test")
    test_ids = sorted(provided.splits["test"][0])
    oneseed = sparse_pairs(provided.pairs("test"), test_ids, "one_seed")
    if TEST_N:
        test_pairs, oneseed = test_pairs[:TEST_N], oneseed[:TEST_N]

    coldset = set(cold_ids.tolist())
    Ks = (20, 100, 500, 2000)

    def cold_stats(Vc, pairs):
        """Full-catalogue rank of each COLD target (canonical score_desc,id_asc tie policy, seeds
        excluded); plus warm Hit@20 sanity."""
        ranks = []
        wh = [0, 0]
        for seeds, gt in pairs:
            sc = Vc @ B[list(seeds)].sum(0)
            gts = sorted(set(gt))
            rr = relevant_ranks(sc, list(seeds), gts)       # tie-aware ranks aligned to gts
            for t, r in zip(gts, rr):
                if t in coldset:
                    ranks.append(int(r))
                elif freq[t] > 2:
                    wh[1] += 1; wh[0] += (r <= 20)
        ranks = np.array(ranks)
        return ({"median_rank": float(np.median(ranks)), "n": len(ranks),
                 **{f"hit@{k}": float((ranks <= k).mean()) for k in Ks}},
                wh[0] / max(1, wh[1]))

    out = {"dataset": dataset, "n": int(n), "dim": DIM, "n_cold": int(cold_mask.sum()), "conditions": {}}
    for cond, pairs in [("supplied", test_pairs), ("one_seed", oneseed)]:
        res = {}
        warm20 = {}
        for name, Vc in Veff.items():
            res[name], warm20[name] = cold_stats(Vc, pairs)
        out["conditions"][cond] = {"cold": res, "warm_hit20": warm20}
        log(f"  [{cond}] {len(pairs)} queries | COLD-target rank (n={res['collab']['n']}), warm Hit@20={warm20['collab']:.3f}:")
        log(f"      {'method':8s} {'med_rank':>9} " + " ".join(f"{'H@'+str(k):>8}" for k in Ks))
        for name in res:
            r = res[name]
            log(f"      {name:8s} {r['median_rank']:9.0f} " + " ".join(f"{r['hit@'+str(k)]:8.4f}" for k in Ks))
    log(f"  ({dataset} done in {time.time()-t0:.0f}s)")
    return out


def main():
    results = [run(ds) for ds in DATASETS]
    out_dir = os.path.join(PROJECT_ROOT, "results", "sefp")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "cold_attention.json")
    json.dump(results, open(outp, "w"), indent=2)
    log(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
