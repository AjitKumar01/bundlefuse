"""Table 1 -- MMFuse vs CLHE on the CANONICAL full-ranking protocol (Recall@/NDCG@ {10,20}).

Puts the new attention-fusion architecture on the SAME metric, splits, candidate universe (all items
minus seeds), supplied partial-bundle protocol and tie policy that produced results/matrix.csv -- so the
numbers sit directly beside the reproduced CLHE row. Uses bundlefuse.metrics.ranking_metrics, called
per-query with a 1-column score matrix and averaged (byte-identical semantics, memory-safe at 213k items).

Reports MMFuse no-dropout (model_nd) and DropoutNet (model_dp), plus a collab-SVD sanity row, against the
hard-coded CLHE and Ours bars from matrix.csv. Full test set (no TEST_N truncation). Honest framing: this
is a LEARNED model vs CLHE (a learned model) -- the zero-parameter story does not apply here; the axes are
accuracy + param budget (reported). Env (via mmfuse): DIM, HEADS, EPOCHS, DROP_COLLAB, LR, NEG, HARD_FRAC.
Run: DATASETS=pog_dense,spotify_sparse DROP_COLLAB=0.3 python3 reports/mmfuse_vs_clhe.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reports.mmfuse as M
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import ranking_metrics

DATASETS = os.environ.get("DATASETS", "pog_dense,spotify_sparse").split(",")

# Bars from results/matrix.csv  (R@10, N@10, R@20, N@20)
BARS = {
    "pog_dense":      {"CLHE": (0.3438, 0.2662, 0.3815, 0.2771), "Ours": (0.3539, 0.2793, 0.3969, 0.2914)},
    "spotify_sparse": {"CLHE": (0.0608, 0.1164, 0.0969, 0.1150), "Ours": (0.1063, 0.2411, 0.1620, 0.2265)},
    "spotify":        {"CLHE": (0.0689, 0.1950, 0.1082, 0.1806), "Ours": (0.0964, 0.2790, 0.1486, 0.2532)},
}


def log(*a):
    print(*a, flush=True)


def fullrank(score_fn, pairs, Ks=(10, 20)):
    """Macro Recall@/NDCG@ averaged over queries, via the canonical ranking_metrics (1 col/query)."""
    acc = {k: np.zeros(2, np.float64) for k in Ks}
    for seeds, gt in pairs:
        sc = np.asarray(score_fn(seeds), np.float32)[:, None]      # (n,1)
        m = ranking_metrics(sc, [(list(seeds), list(gt))], Ks)
        for k in Ks:
            acc[k] += np.asarray(m[k])
    return {k: tuple(acc[k] / len(pairs)) for k in Ks}


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
    svd = TruncatedSVD(n_components=min(M.DIM, n - 2), random_state=0)
    A = svd.fit_transform(W).astype(np.float32); Bs = svd.components_.T.astype(np.float32)
    du = A.shape[1]
    U = torch.from_numpy(A); Bside = torch.from_numpy(Bs); E = torch.from_numpy(content)
    log(f"\n### {dataset}: n={n:,} du={du} de={de} DIM={M.DIM} heads={M.HEADS} drop={M.DROP_COLLAB} | setup {time.time()-t0:.0f}s")

    test_pairs = provided.pairs("test")
    val_pairs = provided.pairs("valid")[:M.VAL_N]
    bundles = list(provided.train.values())
    nparam = sum(p.numel() for p in M.MMFuse(du, de, M.DIM, M.HEADS).parameters())

    log(f"  training MMFuse no-dropout (params={nparam:,})...")
    model_nd = M.train(M.MMFuse(du, de, M.DIM, M.HEADS), U, Bside, E, C, bundles, val_pairs, n, freq, drop_collab=0.0)
    log(f"  training MMFuse dropout={M.DROP_COLLAB}...")
    model_dp = M.train(M.MMFuse(du, de, M.DIM, M.HEADS), U, Bside, E, C, bundles, val_pairs, n, freq, drop_collab=M.DROP_COLLAB)

    def s_collab(seeds):
        return A @ Bs[list(seeds)].sum(0)

    def s_nd(seeds):
        return M.score_all(model_nd, U, Bside, E, seeds, n)

    def s_dp(seeds):
        return M.score_all(model_dp, U, Bside, E, seeds, n)

    log(f"  scoring {len(test_pairs)} test queries (full catalogue)...")
    rows = {}
    for name, fn in [("collab-SVD", s_collab), ("MMFuse_nd", s_nd), ("MMFuse_dp", s_dp)]:
        te = time.time(); rows[name] = fullrank(fn, test_pairs); log(f"    {name:11s} done ({time.time()-te:.0f}s)")

    log(f"\n  {dataset}  (full-ranking, supplied protocol, {len(test_pairs)} queries)")
    log(f"    {'method':14s} {'R@10':>8} {'N@10':>8} {'R@20':>8} {'N@20':>8}")
    for name in ("CLHE", "Ours"):
        b = BARS.get(dataset, {}).get(name)
        if b:
            log(f"    {name:14s} {b[0]:8.4f} {b[1]:8.4f} {b[2]:8.4f} {b[3]:8.4f}   (matrix.csv)")
    for name, r in rows.items():
        log(f"    {name:14s} {r[10][0]:8.4f} {r[10][1]:8.4f} {r[20][0]:8.4f} {r[20][1]:8.4f}")
    log(f"  ({dataset} done in {time.time()-t0:.0f}s)")
    def flat(r):
        out = {}
        for kk, v in r.items():
            out[f"R@{kk}"] = v[0]; out[f"N@{kk}"] = v[1]
        return out
    return {"dataset": dataset, "n": int(n), "params": int(nparam),
            "rows": {k: flat(r) for k, r in rows.items()},
            "bars": BARS.get(dataset, {})}


def main():
    out = [run(ds) for ds in DATASETS]
    d = os.path.join(PROJECT_ROOT, "results", "sefp"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "mmfuse_vs_clhe.json"); json.dump(out, open(p, "w"), indent=2)
    log(f"\nwrote {p}")


if __name__ == "__main__":
    main()
