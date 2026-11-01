"""Input-swap ablation for MMFuse, per the per-signal diagnostic: the architecture was fine, the INPUTS
were the weakest available. On pog_dense the rare/mid signal lives in PPMI (0.127/0.139) and title-TEXT
content (bridge 0.10), NOT in W_pair-SVD (0.040/0.082, what MMFuse used) or raw IMAGE content (~0).

Swap the two frozen inputs, architecture unchanged, and re-measure per-target Hit@20 by TRAIN frequency
bucket -- SAME buckets/splits as the BundleFuse GAP_DECOMP reference. Clean 2x2 (collab kernel x content
modality) + a concatenated-kernel row:
  collab:  wpair  = TruncatedSVD(W_pair)   [control]       ppmi = TruncatedSVD(PPMI)       cat = [wpair|ppmi]
  content: image  = content_feature.pt     [control]       title = title_emb.npy
Falsifiable prediction: (ppmi,title) lifts rare/mid toward BundleFuse's 0.094/0.158. Env (via mmfuse):
DIM, HEADS, EPOCHS. Run: DATASETS=pog_dense EPOCHS=20 python3 reports/mmfuse_inputswap.py
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
from bundlefuse.metrics import top_items

DATASETS = os.environ.get("DATASETS", "pog_dense").split(",")
EDGES = [(0, 0, "cold(f=0)"), (1, 2, "rare(1-2)"), (3, 10, "mid(3-10)"), (11, 10**12, "head(>10)")]
ORDER = ["cold(f=0)", "rare(1-2)", "mid(3-10)", "head(>10)"]
# BundleFuse reference (gap_bundlefuse_pog_dense.json), for the same buckets
BF = {"pog_dense": {"cold(f=0)": 0.0000, "rare(1-2)": 0.0935, "mid(3-10)": 0.1578, "head(>10)": 0.7503}}


def log(*a):
    print(*a, flush=True)


def bucket(f):
    for lo, hi, name in EDGES:
        if lo <= f <= hi:
            return name
    return "head(>10)"


def svd_embed(K, d, n):
    s = TruncatedSVD(n_components=min(d, n - 2), random_state=0)
    A = s.fit_transform(K).astype(np.float32)        # candidate side U*S
    B = s.components_.T.astype(np.float32)           # seed side V
    return A, B


def norm_rows(x):
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def hits_by_bucket(score_fn, pairs, freq, k=20):
    buck = {name: [0, 0] for _, _, name in EDGES}
    for seeds, gt in pairs:
        sc = np.asarray(score_fn(seeds), np.float32)
        top = set(top_items(sc, list(seeds), k).tolist())
        for t in set(gt):
            b = buck[bucket(int(freq[t]))]; b[1] += 1; b[0] += int(t in top)
    return {name: (v[0] / v[1] if v[1] else float("nan")) for name, v in buck.items()}


def run(dataset):
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    provided = ProvidedSplits(dd, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    freq = graph.freq.astype(np.float64)
    Wp = graph.W_pair.tocsr(); PP = graph.PPMI.tocsr()
    img = norm_rows(torch.load(os.path.join(dd, "content_feature.pt"), map_location="cpu", weights_only=False).detach().float().numpy())
    title = norm_rows(np.load(os.path.join(dd, "title_emb.npy")))
    log(f"\n### {dataset}: n={n:,} | building collab embeddings... (setup {time.time()-t0:.0f}s)")

    Aw, Bw = svd_embed(Wp, M.DIM, n)
    Ap, Bp = svd_embed(PP, M.DIM, n)
    collabs = {
        "wpair": (Aw, Bw),
        "ppmi": (Ap, Bp),
        "cat": (np.concatenate([Aw, Ap], 1), np.concatenate([Bw, Bp], 1)),
    }
    contents = {"image": img, "title": title}

    test_pairs = provided.pairs("test")
    val_pairs = provided.pairs("valid")[:M.VAL_N]
    bundles = list(provided.train.values())

    configs = [("wpair", "image"), ("ppmi", "image"), ("wpair", "title"), ("ppmi", "title"), ("cat", "title")]
    results = {}
    for ck, ctk in configs:
        A, Bs = collabs[ck]; E = contents[ctk]
        du, de = A.shape[1], E.shape[1]
        U = torch.from_numpy(A); Bside = torch.from_numpy(Bs); Et = torch.from_numpy(E)
        tag = f"{ck}+{ctk}"
        log(f"  [{tag}] training MMFuse_nd (du={du} de={de})...")
        model = M.train(M.MMFuse(du, de, M.DIM, M.HEADS), U, Bside, Et, graph.C.tocsr(), bundles, val_pairs, n, freq, drop_collab=0.0)
        fn = lambda seeds, _m=model, _U=U, _B=Bside, _E=Et: M.score_all(_m, _U, _B, _E, seeds, n)
        te = time.time(); results[tag] = hits_by_bucket(fn, test_pairs, freq); log(f"    eval done ({time.time()-te:.0f}s)")

    log(f"\n  {dataset}  MMFuse input-swap: per-target Hit@20 by TRAIN frequency ({len(test_pairs)} queries)")
    log(f"    {'config':14s} " + " ".join(f"{b:>11}" for b in ORDER))
    bf = BF.get(dataset)
    if bf:
        log(f"    {'BundleFuse':14s} " + " ".join(f"{bf[b]:11.4f}" for b in ORDER) + "   (reference)")
    for tag, r in results.items():
        marker = "  <- both swaps" if tag == "ppmi+title" else ("  (control)" if tag == "wpair+image" else "")
        log(f"    {tag:14s} " + " ".join(f"{r[b]:11.4f}" for b in ORDER) + marker)
    log(f"  ({dataset} done in {time.time()-t0:.0f}s)")
    return {"dataset": dataset, "configs": results, "bundlefuse": bf}


def main():
    res = [run(ds) for ds in DATASETS]
    d = os.path.join(PROJECT_ROOT, "results", "sefp"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "mmfuse_inputswap.json"); json.dump(res, open(p, "w"), indent=2)
    log(f"\nwrote {p}")


if __name__ == "__main__":
    main()
