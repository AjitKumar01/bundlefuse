"""MMFuse side of the gap decomposition: per-target Hit@K by TRAIN co-occurrence frequency bucket,
on the SAME test pairs and SAME buckets as the BundleFuse GAP_DECOMP block in eval/fullrank.py.

Localises where MMFuse (and the collab-SVD floor, and content-only) wins/loses recall, so the gap to
BundleFuse can be attributed to a frequency regime (cold / rare-warm / mid / head) before adding any
modelling. Env (via mmfuse): DIM, HEADS, EPOCHS, DROP_COLLAB. Run:
  DATASETS=pog_dense EPOCHS=20 python3 reports/gap_decomp_mmfuse.py
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


def log(*a):
    print(*a, flush=True)


def bucket(f):
    for lo, hi, name in EDGES:
        if lo <= f <= hi:
            return name
    return "head(>10)"


def hits_by_bucket(score_fn, pairs, freq, Ks=(10, 20)):
    buck = {name: {f"hit{k}": 0 for k in Ks} | {"n": 0} for _, _, name in EDGES}
    for seeds, gt in pairs:
        sc = np.asarray(score_fn(seeds), np.float32)
        top20 = top_items(sc, list(seeds), max(Ks))
        tops = {k: set(top20[:k].tolist()) for k in Ks}
        for t in set(gt):
            b = buck[bucket(int(freq[t]))]; b["n"] += 1
            for k in Ks:
                b[f"hit{k}"] += int(t in tops[k])
    return {name: {"n": v["n"], **{f"hit@{k}": (v[f"hit{k}"] / v["n"] if v["n"] else float("nan")) for k in Ks}}
            for name, v in buck.items()}


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
    log(f"\n### {dataset}: n={n:,} du={du} de={de} | setup {time.time()-t0:.0f}s")

    test_pairs = provided.pairs("test")
    val_pairs = provided.pairs("valid")[:M.VAL_N]
    bundles = list(provided.train.values())
    log(f"  training MMFuse no-dropout...")
    model_nd = M.train(M.MMFuse(du, de, M.DIM, M.HEADS), U, Bside, E, C, bundles, val_pairs, n, freq, drop_collab=0.0)

    # content-only projection (frozen content cosine, no collab) for reference
    Econt = content

    def s_collab(seeds):
        return A @ Bs[list(seeds)].sum(0)

    def s_nd(seeds):
        return M.score_all(model_nd, U, Bside, E, seeds, n)

    def s_content(seeds):
        return Econt @ Econt[list(seeds)].sum(0)

    out = {"dataset": dataset, "n": int(n), "edges": [(e[0], e[1], e[2]) for e in EDGES], "models": {}}
    for name, fn in [("collab-SVD", s_collab), ("content-cos", s_content), ("MMFuse_nd", s_nd)]:
        te = time.time(); out["models"][name] = hits_by_bucket(fn, test_pairs, freq); log(f"    {name:12s} done ({time.time()-te:.0f}s)")

    log(f"\n  {dataset}  MMFuse-side per-target Hit@20 by TRAIN frequency ({len(test_pairs)} queries)")
    order = ["cold(f=0)", "rare(1-2)", "mid(3-10)", "head(>10)"]
    log(f"    {'bucket':12s} {'n':>7} " + " ".join(f"{m:>12}" for m in out["models"]))
    for b in order:
        log(f"    {b:12s} {out['models']['collab-SVD'][b]['n']:7d} "
            + " ".join(f"{out['models'][m][b]['hit@20']:12.4f}" for m in out["models"]))
    log(f"  ({dataset} done in {time.time()-t0:.0f}s)")
    return out


def main():
    res = [run(ds) for ds in DATASETS]
    d = os.path.join(PROJECT_ROOT, "results", "sefp"); os.makedirs(d, exist_ok=True)
    for r in res:
        p = os.path.join(d, f"gap_mmfuse_{r['dataset']}.json"); json.dump(r, open(p, "w"), indent=2)
        log(f"wrote {p}")


if __name__ == "__main__":
    main()
