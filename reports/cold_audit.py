"""Cold-start audit: is the parameter-only SVD-of-W_pair model (purely collaborative) blind to items
unseen in training, and does the benchmark even exercise cold start?

A collaborative embedding v_c = (U S)_c = C_{c,:} V_d is ZERO for an item with no training
co-occurrence, so cold candidates score 0 (invisible) and cold seeds contribute nothing to the pool.
This script quantifies the blind spot WITHOUT assuming it matters:
  1. fraction of test TARGETS that are cold (train freq==0) or rare (freq<=2), and cold SEEDS.
  2. micro-averaged Hit@20 of the collaborative model, split by cold vs warm TARGET.
  3. the same for a CONTENT-only score (CLAP cosine sum over seeds) -- preview of whether content,
     which exists for any item, can retrieve the cold targets the collaborative model misses.

If almost no test targets are cold, the headline recall numbers simply say nothing about cold start
(the split doesn't test it) -- itself an honest finding. Env: DATASETS, DIM, TEST_N.
Run: python3 reports/cold_audit.py
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.paths import DATA_ROOT, PROJECT_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import top_items
from bundlefuse.robust_protocol import sparse_pairs

DATASETS = os.environ.get("DATASETS", "spotify,pog_dense").split(",")
DIM = int(os.environ.get("DIM", "256"))
TEST_N = int(os.environ.get("TEST_N", "0"))


def log(*a):
    print(*a, flush=True)


def load_content(dd, n):
    try:
        import torch
        x = torch.load(os.path.join(dd, "content_feature.pt"), map_location="cpu",
                       weights_only=False).detach().float().numpy()
        return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    except Exception as e:
        log(f"  (no content features: {e})")
        return None


def hits_by_bucket(score_fn, pairs, freq, k=20):
    """Micro-averaged Hit@k of targets, bucketed by target coldness. Returns dict bucket->(hits,total)."""
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
    return buckets


def run(dataset):
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", dataset)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    provided = ProvidedSplits(dd, n)
    graph = SimplicialGraphOperator(provided.train, n_items=n, svd_dim=2)
    freq = graph.freq.astype(np.float64)
    W = graph.W_pair.tocsr()
    content = load_content(dd, n)
    svd = TruncatedSVD(n_components=min(DIM, n - 2), random_state=0)
    A = svd.fit_transform(W).astype(np.float32)
    B = svd.components_.T.astype(np.float32)
    log(f"\n### {dataset}: n={n} | SVD-wpair d={DIM} built in {time.time()-t0:.0f}s | "
        f"catalogue cold(f=0)={int((freq==0).sum()):,} ({(freq==0).mean()*100:.1f}%)")

    test_pairs = provided.pairs("test")
    test_ids = sorted(provided.splits["test"][0])
    oneseed = sparse_pairs(provided.pairs("test"), test_ids, "one_seed")
    if TEST_N:
        test_pairs, oneseed = test_pairs[:TEST_N], oneseed[:TEST_N]

    def collab(seeds):
        return A @ B[seeds].sum(0)

    def cont(seeds):
        return content @ content[seeds].sum(0) if content is not None else np.zeros(n, np.float32)

    out = {"dataset": dataset, "n": int(n), "dim": DIM,
           "catalogue_cold_frac": float((freq == 0).mean()), "conditions": {}}
    for cond, pairs in [("supplied", test_pairs), ("one_seed", oneseed)]:
        # target / seed coldness exposure
        tf = np.array([freq[t] for _, gt in pairs for t in gt])
        sf = np.array([freq[s] for seeds, _ in pairs for s in seeds])
        allcold_q = sum(1 for seeds, _ in pairs if all(freq[s] == 0 for s in seeds))
        bc = hits_by_bucket(collab, pairs, freq)
        bk = hits_by_bucket(cont, pairs, freq) if content is not None else None
        def rate(b):
            return {k: (v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in b.items()}
        out["conditions"][cond] = {
            "n_queries": len(pairs),
            "target_cold_frac": float((tf == 0).mean()), "seed_cold_frac": float((sf == 0).mean()),
            "queries_all_seeds_cold": int(allcold_q),
            "collab_hit20": rate(bc), "content_hit20": rate(bk) if bk else None}
        log(f"  [{cond}] {len(pairs)} queries | targets cold={float((tf==0).mean())*100:.1f}% "
            f"rare={float(((tf>0)&(tf<=2)).mean())*100:.1f}% | seeds cold={float((sf==0).mean())*100:.1f}% "
            f"| all-seeds-cold queries={allcold_q}")
        rc = rate(bc)
        log(f"      collab  Hit@20:  " + "  ".join(f"{k} {v[0]:.3f}(n={v[1]})" for k, v in rc.items()))
        if bk:
            rk = rate(bk)
            log(f"      content Hit@20:  " + "  ".join(f"{k} {v[0]:.3f}(n={v[1]})" for k, v in rk.items()))
    return out


def main():
    results = []
    for ds in DATASETS:
        results.append(run(ds))
    out_dir = os.path.join(PROJECT_ROOT, "results", "sefp")
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, "cold_audit.json")
    json.dump(results, open(outp, "w"), indent=2)
    log(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
