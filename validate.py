#!/usr/bin/env python3
"""validate.py -- verify the clean BundleFuse class against the original research matrix.

Runs the configured model (configs/<dataset>.json) on every dataset -- the full provided test split for
bundlemllm, and a 5-seed average for dunnhumby / BundleRec (matching the research protocol's multi-seed
reporting) -- and reports Recall@20 / NDCG@20 next to the committed research numbers
(results/matrix_zeroparam.csv on the compressed-inference branch), with the honest delta.

Writes results/validation.json (the golden numbers consumed by tests/test_golden.py).
Run: python validate.py            Env: VAL_SEEDS (default 5), BUNDLEMLLM_TESTN (default full).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundlefuse import datasets
from bundlefuse.model import BundleFuse
from bundlefuse.metrics import ranking_metrics
from bundlefuse.paths import PROJECT_ROOT

# research "Ours" numbers from results/matrix_zeroparam.csv (R@20, N@20, R@20_std)
RESEARCH = {
    "pog": (0.0681, 0.0406, 0.0026), "pog_dense": (0.3752, 0.2631, 0.0065),
    "spotify": (0.1567, 0.2614, 0.0018), "spotify_sparse": (0.1680, 0.2351, 0.0044),
    "electronic": (0.5163, 0.3708, 0.0212), "food": (0.5939, 0.4455, 0.0281),
    "clothing": (0.5881, 0.4544, 0.0326), "dunnhumby": (0.1504, 0.1079, 0.0044),
}
MULTISEED = {"pog", "pog_dense", "spotify", "spotify_sparse", "dunnhumby", "electronic", "food", "clothing"}      # research averaged 5 random splits
N_SEEDS = int(os.environ.get("VAL_SEEDS", "5"))
BUNDLEMLLM_TESTN = int(os.environ.get("BUNDLEMLLM_TESTN", "100000"))   # effectively full provided test
DUNN_TESTN = int(os.environ.get("DUNN_TESTN", "2000"))
ORDER = ["pog", "pog_dense", "spotify", "spotify_sparse", "dunnhumby", "electronic", "food", "clothing"]


def evaluate(model, pairs, users, chunk=200):
    agg = np.zeros(2); tot = 0
    for off in range(0, len(pairs), chunk):
        ch = pairs[off:off + chunk]
        us = users[off:off + chunk] if users else [None] * len(ch)
        sc = np.stack([model.score(s, user=u) for (s, _), u in zip(ch, us)], axis=1)
        agg += np.asarray(ranking_metrics(sc, ch, (20,))[20]) * len(ch); tot += len(ch)
    return agg / tot


def run_once(name, split_seed, test_n, val_n=800):
    mod = datasets.get(name)
    if name in MULTISEED:
        ds = mod.load(val_n=val_n, test_n=test_n, split_seed=split_seed)
    else:
        ds = mod.load(val_n=val_n, test_n=test_n)
    cfg = mod.config(ds.n_items)
    m = BundleFuse(cfg)
    m.fit(ds.train, ds.val_pairs, features=ds.features, cat_labels=ds.cat_labels,
          music=ds.music, alb_labels=ds.alb_labels, cluster_feat=ds.cluster_feat,
          aux_incidences=ds.aux_incidences, group_labels=ds.group_labels,
          user_items=ds.user_items, val_users=ds.val_users)
    return np.asarray(evaluate(m, ds.test_pairs, ds.test_users)), len(ds.test_pairs)


def main():
    out = {}
    print(f"{'dataset':16s} {'clean R@20':>14} {'research R@20':>14} {'ΔR@20':>9} {'clean N@20':>11} {'research N@20':>14}", flush=True)
    for name in ORDER:
        t = time.time()
        if name in MULTISEED:
            tn = DUNN_TESTN if name == "dunnhumby" else BUNDLEMLLM_TESTN
            runs = [run_once(name, s, tn)[0] for s in range(N_SEEDS)]
            arr = np.array(runs); mean, std = arr.mean(0), arr.std(0); nq = f"{N_SEEDS}seed"
        else:
            mean, nqn = run_once(name, 0, BUNDLEMLLM_TESTN); std = np.zeros(2); nq = str(nqn)
        r = RESEARCH[name]
        out[name] = {"clean_r20": float(mean[0]), "clean_n20": float(mean[1]),
                     "clean_r20_std": float(std[0]), "n_test": nq,
                     "research_r20": r[0], "research_n20": r[1], "research_r20_std": r[2],
                     "delta_r20": float(mean[0] - r[0]), "delta_n20": float(mean[1] - r[1])}
        print(f"{name:16s} {mean[0]:>8.4f}±{std[0]:.3f} {r[0]:>8.4f}±{r[2]:.3f} {mean[0]-r[0]:>+9.4f} "
              f"{mean[1]:>11.4f} {r[1]:>14.4f}   ({nq}, {time.time()-t:.0f}s)", flush=True)
    outp = os.path.join(PROJECT_ROOT, "results", "validation.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
