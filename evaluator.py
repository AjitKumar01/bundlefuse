#!/usr/bin/env python3
"""evaluator.py -- train BundleFuse on a dataset from the command line and store a checkpoint.

This sits outside the package and is the single entry point: it reads the dataset to train and the
fit configuration from command-line arguments, builds the per-dataset BundleFuse object, fits it,
(optionally) evaluates Recall@20 / NDCG@20 on the test split, and writes a checkpoint.

Examples
--------
    python evaluator.py pog_dense
    python evaluator.py dunnhumby --test-n 2000 --ckpt-dir checkpoints
    python evaluator.py electronic --families collaborative,content,taxonomy,popularity --svd-dim 64
    python evaluator.py spotify --beta-clique 1.0 --seed-cap 20 --no-eval      # train + checkpoint only
    python evaluator.py food --save my_ckpts/food.pt
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundlefuse import datasets
from bundlefuse.model import BundleFuse
from bundlefuse.baselines import BASELINES
from bundlefuse.metrics import ranking_metrics

MODELS = ["bundlefuse"] + sorted(BASELINES)

# fit-config overrides exposed on the command line -> BundleFuseConfig fields
CONFIG_OVERRIDES = ("alpha_deg", "gamma_vol", "beta_clique", "alpha_bridge", "fit_rounds", "seed_cap")


def evaluate(model, pairs, users=None, chunk=200):
    agg = np.zeros(2); tot = 0
    for off in range(0, len(pairs), chunk):
        ch = pairs[off:off + chunk]
        us = users[off:off + chunk] if users else [None] * len(ch)
        scores = np.stack([model.score(s, user=u) for (s, _), u in zip(ch, us)], axis=1)
        agg += np.asarray(ranking_metrics(scores, ch, (20,))[20]) * len(ch); tot += len(ch)
    return agg / tot


def build_parser():
    ap = argparse.ArgumentParser(description="Train BundleFuse on a dataset and store a checkpoint.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("dataset", choices=sorted(datasets.REGISTRY), help="which dataset to train on")
    ap.add_argument("--model", choices=MODELS, default="bundlefuse", help="BundleFuse or a baseline")
    # data / fit configuration (BundleFuse only; ignored by baselines)
    ap.add_argument("--families", default=None, help="comma-separated families to enable (default: the dataset's own)")
    ap.add_argument("--svd-dim", type=int, default=None, help="collaborative SVD dimension (default: dataset's own)")
    ap.add_argument("--val-n", type=int, default=800, help="number of validation queries for weight selection")
    ap.add_argument("--test-n", type=int, default=2000, help="number of test queries to evaluate")
    ap.add_argument("--alpha-deg", type=float, default=None, help="degree-normalisation exponent")
    ap.add_argument("--gamma-vol", type=float, default=None, help="bundle-volume exponent")
    ap.add_argument("--beta-clique", type=float, default=None, help="higher-order clique coefficient")
    ap.add_argument("--alpha-bridge", type=float, default=None, help="content-bridge mixing")
    ap.add_argument("--fit-rounds", type=int, default=None, help="greedy weight-selection rounds")
    ap.add_argument("--seed-cap", type=int, default=None, help="truncate seed sets to this many items (0 = all)")
    # checkpoint / eval control
    ap.add_argument("--ckpt-dir", default="checkpoints", help="directory to write <dataset>.pt into")
    ap.add_argument("--save", default=None, help="explicit checkpoint path (overrides --ckpt-dir)")
    ap.add_argument("--no-save", action="store_true", help="train/evaluate but do not write a checkpoint")
    ap.add_argument("--no-eval", action="store_true", help="train + checkpoint only, skip evaluation")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.time()

    mod = datasets.get(args.dataset)
    ds = mod.load(val_n=args.val_n, test_n=args.test_n)
    if args.model == "bundlefuse":
        fams = args.families.split(",") if args.families else None
        cfg = mod.config(ds.n_items, families=fams, svd_dim=args.svd_dim)
        for field in CONFIG_OVERRIDES:
            val = getattr(args, field)
            if val is not None:
                setattr(cfg, field, val)
        model = BundleFuse(cfg)
        desc = f"families={cfg.families} svd_dim={cfg.svd_dim}"
    else:
        model = BASELINES[args.model](ds.n_items)
        desc = f"baseline={args.model}"
    print(f"{ds.name} [{args.model}]: n={ds.n_items} train={len(ds.train)} val={len(ds.val_pairs)} "
          f"test={len(ds.test_pairs)} | {desc} | load {time.time()-t0:.0f}s", flush=True)

    model.fit(ds.train, ds.val_pairs, content=ds.content, cat_labels=ds.cat_labels,
              user_items=ds.user_items, val_users=ds.val_users)
    if hasattr(model, "signals_"):
        print(f"  fitted in {time.time()-t0:.0f}s | signals={model.signals_} "
              f"| weights={ {k: round(v,2) for k,v in model.weights_.items()} } kappa_f={model.kappa_f_:.3g}", flush=True)
    else:
        print(f"  fitted in {time.time()-t0:.0f}s", flush=True)

    if not args.no_eval:
        r, nd = evaluate(model, ds.test_pairs, ds.test_users)
        print(f"  TEST ({len(ds.test_pairs)} queries): Recall@20={r:.4f}  NDCG@20={nd:.4f}", flush=True)

    if not args.no_save:
        stem = args.dataset if args.model == "bundlefuse" else f"{args.dataset}_{args.model}"
        path = args.save or os.path.join(args.ckpt_dir, f"{stem}.pt")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model.save(path)
        print(f"  checkpoint -> {path} ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)
    print(f"  (done in {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
