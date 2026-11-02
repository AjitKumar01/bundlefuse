#!/usr/bin/env python3
"""inference.py -- use a trained BundleFuse checkpoint (no training here).

Three modes, chosen by subcommand:

  predict   complete a seed bundle -> top-k items
      python inference.py predict pog_dense --seeds 7917,15610 --k 5
      python inference.py predict dunnhumby --from-test 0 --k 10 --user 42

  evaluate  score the test set under ONE metric
      python inference.py evaluate pog_dense --metric recall@20
      python inference.py evaluate spotify  --metric ndcg@50 --test-n 2000

  report    full evaluation (Recall/NDCG @10/20/50 overall + by seed count) -> JSON
      python inference.py report dunnhumby --out reports/dunnhumby.json

The checkpoint is checkpoints/<dataset>.pt by default (override with --ckpt).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bundlefuse import datasets
from bundlefuse.model import load_model
from bundlefuse.baselines import BASELINES
from bundlefuse.metrics import ranking_metrics

MODELS = ["bundlefuse"] + sorted(BASELINES)


def _ckpt_path(args):
    model = getattr(args, "model", "bundlefuse")
    stem = args.dataset if model == "bundlefuse" else f"{args.dataset}_{model}"
    return args.ckpt or os.path.join(args.ckpt_dir, f"{stem}.pt")


def _load_model(args):
    path = _ckpt_path(args)
    if not os.path.exists(path):
        hint = args.dataset + ("" if getattr(args, "model", "bundlefuse") == "bundlefuse" else f" --model {args.model}")
        sys.exit(f"checkpoint not found: {path}  (train it first: python evaluator.py {hint})")
    return load_model(path)


def _parse_metric(s):
    s = s.strip().lower()
    name, _, k = s.partition("@")
    if name not in ("recall", "ndcg"):
        raise argparse.ArgumentTypeError("metric must be recall@K or ndcg@K")
    return name, (int(k) if k else 20)


def _eval(model, pairs, users, Ks, chunk=200):
    """Macro Recall/NDCG at each cutoff over (seeds, gt) pairs. Returns {K: (recall, ndcg)}."""
    agg = {k: np.zeros(2) for k in Ks}; tot = 0
    for off in range(0, len(pairs), chunk):
        ch = pairs[off:off + chunk]
        us = users[off:off + chunk] if users else [None] * len(ch)
        scores = np.stack([model.score(s, user=u) for (s, _), u in zip(ch, us)], axis=1)
        m = ranking_metrics(scores, ch, Ks)
        for k in Ks:
            agg[k] += np.asarray(m[k]) * len(ch)
        tot += len(ch)
    return {k: tuple(agg[k] / max(tot, 1)) for k in Ks}


def _cut(pairs, k):
    """Truncate each query's seeds to its sorted first-k items (0 = keep all)."""
    if not k:
        return pairs
    return [(sorted(s)[:k], g) for s, g in pairs]


# ------------------------------------------------------------------ modes
def cmd_predict(args):
    model = _load_model(args)
    if args.from_test is not None:
        ds = datasets.load(args.dataset, test_n=args.from_test + 1)
        seeds = list(ds.test_pairs[args.from_test][0])
        user = (ds.test_users[args.from_test] if ds.test_users else None) if args.user is None else args.user
        gt = set(ds.test_pairs[args.from_test][1])
    else:
        if not args.seeds:
            sys.exit("predict needs --seeds a,b,c  (or --from-test INDEX)")
        seeds = [int(x) for x in args.seeds.split(",")]
        user = args.user; gt = None
    sc = model.score(seeds, user=user)
    top = model.predict(seeds, k=args.k, user=user)
    print(f"[{args.dataset}] seeds={seeds}" + (f" user={user}" if user is not None else "") + f" -> top-{args.k}:")
    for rank, i in enumerate(top, 1):
        mark = "  <- held-out" if gt and i in gt else ""
        print(f"  {rank:>2}. item {i:<8} score={sc[i]:.4f}{mark}")
    if gt:
        hit = len(set(top) & gt)
        print(f"  (held-out hits in top-{args.k}: {hit}/{len(gt)})")


def cmd_evaluate(args):
    model = _load_model(args)
    name, K = args.metric
    ds = datasets.load(args.dataset, test_n=args.test_n)
    res = _eval(model, ds.test_pairs, ds.test_users, (K,))
    val = res[K][0 if name == "recall" else 1]
    print(f"[{args.dataset}] {name}@{K} = {val:.4f}   ({len(ds.test_pairs)} test queries)")


def cmd_report(args):
    t0 = time.time()
    model = _load_model(args)
    ds = datasets.load(args.dataset, test_n=args.test_n)
    Ks = (10, 20, 50)
    overall = _eval(model, ds.test_pairs, ds.test_users, Ks)
    by_seed = {}
    for cut, label in [(0, "supplied"), (1, "1-seed"), (2, "2-seed"), (3, "3-seed")]:
        r = _eval(model, _cut(ds.test_pairs, cut), ds.test_users, Ks)
        by_seed[label] = {f"recall@{k}": round(r[k][0], 4) for k in Ks} | {f"ndcg@{k}": round(r[k][1], 4) for k in Ks}
    out = {
        "dataset": args.dataset,
        "model": args.model,
        "checkpoint": _ckpt_path(args),
        "n_items": ds.n_items,
        "n_test": len(ds.test_pairs),
        "families": getattr(getattr(model, "cfg", None), "families", None),
        "signals": getattr(model, "signals_", [getattr(model, "name", args.model)]),
        "weights": {k: round(v, 3) for k, v in getattr(model, "weights_", {}).items()},
        "kappa_f": round(float(getattr(model, "kappa_f_", 0.0)), 4),
        "overall": {f"recall@{k}": round(overall[k][0], 4) for k in Ks} | {f"ndcg@{k}": round(overall[k][1], 4) for k in Ks},
        "by_seed_count": by_seed,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[{args.dataset}] full report ({len(ds.test_pairs)} queries) -> {args.out}  ({time.time()-t0:.0f}s)")
    print(f"  overall: " + "  ".join(f"{m}={v}" for m, v in out["overall"].items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Use a trained BundleFuse checkpoint.")
    ap.add_argument("--ckpt-dir", default="checkpoints", help="where checkpoints live")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (overrides --ckpt-dir)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("predict", help="complete a seed bundle -> top-k items")
    p.add_argument("dataset", choices=sorted(datasets.REGISTRY))
    p.add_argument("--model", choices=MODELS, default="bundlefuse", help="which trained model to use")
    p.add_argument("--seeds", default=None, help="comma-separated seed item indices")
    p.add_argument("--from-test", type=int, default=None, help="use the seeds of test bundle #INDEX")
    p.add_argument("--k", type=int, default=10, help="how many items to return (e.g. 5 or 10)")
    p.add_argument("--user", type=int, default=None, help="user id for personalized datasets")
    p.set_defaults(func=cmd_predict)

    e = sub.add_parser("evaluate", help="score the test set under one metric")
    e.add_argument("dataset", choices=sorted(datasets.REGISTRY))
    e.add_argument("--model", choices=MODELS, default="bundlefuse", help="which trained model to use")
    e.add_argument("--metric", type=_parse_metric, default=_parse_metric("recall@20"), help="recall@K or ndcg@K")
    e.add_argument("--test-n", type=int, default=2000)
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("report", help="full evaluation (R/N @10/20/50, overall + by seed count) -> JSON")
    r.add_argument("dataset", choices=sorted(datasets.REGISTRY))
    r.add_argument("--model", choices=MODELS, default="bundlefuse", help="which trained model to use")
    r.add_argument("--out", default=None, help="output JSON path (default reports/<dataset>_<model>.json)")
    r.add_argument("--test-n", type=int, default=2000)
    r.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    if args.mode == "report" and args.out is None:
        stem = args.dataset if args.model == "bundlefuse" else f"{args.dataset}_{args.model}"
        args.out = os.path.join("reports", f"{stem}.json")
    args.func(args)


if __name__ == "__main__":
    main()
