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
from bundlefuse.metrics import top_items
from bundlefuse.cli_data import add_data_arguments, load_data
from bundlefuse.evaluation import evaluate, flat_metrics, describe_model, sparse_pairs
from bundlefuse.provenance import write_json, run_metadata
import torch

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
    model = load_model(path)
    spec = getattr(model, 'data_spec', None)
    if spec and spec['dataset'] != args.dataset:
        raise ValueError('Checkpoint belongs to a different dataset')
    name = describe_model(model)['name']
    if name != args.model:
        raise ValueError(f'Checkpoint model is {name}; select --model {name}')
    return model


def _parse_metric(s):
    s = s.strip().lower()
    name, _, k = s.partition("@")
    if name not in ("recall", "ndcg"):
        raise argparse.ArgumentTypeError("metric must be recall@K or ndcg@K")
    try:
        cutoff = int(k) if k else 20
    except ValueError:
        raise argparse.ArgumentTypeError('Metric cutoff must be a positive integer') from None
    if cutoff < 1:
        raise argparse.ArgumentTypeError('Metric cutoff must be positive')
    return name, cutoff


def _eval(model, pairs, users, Ks, chunk=32):
    """Macro Recall/NDCG at each cutoff over (seeds, gt) pairs. Returns {K: (recall, ndcg)}."""
    return evaluate(model, pairs, users, Ks, chunk)


def _cut(pairs, k):
    """Truncate each query's seeds to its sorted first-k items (0 = keep all)."""
    return sparse_pairs(pairs, k)


# ------------------------------------------------------------------ modes
def cmd_predict(args):
    model = _load_model(args)
    if args.from_test is not None:
        if args.from_test < 0:
            raise ValueError('Test query index must be nonnegative')
        ds = load_data(args, model, test_n=args.from_test + 1)
        seeds = list(ds.test_pairs[args.from_test][0])
        user = (ds.test_users[args.from_test] if ds.test_users else None) if args.user is None else args.user
        gt = set(ds.test_pairs[args.from_test][1])
    else:
        if not args.seeds:
            sys.exit("predict needs --seeds a,b,c  (or --from-test INDEX)")
        seeds = [int(x) for x in args.seeds.split(",")]
        user = args.user; gt = None
    sc = model.score(seeds, user=user)
    top = top_items(sc, seeds, args.k).tolist()
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
    ds = load_data(args, model)
    res = _eval(model, ds.test_pairs, ds.test_users, (K,), args.chunk)
    val = res[K][0 if name == "recall" else 1]
    print(f"[{args.dataset}] {name}@{K} = {val:.4f}   ({len(ds.test_pairs)} test queries)")


def cmd_report(args):
    t0 = time.time()
    model = _load_model(args)
    ds = load_data(args, model)
    Ks = (10, 20, 50)
    overall = _eval(model, ds.test_pairs, ds.test_users, Ks, args.chunk)
    by_seed, stress = {}, {}
    for count in sorted({len(s) for s, _ in ds.test_pairs}):
        indices = [i for i, (s, _) in enumerate(ds.test_pairs) if len(s) == count]
        us = [ds.test_users[i] for i in indices] if ds.test_users is not None else None
        by_seed[str(count)] = {'queries': len(indices), **flat_metrics(_eval(
            model, [ds.test_pairs[i] for i in indices], us, Ks, args.chunk))}
    for count in (1, 2, 3):
        pairs = _cut(ds.test_pairs, count)
        stress[f'{count}-seed'] = {'mean_revealed': float(np.mean([len(s) for s, _ in pairs])),
                                  'mean_targets': float(np.mean([len(g) for _, g in pairs])),
                                  **flat_metrics(_eval(model, pairs, ds.test_users, Ks, args.chunk))}
    info = describe_model(model)
    out = {
        "dataset": args.dataset,
        "model": args.model,
        "checkpoint": _ckpt_path(args),
        "n_items": ds.n_items,
        "n_test": len(ds.test_pairs),
        'model_info': info, 'data': ds.metadata, 'test_query_ids': ds.test_ids,
        'signals': info.get('signals', [info['name']]), 'weights': info.get('weights', {}),
        'kappa_f': info.get('kappa_f', 0.),
        'overall': flat_metrics(overall),
        "by_seed_count": by_seed,
        'seed_stress': stress,
        'fit_runtime': getattr(model, 'run_info', None),
        'inference_runtime': run_metadata(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    from bundlefuse.provided_splits import file_hash
    out['checkpoint_sha256'] = file_hash(_ckpt_path(args))
    write_json(args.out, out)
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
    add_data_arguments(p)

    e = sub.add_parser("evaluate", help="score the test set under one metric")
    e.add_argument("dataset", choices=sorted(datasets.REGISTRY))
    e.add_argument("--model", choices=MODELS, default="bundlefuse", help="which trained model to use")
    e.add_argument("--metric", type=_parse_metric, default=_parse_metric("recall@20"), help="recall@K or ndcg@K")
    add_data_arguments(e)
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("report", help="full evaluation (R/N @10/20/50, overall + by seed count) -> JSON")
    r.add_argument("dataset", choices=sorted(datasets.REGISTRY))
    r.add_argument("--model", choices=MODELS, default="bundlefuse", help="which trained model to use")
    r.add_argument("--out", default=None, help="output JSON path (default reports/<dataset>_<model>.json)")
    add_data_arguments(r)
    r.set_defaults(func=cmd_report)
    for parser in (p, e, r):
        parser.add_argument('--chunk', type=int, default=32)
        parser.add_argument('--threads', type=int, default=1)

    args = ap.parse_args(argv)
    if args.chunk < 1 or args.threads < 1:
        ap.error('Chunk size and thread count must be positive')
    torch.set_num_threads(args.threads)
    if args.mode == "report" and args.out is None:
        stem = args.dataset if args.model == "bundlefuse" else f"{args.dataset}_{args.model}"
        args.out = os.path.join("reports", f"{stem}.json")
    args.func(args)


if __name__ == "__main__":
    main()
