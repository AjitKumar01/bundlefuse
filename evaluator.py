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
from bundlefuse.model import BundleFuse, load_model
from bundlefuse.baselines import BASELINES
from bundlefuse.evaluation import evaluate as evaluate_metrics, flat_metrics, describe_model
from bundlefuse.cli_data import add_data_arguments, load_data
from bundlefuse.provenance import run_metadata, write_json
import torch

MODELS = ["bundlefuse"] + sorted(BASELINES)

# fit-config overrides exposed on the command line -> BundleFuseConfig fields
CONFIG_OVERRIDES = ("alpha_deg", "gamma_vol", "beta_clique", "alpha_bridge", "fit_rounds", "seed_cap")


def evaluate(model, pairs, users=None, chunk=32):
    return evaluate_metrics(model, pairs, users, (20,), chunk)[20]


def build_parser():
    ap = argparse.ArgumentParser(description="Train BundleFuse on a dataset and store a checkpoint.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("dataset", choices=sorted(datasets.REGISTRY), help="which dataset to train on")
    ap.add_argument("--model", choices=MODELS, default="bundlefuse", help="BundleFuse or a baseline")
    ap.add_argument("--config", default=None, help="dataset config JSON (default: configs/<dataset>.json)")
    ap.add_argument("--no-config", action="store_true", help="ignore the config file; use built-in defaults")
    # data / fit configuration (BundleFuse only; ignored by baselines). These override the config file.
    ap.add_argument("--families", default=None, help="comma-separated families to enable (default: the dataset's own)")
    ap.add_argument("--svd-dim", type=int, default=None, help="collaborative SVD dimension (default: dataset's own)")
    ap.add_argument('--signal-profile', choices=('research','extended'), default=None,
                    help='historical research scorer or expanded signal pool')
    add_data_arguments(ap, training=True)
    ap.add_argument("--chunk", type=int, default=32, help="bounded validation/evaluation query block")
    ap.add_argument("--threads", type=int, default=1, help="CPU computation threads")
    ap.add_argument("--report", default=None, help="JSON run report (default: checkpoint path + .json)")
    ap.add_argument("--alpha-deg", type=float, default=None, help="degree-normalisation exponent")
    ap.add_argument("--gamma-vol", type=float, default=None, help="bundle-volume exponent")
    ap.add_argument("--beta-clique", type=float, default=None, help="higher-order clique coefficient")
    ap.add_argument("--alpha-bridge", type=float, default=None, help="content-bridge mixing")
    ap.add_argument("--fit-rounds", type=int, default=None, help="greedy weight-selection rounds")
    ap.add_argument("--seed-cap", type=int, default=None, help="limit seeds used for scoring; every revealed item remains excluded (0 = all)")
    ap.add_argument("--content-rank", type=int, default=None, help="low-rank content dim for compressed checkpoints")
    ap.add_argument("--no-compress", action="store_true", help="store the full (uncompressed) checkpoint")
    # checkpoint / eval control
    ap.add_argument("--ckpt-dir", default="checkpoints", help="directory to write <dataset>.pt into")
    ap.add_argument("--save", default=None, help="explicit checkpoint path (overrides --ckpt-dir)")
    ap.add_argument("--no-save", action="store_true", help="train/evaluate but do not write a checkpoint")
    ap.add_argument("--no-eval", action="store_true", help="train + checkpoint only, skip evaluation")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.time()
    if args.threads < 1 or args.chunk < 1:
        raise ValueError("Threads and chunk size must be positive")
    if args.config and args.no_config:
        raise ValueError("Choose --config or --no-config")
    torch.set_num_threads(args.threads)

    mod = datasets.get(args.dataset)
    ds = load_data(args)
    if args.model == "bundlefuse":
        fams = args.families.split(",") if args.families else None
        cfg = mod.config(ds.n_items, families=fams, svd_dim=args.svd_dim,
                         use_config=not args.no_config, config_file=args.config)
        for field in CONFIG_OVERRIDES:          # CLI flags override the config file
            val = getattr(args, field)
            if val is not None:
                setattr(cfg, field, val)
        if args.content_rank is not None:
            cfg.content_rank = args.content_rank
        if args.signal_profile is not None:
            cfg.signal_profile = args.signal_profile
        if args.no_compress:
            cfg.compressed = False
        cfg.validation_chunk = args.chunk
        cfg.__post_init__()
        model = BundleFuse(cfg)
        src = "built-in" if args.no_config else (args.config or f"configs/{args.dataset}.json")
        desc = f"families={cfg.families} svd_dim={cfg.svd_dim} content_rank={cfg.content_rank} [config: {src}]"
    else:
        model = BASELINES[args.model](ds.n_items)
        desc = f"baseline={args.model}"
    print(f"{ds.name} [{args.model}]: n={ds.n_items} train={len(ds.train)} val={len(ds.val_pairs)} "
          f"test={len(ds.test_pairs)} | {desc} | load {time.time()-t0:.0f}s", flush=True)

    model.fit(ds.train, ds.val_pairs, **ds.fit_kwargs())
    model.run_info = run_metadata()
    info = describe_model(model)
    print(f"  fitted in {time.time()-t0:.0f}s | " + str(info), flush=True)
    result = {'schema_version': 1, 'dataset': ds.name, 'model': info,
              'data': ds.metadata, 'runtime': model.run_info, 'test_query_ids': ds.test_ids}
    if not args.no_eval:
        result['test'] = flat_metrics(evaluate_metrics(model, ds.test_pairs, ds.test_users, (10,20,50), args.chunk))
        print(f"  TEST ({len(ds.test_pairs)} queries): {result['test']}", flush=True)

    if not args.no_save:
        stem = args.dataset if args.model == "bundlefuse" else f"{args.dataset}_{args.model}"
        path = args.save or os.path.join(args.ckpt_dir, f"{stem}.pt")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Probe validation, including --no-eval runs, without changing or selecting on test.
        probe = [s for s, _ in ds.val_pairs[:args.chunk]]
        us = ds.val_users[:args.chunk] if ds.val_users is not None else None
        before = model.score_batch(probe, us).copy()
        model.save(path)
        loaded = load_model(path)
        if not np.array_equal(before, model.score_batch(probe, us)) or not np.array_equal(before, loaded.score_batch(probe, us)):
            raise RuntimeError('Checkpoint changed scores; refusing to publish a successful run report')
        from bundlefuse.splits import file_hash
        result['checkpoint'] = {'path': path, 'sha256': file_hash(path), 'bytes': os.path.getsize(path),
                                'validation_probe_queries': len(probe), 'score_bit_identity': True}
        print(f"  checkpoint -> {path} ({os.path.getsize(path)/1e6:.1f} MB), score parity verified", flush=True)
    result['seconds'] = time.time() - t0
    report = args.report or (path + '.json' if not args.no_save else os.path.join('reports', f'{ds.name}_{args.model}_fit.json'))
    write_json(report, result)
    print(f"  run report -> {report}", flush=True)
    print(f"  (done in {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
