"""Train and evaluate BundleFuse on a dataset -- the single entry point.

Every dataset is trained through the one :class:`bundlefuse.model.BundleFuse` class: this script just
loads the data (:mod:`bundlefuse.data`), fits the model, evaluates Recall@20 / NDCG@20 on the test
split, and optionally writes a checkpoint.

    python -m bundlefuse.run pog_dense
    python -m bundlefuse.run dunnhumby --save ckpt/dunnhumby.pt
    python -m bundlefuse.run electronic --families collaborative,content,taxonomy,popularity
"""
import argparse
import time

import numpy as np

from bundlefuse.data import load_dataset
from bundlefuse.model import BundleFuse, BundleFuseConfig
from bundlefuse.metrics import ranking_metrics


def evaluate(model, pairs, users=None, chunk=200):
    agg = np.zeros(2); tot = 0
    for off in range(0, len(pairs), chunk):
        ch = pairs[off:off + chunk]
        us = users[off:off + chunk] if users else [None] * len(ch)
        scores = np.stack([model.score(s, user=u) for (s, _), u in zip(ch, us)], axis=1)
        agg += np.asarray(ranking_metrics(scores, ch, (20,))[20]) * len(ch); tot += len(ch)
    return agg / tot


def main():
    ap = argparse.ArgumentParser(description="Train + evaluate BundleFuse on a dataset.")
    ap.add_argument("dataset", help="pog, pog_dense, spotify, spotify_sparse, dunnhumby, electronic, food, clothing")
    ap.add_argument("--families", default=None, help="comma-separated subset to enable (default: dataset's own)")
    ap.add_argument("--val-n", type=int, default=800)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--svd-dim", type=int, default=128)
    ap.add_argument("--save", default=None, help="write a checkpoint to this path")
    args = ap.parse_args()

    t0 = time.time()
    ds = load_dataset(args.dataset, val_n=args.val_n, test_n=args.test_n)
    fams = args.families.split(",") if args.families else ds.families
    print(f"{ds.name}: n={ds.n_items} train={len(ds.train)} val={len(ds.val_pairs)} test={len(ds.test_pairs)} "
          f"| families={fams} | load {time.time()-t0:.0f}s", flush=True)

    model = BundleFuse(BundleFuseConfig(n_items=ds.n_items, families=fams, svd_dim=args.svd_dim))
    model.fit(ds.train, ds.val_pairs, content=ds.content, cat_labels=ds.cat_labels,
              user_items=ds.user_items, val_users=ds.val_users)
    print(f"  fitted in {time.time()-t0:.0f}s | signals={model.signals_} "
          f"| weights={ {k: round(v,2) for k,v in model.weights_.items()} } kappa_f={model.kappa_f_:.3g}", flush=True)

    r, nd = evaluate(model, ds.test_pairs, ds.test_users)
    print(f"\n  TEST ({len(ds.test_pairs)} queries): Recall@20={r:.4f}  NDCG@20={nd:.4f}", flush=True)

    if args.save:
        model.save(args.save)
        print(f"  checkpoint -> {args.save}", flush=True)
    print(f"  (done in {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
