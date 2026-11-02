"""Golden-number regression test.

The configured model (configs/<dataset>.json) must reproduce recorded per-dataset Recall@20 / NDCG@20
(results/golden.json) within tolerance, at a FIXED fast config (test_n=500, val_n=300, split_seed=0).
This guards the scoring math + configs against silent regressions (the kind the content_rank bug was).

Datasets whose raw data is not present are skipped, so this runs anywhere. Regenerate the golden file
after an *intended* change with:  python tests/test_golden.py --update
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse import datasets
from bundlefuse.model import BundleFuse
from bundlefuse.metrics import ranking_metrics
from bundlefuse.paths import PROJECT_ROOT

GOLDEN = os.path.join(PROJECT_ROOT, "results", "golden.json")
MULTISEED = {"dunnhumby", "electronic", "food", "clothing"}
TEST_N, VAL_N, SEED, TOL = 500, 300, 0, 0.01
ORDER = ["pog", "pog_dense", "spotify", "spotify_sparse", "dunnhumby", "electronic", "food", "clothing"]


def _evaluate(model, pairs, users, chunk=200):
    agg = np.zeros(2); tot = 0
    for off in range(0, len(pairs), chunk):
        ch = pairs[off:off + chunk]
        us = users[off:off + chunk] if users else [None] * len(ch)
        sc = np.stack([model.score(s, user=u) for (s, _), u in zip(ch, us)], axis=1)
        agg += np.asarray(ranking_metrics(sc, ch, (20,))[20]) * len(ch); tot += len(ch)
    return agg / tot


def _run(name):
    """Return (R@20, N@20) for the configured model, or None if the dataset's data is absent."""
    mod = datasets.get(name)
    try:
        if name in MULTISEED:
            ds = mod.load(val_n=VAL_N, test_n=TEST_N, split_seed=SEED)
        else:
            ds = mod.load(val_n=VAL_N, test_n=TEST_N)
    except (FileNotFoundError, OSError):
        return None
    m = BundleFuse(mod.config(ds.n_items))
    m.fit(ds.train, ds.val_pairs, features=ds.features, cat_labels=ds.cat_labels,
          music=ds.music, alb_labels=ds.alb_labels, cluster_feat=ds.cluster_feat,
          aux_incidences=ds.aux_incidences, group_labels=ds.group_labels,
          user_items=ds.user_items, val_users=ds.val_users)
    return tuple(float(x) for x in _evaluate(m, ds.test_pairs, ds.test_users))


def test_golden():
    assert os.path.exists(GOLDEN), f"missing {GOLDEN}; regenerate with: python tests/test_golden.py --update"
    golden = json.load(open(GOLDEN))
    checked = skipped = 0
    for name in ORDER:
        r = _run(name)
        if r is None:
            skipped += 1
            print(f"  {name:16s} SKIP (data absent)")
            continue
        g = golden[name]
        dr, dn = abs(r[0] - g["r20"]), abs(r[1] - g["n20"])
        status = "OK" if (dr < TOL and dn < TOL) else "REGRESSED"
        print(f"  {name:16s} R@20={r[0]:.4f} (golden {g['r20']:.4f}, Δ{r[0]-g['r20']:+.4f})  "
              f"N@20={r[1]:.4f} (golden {g['n20']:.4f})  {status}")
        assert dr < TOL, f"{name} R@20 regressed: {r[0]:.4f} vs golden {g['r20']:.4f} (tol {TOL})"
        assert dn < TOL, f"{name} N@20 regressed: {r[1]:.4f} vs golden {g['n20']:.4f} (tol {TOL})"
        checked += 1
    print(f"golden: {checked} checked, {skipped} skipped")
    assert checked > 0, "no datasets available to check"


def _update():
    out = {}
    for name in ORDER:
        r = _run(name)
        if r is None:
            print(f"  {name}: data absent, skipped")
            continue
        out[name] = {"r20": round(r[0], 4), "n20": round(r[1], 4),
                     "test_n": TEST_N, "val_n": VAL_N, "split_seed": SEED}
        print(f"  {name}: R@20={r[0]:.4f} N@20={r[1]:.4f}")
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    json.dump(out, open(GOLDEN, "w"), indent=2)
    print(f"wrote {GOLDEN} ({len(out)} datasets)")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update()
    else:
        test_golden()
        print("all golden checks passed")
