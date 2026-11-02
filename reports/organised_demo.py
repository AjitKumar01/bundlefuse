"""Demo + verification for the new BundleFuse class (bundlefuse/model.py).

Loads pog_dense (thin, dataset-specific loading stays outside the model), fits one BundleFuse object,
reports test Recall@20/NDCG@20, then checkpoints and reloads it and verifies predict() is identical.
Run: python3 reports/organised_demo.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.model import BundleFuse, BundleFuseConfig
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.metrics import ranking_metrics

DS = os.environ.get("BUNDLE_DS", "pog_dense")
TEST_N = int(os.environ.get("TEST_N", "1000"))
VAL_N = int(os.environ.get("VAL_N", "400"))


def main():
    t0 = time.time()
    dd = os.path.join(DATA_ROOT, "bundlemllm", "datasets", DS)
    n = json.load(open(os.path.join(dd, "count.json")))["#I"]
    sp_ = ProvidedSplits(dd, n)
    train = list(sp_.train.values())
    val_pairs = sp_.pairs("valid")[:VAL_N]
    test_pairs = sp_.pairs("test")[:TEST_N]

    content = torch.load(os.path.join(dd, "content_feature.pt"), map_location="cpu", weights_only=False).detach().float().numpy()
    cat = MiniBatchKMeans(n_clusters=400, random_state=0, n_init=3, batch_size=4096).fit(content).labels_
    print(f"{DS}: n={n} train={len(train)} val={len(val_pairs)} test={len(test_pairs)} | load {time.time()-t0:.0f}s", flush=True)

    cfg = BundleFuseConfig(n_items=n, families=["collaborative", "content", "taxonomy", "popularity"])
    model = BundleFuse(cfg)
    model.fit(train, val_pairs, content=content, cat_labels=cat)
    print(f"  fitted in {time.time()-t0:.0f}s | active signals = {model.signals_}", flush=True)
    print(f"  selected weights = { {k: round(v,2) for k,v in model.weights_.items()} }  kappa_f={model.kappa_f_:.3g}", flush=True)

    def evaluate(m, pairs, chunk=200):
        agg = np.zeros(2); tot = 0
        for off in range(0, len(pairs), chunk):
            ch = pairs[off:off + chunk]
            scores = np.stack([m.score(s) for s, _ in ch], axis=1)
            agg += np.asarray(ranking_metrics(scores, ch, (20,))[20]) * len(ch); tot += len(ch)
        return agg / tot

    r, nd = evaluate(model, test_pairs)
    print(f"\n  TEST ({len(test_pairs)} queries): Recall@20={r:.4f}  NDCG@20={nd:.4f}", flush=True)

    # checkpoint round-trip
    ckpt = os.path.join(os.environ.get("SCRATCH", "/tmp"), f"bundlefuse_{DS}.pt")
    model.save(ckpt)
    sz = os.path.getsize(ckpt) / 1e6
    model2 = BundleFuse.load(ckpt)
    same = all(model.predict(s) == model2.predict(s) for s, _ in test_pairs[:50])
    print(f"  checkpoint: saved {sz:.1f}MB -> reloaded; predict identical on 50 queries: {same}", flush=True)
    print(f"  one predict() example seeds={test_pairs[0][0]} -> top5 {model.predict(test_pairs[0][0], k=5)}", flush=True)
    print(f"  (done in {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
