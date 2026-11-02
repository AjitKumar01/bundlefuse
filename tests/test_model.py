"""Self-contained tests for the BundleFuse model class (no external data)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bundlefuse.model import BundleFuse, BundleFuseConfig
from bundlefuse import protocol


def _synthetic(n=60, n_bundles=800, seed=0):
    """Random bundles with planted co-occurrence structure (themes of nearby item ids)."""
    rng = np.random.default_rng(seed)
    bundles = []
    for _ in range(n_bundles):
        c = int(rng.integers(0, n - 6)); size = int(rng.integers(4, 7))
        b = sorted(set(int(c + rng.integers(0, 6)) for _ in range(size)))
        if len(b) >= 3:
            bundles.append(b)
    ntr, nva = int(0.7 * len(bundles)), int(0.1 * len(bundles))
    return n, bundles, ntr, nva


def test_fit_predict_and_checkpoint():
    n, bundles, ntr, nva = _synthetic()
    tr, va, te = bundles[:ntr], bundles[ntr:ntr + nva], bundles[ntr + nva:]
    val_pairs = protocol.mask(va, 1, 0.5)
    test_pairs = protocol.mask(te, 2, 0.5)
    content = np.random.default_rng(1).standard_normal((n, 16)).astype(np.float32)
    cat = (np.arange(n) // 6).astype(int)

    model = BundleFuse(BundleFuseConfig(n_items=n, svd_dim=8,
                                        families=["collaborative", "content", "taxonomy", "popularity"]))
    model.fit(tr, val_pairs, features={"content": content}, cat_labels=cat)
    assert model.fitted_ and model.SIG, "model did not fit"
    assert any(model.w.values()), "no signals selected"

    seeds = test_pairs[0][0]
    top = model.predict(seeds, k=10)
    assert len(top) == 10
    assert not (set(top) & set(seeds)), "seeds must be excluded from predictions"
    assert len(set(top)) == 10, "predictions must be distinct"

    # checkpoint round-trips to identical predictions
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        model.save(p)
        m2 = BundleFuse.load(p)
        for s, _ in test_pairs[:20]:
            assert model.predict(s) == m2.predict(s), "reloaded model predicts differently"
    print("test_fit_predict_and_checkpoint OK")


def test_family_switch():
    """Disabling a family removes its signals from the active menu."""
    n, bundles, ntr, nva = _synthetic()
    val_pairs = protocol.mask(bundles[ntr:ntr + nva], 1, 0.5)
    m = BundleFuse(BundleFuseConfig(n_items=n, svd_dim=8, families=["collaborative", "popularity"]))
    m.fit(bundles[:ntr], val_pairs)                       # no content/cat passed
    assert "style" not in m.SIG and "cat" not in m.SIG, m.SIG
    assert "cooc" in m.SIG and "pop" in m.SIG, m.SIG
    print("test_family_switch OK")


if __name__ == "__main__":
    test_fit_predict_and_checkpoint()
    test_family_switch()
    print("all tests passed")
