"""Mandatory data-independent reference recorded from 15bd250 before the pipeline fixes.

External benchmark results are not fabricated or regenerated to make this test pass.
Use validate.py with an explicit protocol for dataset-level integration measurements.
"""
import json
from pathlib import Path
import numpy as np
from bundlefuse.model import BundleFuse, BundleFuseConfig
from bundlefuse import protocol
from bundlefuse.metrics import ranking_metrics
from test_model import _synthetic


def test_pre_fix_full_scorer_golden():
    golden = json.loads((Path(__file__).parent / 'fixtures/model_golden.json').read_text())
    n, bundles, ntr, nva = _synthetic()
    valid = protocol.mask(bundles[ntr:ntr+nva], 1, .5)
    test = protocol.mask(bundles[ntr+nva:], 2, .5)
    content = np.random.default_rng(1).standard_normal((n,16)).astype(np.float32)
    model = BundleFuse(BundleFuseConfig(n_items=n, svd_dim=8,
                       families=['collaborative','content','taxonomy','popularity']))
    model.fit(bundles[:ntr], valid, features={'content':content}, cat_labels=np.arange(n)//6)
    assert model.w == golden['weights']
    np.testing.assert_allclose(model.best_eb, golden['backoff'], atol=1e-12, rtol=0)
    assert list(model.best_ho) == golden['higher_order']
    scores = np.stack([model.score(s) for s,_ in test], 1)
    np.testing.assert_allclose(scores[:,:3], golden['first_three_scores'], atol=1e-6, rtol=0)
    for k, values in ranking_metrics(scores,test,(10,20,50)).items():
        np.testing.assert_allclose(values, golden['metrics'][str(k)], atol=1e-12, rtol=0)
