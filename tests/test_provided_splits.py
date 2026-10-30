import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.provided_splits import ProvidedSplits
from bundlefuse.eval.provided import DiskSignals
from bundlefuse.fusion_selection import ValidationSignals, fit_fusion, fusion_parts, apply_fusion
from bundlefuse.operators import EmpiricalBayesRouter


def fixture_files(root):
    (root / 'count.json').write_text(json.dumps({'#I': 60}))
    # Identical bundles with different IDs must remain separate training examples.
    for name, contents in {'train': '8, 0, 1\n9, 0, 1\n', 'valid_input': '4, 1\n2, 0\n',
                           'valid_gt': '2, 2\n4, 3\n', 'test_input': '11, 1\n',
                           'test_gt': '11, 2\n'}.items():
        (root / f'bi_{name}.txt').write_text(contents)


def test_provided_split_preserves_duplicate_bundles_ids_and_original_masks(tmp_path):
    fixture_files(tmp_path)
    split = ProvidedSplits(tmp_path, 60)
    assert split.train == {8: [0, 1], 9: [0, 1]}
    assert split.pairs('valid') == [([0], [2]), ([1], [3])]
    assert split.manifest()['test_queries'] == 1
    assert split.manifest()['query_cap'] is None
    (tmp_path / 'bi_full.txt').write_text('0, 50, 51\n')
    assert ProvidedSplits(tmp_path, 60).train == split.train


@pytest.mark.parametrize('name,text', [('bi_test_gt.txt', '11, 1\n'),
                                        ('bi_test_input.txt', '8, 1\n'),
                                        ('bi_train.txt', '8, 0, 0\n'),
                                        ('bi_train.txt', '8, 60\n')])
def test_provided_invalid_protocol_fails_instead_of_silently_filtering(tmp_path, name, text):
    fixture_files(tmp_path)
    (tmp_path / name).write_text(text)
    with pytest.raises(ValueError):
        ProvidedSplits(tmp_path, 60)


def test_disk_and_dense_selection_use_all_queries_and_agree(tmp_path):
    torch.manual_seed(13)
    keys = ['simp', 'diff2', 'cat', 'bridge', 'pair', 'diff2_pair', 'rao']
    signals = {key: torch.randn(60, 7) for key in keys}
    pairs = [([j], [(j + 12) % 60, (j + 16) % 60]) for j in range(7)]
    freq = np.arange(60, dtype=np.float32) % 6
    dense = ValidationSignals(lambda: iter([(0, signals)]), pairs, freq)
    def scorer(bundles):
        ids = [s[0] for s in bundles]
        return {}, {key: values[:, ids] for key, values in signals.items()}
    cache = DiskSignals(tmp_path / 'cache', pairs, scorer, keys, 60, 3)
    disk = ValidationSignals(cache.blocks, pairs, freq)
    try:
        w = {'simp': 1.0, 'cat': .5, 'bridge': .25}
        ho, priors, eb = (.5, .2), ['cat', 'bridge'], (1.0, .8)
        assert disk.evaluate(w, ho, eb, priors) == pytest.approx(dense.evaluate(w, ho, eb, priors))
        assert disk.prior_scale(w, ho, priors) == pytest.approx(dense.prior_scale(w, ho, priors))
        observed, prior = fusion_parts(signals, w, priors, ho)
        expected = EmpiricalBayesRouter.fit_variance_components(freq, observed, prior * eb[1])
        actual = disk.variance_components(w, ho, priors, eb[1])
        for key in expected:
            assert actual[key] == pytest.approx(expected[key], abs=1e-6)
        chosen_dense = fit_fusion(dense, keys[:4], ['cat', 'bridge'])
        chosen_disk = fit_fusion(disk, keys[:4], ['cat', 'bridge'])
        assert chosen_disk[:2] == chosen_dense[:2]
        np.testing.assert_allclose(chosen_disk[2:4], chosen_dense[2:4], rtol=1e-6)
        w, priors, eb, ho, _ = chosen_dense
        expected_scores = apply_fusion(signals, w, priors, eb, torch.as_tensor(freq)[:, None], ho)
        rebuilt = torch.cat([apply_fusion(z, w, priors, eb, torch.as_tensor(freq)[:, None], ho)
                             for _, z in cache.blocks()], dim=1)
        torch.testing.assert_close(rebuilt, expected_scores)
    finally:
        cache.close()
    assert not (tmp_path / 'cache').exists()
