import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.baselines.hybrid import SimpleHybrid
from bundlefuse.eval.components import CONTROL_KEYS, METHODS, apply_method, select_controls, query_groups, selected_metrics
from bundlefuse.eval.provided import DiskSignals
from bundlefuse.fusion_selection import ValidationSignals, fit_fusion
from bundlefuse.metrics import ranking_metrics
from bundlefuse.paired_stats import paired_comparison, holm_adjust
from bundlefuse.provided_splits import ProvidedSplits


def validation_fixture():
    torch.manual_seed(17)
    keys = ['simp', 'diff2', 'cat', 'bridge', 'pair', 'diff2_pair', 'rao']
    z = {key: torch.randn(60, 7) for key in keys}
    pairs = [([j], [j + 12, j + 16]) for j in range(7)]
    freq = np.arange(60, dtype=np.float32) % 6
    return keys, z, pairs, freq


def test_batched_candidate_evaluation_scans_blocks_once_and_matches_sequential():
    keys, z, pairs, freq = validation_fixture()
    passes = []
    def blocks():
        passes.append(True)
        for start in range(0, len(pairs), 3):
            yield start, {key: value[:, start:start + 3] for key, value in z.items()}
    candidates = [({'simp': 1.0}, (0.0, 0.0), (0.0, 1.0), ()),
                  ({'cat': .5, 'simp': .25}, (.5, .2), (1.0, .8), ('cat',))]
    batch = ValidationSignals(blocks, pairs, freq)
    actual = batch.evaluate_many(candidates)
    assert len(passes) == 1
    dense = ValidationSignals(lambda: iter([(0, z)]), pairs, freq)
    expected = [dense.evaluate(*candidate) for candidate in candidates]
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert batch.evaluate_many(candidates) == actual
    assert len(passes) == 1


@pytest.mark.parametrize('constraints,removed', [({'incidence': (0.0,)}, 'incidence'),
                                                 ({'rao': (0.0,)}, 'rao'),
                                                 ({'shrinkage': False}, 'shrinkage'),
                                                 ({}, 'bridge')])
def test_ablations_cannot_reenable_removed_components(constraints, removed):
    keys, z, pairs, freq = validation_fixture()
    available = [key for key in keys[:4] if removed != 'bridge' or key != 'bridge']
    validation = ValidationSignals(lambda: iter([(0, z)]), pairs, freq)
    w, prior, eb, ho, _ = fit_fusion(validation, available, ['cat', 'bridge'], **constraints)
    if removed == 'incidence':
        assert ho[0] == 0
    if removed == 'rao':
        assert ho[1] == 0
    if removed == 'shrinkage':
        assert eb[0] == 0
    if removed == 'bridge':
        assert 'bridge' not in w and 'bridge' not in prior


def test_chunked_hybrid_selection_matches_existing_control(tmp_path):
    _, z, pairs, freq = validation_fixture()
    raw = {name: z[key].numpy() for name, key in zip(('cooc', 'content', 'category', 'popularity'),
                                                    ('pair', 'diff2_pair', 'cat', 'bridge'))}
    class Graph:
        pass
    graph = Graph(); graph.freq = freq
    hybrid = SimpleHybrid(graph, None, None)
    hybrid.signals = lambda seeds: {key: value[:, [s[0] for s in seeds]] for key, value in raw.items()}
    expected = hybrid.select(ranking_metrics, pairs, pairs)
    all_signals = dict(z, **{'raw_' + key: torch.as_tensor(value) for key, value in raw.items()})
    def scorer(seeds):
        return {}, {key: value[:, [s[0] for s in seeds]] for key, value in all_signals.items()}
    cache = DiskSignals(tmp_path / 'signals', pairs, scorer, list(all_signals), 60, 3)
    try:
        validation = ValidationSignals(cache.blocks, pairs, freq)
        selected = select_controls(validation, cache, torch.as_tensor(freq)[:, None])
        actual, per_query = selected_metrics(cache.blocks, pairs, selected, torch.as_tensor(freq)[:, None], True)
        for method in ('SimpleHybrid', 'HardSwitchHybrid'):
            assert len(per_query[method]) == len(pairs)
            assert actual[method]['R@20'] == pytest.approx(expected[method]['test'][20][0])
            assert actual[method]['N@20'] == pytest.approx(expected[method]['test'][20][1])
            assert selected[method]['val_ndcg20'] == pytest.approx(expected[method]['val_ndcg20'])
        assert selected['HardSwitchHybrid']['gate'] == expected['HardSwitchHybrid']['hp']['gate']
    finally:
        cache.close()


def test_query_clusters_ignore_masks_and_item_order_but_preserve_rows():
    assert query_groups([([0], [1, 2]), ([2, 0], [1]), ([0], [3])]) == [0, 0, 1]


def test_paired_resampling_is_deterministic_handles_null_and_weights_duplicate_rows():
    kwargs = {'bootstrap': 100, 'permutations': 200, 'seed': 9}
    first = [.1, .1, .9]; second = [0, 0, 0]; groups = [0, 0, 1]
    result = paired_comparison(first, second, groups, **kwargs)
    assert result == paired_comparison(first, second, groups, **kwargs)
    assert result['delta'] == pytest.approx(1.1 / 3)  # Query mean, not (.1 + .9) / 2.
    assert result['clusters'] == 2 and result['queries'] == 3
    null = paired_comparison(first, first, groups, **kwargs)
    assert null['delta'] == 0 and null['ci95'] == [0, 0] and null['p_value'] == 1
    with pytest.raises(ValueError):
        paired_comparison([1], [], [])


def test_holm_correction_uses_the_prespecified_family():
    records = [{'p_value': .001}, {'p_value': .03}, {'p_value': .5}]
    holm_adjust(records, planned_tests=4)
    assert [r['p_holm'] for r in records] == pytest.approx([.004, .09, 1])
    assert [r['reject_05'] for r in records] == [True, False, False]


def test_complete_component_pipeline_and_report_keep_all_methods_and_query_ids(tmp_path, monkeypatch):
    import json
    from bundlefuse.eval.components import run_components
    from bundlefuse import component_report
    (tmp_path / 'count.json').write_text(json.dumps({'#I': 60}))
    rows = {'train': '8, 0, 1\n9, 2, 3\n', 'valid_input': '0, 0\n1, 1\n',
            'valid_gt': '0, 12\n1, 13\n', 'test_input': '2, 2\n3, 3\n4, 4\n',
            'test_gt': '2, 14\n3, 15\n4, 16\n'}
    for name, text in rows.items():
        (tmp_path / f'bi_{name}.txt').write_text(text)
    split = ProvidedSplits(tmp_path, 60)
    keys, z, _, freq = validation_fixture()
    def scorer(seeds):
        return {}, {key: value[:, [s[0] for s in seeds]] for key, value in z.items()}
    class Hybrid:
        def signals(self, seeds):
            return {name: z[key][:, [s[0] for s in seeds]].numpy() for name, key in
                    zip(('cooc', 'content', 'category', 'popularity'), ('pair', 'diff2_pair', 'cat', 'bridge'))}
    monkeypatch.setenv('PROVIDED_CHUNK', '2')
    output = tmp_path / 'output'
    run_components('pog', split, scorer, keys[:4], freq, False, 'ndcg20', output / 'pog', Hybrid())
    result = json.loads((output / 'pog' / 'result.json').read_text())
    assert set(result['methods']) == set(METHODS)
    assert result['test_query_ids'] == [2, 3, 4]
    assert all(len(rows) == 3 for rows in result['per_query'].values())
    assert not (output / 'pog' / 'validation_cache').exists()
    monkeypatch.setattr(component_report, 'PROJECT_ROOT', str(tmp_path))
    (tmp_path / 'README.md').write_text('# Test\n')
    component_report.component_report({'jobs': [{'dataset': 'pog', 'status': 'complete'}]}, output)
    stats = json.loads((output / 'paired.json').read_text())
    assert stats['planned_tests'] == 14
    assert set(stats['datasets']['pog']) == set(METHODS) - {'BundleFuse'}
    assert '1/1 datasets complete' in (tmp_path / 'README.md').read_text()
