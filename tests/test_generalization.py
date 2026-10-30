import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.baselines.metadata_ranker import MetadataRanker, training_batch, train_ranker
from bundlefuse.robust_protocol import CASES, sparse_pairs
from bundlefuse.temporal_data import load_temporal
from bundlefuse.profiling import profile_scorers


def test_sparse_queries_keep_membership_move_dropped_seeds_to_truth_and_are_nested():
    pairs = [([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]), ([0], [1])]
    cases = {case: sparse_pairs(pairs, [7, 8], case) for case in CASES}
    assert cases['supplied'] == pairs
    for rows in cases.values():
        for (seeds, truth), (old_seeds, old_truth) in zip(rows, pairs):
            assert seeds and truth and not set(seeds).intersection(truth)
            assert set(seeds) | set(truth) == set(old_seeds) | set(old_truth)
        assert rows[1] == pairs[1]
    assert set(cases['one_seed'][0][0]) <= set(cases['quarter_seeds'][0][0]) <= set(cases['half_seeds'][0][0])
    assert cases['half_seeds'] == sparse_pairs(pairs, [7, 8], 'half_seeds')


def test_training_batches_exclude_every_known_positive_and_never_use_revealed_targets():
    seeds, mask, candidates = training_batch([[0, 1, 2, 3], [4, 5]], 30, np.random.default_rng(4), 32,
                                            exclusions=[list(range(10)), [4, 5, 10, 11]])
    for j, known in enumerate((set(range(10)), {4, 5, 10, 11})):
        observed = set(seeds[j][mask[j].bool()].tolist())
        assert candidates[j, 0].item() not in observed
        assert not set(candidates[j, 1:].tolist()).intersection(known)
    with pytest.raises(ValueError):
        training_batch([[0, 1]], 2, np.random.default_rng(0), 3)


@pytest.mark.parametrize('sparse', [False, True])
def test_learned_ranker_updates_parameters_has_zero_cold_id_effect_and_roundtrips(sparse):
    torch.set_num_threads(1); torch.manual_seed(3)
    content = np.random.default_rng(0).normal(size=(40, 12)).astype(np.float32)
    content = sp.csr_matrix(content) if sparse else content
    frequency = np.ones(40, np.float32); frequency[-1] = 0
    labels = [np.arange(40) % 5]
    model = MetadataRanker(content, labels, frequency, dim=8)
    optimizers = model.optimizers(.003, 1e-3)
    before = model.query_encoder[-1].weight.detach().clone()
    data = training_batch([[0, 1, 2], [4, 5, 6]], 40, np.random.default_rng(0), 8)
    loss = model.loss(*data, 1e-3); loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before, model.query_encoder[-1].weight)
    vector, bias, _ = model.item_vectors(torch.tensor([39]))
    with torch.no_grad():
        model.item_embedding.weight[39] += 100
    after, after_bias, _ = model.item_vectors(torch.tensor([39]))
    torch.testing.assert_close(after, vector); torch.testing.assert_close(after_bias, bias)
    catalogue = model.catalogue()
    score = model.score([[1, 2], [3]], catalogue)
    torch.testing.assert_close(score[:, :1], model.score([[2, 1]], catalogue))
    replacement = MetadataRanker(content, labels, frequency, dim=8)
    replacement.load_state_dict(model.state_dict())
    torch.testing.assert_close(score, replacement.score([[1, 2], [3]], replacement.catalogue()))
    assert 'content' not in model.state_dict()


def test_checkpoint_selection_uses_validation_and_restores_best_epoch(tmp_path):
    torch.set_num_threads(1)
    content = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    values = iter([.2, .5, .3])
    model, record = train_ranker(content, [np.arange(40) % 3], np.ones(40),
                                 [[0, 1, 2], [3, 4, 5]], lambda _: next(values),
                                 {'dim': 8, 'lr': .003, 'regularization': 1e-3, 'negatives': 4,
                                  'batch_size': 2, 'max_epochs': 3, 'evaluate_every': 1, 'patience': 2}, 0, tmp_path)
    assert record['best_epoch'] == 2 and record['val_ndcg20'] == .5
    assert not record['best_at_final_check']
    saved = torch.load(tmp_path / 'best.pt', weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[key])


def write_retail_fixture(root):
    with (root / 'product.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['PRODUCT_ID', 'DEPARTMENT', 'COMMODITY_DESC', 'SUB_COMMODITY_DESC', 'BRAND'])
        writer.writeheader()
        for item in range(8):
            writer.writerow(dict(PRODUCT_ID=item, DEPARTMENT='food', COMMODITY_DESC='goods',
                                 SUB_COMMODITY_DESC=f'category{item % 3}', BRAND='brand'))
    with (root / 'transaction_data.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['BASKET_ID', 'PRODUCT_ID', 'household_key', 'DAY'])
        writer.writeheader()
        for day in range(1, 11):
            for basket in range(2):
                # Item 6 appears only after the training cutoff; item 7 never has an interaction.
                members = [0, 1, 2 if day <= 7 else 6]
                for item in members:
                    writer.writerow(dict(BASKET_ID=day * 10 + basket, PRODUCT_ID=item,
                                         household_key=basket, DAY=day))


def test_temporal_folds_do_not_share_days_or_filter_future_cold_catalogue_items(tmp_path):
    write_retail_fixture(tmp_path)
    data = load_temporal(tmp_path, caps=(0, 0, 0))
    ranges = data['protocol']['day_ranges']
    assert ranges['train'][1] < ranges['valid'][0] <= ranges['valid'][1] < ranges['test'][0]
    assert len(data['categories']) == 8
    assert 6 not in {i for bundle in data['train'] for i in bundle}
    assert 6 in {i for seeds, truth in data['test'] for i in seeds + truth}
    assert len(data['train']) == 14 and len(data['test']) == 4  # No future-dependent deduplication.
    assert len(data['groups']) == len(data['test'])


def test_profile_records_scoring_and_metrics_separately_with_fixed_queries():
    pairs = [([i], [(i + 1) % 60]) for i in range(4)]
    report = profile_scorers({'example': lambda seeds: np.zeros((60, len(seeds)), np.float32)},
                             pairs, repeats=2, query_count=3)['example']
    assert report['queries'] == 3 and len(report['scoring_ms_per_query']) == 2
    assert report['median_scoring_plus_metrics_ms_per_query'] >= report['median_scoring_ms_per_query']


def test_retail_signal_adapter_matches_the_existing_retail_operators():
    import ast
    from bundlefuse.retail_signals import RetailSignals
    from bundlefuse.operators import SimplicialGraphOperator, zcols
    from bundlefuse.baselines.hybrid import SimpleHybrid, category_scores
    text = (Path(__file__).resolve().parents[1] / 'bundlefuse' / 'eval' / 'dunnhumby.py').read_text()
    functions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)
                 and node.name in ('build', 'score_all', 'make_hybrid')]
    rng = np.random.default_rng(4)
    content = rng.normal(size=(40, 8)).astype(np.float32)
    content /= np.linalg.norm(content, axis=1, keepdims=True)
    tfidf = sp.csr_matrix(np.abs(content))
    categories = np.arange(40) % 5
    train = [rng.choice(40, size=5, replace=False).tolist() for _ in range(30)]
    users = [j % 5 for j in range(30)]
    adapter = RetailSignals(content, tfidf, categories, 5).build(train, users)
    namespace = {'np': np, 'sp': sp, 'SimplicialGraphOperator': SimplicialGraphOperator,
                 'n': 40, 'nusers': 5, 'cat_lbl': categories, 'NCc': 5, 'CONTENT': content,
                 'tf_mat': tfidf, 'ENCODER': False, 'SIG': adapter.keys,
                 'zcols': zcols, 'SimpleHybrid': SimpleHybrid, 'category_scores': category_scores}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<retail-reference>', 'exec'), namespace)
    built = namespace['build'](train, users)
    pairs = [([0, 1], [2]), ([3], [4])]
    expected_raw, expected = namespace['score_all'](pairs, built)
    actual_raw, actual = adapter.score([s for s, _ in pairs])
    for key in adapter.keys + ['pair', 'diff2_pair', 'rao']:
        np.testing.assert_allclose(actual_raw[key], expected_raw[key], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, rtol=1e-6)
    old_hybrid = namespace['make_hybrid'](built).signals([s for s, _ in pairs])
    new_hybrid = adapter.hybrid().signals([s for s, _ in pairs])
    for key in old_hybrid:
        np.testing.assert_allclose(old_hybrid[key], new_hybrid[key], atol=1e-6, rtol=1e-6)


def test_learned_robustness_pipeline_and_report_end_to_end(tmp_path, monkeypatch):
    from bundlefuse.eval.generalization import run_generalization, METHODS
    from bundlefuse.eval.components import query_groups
    from bundlefuse import generalization_report
    torch.set_num_threads(1)
    rng = np.random.default_rng(5)
    keys = ['simp', 'diff2', 'cat', 'bridge', 'content', 'pair', 'diff2_pair', 'rao']
    signals = {key: torch.as_tensor(rng.normal(size=(60, 8)).astype(np.float32)) for key in keys}
    def scorer(seeds):
        return {}, {key: value[:, [s[0] for s in seeds]] for key, value in signals.items()}
    class Hybrid:
        def signals(self, seeds):
            return {key: signals[value][:, [s[0] for s in seeds]].numpy() for key, value in
                    zip(('cooc', 'content', 'category', 'popularity'), ('pair', 'content', 'cat', 'bridge'))}
    valid = [([0, 1], [12]), ([2], [13])]
    test = [([3, 4], [14]), ([5, 6, 7], [15]), ([0], [16])]
    grid = ({'dim': 8, 'lr': .003, 'regularization': 1e-3, 'negatives': 4, 'batch_size': 4,
             'max_epochs': 2, 'evaluate_every': 1, 'patience': 2},)
    root = tmp_path / 'study'; output = root / 'toy'
    run_generalization('toy', [[0, 1, 2], [3, 4, 5]], valid, test, [3, 4, 5], query_groups(test),
                       {'protocol': 'test'}, scorer, keys[:5], np.ones(60, np.float32), Hybrid(),
                       rng.normal(size=(60, 8)).astype(np.float32), [np.arange(60) % 4], output,
                       chunk=2, grid=grid)
    result = json.loads((output / 'result.json').read_text())
    assert set(result['cases']) == set(CASES)
    for case in CASES:
        rows = json.loads((output / result['cases'][case]['per_query_file']).read_text())
        assert set(rows['per_query']) == set(METHODS)
        assert all(len(value) == len(test) for value in rows['per_query'].values())
    assert not (output / 'validation_cache').exists()
    assert result['cases']['one_seed']['mean_revealed'] == 1
    monkeypatch.setattr(generalization_report, 'PROJECT_ROOT', str(tmp_path))
    (tmp_path / 'README.md').write_text('# Test\n')
    generalization_report.report_generalization({'jobs': [{'dataset': 'toy', 'status': 'complete'}]}, root)
    assert json.loads((root / 'paired.json').read_text())['planned_tests'] == 40
    assert '1/1 dataset jobs complete' in (tmp_path / 'README.md').read_text()
