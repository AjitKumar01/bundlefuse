import ast
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.baselines.compact_ranker import CompactRanker
from bundlefuse.baselines.metadata_ranker import training_batch
from bundlefuse.compact_features import build_features
from bundlefuse.fusion_selection import apply_fusion, required_fusion_signals
from bundlefuse.operators import SimplicialGraphOperator, DualSpectrumOntology, zcols
from bundlefuse.selected_signals import SelectedSignals


def fixture(music):
    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    n = 60
    train = [rng.choice(n, size=5, replace=False).tolist() for _ in range(45)]
    categories = np.arange(n) % 6
    albums = np.arange(n) % 11 if music else None
    content = rng.normal(size=(n, 9)).astype(np.float32)
    graph = SimplicialGraphOperator(train, n, svd_dim=8)
    onto = DualSpectrumOntology(train, categories, n, alb_lbl=albums,
                                cluster_feat=None if music else content, n_soft_clusters=4)
    names = ['csvd', 'bridge', 'audiostyle'] if music else ['csvd', 'bridge', 'style', 'imgstyle']
    features = {key: torch.as_tensor(rng.normal(size=(n, 8)).astype(np.float32)) for key in names}
    keys = ['simp', 'diff2', 'pop', 'cooc', 'ppmi', 'cat'] + (['homo'] if music else ['softcat']) + names
    namespace = dict(np=np, torch=torch, graph_op=graph, onto_op=onto, n=n, freq=graph.freq,
                     cat_lbl=categories, rao_dense_feat=content, MUSIC=music, DEVICE=torch.device('cpu'),
                     ENCODER=False, FEAT_T=features, SIG=keys, zcols=zcols)
    text = (Path(__file__).resolve().parents[1] / 'bundlefuse/eval/fullrank.py').read_text()
    function = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)
                    and node.name == 'score_all_signals')
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<legacy-fullrank>', 'exec'), namespace)
    scorer = SelectedSignals(graph, onto, features, categories, content, music, torch.device('cpu'))
    return scorer, namespace['score_all_signals'], keys, train, content


@pytest.mark.parametrize('music', [False, True])
def test_pruned_signals_and_fused_scores_are_bit_identical_to_independent_legacy(music):
    scorer, reference, keys, _, _ = fixture(music)
    bundles = [[0, 1, 3], [2], [4, 5], [0, 1, 3]]
    raw, z = reference(bundles)
    selected_raw, selected_z = scorer.score(bundles, keys + ['pair', 'diff2_pair', 'rao', 'delta_inc'])
    for key in selected_raw:
        assert np.array_equal(selected_raw[key].numpy().view(np.uint32), raw[key].numpy().view(np.uint32)), key
    for key in selected_z:
        expected = zcols(raw[key])
        assert np.array_equal(selected_z[key].numpy().view(np.uint32), expected.numpy().view(np.uint32)), key
    for alpha in (0., .5, 1.):
        for gamma in (0., .2):
            for kappa in (0., .5):
                w, prior, eb, ho = {'cat': .5, 'simp': .25, 'pop': .25}, ['bridge'], (kappa, 1.3), (alpha, gamma)
                required = required_fusion_signals(w, prior, eb, ho)
                _, pruned = scorer.score(bundles, required)
                expected = apply_fusion(z, w, prior, eb, torch.as_tensor(scorer.graph.freq)[:, None], ho)
                actual = apply_fusion(pruned, w, prior, eb, torch.as_tensor(scorer.graph.freq)[:, None], ho)
                assert np.array_equal(expected.numpy().view(np.uint32), actual.numpy().view(np.uint32))
                # Independent pre-optimization fusion expression, including the original +0*Rao.
                from bundlefuse.operators import EmpiricalBayesRouter
                pair = dict(z, simp=z['pair'], diff2=z['diff2_pair'])
                observed = sum(w[k] * pair[k] for k in w)
                if alpha:
                    observed = (1 - alpha) * observed + alpha * sum(w[k] * z[k] for k in w)
                if gamma:
                    observed = observed + gamma * z['rao']
                if kappa:
                    mu = sum(w.get(k, .5) * z[k] for k in prior) + gamma * z['rao'] + w['pop'] * z['pop']
                    observed = EmpiricalBayesRouter.conjugate_posterior(
                        observed, mu * eb[1], torch.as_tensor(scorer.graph.freq)[:, None], kappa)
                assert np.array_equal(observed.numpy().view(np.uint32), actual.numpy().view(np.uint32))


def test_unused_rao_diffusion_and_incidence_are_never_evaluated(monkeypatch):
    scorer, _, _, _, _ = fixture(False)
    def fail(*args, **kwargs):
        raise AssertionError('Unused expensive operator evaluated')
    monkeypatch.setattr(scorer.graph, 'propagate', fail)
    monkeypatch.setattr(scorer.graph, 'score_rao_concordance', fail)
    monkeypatch.setattr(scorer.graph, 'score_simplicial_incidence_only', fail)
    scorer.score([[0, 1]], {'pair', 'cat'})
    with pytest.raises(ValueError):
        scorer.score([[0]], {'not_a_signal'})


def test_feature_packs_share_raw_columns_and_are_deterministic():
    scorer, _, _, _, content = fixture(False)
    args = (content, [scorer.categories], scorer.graph.freq, scorer.graph, scorer.ontology,
            {key: value.numpy() for key, value in scorer.features.items()}, False)
    packs, record = build_features(*args)
    repeated, other = build_features(*args)
    assert packs['Raw'].shape == packs['Engineered'].shape == (60, 256)
    np.testing.assert_array_equal(packs['Raw'][:, :97], packs['Engineered'][:, :97])
    assert packs['Raw'][:, 97:].any() and packs['Engineered'][:, 97:].any()
    assert not np.array_equal(packs['Raw'][:, 97:], packs['Engineered'][:, 97:])
    assert record['sha256'] == other['sha256']
    for key in packs:
        np.testing.assert_array_equal(packs[key], repeated[key])


def test_compact_model_trains_uses_same_parameter_count_and_cached_scores_match():
    torch.set_num_threads(1); torch.manual_seed(1)
    features = np.random.default_rng(3).normal(size=(50, 256)).astype(np.float32)
    freq = np.arange(50, dtype=np.float32)
    model = CompactRanker(features, freq)
    assert sum(p.numel() for p in model.parameters()) == 58314
    assert not any(isinstance(layer, torch.nn.Embedding) for layer in model.modules())
    before = model.item_projection.weight.detach().clone()
    seeds, mask, candidates = training_batch([[0, 1, 2], [3, 4, 5, 6]], 50, np.random.default_rng(0), 8)
    loss = model.loss(seeds, mask, candidates, 1e-4)
    loss.backward(); model.optimizers(.003, 1e-4)[0].step()
    assert torch.isfinite(loss) and not torch.equal(before, model.item_projection.weight)
    catalogue = model.catalogue()
    torch.testing.assert_close(model.score([[0, 1], [3]], catalogue), model.score([[1, 0], [3]], catalogue))
    torch.testing.assert_close(model.score([[3]], catalogue), model.score([[0, 1], [3]], catalogue)[:, 1:])
    assert 'features' not in model.state_dict()
    raw = features.copy()
    other = CompactRanker(raw, freq)
    assert sum(p.numel() for p in other.parameters()) == 58314


def test_compact_pilot_runs_parity_training_frozen_test_and_live_report(tmp_path, monkeypatch):
    import json
    from bundlefuse.artifacts import write_json, flat_metrics
    from bundlefuse.eval.compact import run_compact
    from bundlefuse.eval.components import apply_method
    from bundlefuse.metrics import ranking_metrics
    from bundlefuse import compact_report
    scorer, reference, _, train, content = fixture(False)
    valid, test = [([0, 1], [12]), ([2], [13])], [([3, 4], [14]), ([5, 6], [15])]
    protocol = {'sha256': {'fixture': 'immutable'}, 'protocol': 'provided'}
    class Splits:
        splits = {'test': ({45: [3, 4], 46: [5, 6]}, {45: [14], 46: [15]})}
        def manifest(self):
            return protocol
        def pairs(self, split):
            return valid if split == 'valid' else test
    splits = Splits(); splits.train = dict(enumerate(train))
    selected = {'kind': 'fusion', 'weights': {'cat': .5, 'simp': .5}, 'prior_keys': ['bridge'],
                'kappa': .5, 'prior_scale': 1., 'alpha_inc': .5, 'gamma_rao': .2}
    def metrics(pairs):
        _, z = reference([s for s, _ in pairs])
        score = apply_method(z, selected, torch.as_tensor(scorer.graph.freq)[:, None])
        rows = [flat_metrics(ranking_metrics(score[:, j:j + 1], [pair], Ks=(10, 20, 50)))
                for j, pair in enumerate(pairs)]
        return {key: np.mean([row[key] for row in rows]) for key in rows[0]}, rows
    val_metrics, _ = metrics(valid); test_metrics, rows = metrics(test)
    # Avoid a zero denominator in this deliberately small random fixture.
    val_metrics['R@20'] = max(.01, val_metrics['R@20']); val_metrics['N@20'] = max(.01, val_metrics['N@20'])
    frozen = {'protocol': protocol, 'test_query_ids': [45, 46], 'selected': {'BundleFuse': selected},
              'per_query': {'BundleFuse': rows}, 'validation': {'BundleFuse': val_metrics},
              'methods': {'BundleFuse': test_metrics}, 'query_groups': [0, 1]}
    write_json(tmp_path / 'reference.json', frozen)
    def score(seeds, required=None):
        return reference(seeds) if required is None else scorer.score(seeds, required)
    grid = ({'dim': 8, 'lr': .003, 'regularization': 1e-4, 'negatives': 4, 'batch_size': 16,
             'max_epochs': 1, 'evaluate_every': 1, 'patience': 1},)
    output = tmp_path / 'pilot' / 'toy'
    run_compact('toy', splits, score, scorer.graph, scorer.ontology,
                {key: value.numpy() for key, value in scorer.features.items()}, content, scorer.categories,
                None, False, output, grid=grid, chunk=2, reference_path=tmp_path / 'reference.json')
    result = json.loads((output / 'result.json').read_text())
    assert len(result['methods']) == 7 and all(len(rows) == 2 for rows in result['per_query'].values())
    assert result['parity']['queries_verified'] == 2 and result['parity']['score_bit_identity']
    assert result['parity']['fused_score_sha256_reference'] == result['parity']['fused_score_sha256_optimized']
    assert all(record['stopped'] == 'epoch_budget' for record in result['selected']['selected'].values())
    monkeypatch.setattr(compact_report, 'PROJECT_ROOT', str(tmp_path))
    (tmp_path / 'README.md').write_text('# Test\n')
    compact_report.report_compact({'jobs': [{'dataset': 'toy', 'status': 'complete'}]}, output.parent)
    assert '1/1 dataset jobs complete' in (tmp_path / 'README.md').read_text()
    assert '| Dataset | Status |' in (tmp_path / 'README.md').read_text()
    assert json.loads((output.parent / 'paired.json').read_text())['planned_tests'] == 6
