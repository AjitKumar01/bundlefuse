"""Regression tests for the ranking, baseline and experiment failures found in review."""
import itertools
import math
import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bundlefuse.metrics import top_items, ranking_metrics, full_ndcg, mean_reciprocal_rank
from bundlefuse.selection import sel_score
from bundlefuse.baselines.randomwalk import p3alpha_W, rp3beta_W
from bundlefuse.baselines.bundleknn import bundleknn_scores
from bundlefuse.baselines.base import topk_per_column, evaluate_scorer
from bundlefuse.experiment_io import run_checked, parse_matrix
from bundlefuse.artifacts import write_json


def test_all_tied_metrics_match_an_actual_ranking():
    scores = np.zeros((5, 1), dtype=np.float32)
    pairs = [([0], [1, 2])]
    assert full_ndcg(scores, pairs) == 1.0
    assert mean_reciprocal_rank(scores, pairs) == 1.0
    assert ranking_metrics(scores, pairs, Ks=(1, 2))[1] == (0.5, 1.0)
    # MRR is first-relevant, not the average over both positives.
    scores = np.array([[9], [3], [2], [1], [0]], dtype=np.float32)
    assert mean_reciprocal_rank(scores, [([0], [2, 4])]) == 0.5


def test_tied_topk_and_full_ndcg_agree_with_brute_force():
    for values in itertools.product((0.0, 1.0), repeat=5):
        column = np.array(values)
        expected = sorted(range(1, 5), key=lambda i: (-column[i], i))
        assert top_items(column, [0], 3).tolist() == expected[:3]
        dcg = sum(1 / math.log2(expected.index(i) + 2) for i in (2, 4))
        ideal = 1 + 1 / math.log2(3)
        assert full_ndcg(column[:, None], [([0], [2, 4])]) == pytest.approx(dcg / ideal)
        assert 0 <= full_ndcg(column[:, None], [([0], [2, 4])]) <= 1


def test_metrics_do_not_depend_on_rng_batching_or_tensor_backend():
    scores = np.zeros((60, 3), dtype=np.float32)
    pairs = [([0], [1, 2]), ([1], [4]), ([2], [5, 6])]
    before = scores.copy()
    np.random.seed(1); torch.manual_seed(1)
    whole = ranking_metrics(scores, pairs)
    np.random.seed(99); torch.manual_seed(99)
    assert ranking_metrics(torch.tensor(scores), pairs) == whole
    pieces = [ranking_metrics(scores[:, j:j + 1], pairs[j:j + 1]) for j in range(3)]
    for k in whole:
        np.testing.assert_allclose(whole[k], np.mean([piece[k] for piece in pieces], axis=0))
    np.testing.assert_array_equal(scores, before)


def test_powered_random_walk_adds_individually_powered_paths():
    # Item 0 reaches item 1 through bundles of sizes 3 and 2; both frequencies are 2.
    B = sp.csr_matrix(np.array([[1, 1, 1], [1, 1, 0]], dtype=np.float32))
    freq = np.asarray(B.sum(0)).ravel()
    W = p3alpha_W(B, freq, alpha=0.5, topk=0).toarray()
    expected = math.sqrt(1 / 2) * (math.sqrt(1 / 3) + math.sqrt(1 / 2))
    assert W[1, 0] == pytest.approx(expected)
    assert not np.isclose(W[1, 0], math.sqrt(1 / 6 + 1 / 4))
    assert not np.any(W.diagonal())
    np.testing.assert_allclose(rp3beta_W(B, freq, .5, 0, 0).toarray(), W)
    np.testing.assert_allclose(rp3beta_W(B, freq, .5, 1, 0).toarray(), W / freq[:, None])


def test_bundleknn_keeps_exactly_one_of_tied_neighbours():
    B = sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 1]], dtype=np.float32))
    indicator = sp.csr_matrix(np.array([[1], [0], [0]], dtype=np.float32))
    actual = bundleknn_scores(B, indicator, topn=1)[:, 0]
    np.testing.assert_allclose(actual, [1 / math.sqrt(2), 1 / math.sqrt(2), 0])


def test_sparse_topk_uses_stable_item_ids_at_the_boundary():
    W = sp.csr_matrix(np.ones((5, 2), dtype=np.float32))
    actual = topk_per_column(W, 2).toarray()
    np.testing.assert_array_equal(actual[:2], 1)
    assert not actual[2:].any()


def test_unknown_objective_and_invalid_queries_fail():
    with pytest.raises(ValueError, match="Unknown validation"):
        sel_score(np.zeros((3, 1)), [([0], [1])], "typo", ranking_metrics)
    with pytest.raises(ValueError, match="disjoint"):
        ranking_metrics(np.zeros((3, 1)), [([0], [0])])


def test_failed_or_partial_experiment_cannot_become_a_table():
    with pytest.raises(RuntimeError, match="exit 7"):
        run_checked([sys.executable, "-c", "print('Ours 0.1 0.1 0.2 0.2'); raise SystemExit(7)"])
    with pytest.raises(ValueError, match="Missing experiment rows"):
        parse_matrix("Ours 0.1 0.1 0.2 0.2")


def test_nonfinite_artifact_is_rejected(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"R@20": 0.5})
    with pytest.raises(ValueError):
        write_json(path, {"R@20": float("nan")})
    assert '0.5' in path.read_text()


def test_chunk_aggregation_weights_the_short_final_chunk():
    scores = np.arange(60 * 3).reshape(60, 3).astype(float)
    pairs = [([0], [59]), ([1], [58]), ([2], [3])]
    scorer = lambda seeds: scores[:, [s[0] for s in seeds]]
    actual = evaluate_scorer(scorer, ranking_metrics, pairs, chunk_size=2)
    expected = ranking_metrics(scores, pairs)
    for k in expected:
        np.testing.assert_allclose(actual[k], expected[k])


def test_hybrid_can_recover_unexposed_items_and_never_selects_on_test_labels():
    from bundlefuse.operators import SimplicialGraphOperator
    from bundlefuse.baselines.hybrid import SimpleHybrid
    graph = SimplicialGraphOperator([[0, 1], [0, 2], [3, 4]], n_items=60, svd_dim=2)
    content = np.zeros((60, 2), dtype=np.float32)
    content[[0, 50, 51], 0] = 1.0
    content[[1, 2, 3, 4], 1] = 1.0
    hybrid = SimpleHybrid(graph, content, lambda seeds: np.zeros((60, len(seeds)), dtype=np.float32))
    assert graph.freq[50] == graph.freq[51] == 0
    first = hybrid.select(ranking_metrics, [([0], [50])], [([0], [51])])
    changed_test = hybrid.select(ranking_metrics, [([0], [50])], [([0], [59])])
    assert first['SimpleHybrid']['test'][20][0] == 1.0
    for name in first:
        assert first[name]['hp'] == changed_test[name]['hp']


def test_hybrid_has_access_to_auxiliary_evidence_for_cold_candidates():
    from bundlefuse.operators import SimplicialGraphOperator
    from bundlefuse.baselines.hybrid import SimpleHybrid
    graph = SimplicialGraphOperator([[0, 1], [1, 2]], n_items=60, svd_dim=2)
    aux = sp.csr_matrix(([1.0], ([50], [0])), shape=(60, 60))
    hybrid = SimpleHybrid(graph, np.zeros((60, 2)), lambda seeds: np.zeros((60, len(seeds))), {'aux': aux})
    signals = hybrid.signals([[0]])
    assert graph.freq[50] == 0 and signals['aux'][50, 0] > signals['aux'][1, 0]
