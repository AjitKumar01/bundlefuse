"""Small mathematical checks for the score-sufficiency analysis.

Synthetic constructions only. No benchmark data, fitting, or checkpoint changes.
Run from the repository root: PYTHONPATH=. .venv/bin/python scripts/check_score_sufficiency.py
"""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from bundlefuse.fusion import apply_fusion
from bundlefuse.operators import (
    BilateralContentBridge,
    DualSpectrumOntology,
    SimplicialGraphOperator,
)
from bundlefuse.selected_signals import SelectedSignals


ROOT = Path(__file__).resolve().parents[1]


def corpus(q, shift):
    return [[a, q + b, 2*q + c, 3*q + (a+b+c+shift) % q]
            for a, b, c in itertools.product(range(q), repeat=3)]


def counts(bundles, order):
    return Counter(subset for bundle in bundles
                   for subset in itertools.combinations(bundle, order))


def score_corpus(q, shift):
    train = corpus(q, shift)
    labels = np.repeat(np.arange(4), q)
    descriptors = np.eye(4, dtype=np.float32)[labels]
    graph = SimplicialGraphOperator(train, 4*q, svd_dim=min(6, 4*q-2))
    ontology = DualSpectrumOntology(
        train, labels, 4*q, cluster_feat=descriptors, n_soft_clusters=4)
    style = BilateralContentBridge.disentangle(descriptors, labels, 4)
    bridge = BilateralContentBridge.build_dense_bridge(style, graph)
    features = {k: torch.tensor(v) for k, v in {
        'style': style, 'imgstyle': style, 'bridge': bridge,
        'csvd': graph.Ucsvd}.items()}
    provider = SelectedSignals(
        graph, ontology, features, labels, style, False, torch.device('cpu'))
    required = {'pair', 'simp', 'diff2_pair', 'diff2', 'cooc', 'ppmi',
                'csvd', 'cat', 'softcat', 'style', 'imgstyle', 'bridge',
                'pop', 'rao'}
    query = [0, q, 2*q]
    raw, standard = provider.score([query], required)
    recipe = json.loads((ROOT/'results/publication_mechanisms/pog_dense/full.json').read_text())
    final = apply_fusion(
        standard, recipe['weights'], recipe['prior'], recipe['backoff'],
        torch.tensor(graph.freq)[:, None], recipe['higher_order'])
    candidates = np.arange(3*q, 4*q)
    joint = [sum(set(query + [int(c)]).issubset(b) for b in train)
             for c in candidates]
    arrays = {k: v.detach().numpy() for k, v in raw.items()}
    arrays['final_dense_recipe'] = final.detach().numpy()
    return train, arrays, candidates, joint


def check_missing_four_item_relation():
    output = []
    for q in (2, 21):
        left, lscore, candidates, ljoint = score_corpus(q, 0)
        right, rscore, _, rjoint = score_corpus(q, 1)
        equal = {str(order): counts(left, order) == counts(right, order)
                 for order in (1, 2, 3)}
        assert all(equal.values())
        differences = {key: float(np.max(np.abs(lscore[key]-rscore[key])))
                       for key in lscore}
        ties = {key: float(np.ptp(lscore[key][candidates, 0])) for key in lscore}
        assert max(differences.values()) < 2e-5, differences
        assert max(ties.values()) < 2e-5, ties
        assert ljoint == [1] + [0]*(q-1)
        assert rjoint == [0, 1] + [0]*(q-2)
        output.append({
            'values_per_group': q, 'items': 4*q, 'bundles_per_corpus': len(left),
            'first_three_count_orders_equal': equal,
            'max_score_difference_between_corpora': differences,
            'candidate_score_ranges_in_first_corpus': ties,
            'query': [0, q, 2*q], 'candidates': candidates.tolist(),
            'four_item_counts_first_corpus': ljoint,
            'four_item_counts_second_corpus': rjoint,
            'note': 'Floating-point ties represent exact algebraic equality. '
                    'The q=21 construction has more tied completion candidates than cutoff 20.'})
    return output


def check_gaussian_information():
    rng = np.random.default_rng(4711)
    max_error = 0.0
    for _ in range(100):
        d = 4
        mat = rng.normal(size=(d+1, d+1))
        covariance = mat @ mat.T + np.eye(d+1)
        loading = rng.normal(size=d+1)
        old = covariance[:d, :d]
        cross = covariance[:d, d]
        old_precision = loading[:d] @ np.linalg.solve(old, loading[:d])
        residual_loading = loading[d] - cross @ np.linalg.solve(old, loading[:d])
        residual_variance = covariance[d, d] - cross @ np.linalg.solve(old, cross)
        increment = residual_loading**2 / residual_variance
        full = loading @ np.linalg.solve(covariance, loading)
        max_error = max(max_error, abs(full-old_precision-increment))
        np.testing.assert_allclose(full, old_precision+increment, atol=1e-11)
    return {'cases': 100, 'max_precision_identity_error': max_error}


def check_ranking_and_gain_bound():
    # Two equally probable additional observations, four eligible candidates,
    # binary hidden sets. The old representation cannot distinguish the states.
    hidden = list(itertools.chain.from_iterable(
        itertools.combinations(range(4), k) for k in range(1, 5)))
    permutations = list(itertools.permutations(range(4)))
    rng = np.random.default_rng(72)
    max_sort_error = 0.0
    largest_bound_ratio = 0.0
    for _ in range(100):
        probabilities = rng.dirichlet(np.ones(len(hidden)), size=2)
        for metric in ('recall', 'ndcg'):
            k = 2
            discounts = np.ones(k) if metric == 'recall' else 1/np.log2(np.arange(2, k+2))
            values = []
            for group in hidden:
                denominator = len(group) if metric == 'recall' else discounts[:min(k, len(group))].sum()
                values.append(np.array([int(c in group) for c in range(4)])/denominator)
            values = np.asarray(values)
            conditional = probabilities @ values
            coarse = conditional.mean(axis=0)
            utility = lambda eta, order: float(eta[list(order[:k])] @ discounts)
            old_order = np.argsort(-coarse, kind='stable')
            with_extra = 0.0
            without_extra = 0.0
            for state in range(2):
                direct = [sum(probabilities[state, j] * utility(values[j], order)
                              for j in range(len(hidden))) for order in permutations]
                best_order = np.argsort(-conditional[state], kind='stable')
                best = utility(conditional[state], best_order)
                max_sort_error = max(max_sort_error, abs(best-max(direct)))
                np.testing.assert_allclose(best, max(direct), atol=1e-12)
                with_extra += .5*best
                without_extra += .5*utility(conditional[state], old_order)
            delta = float(np.square(conditional-coarse).sum(axis=1).mean())
            bound = 2*np.linalg.norm(discounts)*np.sqrt(delta)
            gain = with_extra-without_extra
            assert -1e-12 <= gain <= bound+1e-12
            largest_bound_ratio = max(largest_bound_ratio, gain/bound if bound else 0)
    return {'cases_per_metric': 100, 'permutations_per_case': len(permutations),
            'max_optimal_sort_error': max_sort_error,
            'max_actual_gain_divided_by_bound': largest_bound_ratio}


def main():
    result = {
        'scope': 'Synthetic algebra checks; not a new benchmark or a sufficiency certificate.',
        'four_item_counterexample': check_missing_four_item_relation(),
        'correlated_measurement_identity': check_gaussian_information(),
        'ranking_target_and_information_gain': check_ranking_and_gain_bound(),
        'source_hashes': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'scripts/check_score_sufficiency.py', 'bundlefuse/operators.py',
            'bundlefuse/fusion.py', 'bundlefuse/selected_signals.py',
            'results/publication_mechanisms/pog_dense/full.json')},
    }
    destination = ROOT/'results/algebraic_diagnostics/score_sufficiency.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print('Passed: 2 production-score counterexamples, 100 correlated-measurement identities, '
          'and 200 exhaustive small ranking checks.')
    print('Saved:', destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
