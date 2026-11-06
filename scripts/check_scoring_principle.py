"""Check structural claims against the scoring implementation, without fitting data.

Run from any directory: python3 /path/to/scripts/check_scoring_principle.py
Writes an evidence record; does not modify checkpoints or experiment source.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

from bundlefuse.fusion import apply_fusion
from bundlefuse.operators import BilateralContentBridge, SimplicialGraphOperator, zcols


def main():
    errors = {}

    def check(name, actual, expected, atol=3e-6):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=atol)
        errors[name] = float(np.max(np.abs(actual - expected)))

    # A pair-balanced exchange: the same item degrees, bundle sizes and pairs,
    # but different triples. Items a,b are the query; c,d are candidates.
    left = [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]]
    right = [[0, 1, 3], [0, 2, 4], [1, 2, 5], [3, 4, 5]]
    graphs = [SimplicialGraphOperator(t, 6, svd_dim=3) for t in (left, right)]
    g0, g1 = graphs
    check('identical_raw_gram', (g0.B.T @ g0.B).toarray(), (g1.B.T @ g1.B).toarray())
    check('identical_ppmi', g0.PPMI.toarray(), g1.PPMI.toarray())
    check('identical_csvd_kernel', g0.Ucsvd @ g0.Ucsvd.T, g1.Ucsvd @ g1.Ucsvd.T)
    pairs, groups = [], []
    for g in graphs:
        pair, _ = g.score_pairwise_and_diff2([[0, 1]])
        delta, _ = g.score_simplicial_incidence_only([[0, 1]])
        pairs.append(pair[:, 0])
        groups.append((pair + delta)[:, 0])
    check('identical_pair_scores', pairs[0], pairs[1])
    assert groups[0][2] > groups[0][3]
    assert groups[1][2] < groups[1][3]

    # csvd is a projected row-profile Gram matrix, not the rank-k PPMI matrix.
    train = [[0, 1, 2], [0, 3], [1, 4, 5], [2, 3, 6],
             [0, 4, 6], [1, 3, 5], [2, 6]]
    g = SimplicialGraphOperator(train, 9, svd_dim=3)
    model = TruncatedSVD(n_components=3, random_state=0)
    embedding = model.fit_transform(g.PPMI).astype(np.float32)
    projected = g.PPMI @ model.components_.T
    check('svd_fit_transform_is_projection', embedding, projected)
    invnorm = 1 / (np.linalg.norm(embedding, axis=1) + 1e-8)
    projection_gram = projected @ projected.T
    kernel = invnorm[:, None] * projection_gram * invnorm[None, :]
    check('csvd_projected_profile_kernel', g.Ucsvd @ g.Ucsvd.T, kernel)
    reconstruction = embedding @ model.components_
    reconstruction_gap = float(np.linalg.norm(embedding @ embedding.T - reconstruction))
    assert reconstruction_gap > .1

    rng = np.random.default_rng(20261105)
    B = g.B.toarray().astype(float)
    frequency, size = B.sum(0), B.sum(1)
    d, v = (frequency + 5) ** -.5, (size + 1) ** -.5
    queries = [[0], [0, 1], [2, 4, 7]]
    X = np.zeros((9, len(queries)))
    for j, seeds in enumerate(queries):
        X[seeds, j] = 1
    pair, _ = g.score_pairwise_and_diff2(queries)
    delta, _ = g.score_simplicial_incidence_only(queries)
    minimum_eigenvalues = []
    for j in range(len(queries)):
        overlap = B @ X[:, j]
        query_weight = v * (1 + g.beta_clique * np.maximum(overlap - 1, 0))
        H = B.T @ (query_weight[:, None] * B)
        check(f'query_conditioned_kernel_{j}', H @ (d * X[:, j]), (pair + delta)[:, j])
        symmetric = np.sqrt(d)[:, None] * H * np.sqrt(d)[None, :]
        minimum_eigenvalues.append(float(np.linalg.eigvalsh(symmetric).min()))
        assert minimum_eigenvalues[-1] > -1e-10
        weight = query_weight * (B @ (d * X[:, j]))
        if weight.sum() > 0:
            check(f'weighted_history_inclusion_{j}', B.T @ (weight / weight.sum()),
                  (pair + delta)[:, j] / weight.sum())

    # The category-centering step solves a least-squares projection problem.
    labels = np.arange(9) % 3
    G = np.eye(3)[labels]
    features = rng.normal(size=(9, 5)).astype(np.float32)
    centroids = np.linalg.solve(G.T @ G, G.T @ features)
    residual = features - G @ centroids
    check('category_residual_orthogonality', G.T @ residual, np.zeros((3, 5)))
    E = residual / (np.linalg.norm(residual, axis=1, keepdims=True) + 1e-8)
    actual_E = BilateralContentBridge.disentangle(features, labels, 3)
    check('category_projection_matches_operator', actual_E, E)
    propagated = g.propagate(actual_E)
    N = propagated / (np.linalg.norm(propagated, axis=1, keepdims=True) + 1e-8)
    unnormalized = actual_E + .5 * N
    invnorm = 1 / (np.linalg.norm(unnormalized, axis=1) + 1e-8)
    expanded = (actual_E @ actual_E.T + .5 * actual_E @ N.T
                + .5 * N @ actual_E.T + .25 * N @ N.T)
    bridge = BilateralContentBridge.build_dense_bridge(actual_E, g)
    check('bridge_four_kernel_terms', bridge @ bridge.T,
          invnorm[:, None] * expanded * invnorm[None, :])

    # Use the actual dense recipe and fusion routine for the convex objective.
    recipe_path = ROOT / 'results/publication_mechanisms/pog_dense/frozen_recipes.json'
    recipe = json.loads(recipe_path.read_text())['full']
    signals = {key: zcols(torch.tensor(rng.normal(size=(9, 3)), dtype=torch.float32))
               for key in ('pair', 'simp', 'cat', 'bridge', 'pop', 'csvd')}
    theta = apply_fusion(signals, recipe['weights'], recipe['prior'], recipe['backoff'],
                         frequency, recipe['higher_order']).numpy()
    w = recipe['weights']
    prior = sum(w[k] * signals[k].numpy() for k in ('cat', 'bridge', 'pop'))
    history = sum(w[k] * signals[k].numpy() for k in ('simp', 'csvd'))
    kappa, scale = recipe['backoff']
    observed = prior + history
    gradient = frequency[:, None] * (theta - observed) + kappa * (theta - scale * prior)
    check('convex_objective_stationarity', gradient, np.zeros_like(gradient))
    rho = frequency[:, None] / (frequency[:, None] + kappa)
    check('overlapping_prior_decomposition', theta, (scale + (1-scale)*rho)*prior + rho*history)
    energy = lambda t: float(np.sum(frequency[:, None]*(t-observed)**2 + kappa*(t-scale*prior)**2))
    energy_min = energy(theta)
    assert all(energy(theta + rng.normal(size=theta.shape)) > energy_min for _ in range(100))

    source_paths = ['bundlefuse/operators.py', 'bundlefuse/selected_signals.py',
                    'bundlefuse/fusion.py', 'bundlefuse/model.py',
                    'scripts/check_scoring_principle.py',
                    'results/publication_mechanisms/pog_dense/frozen_recipes.json']
    output = {
        'status': 'passed', 'scope': 'Structural identities on synthetic inputs, not a new accuracy experiment.',
        'maximum_absolute_errors': errors,
        'pair_indistinguishable_corpora': {
            'item_names': ['a', 'b', 'c', 'd', 'e', 'f'],
            'left': left, 'right': right, 'query': [0, 1], 'candidates': [2, 3],
            'left_pair': pairs[0][[2, 3]].tolist(), 'right_pair': pairs[1][[2, 3]].tolist(),
            'left_group': groups[0][[2, 3]].tolist(), 'right_group': groups[1][[2, 3]].tolist()},
        'projected_gram_vs_reconstruction_frobenius_gap': reconstruction_gap,
        'query_kernel_minimum_eigenvalues': minimum_eigenvalues,
        'convex_objective_random_perturbations_checked': 100,
        'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}}
    destination = ROOT / 'results/algebraic_diagnostics/scoring_principle.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
