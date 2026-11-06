"""Check the overlap interaction on a tiny, fully specified bundle corpus."""
import json
from pathlib import Path

import numpy as np

from bundlefuse.operators import EmpiricalBayesRouter, SimplicialGraphOperator


def main():
    # Seeds a,b; candidates c,d; four different filler items.
    a, b, c, d, p, q, r, t = range(8)
    train = [[a, b, c], [a, d, p], [a, d, q],
             [b, d, r], [b, d, t]]
    graph = SimplicialGraphOperator(train, n_items=8, beta_clique=1.5, svd_dim=2)
    queries = [[a], [b], [a, b]]
    pair, _ = graph.score_pairwise_and_diff2(queries)
    delta, _ = graph.score_simplicial_incidence_only(queries)
    simp = pair + delta
    np.testing.assert_allclose(delta[:, :2], 0)
    assert simp[c, 0] < simp[d, 0]
    assert simp[c, 1] < simp[d, 1]
    assert simp[c, 2] > simp[d, 2]
    # An additive zero-intercept item-item score has zero mixed difference.
    gaps = simp[c] - simp[d]
    mixed = float(gaps[2] - gaps[0] - gaps[1])
    assert mixed > 0
    # Check the full quadratic expansion for every candidate, not just c and d.
    B = graph.B_bin.toarray().astype(np.float64)
    sizes = B.sum(axis=1)
    volume = (sizes + 1) ** (-graph.gamma_vol)
    frequency = B.sum(axis=0)
    seed_weight = (frequency + 5) ** (-graph.alpha_deg)
    expected = np.zeros(8)
    for left in (a, b):
        expected += seed_weight[left] * (B.T @ (volume * B[:, left]))
    for left, right in ((a, b),):
        expected += graph.beta_clique * (seed_weight[left] + seed_weight[right]) * (
            B.T @ (volume * B[:, left] * B[:, right]))
    np.testing.assert_allclose(simp[:, 2], expected, atol=1e-6)
    # Verify the exact query-mean / candidate-frequency decomposition.
    rng = np.random.default_rng(42)
    for _ in range(100):
        raw = rng.normal(size=(8, 4)).astype(np.float32)
        mean, std = raw.mean(axis=0), raw.std(axis=0, ddof=1) + 1e-9
        z = (raw - mean) / std
        weights = np.array([.5, .25, .25], dtype=np.float32)
        shared = z[:, :3] @ weights
        pair_z = .25 * z[:, 3]
        rho = frequency / (frequency + .5)
        scale = 1.0424
        observed = EmpiricalBayesRouter.conjugate_posterior(
            shared + pair_z, scale * shared, frequency, .5)
        H = (raw[:, :3] / std[:3]) @ weights
        U = .25 * raw[:, 3] / std[3]
        A = (mean[:3] / std[:3]) @ weights
        J = .25 * mean[3] / std[3]
        qscale = scale + (1 - scale) * rho
        reconstructed = qscale * H + rho * U - qscale * A - rho * J
        np.testing.assert_allclose(observed, reconstructed, rtol=1e-5, atol=1e-6)
    result = {'bundles': train, 'queries': queries, 'candidate_c': c,
              'candidate_d': d, 'pair_scores_c_d': pair[[c, d]].tolist(),
              'incidence_scores_c_d': delta[[c, d]].tolist(),
              'combined_scores_c_d': simp[[c, d]].tolist(),
              'mixed_difference_c_minus_d': mixed,
              'quadratic_expansion_verified': True,
              'frequency_centering_random_cases_verified': 100}
    destination = Path('results/algebraic_diagnostics/higher_order_toy.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
