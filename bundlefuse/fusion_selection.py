"""Fusion selection shared by in-memory and disk-backed validation signals."""
import numpy as np
import torch

from .metrics import ranking_metrics, as_numpy
from .operators import EmpiricalBayesRouter
from .selection import sel_score


def fusion_observed(signals, weights, higher_order):
    pair = dict(signals, simp=signals['pair'], diff2=signals['diff2_pair'])
    base_pair = sum(weights[k] * pair[k] for k in weights if weights[k])
    alpha, gamma = higher_order
    observed = base_pair
    if alpha:
        base_inc = sum(weights[k] * signals[k] for k in weights if weights[k])
        observed = (1 - alpha) * base_pair + alpha * base_inc
    if gamma:
        observed = observed + gamma * signals['rao']
    return observed


def fusion_parts(signals, weights, prior_keys, higher_order):
    observed = fusion_observed(signals, weights, higher_order)
    _, gamma = higher_order
    prior = sum(weights.get(k, 0.5) * signals[k] for k in prior_keys) + gamma * signals['rao']
    if weights.get('pop', 0) > 0:
        prior = prior + weights['pop'] * signals['pop']
    return observed, prior


def apply_fusion(signals, weights, prior_keys, backoff, freq, higher_order):
    kappa, scale = backoff
    if kappa <= 0:
        return fusion_observed(signals, weights, higher_order)
    observed, prior = fusion_parts(signals, weights, prior_keys, higher_order)
    return EmpiricalBayesRouter.conjugate_posterior(observed, prior * scale, freq, kappa)


class ValidationSignals:
    """Replay bounded query blocks without changing the validation population."""
    def __init__(self, blocks, pairs, freq, objective='ndcg20', metric=None, progress=None):
        self.blocks = blocks
        self.pairs = pairs
        self.freq = np.asarray(freq)
        self.objective = objective
        self.metric = metric
        self.memo = {}
        self.progress = progress
        self.scales = {}
        self.variances = {}

    @staticmethod
    def key(weights, higher_order=(0.0, 0.0), backoff=(0.0, 1.0), prior_keys=()):
        return (tuple(weights.items()), tuple(higher_order), tuple(backoff), tuple(prior_keys))

    def record(self, key, value):
        self.memo[key] = value
        if self.progress and len(self.memo) % 10 == 0:
            self.progress(f'validation candidates evaluated: {len(self.memo)}; latest {self.objective}={value:.6f}')

    def score(self, signals, start, candidate):
        first = next(iter(signals.values()))
        freq = torch.as_tensor(self.freq, device=first.device)[:, None]
        weights, higher_order, backoff, prior_keys = candidate
        scores = apply_fusion(signals, weights, prior_keys, backoff, freq, higher_order)
        count = scores.shape[1]
        value = (self.metric(scores) if self.metric else
                 sel_score(as_numpy(scores), self.pairs[start:start + count], self.objective, ranking_metrics))
        return value * count

    def evaluate(self, weights, higher_order=(0.0, 0.0), backoff=(0.0, 1.0), prior_keys=()):
        key = self.key(weights, higher_order, backoff, prior_keys)
        if key in self.memo:
            return self.memo[key]
        total = 0.0
        for start, signals in self.blocks():
            total += self.score(signals, start, (weights, higher_order, backoff, prior_keys))
        result = total / len(self.pairs)
        self.record(key, result)
        return result

    def evaluate_many(self, candidates):
        """Evaluate a coordinate/grid in one cache pass, preserving candidate order and ties."""
        pending = {self.key(*candidate): candidate for candidate in candidates
                   if self.key(*candidate) not in self.memo}
        totals = dict.fromkeys(pending, 0.0)
        if pending:
            for start, signals in self.blocks():
                for key, candidate in pending.items():
                    totals[key] += self.score(signals, start, candidate)
            for key, total in totals.items():
                self.record(key, total / len(self.pairs))
        return [self.memo[self.key(*candidate)] for candidate in candidates]

    def prior_scale(self, weights, higher_order, prior_keys):
        key = self.key(weights, higher_order, prior_keys=prior_keys)
        if key in self.scales:
            return self.scales[key]
        totals = np.zeros((2, 2), dtype=np.float64)
        size = 0
        for _, signals in self.blocks():
            observed, prior = fusion_parts(signals, weights, prior_keys, higher_order)
            size += observed.numel()
            for i, values in enumerate((observed, prior)):
                # Accumulate in float64; the same Bessel correction as torch.std().
                array = as_numpy(values).astype(np.float64)
                totals[i] += (array.sum(), np.square(array).sum())
        std = np.sqrt(np.maximum(0, (totals[:, 1] - totals[:, 0] ** 2 / size) / (size - 1)))
        scale = float((std[0] + 1e-9) / (std[1] + 1e-9))
        self.scales[key] = scale
        return scale

    def variance_components(self, weights, higher_order, prior_keys, scale):
        key = self.key(weights, higher_order, (0.0, scale), prior_keys)
        if key in self.variances:
            return self.variances[key]
        mse = np.zeros(len(self.freq), dtype=np.float64)
        for _, signals in self.blocks():
            observed, prior = fusion_parts(signals, weights, prior_keys, higher_order)
            residual = as_numpy(observed - prior * scale)
            mse += np.square(residual).sum(axis=1, dtype=np.float64)
        mse /= len(self.pairs)
        variance = EmpiricalBayesRouter.fit_variance_components_from_mse(self.freq, mse)
        self.variances[key] = variance
        return variance


def fit_fusion(validation, signals, prior_candidates, *, incidence=(0.0, 0.5, 1.0),
               rao=(0.0, 0.2, 0.35), shrinkage=True):
    """The existing greedy weight, higher-order, and shrinkage grids, on all validation queries."""
    weights = dict.fromkeys(signals, 0.0)
    best = -1e18
    for _ in range(4):
        improved = False
        for key in signals:
            grid = [dict(weights, **{key: value}) for value in (0.0, 0.25, 0.5, 1.0, 1.5)]
            grid = [candidate for candidate in grid if sum(candidate.values())]
            scores = validation.evaluate_many([(candidate, (0.0, 0.0), (0.0, 1.0), ()) for candidate in grid])
            for candidate, score in zip(grid, scores):
                if score > best + 2e-4:
                    best, weights, improved = score, candidate, True
        if not improved:
            break
    # Same optional-signal pruning as EmpiricalBayesRouter.prune_weights_val.
    weights = {key: value for key, value in weights.items() if value > 0}
    score = validation.evaluate(weights)
    for key in ('bridge', 'contrast', 'softcat', 'homo'):
        if weights.get(key, 0) <= 0:
            continue
        candidate = {k: v for k, v in weights.items() if k != key}
        if not candidate:
            continue
        value = validation.evaluate(candidate)
        if value >= score - 3e-4:
            weights, score = candidate, value
    prior_keys = [key for key in prior_candidates if key in signals and weights.get(key, 0) > 0]
    if not prior_keys:
        prior_keys = [key for key in prior_candidates if key in signals]
    higher_order = (0.0, 0.0)
    best = validation.evaluate(weights)
    grid = [(alpha, gamma) for alpha in incidence for gamma in rao]
    values = validation.evaluate_many([(weights, ho, (0.0, 1.0), ()) for ho in grid])
    for ho, value in zip(grid, values):
        if value > best + 5e-5:
            best, higher_order = value, ho
    if not shrinkage:
        return weights, prior_keys, (0.0, 1.0), higher_order, {'disabled': True}
    scale = validation.prior_scale(weights, higher_order, prior_keys)
    variance = validation.variance_components(weights, higher_order, prior_keys, scale)
    backoff = (0.0, scale)
    grid = [(kappa, prior_scale) for prior_scale in (scale, 1.0) for kappa in
            EmpiricalBayesRouter.shrinkage_candidates(variance, [0.0, 0.05, 0.5, 1.0, variance['kappa_f'], 5.0])]
    values = validation.evaluate_many([(weights, higher_order, eb, prior_keys) for eb in grid])
    for eb, value in zip(grid, values):
        if value > best + 5e-5:
            best, backoff = value, eb
    return weights, prior_keys, backoff, higher_order, variance
