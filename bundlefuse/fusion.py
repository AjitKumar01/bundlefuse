"""Validation objectives, fusion selection and empirical-Bayes scoring.

Full-ranking uses streaming Torch validation. Retail/commerce use their original NumPy
recipe, including population standard deviations and dataset-specific search tie rules.
"""
import math
import numpy as np
import torch

from .metrics import ranking_metrics, as_numpy, full_ndcg, mean_reciprocal_rank
from .operators import EmpiricalBayesRouter


def softmax_nll_np(F, bundles):
    """Standardised listwise softmax NLL of the held-out items (lower=better).

    Per query: z-standardise the score column so the overall weight scale is irrelevant, mask the
    seeds out of the denominator, average ``-log softmax`` over the held-out items.
    ``bundles`` is a list of ``(seeds, held_out)`` index pairs.
    """
    mu = F.mean(0, keepdims=True); sd = F.std(0, keepdims=True) + 1e-9
    Z = ((F - mu) / sd).astype(np.float64)
    tot, cnt = 0.0, 0
    for j, (s, gt) in enumerate(bundles):
        if not gt:
            continue
        col = Z[:, j].copy(); col[list(s)] = -1e9
        m = float(col.max()); logden = m + math.log(float(np.exp(col - m).sum()))
        tot += float(-np.mean(col[list(gt)] - logden)); cnt += 1
    return tot / max(cnt, 1)


def full_ndcg_np(F, bundles):
    return full_ndcg(F, bundles)


def mrr_np(F, bundles):
    return mean_reciprocal_rank(F, bundles)


def sel_score(F, bundles, sel_obj, eval_mat):
    """Selection objective (higher=better) named by ``sel_obj``.

    ``ndcg<K>``/``recall<K>`` select at cutoff K and delegate to the evaluator's own ``eval_mat``
    (so the cutoff metric is computed identically to how it is reported); ``ndcgfull``/``mrr``/``nll``
    are cutoff-free and computed here.
    """
    if sel_obj == "nll":
        return -softmax_nll_np(F, bundles)
    if sel_obj == "ndcgfull":
        return full_ndcg_np(F, bundles)
    if sel_obj == "mrr":
        return mrr_np(F, bundles)
    if sel_obj.startswith("recall") and sel_obj[6:].isdigit():
        K = int(sel_obj[6:]); return eval_mat(F, bundles, Ks=(K,))[K][0]
    if sel_obj.startswith("ndcg") and sel_obj[4:].isdigit():
        K = int(sel_obj[4:]); return eval_mat(F, bundles, Ks=(K,))[K][1]
    raise ValueError(f"Unknown validation objective: {sel_obj}")


def fusion_observed(signals, weights, higher_order):
    aliases = {'simp': 'pair', 'diff2': 'diff2_pair'}
    base_pair = sum(weights[k] * signals[aliases.get(k, k)] for k in weights if weights[k])
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
    prior = sum(weights[k] * signals[k] for k in prior_keys)
    if gamma:
        prior = prior + gamma * signals['rao']
    if weights.get('pop', 0) > 0:
        prior = prior + weights['pop'] * signals['pop']
    return observed, prior


def has_prior(weights, prior_keys, gamma):
    """The fallback holds only sources the fit selected; with none, the gate has nothing to fall back to."""
    return bool(prior_keys) or bool(gamma) or weights.get('pop', 0) > 0


def apply_fusion(signals, weights, prior_keys, backoff, freq, higher_order):
    kappa, scale = backoff
    if kappa <= 0:
        return fusion_observed(signals, weights, higher_order)
    observed, prior = fusion_parts(signals, weights, prior_keys, higher_order)
    return EmpiricalBayesRouter.frequency_gate(observed, prior * scale, freq, kappa)


def required_fusion_signals(weights, prior_keys, backoff, higher_order):
    """Exact dependencies of a frozen fusion; selection still needs the complete signal set."""
    aliases = {'simp': 'pair', 'diff2': 'diff2_pair'}
    required = {aliases.get(k, k) for k, value in weights.items() if value}
    alpha, gamma = higher_order
    if alpha:
        required.update(k for k, value in weights.items() if value)
    if gamma:
        required.add('rao')
    if backoff[0] > 0:
        required.update(prior_keys)
        if weights.get('pop', 0) > 0:
            required.add('pop')
    return required


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
               rao=(0.0, 0.2, 0.35), shrinkage=True, rounds=4):
    """The existing greedy weight, higher-order, and shrinkage grids, on all validation queries."""
    weights = dict.fromkeys(signals, 0.0)
    best = -1e18
    if rounds < 1 or not signals:
        raise ValueError('Fusion selection needs positive rounds and at least one signal')
    for _ in range(rounds):
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
    higher_order = (0.0, 0.0)
    best = validation.evaluate(weights)
    grid = [(alpha, gamma) for alpha in incidence for gamma in rao]
    values = validation.evaluate_many([(weights, ho, (0.0, 1.0), ()) for ho in grid])
    for ho, value in zip(grid, values):
        if value > best + 5e-5:
            best, higher_order = value, ho
    if not shrinkage or not has_prior(weights, prior_keys, higher_order[1]):
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


def basket_observed(signals, weights, higher_order):
    alpha, gamma = higher_order
    aliases = {'simp': 'pair', 'diff2': 'diff2_pair'}
    base = sum(weights[k] * signals[aliases.get(k,k)] for k in weights)
    result = (1. - alpha) * base
    if alpha:
        result = result + alpha * sum(weights[k] * signals[k] for k in weights)
    if gamma:
        result = result + gamma * signals['rao']
    return result


def apply_basket_fusion(signals, weights, priors, backoff, frequency, higher_order):
    result = basket_observed(signals, weights, higher_order)
    kappa, scale = backoff
    if kappa <= 0:
        return result
    prior = sum(weights[k] * signals[k] for k in priors)
    if higher_order[1]:
        prior = prior + higher_order[1] * signals['rao']
    if weights.get('pop',0) > 0:
        prior = prior + weights['pop'] * signals['pop']
    return EmpiricalBayesRouter.frequency_gate(result, prior * scale, frequency, kappa)


def fit_basket_fusion(signals, keys, pairs, frequency, kind, objective='ndcg20', rounds=4):
    score = lambda values: sel_score(values, pairs, objective, ranking_metrics)
    pair = dict(signals, simp=signals['pair'], diff2=signals['diff2_pair'])
    weights = dict.fromkeys(keys,0.)
    best = -1e18
    for _ in range(rounds):
        improved = False
        for key in keys:
            for value in (0.,.25,.5,1.,1.5):
                candidate = dict(weights, **{key:value})
                if not sum(candidate.values()):
                    continue
                result = score(sum(candidate[k] * pair[k] for k in candidate if candidate[k]))
                if result > best + 2e-4:
                    best, weights, improved = result, candidate, True
        if not improved:
            break
    optional = ('bridge','contrast','cat') if kind == 'retail' else ('contrast','homo')
    weights = EmpiricalBayesRouter.prune_weights_val(weights, pair, score, optional_keys=optional, val_eps=3e-4)
    ho_best, ho_value = ((1.,0.) if kind == 'retail' else (.5,.25)), -1e18
    for alpha in (0.,.5,1.):
        for gamma in ((0.,.15,.25) if kind == 'retail' else (0.,.2,.35)):
            value = score(basket_observed(signals,weights,(alpha,gamma)))
            accept = value >= ho_value - 1e-5 if kind == 'retail' else value > ho_value + 1e-5
            if accept:
                ho_best, ho_value = (alpha,gamma), value
    best = max(best, ho_value)
    candidates = ('cat','content','bridge','contrast') if kind == 'retail' else ('cat','homo','tfidf','bridge','pretrain','contrast','img')
    priors = [k for k in candidates if k in keys and weights.get(k,0) > 0]
    if not has_prior(weights, priors, ho_best[1]):
        return weights, priors, (0.,1.), ho_best, {'disabled': True}
    y = basket_observed(signals,weights,ho_best)
    prior = sum(weights[k] * signals[k] for k in priors) + ho_best[1]*signals['rao']
    if weights.get('pop',0) > 0:
        prior = prior + weights['pop']*signals['pop']
    scale = float(np.std(y)/(np.std(prior)+1e-9))
    variance = EmpiricalBayesRouter.fit_variance_components(frequency,y,prior*scale)
    grid = [0.,.05,.25,variance['kappa_f'],1.] if kind == 'retail' else [0.,.1,.5,variance['kappa_f'],2.]
    kappa = 0.
    for value in EmpiricalBayesRouter.shrinkage_candidates(variance,grid):
        result = score(apply_basket_fusion(signals,weights,priors,(value,scale),frequency,ho_best))
        if result > best + 1e-4:
            best, kappa = result, value
    return weights, priors, (kappa,scale), ho_best, variance
