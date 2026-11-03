"""Batched metric calculation and model descriptions used by all three CLIs."""
from dataclasses import asdict

import numpy as np

from .metrics import ranking_metrics, TIE_POLICY


def evaluate(model, pairs, users=None, Ks=(20,), chunk=32):
    if not pairs or chunk < 1 or (users is not None and len(users) != len(pairs)):
        raise ValueError('Evaluation needs nonempty queries, a positive chunk and aligned users')
    totals = {k: np.zeros(2) for k in Ks}
    for start in range(0, len(pairs), chunk):
        batch = pairs[start:start + chunk]
        us = users[start:start + chunk] if users is not None else None
        scores = model.score_batch([s for s, _ in batch], us)
        metrics = ranking_metrics(scores, batch, Ks)
        for k in Ks:
            totals[k] += np.asarray(metrics[k]) * len(batch)
    return {k: tuple(value / len(pairs)) for k, value in totals.items()}


def flat_metrics(metrics):
    return {f'{name}@{k}': float(values[i]) for k, values in metrics.items()
            for i, name in enumerate(('recall', 'ndcg'))}


def describe_model(model):
    if not hasattr(model, 'cfg'):
        return {'name': model.name, 'n_items': model.n, 'tie_policy': TIE_POLICY}
    return {'name': 'bundlefuse', 'n_items': model.cfg.n_items, 'config': asdict(model.cfg),
            'signals': list(model.SIG), 'weights': dict(model.w), 'prior_keys': list(model.prior_keys),
            'kappa_f': float(model.best_eb[0]), 'prior_scale': float(model.best_eb[1]),
            'alpha_inc': float(model.best_ho[0]), 'gamma_rao': float(model.best_ho[1]),
            'tie_policy': TIE_POLICY,
            'variance_components': getattr(model, 'variance_components_', None)}


def sparse_pairs(pairs, count):
    """Reduce revealed context while preserving complete bundle membership as targets."""
    result = []
    for seeds, targets in pairs:
        ordered = sorted(seeds)
        kept = ordered[:count] if count else ordered
        result.append((kept, list(targets) + ordered[len(kept):]))
    return result
