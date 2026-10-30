"""Simple same-information controls, without graph bridging or higher-order operators.

SimpleHybrid blends raw co-occurrence, content similarity, category PMI, popularity,
and any supplied training-only auxiliary interactions. HardSwitchHybrid replaces
low-frequency candidates' scores with the content/category prior. Both tune on val.
"""
import numpy as np
from ..operators import zcols
from .base import evaluate_scorer, _seeds


def category_scores(P, labels, seed_lists):
    return np.stack([np.asarray(P[labels[s]].sum(0)).ravel()[labels]
                     for s in seed_lists], axis=1).astype(np.float32)


class SimpleHybrid:
    def __init__(self, graph, content, category_scorer, auxiliary=None):
        self.graph = graph
        self.content = content
        self.category_scorer = category_scorer
        self.auxiliary = auxiliary or {}

    def signals(self, seeds):
        indicator = self.graph.build_indicator(seeds)
        product = self.content @ (self.content.T @ indicator)
        product = product.toarray() if hasattr(product, "toarray") else np.asarray(product)
        raw = {
            "cooc": (self.graph.C @ indicator).toarray(),
            "content": product,
            "category": self.category_scorer(seeds),
            "popularity": np.broadcast_to(np.log1p(self.graph.freq)[:, None], (self.graph.n, len(seeds))),
        }
        raw.update({key: (matrix @ indicator).toarray() for key, matrix in self.auxiliary.items()})
        return {key: zcols(np.asarray(value, dtype=np.float32)) for key, value in raw.items()}

    def select(self, eval_mat, vset, tset):
        signals = self.signals(_seeds(vset))
        weights = {key: 0.0 for key in signals}
        best = -np.inf
        for _ in range(4):
            improved = False
            for key in signals:
                for value in (0.0, 0.25, 0.5, 1.0, 1.5):
                    trial = dict(weights, **{key: value})
                    if not any(trial.values()):
                        continue
                    score = sum(trial[k] * signals[k] for k in signals if trial[k])
                    objective = eval_mat(score, vset, Ks=(20,))[20][1]
                    if objective > best + 2e-4:
                        weights, best, improved = trial, objective, True
            if not improved:
                break

        def fused(z):
            return sum(weights[k] * z[k] for k in z if weights[k])

        def prior(z):
            # Keep both metadata channels available even when a collaborative blend ignores one.
            return sum((weights[k] or 0.5) * z[k] for k in ("content", "category"))

        records = {}
        scorer = lambda seeds: fused(self.signals(seeds))
        records["SimpleHybrid"] = {"hp": weights, "val_ndcg20": float(best),
                                   "test": evaluate_scorer(scorer, eval_mat, tset, Ks=(10, 20, 50))}
        baseline = fused(signals)
        best_gate, gate_nd = None, best
        for threshold in (0, 1, 5):
            for scale in (0.5, 1.0, 2.0):
                candidate = np.where(self.graph.freq[:, None] <= threshold, scale * prior(signals), baseline)
                nd = eval_mat(candidate, vset, Ks=(20,))[20][1]
                if nd > gate_nd + 2e-4:
                    best_gate, gate_nd = (threshold, scale), nd

        def switched(seeds):
            z = self.signals(seeds)
            if best_gate is None:
                return fused(z)
            threshold, scale = best_gate
            return np.where(self.graph.freq[:, None] <= threshold, scale * prior(z), fused(z))

        records["HardSwitchHybrid"] = {
            "hp": {"weights": weights, "gate": best_gate}, "val_ndcg20": float(gate_nd),
            "test": evaluate_scorer(switched, eval_mat, tset, Ks=(10, 20, 50)),
        }
        return records
