"""Selection-only adapters: fixed mixtures and resampled validation populations."""
import numpy as np
import torch
from bundlefuse.fusion import ValidationSignals, fusion_parts


class StudyValidation(ValidationSignals):
    def __init__(self, blocks, pairs, freq, indices=None, progress=None):
        self.indices = np.arange(len(pairs)) if indices is None else np.asarray(indices)
        self.source_blocks = blocks
        selected = [pairs[i] for i in self.indices]
        super().__init__(self.replay, selected, freq, progress=progress)

    def replay(self):
        # Sorted bootstrap indices preserve multiplicities and bounded block memory.
        offset = 0
        for start, signals in self.source_blocks():
            width = next(iter(signals.values())).shape[1]
            ix = self.indices[(self.indices >= start) & (self.indices < start + width)] - start
            if len(ix):
                yield offset, {k: v[:, ix] for k, v in signals.items()}
                offset += len(ix)

    def score(self, signals, start, candidate):
        weights, ho, backoff, prior = candidate
        if backoff[0] >= 0:
            return super().score(signals, start, candidate)
        # Negative first coordinate encodes a constant fallback fraction, including 1.
        u = -backoff[0]
        y, p = fusion_parts(signals, weights, prior, ho)
        values = ((1-u)*y + u*backoff[1]*p).numpy()
        from bundlefuse.metrics import ranking_metrics
        count = values.shape[1]
        return ranking_metrics(values, self.pairs[start:start+count], Ks=(20,))[20][1]*count


def cluster_bootstrap(pairs, seed):
    groups = {}
    for i, (s, gt) in enumerate(pairs):
        groups.setdefault(tuple(sorted(set(s) | set(gt))), []).append(i)
    rows = list(groups.values())
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rows[i] for i in rng.integers(len(rows), size=len(rows))]))


def score_recipe(signals, recipe, frequency):
    from bundlefuse.fusion import apply_fusion
    w, prior, eb, ho = (recipe[k] for k in ('weights','prior','backoff','higher_order'))
    if eb[0] >= 0:
        return apply_fusion(signals, w, prior, eb, frequency, ho).numpy()
    y, p = fusion_parts(signals, w, prior, ho)
    u = -eb[0]
    return ((1-u)*y + u*eb[1]*p).numpy()


def recipe(fitted):
    w, prior, eb, ho, variance = fitted
    return dict(weights=w, prior=prior, backoff=list(eb), higher_order=list(ho), variance=variance)
