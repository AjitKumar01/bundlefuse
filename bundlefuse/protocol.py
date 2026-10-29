"""Shared bundle-completion evaluation protocol.

Bundle completion is evaluated by *masking*: for each held-out bundle we reveal a fraction of its
members as the seed set ``S`` and hide the rest as the ground truth the scorer must recover. This
is the same masking used by every evaluator; the e-commerce and retail evaluators call :func:`mask`
directly (the full-ranking POG/Spotify evaluator keeps a dict-returning variant for its loader).
"""
import random


def mask(bundles, seed, mask_frac):
    """Split each bundle into ``(seeds, held_out)`` by hiding ``mask_frac`` of its members.

    A fresh ``random.Random(seed)`` shuffles each bundle, keeps ``round((1-mask_frac)*|b|)`` (at
    least one) items as seeds and hides the rest. If nothing would be hidden (tiny bundle) we hide
    exactly the last item so every query has a non-empty ground truth.
    """
    r = random.Random(seed); out = []
    for b in bundles:
        bb = b[:]; r.shuffle(bb)
        k = max(1, int(round(len(bb) * (1 - mask_frac))))
        s, h = bb[:k], bb[k:]
        if not h:
            s, h = bb[:-1], [bb[-1]]
        out.append((s, h))
    return out
