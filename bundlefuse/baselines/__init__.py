"""Matched baseline suite for bundle completion, one module per method.

Each baseline is a :class:`~bundlefuse.baselines.base.Baseline` (OOP interface:
``make_scorer(graph_op, freq, hp) -> scorer(seed_lists)``), and most are item--item models
(:class:`~bundlefuse.baselines.base.ItemItemBaseline`, a sparse ``W`` scored by ``W @ Ind``):

- :mod:`~bundlefuse.baselines.itemknn`    -- :class:`ItemKNN`
- :mod:`~bundlefuse.baselines.randomwalk` -- :class:`P3alpha`, :class:`RP3beta`
- :mod:`~bundlefuse.baselines.slim`       -- :class:`SLIM`
- :mod:`~bundlefuse.baselines.bundleknn`  -- :class:`BundleKNN`
- :mod:`~bundlefuse.baselines.ials`       -- :class:`IALS`
- :mod:`~bundlefuse.baselines.combiner`   -- the LinearCombiner blend
- :mod:`~bundlefuse.baselines.suite`      -- :func:`run_matched_seed`

The underlying ``*_W`` / ``*_scores`` functions and the ``*_GRID`` constants are re-exported here
for the evaluators' inline baseline paths, so ``from bundlefuse import baselines as BL`` keeps working.
"""
from .base import Baseline, ItemItemBaseline, topk_per_column
from .itemknn import ItemKNN, itemknn_W, ITEMKNN_GRID
from .randomwalk import (P3alpha, RP3beta, bipartite_item_item,
                         p3alpha_W, rp3beta_W, P3ALPHA_GRID, RP3BETA_GRID)
from .slim import SLIM, slim_W, SLIM_GRID
from .bundleknn import BundleKNN, bundleknn_scores, BUNDLEKNN_GRID
from .ials import IALS, ials_item_factors, ials_scores, IALS_GRID
from .combiner import run_linear_combiner
from .suite import run_matched_seed

# the suite, in reported order
ALL_BASELINES = [ItemKNN, P3alpha, RP3beta, SLIM, BundleKNN, IALS]

__all__ = [
    "Baseline", "ItemItemBaseline", "topk_per_column", "ALL_BASELINES",
    "ItemKNN", "itemknn_W", "ITEMKNN_GRID",
    "P3alpha", "RP3beta", "bipartite_item_item", "p3alpha_W", "rp3beta_W", "P3ALPHA_GRID", "RP3BETA_GRID",
    "SLIM", "slim_W", "SLIM_GRID",
    "BundleKNN", "bundleknn_scores", "BUNDLEKNN_GRID",
    "IALS", "ials_item_factors", "ials_scores", "IALS_GRID",
    "run_linear_combiner", "run_matched_seed",
]
