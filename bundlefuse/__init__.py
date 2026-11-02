"""BundleFuse: a zero-parameter, config-driven, regime-adaptive scorer for bundle completion.

A closed-form evidence-fusion model: collaborative co-occurrence + taxonomic ontology +
graph-bridged content, with empirical-Bayes cold-start shrinkage and validation-selected fusion
weights. Signals are organised into families (collaborative / content / taxonomy / popularity /
personalization) switched on by the config.

Train and evaluate any dataset through the one model class::

    from bundlefuse.model import BundleFuse, BundleFuseConfig
    from bundlefuse.data import load_dataset

    ds = load_dataset("pog_dense")
    model = BundleFuse(BundleFuseConfig(n_items=ds.n_items, families=ds.families))
    model.fit(ds.train, ds.val_pairs, content=ds.content, cat_labels=ds.cat_labels)
    items = model.predict(seeds, k=20)

or from the command line::  python -m bundlefuse.run pog_dense
"""
from .model import BundleFuse, BundleFuseConfig
from .operators import (
    SimplicialGraphOperator,
    BilateralContentBridge,
    EmpiricalBayesRouter,
    zcols,
)

__all__ = [
    "BundleFuse", "BundleFuseConfig",
    "SimplicialGraphOperator", "BilateralContentBridge", "EmpiricalBayesRouter", "zcols",
]
