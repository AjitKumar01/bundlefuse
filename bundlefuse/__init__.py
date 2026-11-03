"""BundleFuse: a zero-parameter, config-driven, regime-adaptive scorer for bundle completion.

A closed-form evidence-fusion model: collaborative co-occurrence + taxonomic ontology +
graph-bridged content, with empirical-Bayes cold-start shrinkage and validation-selected fusion
weights. Signals are organised into families (collaborative / content / taxonomy / popularity /
personalization) switched on by the config.

Train and evaluate any dataset through the one model class::

    from bundlefuse.model import BundleFuse, BundleFuseConfig
    from bundlefuse import datasets

    model, ds = datasets.build("pog_dense")
    model.fit(ds.train, ds.val_pairs, **ds.fit_kwargs())
    items = model.predict(seeds, k=20)

or from the command line::  python evaluator.py pog_dense --protocol provided
"""
from .model import BundleFuse, BundleFuseConfig, load_model
from .baselines import BASELINES, Baseline
from .operators import (
    SimplicialGraphOperator,
    BilateralContentBridge,
    EmpiricalBayesRouter,
    zcols,
)

__all__ = [
    "BundleFuse", "BundleFuseConfig", "load_model",
    "BASELINES", "Baseline",
    "SimplicialGraphOperator", "BilateralContentBridge", "EmpiricalBayesRouter", "zcols",
]
