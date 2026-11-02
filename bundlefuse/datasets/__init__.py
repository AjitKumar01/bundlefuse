"""Per-dataset modules: each file (pog, pog_dense, spotify, spotify_sparse, dunnhumby,
electronic, food, clothing) owns one dataset -- its loader and the BundleFuse object
configured for it. ``build(name)`` returns that object plus the loaded data.
"""
from bundlefuse.datasets import (
    pog, pog_dense, spotify, spotify_sparse, dunnhumby, electronic, food, clothing,
)
from bundlefuse.datasets.base import Dataset

REGISTRY = {m.NAME: m for m in (
    pog, pog_dense, spotify, spotify_sparse, dunnhumby, electronic, food, clothing,
)}


def get(name):
    """Return the dataset module for ``name``."""
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def build(name, **kwargs):
    """Build the (unfitted) BundleFuse object and Dataset for ``name`` (see each module's build)."""
    return get(name).build(**kwargs)


def load(name, **kwargs):
    """Load just the Dataset for ``name``."""
    return get(name).load(**kwargs)


__all__ = ["REGISTRY", "get", "build", "load", "Dataset"]
