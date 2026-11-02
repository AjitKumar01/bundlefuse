"""dunnhumby dataset -- builds the BundleFuse object configured for this dataset.

Grocery baskets as bundles, households as users: the personalization family is enabled.
"""
from bundlefuse.datasets.base import load_dunnhumby
from bundlefuse.model import BundleFuse, BundleFuseConfig

NAME = "dunnhumby"
FAMILIES = ["collaborative", "content", "taxonomy", "popularity", "personalization"]
SVD_DIM = 128


def load(val_n=800, test_n=2000, mask_frac=0.5, split_seed=0):
    return load_dunnhumby(val_n, test_n, mask_frac, split_seed)


def config(n_items, families=None, svd_dim=None):
    return BundleFuseConfig(n_items=n_items, families=families or FAMILIES, svd_dim=svd_dim or SVD_DIM)


def build(val_n=800, test_n=2000, families=None, svd_dim=None):
    """Return an (unfitted) BundleFuse object for dunnhumby and its loaded Dataset."""
    ds = load(val_n, test_n)
    return BundleFuse(config(ds.n_items, families, svd_dim)), ds
