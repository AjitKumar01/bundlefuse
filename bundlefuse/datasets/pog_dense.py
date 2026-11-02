"""pog_dense dataset -- builds the BundleFuse object configured for this dataset."""
from bundlefuse.datasets.base import load_bundlemllm
from bundlefuse.model import BundleFuse, BundleFuseConfig

NAME = "pog_dense"
FAMILIES = ["collaborative", "content", "taxonomy", "popularity"]
SVD_DIM = 128


def load(val_n=800, test_n=2000):
    return load_bundlemllm(NAME, val_n, test_n)


def config(n_items, families=None, svd_dim=None):
    return BundleFuseConfig(n_items=n_items, families=families or FAMILIES, svd_dim=svd_dim or SVD_DIM)


def build(val_n=800, test_n=2000, families=None, svd_dim=None):
    """Return an (unfitted) BundleFuse object for pog_dense and its loaded Dataset."""
    ds = load(val_n, test_n)
    return BundleFuse(config(ds.n_items, families, svd_dim)), ds
