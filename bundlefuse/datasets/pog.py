"""pog dataset -- builds the BundleFuse object configured for this dataset (configs/pog.json)."""
from bundlefuse.datasets.base import load_bundlemllm
from bundlefuse.model import BundleFuse
from bundlefuse.config import make_config

NAME = "pog"
FAMILIES = ["collaborative", "content", "taxonomy", "popularity"]   # fallback if no config file
SVD_DIM = 128


def load(val_n=800, test_n=2000):
    return load_bundlemllm(NAME, val_n, test_n)


def config(n_items, families=None, svd_dim=None, use_config=True, config_file=None):
    return make_config(NAME, n_items, use_config=use_config, config_file=config_file,
                       fallback={"families": FAMILIES, "svd_dim": SVD_DIM},
                       families=families, svd_dim=svd_dim)


def build(val_n=800, test_n=2000, families=None, svd_dim=None, use_config=True, config_file=None):
    """Return an (unfitted) BundleFuse object for pog (from configs/pog.json) and its Dataset."""
    ds = load(val_n, test_n)
    return BundleFuse(config(ds.n_items, families, svd_dim, use_config, config_file)), ds
