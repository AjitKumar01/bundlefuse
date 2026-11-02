"""electronic dataset (BundleRec) -- builds the BundleFuse object configured for this dataset (configs/electronic.json)."""
from bundlefuse.datasets.base import load_bundlerec
from bundlefuse.model import BundleFuse
from bundlefuse.config import make_config

NAME = "electronic"
FAMILIES = ["collaborative", "content", "taxonomy", "popularity", "personalization"]
SVD_DIM = 128


def load(val_n=800, test_n=2000, mask_frac=0.5, split_seed=0):
    return load_bundlerec(NAME, val_n, test_n, mask_frac, split_seed)


def config(n_items, families=None, svd_dim=None, use_config=True, config_file=None):
    return make_config(NAME, n_items, use_config=use_config, config_file=config_file,
                       fallback={"families": FAMILIES, "svd_dim": SVD_DIM},
                       families=families, svd_dim=svd_dim)


def build(val_n=800, test_n=2000, families=None, svd_dim=None, use_config=True, config_file=None):
    """Return an (unfitted) BundleFuse object for electronic (from configs/electronic.json) and its Dataset."""
    ds = load(val_n, test_n)
    return BundleFuse(config(ds.n_items, families, svd_dim, use_config, config_file)), ds
