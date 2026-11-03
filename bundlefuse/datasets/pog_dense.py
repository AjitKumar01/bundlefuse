"""pog_dense dataset -- builds the BundleFuse object configured for this dataset (configs/pog_dense.json)."""
from bundlefuse.datasets.base import load_bundlemllm
from bundlefuse.model import BundleFuse
from bundlefuse.config import make_config

NAME = "pog_dense"
FAMILIES = ["collaborative", "content", "taxonomy", "popularity"]   # fallback if no config file
SVD_DIM = 128


def load(val_n=None, test_n=None, mask_frac=0.5, split_seed=0, protocol_mode="provided"):
    return load_bundlemllm(NAME, val_n, test_n, mask_frac, split_seed, protocol_mode=protocol_mode)


def config(n_items, families=None, svd_dim=None, use_config=True, config_file=None):
    return make_config(NAME, n_items, use_config=use_config, config_file=config_file,
                       fallback={"families": FAMILIES, "svd_dim": SVD_DIM},
                       families=families, svd_dim=svd_dim)


def build(val_n=None, test_n=None, families=None, svd_dim=None, use_config=True, config_file=None,
          mask_frac=0.5, split_seed=0, protocol_mode="provided"):
    """Return an (unfitted) BundleFuse object for pog_dense (from configs/pog_dense.json) and its Dataset."""
    ds = load(val_n, test_n, mask_frac, split_seed, protocol_mode)
    return BundleFuse(config(ds.n_items, families, svd_dim, use_config, config_file)), ds
