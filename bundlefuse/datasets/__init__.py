"""Dataset registry and shared load/build API.

Dataset differences belong in this table and configs/<name>.json, rather than eight
copies of the same load/config/build wrappers. Family I/O lives in base.py and baskets.py.
"""
from dataclasses import dataclass

from bundlefuse.config import make_config
from bundlefuse.model import BundleFuse
from . import base
from .base import Dataset

_DEFAULT = object()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str

    def load(self, val_n=_DEFAULT, test_n=_DEFAULT, mask_frac=0.5, split_seed=0,
             protocol_mode=None, data_profile="research"):
        provided = self.family == 'bundlemllm'
        if val_n is _DEFAULT:
            val_n = None if provided else 800
        if test_n is _DEFAULT:
            test_n = None if provided else 2000
        options = dict(protocol_mode=protocol_mode or ('provided' if provided else 'resplit'),
                       data_profile=data_profile)
        args = (val_n, test_n, mask_frac, split_seed)
        if self.family == 'dunnhumby':
            return base.load_dunnhumby(*args, **options)
        loader = base.load_bundlemllm if provided else base.load_bundlerec
        return loader(self.name, *args, **options)

    def config(self, n_items, families=None, svd_dim=None, use_config=True, config_file=None):
        families_default = ['collaborative', 'content', 'taxonomy', 'popularity']
        if self.family != 'bundlemllm':
            families_default.append('personalization')
        return make_config(self.name, n_items, use_config=use_config, config_file=config_file,
                           fallback={'families': families_default, 'svd_dim': 128},
                           families=families, svd_dim=svd_dim)

    def build(self, val_n=_DEFAULT, test_n=_DEFAULT, families=None, svd_dim=None,
              use_config=True, config_file=None, mask_frac=0.5, split_seed=0,
              protocol_mode=None, data_profile="research"):
        dataset = self.load(val_n, test_n, mask_frac, split_seed, protocol_mode, data_profile)
        config = self.config(dataset.n_items, families, svd_dim, use_config, config_file)
        return BundleFuse(config), dataset


REGISTRY = {name: DatasetSpec(name, family) for family, names in (
    ('bundlemllm', ('pog', 'pog_dense', 'spotify', 'spotify_sparse')),
    ('bundlerec', ('electronic', 'food', 'clothing')),
    ('dunnhumby', ('dunnhumby',)),
) for name in names}


def get(name):
    """Return the dataset definition, with load/config/build methods."""
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def build(name, **kwargs):
    """Return an unfitted model and its loaded dataset."""
    return get(name).build(**kwargs)


def load(name, **kwargs):
    """Load the dataset without constructing a model."""
    return get(name).load(**kwargs)


__all__ = ['REGISTRY', 'get', 'build', 'load', 'Dataset']
