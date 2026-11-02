"""Per-dataset default configs.

Each dataset has a JSON file ``configs/<name>.json`` at the repo root holding the research-matched
defaults (families, svd_dim, content_rank, operator hyper-parameters, compression). ``make_config``
loads it into a :class:`BundleFuseConfig`; command-line / caller overrides win over the file, and a
``fallback`` dict is used when no file exists or ``use_config=False``.
"""
import json
import os

from bundlefuse.model import BundleFuseConfig
from bundlefuse.paths import PROJECT_ROOT

CONFIG_DIR = os.environ.get("BUNDLEFUSE_CONFIG_DIR") or os.path.join(PROJECT_ROOT, "configs")


def config_file_path(name):
    """Default config path for a dataset: ``configs/<name>.json``."""
    return os.path.join(CONFIG_DIR, f"{name}.json")


def load_config_file(name, config_file=None):
    """Return the dataset's config dict, or {} if no file is present."""
    p = config_file or config_file_path(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def make_config(name, n_items, *, use_config=True, config_file=None, fallback=None, **overrides):
    """Build a BundleFuseConfig: start from the dataset's config file (or ``fallback``), then apply
    any non-None ``overrides`` (so explicit CLI/caller values win over the file)."""
    base = load_config_file(name, config_file) if use_config else {}
    if not base and fallback:
        base = dict(fallback)
    base.pop("n_items", None)
    base.update({k: v for k, v in overrides.items() if v is not None})
    return BundleFuseConfig(n_items=n_items, **base)
