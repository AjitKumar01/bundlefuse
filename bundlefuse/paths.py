"""Single source of truth for on-disk locations.

Every evaluator used to recompute these with fragile ``os.path.dirname(...)`` chains whose
depth silently depended on how deep the file was nested. They are now computed once, here,
relative to this file's fixed location inside the package.

Layout::

    <DATA_ROOT>/                     parent of the repo; holds the (un-vendored) datasets
      bundlemllm/datasets/<ds>/      POG / Spotify (fullrank)
      BundleRec/dataset/<ds>/        electronic / food / clothing
      dunnhumby/                     transaction baskets
    <PROJECT_ROOT>/                  the git repo root
      bundlefuse/                    this package
      results/                       logs + json produced by the evaluators
      paper/  notes/  docs/  tests/

Override the dataset root per-run with the env vars the evaluators already read
(``DATA_DIR``/``BR_DIR``/``DUNN_DIR``); this module only supplies the defaults.
"""
import os

# .../<PROJECT_ROOT>/bundlefuse/paths.py  ->  package dir  ->  project root  ->  data root
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_PKG_DIR)
DATA_ROOT = os.path.dirname(PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def dataset_dir(family, ds):
    """Default directory for a dataset, by family.

    family ``bundlemllm`` -> ``<DATA_ROOT>/bundlemllm/datasets/<ds>`` (POG / Spotify)
    family ``bundlerec``  -> ``<DATA_ROOT>/BundleRec/dataset/<ds>``   (electronic / food / clothing)
    family ``dunnhumby``  -> ``<DATA_ROOT>/dunnhumby``
    """
    if family == "bundlemllm":
        return os.path.join(DATA_ROOT, "bundlemllm", "datasets", ds)
    if family == "bundlerec":
        return os.path.join(DATA_ROOT, "BundleRec", "dataset", ds)
    if family == "dunnhumby":
        return os.path.join(DATA_ROOT, "dunnhumby")
    raise ValueError(f"unknown dataset family: {family}")


def results_path(name):
    """Absolute path to a file under ``results/`` (which is created if missing)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)
