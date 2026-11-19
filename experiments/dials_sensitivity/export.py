"""Pack what the Kaggle sensitivity sweep needs: the dials package (code only) and one pickled dials.data.Dataset per
dataset (training bundles, validation and test pairs, projected features, labels, clean test indices), loaded exactly
as the frozen runs load them. Data goes to a private Kaggle dataset, never into the repository.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_sensitivity.export <out_dir>
"""
import triad  # pins BLAS threads before numpy loads
import pickle
import shutil
import sys
from pathlib import Path
from triad import data
from bundlefuse.paths import DATA_ROOT

DATASETS = ('electronic', 'food', 'clothing', 'pog_dense', 'spotify_sparse')

if __name__ == '__main__':
    out = Path(sys.argv[1]);(out/'datasets').mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(dials.__file__).parent, out/'code'/'dials', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for name in DATASETS:
        ds = data.load(name, Path(DATA_ROOT)/'dials'/'cache')
        with open(out/'datasets'/f'{name}.pkl', 'wb') as f:pickle.dump(ds, f, protocol=4)
        print(name, ds.n_items, len(ds.train), len(ds.val_pairs), len(ds.test_pairs), flush=True)
