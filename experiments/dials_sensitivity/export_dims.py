"""Retail datasets with BLIP title features at other sizes, for the feature-size check of the sensitivity sweep.

Everything is built by dials.data.load_resplit as in the frozen runs; only the truncated SVD that projects the
row-normalized 768-d BLIP vectors changes: d = 64 or 256 components, or none (d = 768, the normalized BLIP vectors
themselves). d = 128 is the frozen setting (the sweep's ref runs). Writes <out>/datasets/<dataset>_d<d>.pkl.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_sensitivity.export_dims <out_dir>
"""
import triad  # pins BLAS threads before numpy loads
import pickle
import sys
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from triad import data
from bundlefuse.paths import DATA_ROOT


class Identity:
    def __init__(self, *_, **__):pass

    def fit_transform(self, x):return x.astype('float32')


if __name__ == '__main__':
    out = Path(sys.argv[1])/'datasets';out.mkdir(parents=True, exist_ok=True);original = data.TruncatedSVD
    for d in (64, 256, 768):
        data.TruncatedSVD = Identity if d == 768 else (lambda _n, random_state, d=d: TruncatedSVD(d, random_state=random_state))
        try:
            for name in ('electronic', 'food', 'clothing'):
                ds = data.load(name, Path(DATA_ROOT)/'dials'/'cache');ds.name = f'{name}_d{d}'
                with open(out/f'{ds.name}.pkl', 'wb') as f:pickle.dump(ds, f, protocol=4)
                print(ds.name, ds.features.shape, flush=True)
        finally:data.TruncatedSVD = original
