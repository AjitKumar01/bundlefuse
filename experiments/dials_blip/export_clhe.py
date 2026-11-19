"""Write <dataset>_<variant>_blip: a CLHE/RaMen export (clean by default; also published, for the official RaMen
rerun) with BLIP text features in place of TF-IDF.

Everything is copied from <dataset>_<variant> (split, training-only user-item file and item graph, item_cf_feature.pt from
training bundles), except content_feature.pt and description_feature.pt, which become the 768-d BLIP [CLS] vectors of
the item titles computed in experiments/dials_blip/run.py (the same vectors DIALS projects to 128 dimensions). With no
item images, both of CLHE's feature slots hold the text vector, as they held the TF-IDF vector before. Item order is
the same in both pipelines: raw item IDs from bundle_item.csv, sorted.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_blip.export_clhe --datasets ../bundlemllm/datasets --out <dir> [--variants clean published]
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from triad import data
from experiments.dials_blip.run import BlipVectorizer


def titles(name):
    """The title list in item order, built exactly as dials.data.load_resplit builds it."""
    root = data.DATA_ROOT/'BundleRec'/'dataset'/name
    items = sorted({int(i) for _, i in data.read_csv(root/'bundle_item.csv')});index = {it: k for k, it in enumerate(items)}
    out = ['']*len(items)
    for row in data.read_csv(root/'item_titles.csv'):
        if int(row[0]) in index:out[index[int(row[0])]] = ' '.join(row[1:])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets', required=True);p.add_argument('--out', required=True)
    p.add_argument('--variants', nargs='+', default=['clean'])
    a = p.parse_args()
    for name, variant in ((n, v) for n in ('electronic', 'food', 'clothing') for v in a.variants):
        src, dst = Path(a.datasets)/f'{name}_{variant}', Path(a.out)/f'{name}_{variant}_blip'
        shutil.copytree(src, dst, dirs_exist_ok=True)
        x = torch.as_tensor(BlipVectorizer().fit_transform(titles(name)), dtype=torch.float32)
        old = torch.load(src/'content_feature.pt', weights_only=True)
        assert len(x) == len(old), f'{name}: {len(x)} BLIP rows against {len(old)} items'
        torch.save(x, dst/'content_feature.pt');torch.save(x.clone(), dst/'description_feature.pt')
        print(name, tuple(x.shape), 'zero rows', int((x.abs().sum(1) == 0).sum()), '->', dst)


if __name__ == '__main__':
    main()
