"""Write <dataset>_pretrain_blip for CLHE and RaMen: the <dataset>_clean_blip export (same split, same BLIP title
features) with its user-item graph replaced by BundleRec's user_item_pretrain.csv.

user_item_pretrain.csv is a separate purchase log (1.95 M rows on Electronic, 325 k on Food, 749 k on Clothing). It
shares only 2-3 (user, item) pairs with the bundles' own user_item.csv (which holds every bundle, test bundles
included), so it does not contain the test bundles. Rows on items outside our catalogue are dropped, users left with
no item are dropped, and repeated (user, item) rows count once. From it, as in export.py:
  ui_full.txt          one line per user: the catalogue items it bought
  iui_edge_index.npy   RaMen's item-item graph, E = X^T X, edge where E_ij >= 1 (epsilon = 1, as in RaMen's paper for
                       Food and Electronic)
  item_cf_feature.pt   CLHE's collaborative item feature, a rank-64 SVD of D_u^-1/2 X D_i^-1/2 (item factors)
  count.json           #U and #U-I updated
The report gives the share of test targets that the item-item graph links to one of their seeds (on the clean graph
this is low, on the published one it is 1 by construction).

PYTHONPATH=.:../dials .venv/bin/python -m experiments.ramen_retail.export_pretrain --datasets <dir> --out <dir>
"""
import argparse
import collections
import json
import shutil
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from bundlefuse.paths import DATA_ROOT
from .export import rows, lines, cf_feature

DATASETS = ['electronic', 'food', 'clothing']


def read(path):
    return {int(l.split(', ')[0]): [int(v) for v in l.strip().split(', ')[1:]] for l in open(path) if l.strip()}


def export(name, src_root, out_root):
    src, dst = Path(src_root)/f'{name}_clean_blip', Path(out_root)/f'{name}_pretrain_blip'
    shutil.copytree(src, dst, dirs_exist_ok=True)
    root = Path(DATA_ROOT)/'BundleRec'/'dataset'/name
    items = sorted({int(i) for _, i in rows(root, 'bundle_item.csv')});imap = {it: k for k, it in enumerate(items)};n = len(items)
    users = collections.defaultdict(set)
    for u, i, _ in rows(root, 'user_item_pretrain.csv'):
        if int(i) in imap:users[int(u)].add(imap[int(i)])
    users = [sorted(v) for _, v in sorted(users.items()) if v]
    lines(dst/'ui_full.txt', list(enumerate(users)))
    X = sp.csr_matrix((np.ones(sum(map(len, users)), np.float32), (np.repeat(np.arange(len(users)), list(map(len, users))), np.concatenate(users))),
                      shape=(len(users), n))
    E = (X.T@X).tocoo();keep = (E.data >= 1) & (E.row != E.col)
    np.save(dst/'iui_edge_index.npy', np.vstack([E.row[keep], E.col[keep]]).astype(np.int64))
    torch.save(cf_feature(X), dst/'item_cf_feature.pt')
    count = json.loads((dst/'count.json').read_text());count.update({'#U': len(users), '#U-I': int(X.nnz)});(dst/'count.json').write_text(json.dumps(count))
    train, seeds, truth = read(dst/'bi_train.txt'), read(dst/'bi_test_input.txt'), read(dst/'bi_test_gt.txt')
    E = E.tocsr();linked = [any(E[a, g] > 0 for a in seeds[b]) for b in seeds for g in truth[b]]
    return dict(count, iui_edges=int(keep.sum()), items_with_history=int((np.asarray(X.sum(0)).ravel() > 0).sum()),
                test_targets_linked_to_a_seed=float(np.mean(linked)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets', required=True);p.add_argument('--out', required=True);a = p.parse_args()
    report = {name: export(name, a.datasets, a.out) for name in DATASETS}
    Path('results/ramen_retail/export_pretrain.json').write_text(json.dumps(report, indent=1));print(json.dumps(report, indent=1))
