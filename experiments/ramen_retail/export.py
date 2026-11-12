"""Write our Electronic, Food and Clothing split in RaMen's input format, in two variants.

published: the user-item file is BundleRec's user_item.csv, as RaMen's Table 1 counts imply (888 users and 6,165
           U-I rows on Electronic, 879 and 6,395 on Food). Each user's items there are exactly the items of that
           user's bundles, validation and test bundles included, so the item-item graph RaMen builds from it
           (E = X^T X, edge if E_ij >= 1) links every held-out target to its seeds.
clean:     the same file restricted to the items of each user's training bundles, which is what DIALS sees.

Bundle ids: training bundles first, then validation, then test, each in our order. Content and description features
are both our catalogue title features (TF-IDF, SVD to 128), because Food and Clothing have no images.
item_cf_feature.pt (used by CLHE, ignored by RaMen) stands in for CLHE's LightGCN item feature: a rank-64 SVD of the
variant's own normalized user-item matrix D_u^-1/2 X D_i^-1/2, item factors V sigma^1/2. Items with no user-item
entry get zeros, and CLHE treats them as cold.

PYTHONPATH=. .venv/bin/python -m experiments.ramen_retail.export --out ../baselines/RaMen_repo/datasets
"""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from bundlefuse.paths import DATA_ROOT
from experiments.neural.data import load_data

DATASETS=['electronic','food','clothing']


def rows(root,name):
    with open(os.path.join(root,name)) as f:
        r=csv.reader(f);next(r);return list(r)


def lines(path,entries):
    with open(path,'w') as f:
        for b,items in entries:f.write(', '.join(map(str,[b]+list(items)))+'\n')


def cf_feature(X,rank=64):
    from sklearn.utils.extmath import randomized_svd
    du=np.maximum(np.asarray(X.sum(1)).ravel(),1);di=np.maximum(np.asarray(X.sum(0)).ravel(),1)
    _,sigma,vt=randomized_svd(sp.diags(du**-.5)@X@sp.diags(di**-.5),rank,n_iter=5,random_state=0)
    v=(vt.T*np.sqrt(sigma)).astype(np.float32);v[np.asarray(X.sum(0)).ravel()==0]=0
    return torch.as_tensor(v)


def export(name,out):
    ds,x,_,_,_=load_data(name);n=ds.n_items;root=os.path.join(DATA_ROOT,'BundleRec','dataset',name)
    raw=collections.defaultdict(list)
    for b,i in rows(root,'bundle_item.csv'):raw[int(b)].append(int(i))
    items=sorted({i for v in raw.values() for i in v});imap={it:k for k,it in enumerate(items)}
    if len(items)!=n:raise RuntimeError('Item map differs from the loader')
    key={b:tuple(sorted({imap[i] for i in v})) for b,v in raw.items()}
    train_keys={tuple(sorted(set(b))) for b in ds.train}
    owner=collections.defaultdict(list)
    for u,b,_ in rows(root,'user_bundle.csv'):owner[int(u)].append(int(b))
    published=collections.defaultdict(list)
    for u,i,_ in rows(root,'user_item.csv'):published[int(u)].append(imap[int(i)])
    clean={u:[i for b in bs if key[b] in train_keys for i in key[b]] for u,bs in owner.items()}
    T,V,Q=len(ds.train),len(ds.val_pairs),len(ds.test_pairs)
    report={}
    for variant,users in (('published',published),('clean',clean)):
        users=[v for _,v in sorted(users.items()) if v]
        d=Path(out)/f'{name}_{variant}';d.mkdir(parents=True,exist_ok=True)
        lines(d/'bi_train.txt',[(k,sorted(set(b))) for k,b in enumerate(ds.train)])
        for tag,pairs,start in (('valid',ds.val_pairs,T),('test',ds.test_pairs,T+V)):
            lines(d/f'bi_{tag}_input.txt',[(start+k,sorted(set(s))) for k,(s,_) in enumerate(pairs)])
            lines(d/f'bi_{tag}_gt.txt',[(start+k,sorted(set(g))) for k,(_,g) in enumerate(pairs)])
        lines(d/'ui_full.txt',list(enumerate(users)))
        X=sp.csr_matrix((np.ones(sum(map(len,users)),np.float32),(np.repeat(np.arange(len(users)),list(map(len,users))),np.concatenate(users))),shape=(len(users),n))
        X.data[:]=1;X.sum_duplicates();X.data[:]=1
        E=(X.T@X).tocoo();keep=(E.data>=1)&(E.row!=E.col)
        np.save(d/'iui_edge_index.npy',np.vstack([E.row[keep],E.col[keep]]).astype(np.int64))
        feature=torch.as_tensor(x,dtype=torch.float32)
        torch.save(feature,d/'content_feature.pt');torch.save(feature.clone(),d/'description_feature.pt')
        torch.save(cf_feature(X),d/'item_cf_feature.pt')
        count={'#U':len(users),'#B':T+V+Q,'#I':n,'#B-I':int(sum(len(set(b)) for b in ds.train)),'#U-I':int(X.nnz)}
        (d/'count.json').write_text(json.dumps(count))
        E=E.tocsr();linked=[any(E[a,g]>0 for a in set(s)) for s,gt in ds.test_pairs for g in set(gt)]
        report[variant]=dict(count,iui_edges=int(keep.sum()),test_targets_linked_to_a_seed=float(np.mean(linked)))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);args=p.parse_args()
    report={name:export(name,args.out) for name in DATASETS}
    Path('results/ramen_retail').mkdir(parents=True,exist_ok=True)
    Path('results/ramen_retail/export.json').write_text(json.dumps(report,indent=1))
    print(json.dumps(report,indent=1))
