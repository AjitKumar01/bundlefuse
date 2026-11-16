"""Training sets for the fixed-count bundle-budget test: Electronic, Food and Clothing (family "retail"), Spotify_sparse
and Spotify (family "spotify").

The e-commerce datasets have only about 1,200 training bundles, so the POG test's 10% and 1% would leave 120 and 12
bundles. Fixed counts are used instead, which is also how a retailer would describe its data:
  full            all training bundles (the split of the frozen runs)
  n3000 ... n50   nested random subsets of that many training bundles (one fixed permutation per dataset, seed
                  20261113): 600, 300, 100, 50 for every dataset, and also 3,000 and 1,000 for Spotify, whose full
                  split is larger
Validation and test queries are never changed; only the bundles the methods learn from are. Statistics (counts, label
factors, partner histories) and training rows are the same bundles.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget_retail.conditions
"""
import dials  # pins BLAS threads before numpy loads
import dataclasses
import numpy as np
from functools import lru_cache
from pathlib import Path
from dials import data as dials_data
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import CACHE

FAMILIES=dict(retail=('electronic','food','clothing'),spotify=('spotify_sparse','spotify'))
DATASETS=sum(FAMILIES.values(),())
SMALL=(600,300,100,50);COUNTS=dict(electronic=SMALL,food=SMALL,clothing=SMALL,spotify_sparse=(3000,1000)+SMALL,spotify=(3000,1000)+SMALL)
CONDITIONS=('full','n3000','n1000')+tuple(f'n{k}' for k in SMALL)


def conditions(name):
    """The conditions of one dataset, largest first."""
    return ('full',)+tuple(f'n{k}' for k in COUNTS[name])
SEED=20261113;OUT=Path('results/bundle_budget_retail')


@lru_cache(maxsize=None)
def dataset(name):return dials_data.load(name,CACHE)


@lru_cache(maxsize=None)
def condition(name,cond):
    """The training bundles of one condition."""
    ds=dataset(name)
    if cond=='full':return ds.train
    k=int(cond[1:]);order=np.random.default_rng(SEED).permutation(len(ds.train))
    return [ds.train[i] for i in sorted(order[:k])]


def with_train(ds,bundles):return dataclasses.replace(ds,train=list(bundles))


def coverage(ds,bundles):
    seen=np.zeros(ds.n_items,bool)
    for b in bundles:seen[list(b)]=True
    targets=[g for k in ds.clean_test for g in ds.test_pairs[k][1]];seeds=[s for k in ds.clean_test for s in ds.test_pairs[k][0]]
    return dict(catalogue=float(seen.mean()),test_seeds=float(seen[seeds].mean()),test_targets=float(seen[targets].mean()))


def main():
    report={}
    for name in DATASETS:
        ds=dataset(name);report[name]={}
        for cond in conditions(name):
            b=condition(name,cond)
            r=dict(bundles=len(b),mean_size=float(np.mean([len(set(x)) for x in b])),**coverage(ds,b))
            report[name][cond]=r;print(name,cond,{k:round(v,3) for k,v in r.items()},flush=True)
    write_json(OUT/'conditions.json',dict(note=__doc__,conditions=report))


if __name__=='__main__':main()
