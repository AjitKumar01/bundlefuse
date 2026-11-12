"""Training sets for the bundle-budget test: how does bundle completion degrade as training bundles disappear?

Validation and test queries are never changed; only the bundles the methods learn from are.
  full          all training bundles (the published split)
  frac10, frac1 nested random 10% and 1% of the training bundles (one fixed permutation, seed 20261113)
  pseudo        no bundles at all: each user's interaction history (ui_full.txt of the released dataset) is one
                pseudo-bundle; unique items, users with at least 2 items, histories over 30 items split at random into
                pieces of at most 30
  pseudo_clean  pseudo, minus every user who alone holds all seeds and at least one target of some validation or test
                query (a contamination check; it can only remove evidence)
DIALS trains on at most as many rows per epoch as the full split has bundles: when a condition has more, a fixed
random sample of that size is the training rows, while counts, label factors and partner histories use all of them.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget.conditions
"""
import dials  # pins BLAS threads before numpy loads
import dataclasses
import numpy as np
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from dials import data as dials_data
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import CACHE

DATASETS=('pog','pog_dense');CONDITIONS=('full','frac10','frac1','pseudo','pseudo_clean')
SEED=20261113;CAP=30;OUT=Path('results/bundle_budget')


@lru_cache(maxsize=None)
def dataset(name):return dials_data.load(name,CACHE)


def histories(name):
    rng=np.random.default_rng(SEED);out=[]
    for line in (Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/'ui_full.txt').open():
        v=[int(x) for x in line.replace(' ','').rstrip(',\n').split(',') if x!='']
        items=sorted(set(v[1:]))
        if len(items)<2:continue
        if len(items)<=CAP:out.append(items);continue
        items=rng.permutation(items).tolist()
        for k in range(0,len(items),CAP):
            piece=sorted(items[k:k+CAP])
            if len(piece)>=2:out.append(piece)
    return out


def contaminating(ds,bundles):
    """Indices of pseudo-bundles that alone hold all seeds and a target of a validation or test query."""
    owners=defaultdict(set)
    for u,b in enumerate(bundles):
        for i in b:owners[i].add(u)
    bad=set()
    for seeds,truth in ds.val_pairs+ds.test_pairs:
        common=None
        for i in set(seeds):
            common=set(owners.get(i,())) if common is None else common&owners.get(i,set())
            if not common:break
        if common:bad|={u for u in common if any(u in owners.get(g,()) for g in truth)}
    return bad


@lru_cache(maxsize=None)
def condition(name,cond):
    """(statistics bundles, training rows) for one condition."""
    ds=dataset(name);m=len(ds.train)
    if cond=='full':return ds.train,ds.train
    if cond in ('frac10','frac1'):
        order=np.random.default_rng(SEED).permutation(m);k=int(np.ceil(m*(.1 if cond=='frac10' else .01)))
        part=[ds.train[i] for i in sorted(order[:k])];return part,part
    stats=histories(name)
    if cond=='pseudo_clean':
        bad=contaminating(ds,stats);stats=[b for u,b in enumerate(stats) if u not in bad]
    if len(stats)<=m:return stats,stats
    pick=np.sort(np.random.default_rng(SEED).choice(len(stats),m,replace=False));return stats,[stats[i] for i in pick]


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
        for cond in CONDITIONS:
            stats,rows=condition(name,cond)
            r=dict(bundles=len(stats),training_rows=len(rows),mean_size=float(np.mean([len(set(b)) for b in stats])),**coverage(ds,stats))
            if cond=='pseudo':r['contaminating_users']=len(contaminating(ds,stats))
            report[name][cond]=r;print(name,cond,r,flush=True)
    write_json(OUT/'conditions.json',dict(note=__doc__,conditions=report))


if __name__=='__main__':main()
