"""Tune each simple baseline on validation NDCG@20, then score test once with the chosen setting.

PYTHONPATH=. .venv/bin/python -m experiments.simple_baselines.run --dataset pog
PYTHONPATH=. .venv/bin/python -m experiments.simple_baselines.run --dataset pog --extended

--extended adds a second suite (content plus ItemKNN blend, iALS, SGFCF) to an existing summary.json. The blend uses the
catalogue features DIALS uses (on POG and Spotify read through dials.data, i.e. only the released BLIP / CLAP
features), and ItemKNN fixed at the setting the first suite chose on validation. --only restricts a suite to some methods.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.neural.evaluation import summarize
from experiments.neural_revision.run import log
from .methods import Baselines

RESULTS=Path('results/simple_baselines')
POG=['pog_dense','pog','spotify_sparse','spotify'];RETAIL=['electronic','food','clothing'];BASKETS=['dunnhumby']  # BASKETS: split and features from dials.data


def load(name):
    if name in BASKETS:
        from dials import data
        ds=data.load(name);return ds,ds.clean_test
    if name in RETAIL:
        from experiments.neural.data import load_data
        ds,*_=load_data(name);return ds,list(range(len(ds.test_pairs)))
    from bundlefuse import datasets
    from experiments.neural_revision.pog import overlap_audit
    ds=datasets.load(name,protocol_mode='provided',data_profile='extended',val_n=0,test_n=0);ds.features=None
    return ds,overlap_audit(ds.train,ds.val_pairs,ds.test_pairs)['clean_test_indices']


def evaluate(model,name,p,pairs,batch=128):
    records=[]
    for start in range(0,len(pairs),batch):
        rows=pairs[start:start+batch];records.extend(summarize(model.scores(name,p,[s for s,_ in rows]),rows)['queries'])
    return records


def mean(records,idx=None):
    rows=records if idx is None else [records[i] for i in idx]
    return dict(recall20=float(np.mean([r['recall20'] for r in rows])),ndcg20=float(np.mean([r['ndcg20'] for r in rows])))


def features(name,ds):
    if name in BASKETS:return ds.features
    if name in POG:  # the released features, through the same loader as DIALS
        from dials import data
        other=data.load(name,Path(DATA_ROOT)/'dials'/'cache')
        if [sorted(set(b)) for b in other.train]!=[sorted(set(b)) for b in ds.train]:raise RuntimeError('Training split differs')
        return other.features
    if name in RETAIL:
        from experiments.neural.data import load_data
        _,x,*_=load_data(name)
    else:
        from experiments.neural_lowrank import run as base
        other,x,*_=base.data(name)
        if [sorted(set(b)) for b in other.train]!=[sorted(set(b)) for b in ds.train]:raise RuntimeError('Training split differs')
    if len(x)!=ds.n_items:raise RuntimeError('Feature rows differ from the catalogue')
    return x


def run(args):
    out=RESULTS/args.dataset;out.mkdir(parents=True,exist_ok=True)
    ds,keep=load(args.dataset)
    if args.extended:
        previous=json.loads((out/'summary.json').read_text())
        model=Baselines(ds.train,ds.n_items,features(args.dataset,ds));grid=model.extended_grid(previous['test']['itemknn']['params'])
    else:model=Baselines(ds.train,ds.n_items);grid=model.grid(args.ease_limit)
    if args.only:grid=[(name,p) for name,p in grid if name in args.only]
    log(f'{args.dataset}: n={ds.n_items} train={len(ds.train)} pairs={model.C.nnz}')
    validation={}
    for name,p in grid:
        started=time.monotonic();v=mean(evaluate(model,name,p,ds.val_pairs))
        validation.setdefault(name,[]).append(dict(params=p,**v,seconds=time.monotonic()-started))
        log(f'{args.dataset} {name} {p} VAL N20={v["ndcg20"]:.6f}')
    report=dict(dataset=args.dataset,selection='Best validation NDCG@20 per method; test scored once with that setting.',validation=validation,test={})
    for name,rows in validation.items():
        best=max(rows,key=lambda r:r['ndcg20']);records=evaluate(model,name,best['params'],ds.test_pairs)
        report['test'][name]=dict(params=best['params'],validation_ndcg20=best['ndcg20'],full=mean(records),filtered=mean(records,keep),
                                  queries=[[r['recall20'],r['ndcg20']] for r in records])
        log(f'{args.dataset} {name} TEST filtered {report["test"][name]["filtered"]}')
    if args.extended:
        previous['validation'].update(validation);previous['test'].update(report['test']);report=previous
    write_json(out/'summary.json',report);log(f'{args.dataset} COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,choices=POG+RETAIL+BASKETS)
    p.add_argument('--ease_limit',type=int,default=40000,help='skip EASE when more training items than this (dense inverse)')
    p.add_argument('--extended',action='store_true',help='add the second suite to an existing summary.json')
    p.add_argument('--only',nargs='+',help='run only these methods of the suite')
    run(p.parse_args())
