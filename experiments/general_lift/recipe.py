"""Refit each dataset's current recipe on this source and score the filtered test queries."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from bundlefuse import datasets
from bundlefuse.metrics import ranking_metrics,top_items
from bundlefuse.provenance import write_json
from experiments.sufficiency.engine import metrics
from .run import DATASETS,population,log


def run(args):
    name=args.dataset;out=args.output/name;(out/'queries').mkdir(parents=True,exist_ok=True)
    model,ds=datasets.build(name)
    log(f'{name}: fit recipe on {len(ds.val_pairs)} validation queries')
    model.fit(ds.train,ds.val_pairs,**ds.fit_kwargs())
    train=[list(dict.fromkeys(b)) for b in (ds.train.values() if isinstance(ds.train,dict) else ds.train)]
    _,ti=population(train,ds.val_pairs,ds.test_pairs)
    test=[ds.test_pairs[i] for i in ti]
    row=dict(metrics=[],predictions=[],groups=[])
    for i,pair in enumerate(test):
        values=np.asarray(model.score_batch([list(pair[0])]))[:,0]
        row['metrics'].append(metrics(values,pair).tolist());row['predictions'].append(top_items(values,pair[0],20).tolist())
        row['groups'].append(str(tuple(sorted(set(pair[0])|set(pair[1])))))
        if (i+1)%500==0:log(f'recipe test {i+1}/{len(test)}')
    row['mean']=np.mean(row['metrics'],axis=0).tolist()
    write_json(out/'queries'/'recipe.json',row)
    write_json(out/'recipe.json',dict(weights=model.w,prior_keys=list(model.prior_keys),backoff=list(model.best_eb),
                                      higher_order=list(model.best_ho),filtered_test=row['mean']))
    log(f'{name} recipe COMPLETE filtered test {row["mean"]}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=DATASETS)
    parser.add_argument('--output',type=Path,default=Path('results/general_lift'))
    args=parser.parse_args();torch.set_num_threads(1)
    with torch.inference_mode():run(args)
