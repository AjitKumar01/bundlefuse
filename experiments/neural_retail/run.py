"""Frozen v3.1 on Electronic, Food and Clothing under a plan fixed before training.

PYTHONPATH=. .venv/bin/python -m experiments.neural_retail.run plan
PYTHONPATH=. .venv/bin/python -m experiments.neural_retail.run train --dataset electronic --seed 0
PYTHONPATH=. .venv/bin/python -m experiments.neural_retail.run test
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from bundlefuse.provenance import write_json
from experiments.neural.data import load_data
from experiments.neural_lowrank import run as base
from experiments.neural_relations import run as r3  # sets base.construct to the v3.1 model
from experiments.neural_revision.run import log

DATASETS=['electronic','food','clothing']
RESULTS=Path('results/neural_retail');CHECKPOINTS=Path('checkpoints/neural_retail')
PLAN=dict(question='Does v3.1, frozen after development on POG, POG_DENSE and Spotify_sparse, transfer to small e-commerce bundle data?',
          model='v3.1 exactly as frozen in results/neural_relations/plan.json, no change to any setting; one label level (category).',
          data='Electronic, Food, Clothing: the seed 0 resplit used by the earlier neural baseline; catalogue title features (TF-IDF, SVD to 128). No complete bundle is shared across train, validation and test, so every test query counts.',
          training='Seeds 0, 1, 2; epoch chosen on validation NDCG@20. These datasets were never used in v3.1 development.',
          primary='Per dataset, three-seed per-query means on test: v3.1 vs the best simple baseline chosen on validation NDCG@20 (results/simple_baselines), NDCG@20 and Recall@20, Holm over all six tests.',
          secondary='v3.1 vs the earlier neural baseline (graph_pair_no_category, three seeds) where its test predictions exist. Published RaMen numbers are on a different split and are only indicative.',
          decision='Ahead or behind only with Holm p<.05 on NDCG@20. Otherwise not separated. Test sets are small (346-371 queries).')


def source_hash():
    h=hashlib.sha256()
    for path in [Path(__file__)]+sorted(Path('experiments/neural_relations').glob('*.py'))+[Path('experiments/neural_lowrank/run.py'),Path('experiments/neural_lowrank/model.py')]:h.update(path.read_bytes())
    return h.hexdigest()


def data(name):
    ds,x,c,f,_=load_data(name)
    return ds,x,c,f,dict(clean_test_indices=list(range(len(ds.test_pairs))))


def plan(_):
    path=RESULTS/'plan.json'
    if path.exists():raise RuntimeError('Plan already fixed')
    RESULTS.mkdir(parents=True,exist_ok=True);write_json(path,dict(PLAN,config=r3.FROZEN,source_sha256=source_hash()))


def train(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if json.loads((RESULTS/'plan.json').read_text())['source_sha256']!=source_hash():raise RuntimeError('Source changed after the plan was fixed')
    ds,x,c,_,_=data(args.dataset);key=f'v31_loo_seed{args.seed}'
    base.fit(ds,x,c,'loo',args.seed,r3.FROZEN,RESULTS/args.dataset/key,CHECKPOINTS/args.dataset/key)
    log(f'{args.dataset} {key} COMPLETE')


def test(_):
    from experiments.mechanisms.statistics import paired_comparison,holm_adjust
    from experiments.neural_revision.probes import target_frequency_metrics
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    plan_=json.loads((RESULTS/'plan.json').read_text())
    if plan_['source_sha256']!=source_hash():raise RuntimeError('Source changed after the plan was fixed')
    report=dict(plan=plan_['decision'],datasets={});records=[]
    for name in DATASETS:
        ds,x,c,f,_=data(name);rows=[]
        for s in (0,1,2):
            choice=json.loads((RESULTS/name/f'v31_loo_seed{s}'/'selected.json').read_text())
            if choice['config']!=r3.FROZEN:raise RuntimeError('Configuration differs from the frozen one')
            path=RESULTS/name/f'v31_loo_seed{s}'/'test.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                result=base.evaluate(base.load(ds,x,c,choice),ds.test_pairs);result['by_target_frequency']=target_frequency_metrics(result,ds.test_pairs,f)
                write_json(path,result)
            rows.append(result);log(f'{name} seed {s} test R20={result["recall20"]:.4f} N20={result["ndcg20"]:.4f}')
        new=np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in rows],axis=0)
        simple=json.loads(Path(f'results/simple_baselines/{name}/summary.json').read_text())['test']
        best=max(simple,key=lambda k:simple[k]['validation_ndcg20']);other=np.asarray(simple[best]['queries'])
        ids=base.groups(ds.test_pairs)
        report['datasets'][name]=dict(v31=new.mean(0).tolist(),seeds=[[r['recall20'],r['ndcg20']] for r in rows],
                                      best_simple=dict(method=best,params=simple[best]['params'],test=other.mean(0).tolist()),
                                      all_simple={k:[v['full']['recall20'],v['full']['ndcg20']] for k,v in simple.items()},
                                      by_target_frequency=[r['by_target_frequency'] for r in rows])
        for j,metric in ((1,'ndcg20'),(0,'recall20')):
            records.append(dict(paired_comparison(new[:,j],other[:,j],ids,bootstrap=2000,permutations=10000,seed=20261108),dataset=name,metric=metric,first='v31',second=f'simple:{best}'))
    holm_adjust(records,len(records));report['primary']=records
    write_json(RESULTS/'summary.json',report);log('test COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(required=True)
    sub.add_parser('plan').set_defaults(run=plan);sub.add_parser('test').set_defaults(run=test)
    t=sub.add_parser('train');t.set_defaults(run=train);t.add_argument('--dataset',required=True,choices=DATASETS)
    t.add_argument('--seed',type=int,required=True);t.add_argument('--threads',type=int,default=1)
    args=p.parse_args();args.run(args)
