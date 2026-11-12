"""Test a learned bilinear form and a category table on POG and POG_DENSE under a plan fixed before training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.neural_revision import training
from experiments.neural_revision.pog import prepare,metric_subset
from experiments.neural_revision.probes import target_frequency_metrics
from experiments.neural_revision.run import log
from .model import BilinearNetwork

DATASETS=['pog_dense','pog']
ARMS=dict(bilinear=dict(category_mode='off',bilinear=True),category=dict(category_mode='free',bilinear=False),both=dict(category_mode='free',bilinear=True))
CONFIG=dict(epochs=100,batch=32,eval_every=2,patience=6,link_lr=.03,shared_lr=.001,weight_decay=.001,log_every_batches=50)
RESULTS=Path('results/neural_bilinear');CHECKPOINTS=Path('checkpoints/neural_bilinear');BASELINE=Path('results/neural_pog')
PLAN=dict(question='Do a learned bilinear form V M V^T and a learned category table close the POG gap in the neural baseline?',
          arms={'bilinear':'M starts at the identity, unconstrained, shared learning rate','category':'free signed category table, starts at zero, link learning rate, no cold mask','both':'both changes'},
          unchanged='All other settings, features, masks, and epoch selection match the baseline. No BundleFuse signal is an input.',
          selection='On seed 0 validation NDCG@20, choose the one arm with the highest mean ratio to baseline across POG and POG_DENSE. The baseline is a candidate. Test is not used.',
          replication='Train the chosen arm at seeds 1 and 2; compare per-query three-seed means with the baseline on the test set.',
          primary='Filtered test queries. Selected vs baseline and selected vs BundleFuse, on NDCG@20 and Recall@20, per dataset. Holm over 8 tests.',
          decision=dict(helps='Selected beats baseline on NDCG with Holm p<.05 on both datasets.',
                        versus_bundlefuse='Ahead or behind only with Holm p<.05 on NDCG. Otherwise not separated, which is not equivalence.'),
          scope='Exploratory: the splits were inspected earlier, and these changes came from a validation audit of the same data.')


def source_hash():
    h=hashlib.sha256()
    for path in sorted(Path('experiments/neural_bilinear').glob('*.py')):h.update(path.read_bytes())
    return h.hexdigest()


def construct(ds,features,cats,condition):
    return BilinearNetwork(features,cats,ds.train,category_mode=condition['category_mode'],bilinear=condition['bilinear'])


training.construct=construct  # fit and load_selected build models through this name; neural_revision source stays unchanged


def condition(arm):
    return dict(name=arm,variant='graph_pair',cold_probability=.25,project_links=True,**ARMS[arm])


def data(name,smoke=False):
    out=RESULTS/name;out.mkdir(parents=True,exist_ok=True)
    ds,x,c,f,_,audit=prepare(name,Path('checkpoints/neural_pog')/name,out,False)
    if smoke:ds.train=ds.train[:64];ds.val_pairs=ds.val_pairs[:8];ds.test_pairs=ds.test_pairs[:8]
    return ds,x,c,f,audit


def plan(_):
    path=RESULTS/'plan.json'
    if path.exists():raise RuntimeError('Plan already fixed')
    RESULTS.mkdir(parents=True,exist_ok=True);write_json(path,dict(PLAN,config=CONFIG,source_sha256=source_hash()))


def train(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    results=args.results;checkpoints=args.checkpoints
    if not args.smoke and json.loads((RESULTS/'plan.json').read_text())['source_sha256']!=source_hash():raise RuntimeError('Source changed after the plan was fixed')
    ds,x,c,_,_=data(args.dataset,args.smoke)
    key=f'{args.arm}_seed{args.seed}'
    training.fit(ds,x,c,condition(args.arm),args.seed,CONFIG if not args.epochs else dict(CONFIG,epochs=args.epochs),results/args.dataset/key,checkpoints/args.dataset/key,log)
    log(f'{args.dataset} {key} COMPLETE')


def chosen(dataset,arm,seed):
    return json.loads((RESULTS/dataset/f'{arm}_seed{seed}'/'selected.json').read_text())


def select(_):
    table={};scores={}
    base={d:json.loads((BASELINE/d/'graph_pair_no_category_seed0'/'selected.json').read_text())['validation']['ndcg20'] for d in DATASETS}
    for arm in ARMS:
        table[arm]={d:chosen(d,arm,0)['validation']['ndcg20'] for d in DATASETS}
        scores[arm]=float(np.mean([table[arm][d]/base[d] for d in DATASETS]))
    scores['baseline']=1.;winner=max(scores,key=scores.get)
    write_json(RESULTS/'selection.json',dict(baseline=base,validation_ndcg20=table,mean_ratio=scores,selected=winner));log(f'selected {winner} {scores}')


def groups(pairs):return [str(tuple(sorted(set(s)|set(g)))) for s,g in pairs]


def test(_):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    winner=json.loads((RESULTS/'selection.json').read_text())['selected']
    if winner=='baseline':raise RuntimeError('Baseline selected: nothing advances to test')
    report=dict(selected=winner,plan=json.loads((RESULTS/'plan.json').read_text())['decision'],datasets={});records=[]
    for name in DATASETS:
        ds,x,c,f,audit=data(name);keep=audit['clean_test_indices'];rows=[]
        for seed in (0,1,2):
            path=RESULTS/name/f'{winner}_seed{seed}'/'test.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                result=training.evaluate(training.load_selected(ds,x,c,chosen(name,winner,seed)),ds.test_pairs)
                result['overlap_filtered']=metric_subset(result['queries'],keep);result['by_target_frequency']=target_frequency_metrics(result,ds.test_pairs,f)
                write_json(path,result)
            rows.append(result)
        baseline=[json.loads((BASELINE/name/f'graph_pair_no_category_seed{s}'/'test.json').read_text()) for s in (0,1,2)]
        reference=np.asarray(json.loads(Path(f'results/publication_mechanisms/{name}/queries.json').read_text())['metrics']['full'],float)
        mean=lambda runs:np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in runs],axis=0)
        new,old=mean(rows),mean(baseline);ids=groups(ds.test_pairs);clean=[ids[i] for i in keep]
        report['datasets'][name]=dict(
            seeds={k:[dict(full=[r['recall20'],r['ndcg20']],filtered=r['overlap_filtered'],by_target_frequency=r['by_target_frequency']) for r in runs] for k,runs in (('selected',rows),('baseline',baseline))},
            mean={k:dict(full=v.mean(0).tolist(),filtered=v[keep].mean(0).tolist()) for k,v in (('selected',new),('baseline',old),('bundlefuse',reference))})
        for second,other in (('baseline',old),('bundlefuse',reference)):
            for j,metric in ((1,'ndcg20'),(0,'recall20')):
                records.append(dict(paired_comparison(new[keep,j],other[keep,j],clean,bootstrap=2000,permutations=10000,seed=20261108),dataset=name,metric=metric,first='selected',second=second))
    holm_adjust(records,8);report['primary']=records
    write_json(RESULTS/'summary.json',report);log('test COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(required=True)
    sub.add_parser('plan').set_defaults(run=plan);sub.add_parser('select').set_defaults(run=select);sub.add_parser('test').set_defaults(run=test)
    t=sub.add_parser('train');t.set_defaults(run=train);t.add_argument('--dataset',required=True,choices=DATASETS);t.add_argument('--arm',required=True,choices=list(ARMS));t.add_argument('--seed',type=int,default=0)
    t.add_argument('--smoke',action='store_true');t.add_argument('--epochs',type=int,default=0);t.add_argument('--results',type=Path,default=RESULTS);t.add_argument('--checkpoints',type=Path,default=CHECKPOINTS)
    args=parser.parse_args();args.run(args)
