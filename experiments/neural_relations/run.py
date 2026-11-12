"""Development runs of the relation-family model (v3). Validation only; test needs a separate fixed plan.

PYTHONPATH=. .venv/bin/python -m experiments.neural_relations.run train --dataset spotify_sparse --tag v3
PYTHONPATH=. .venv/bin/python -m experiments.neural_relations.run inspect --dataset spotify_sparse --tag v3
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from bundlefuse.provenance import write_json
from experiments.neural_lowrank import run as base
from experiments.neural_revision.run import log
from .model import RelationFamily

RESULTS=Path('results/neural_relations');CHECKPOINTS=Path('checkpoints/neural_relations')
CONFIG=dict(base.CONFIG,group_lr=.01,use_album=True,graph_rank=64,model='relation_family_v3')


def levels(ds,cats,config):
    out=[cats]
    if config.get('use_album',True) and getattr(ds,'alb_labels',None) is not None:out.append(np.asarray(ds.alb_labels))
    return out


def spectral(ds,features_levels,config):
    """Cache the cross-fitted graph and label SVDs; they depend only on the training rows, labels and ranks."""
    import hashlib
    from bundlefuse.provenance import digest_rows
    from .model import cross_fitted
    h=hashlib.sha256(digest_rows(ds.train).encode())
    for l in features_levels:h.update(np.asarray(l,dtype='<i8').tobytes())
    path=CHECKPOINTS/ds.name/f'spectral_g{config["graph_rank"]}_l{config["group_rank"]}_folds5_{h.hexdigest()[:16]}.npz'
    if path.exists():
        saved=np.load(path)
        unpack=lambda key:((saved[f'{key}_left'],saved[f'{key}_right']),list(zip(saved[f'{key}_fold_left'],saved[f'{key}_fold_right'])))
        return dict(graph=unpack('graph'),labels=[unpack(f'label{i}') for i in range(len(features_levels))])
    out=cross_fitted(ds.train,ds.n_items,config['graph_rank'],features_levels,config['group_rank'])
    arrays={}
    for key,((left,right),folds) in [('graph',out['graph'])]+[(f'label{i}',v) for i,v in enumerate(out['labels'])]:
        arrays.update({f'{key}_left':left,f'{key}_right':right,f'{key}_fold_left':np.stack([f[0] for f in folds]),f'{key}_fold_right':np.stack([f[1] for f in folds])})
    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**arrays)
    return out


def construct(ds,features,cats,config=CONFIG):
    groups=levels(ds,cats,config)
    model=RelationFamily(features,groups,ds.train,rank=config['rank'],group_rank=config['group_rank'],hidden=config['hidden'],
                         graph_rank=config['graph_rank'],spectral=spectral(ds,groups,config),label_mode=config.get('label_mode','free'))
    model.disabled=set(config.get('disabled',()))
    return model


base.construct=construct  # base.fit, base.load and base.evaluate build models through this name


def train(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    ds,x,c,_,_=base.data(args.dataset)
    config=dict(CONFIG)
    if args.disable:config['disabled']=sorted(args.disable)
    for k in ('group_lr','fast_weight_decay','eval_every','label_mode'):
        if getattr(args,k) is not None:config[k]=getattr(args,k)
    if args.epochs:config['epochs']=args.epochs
    key=f'{args.tag}_{args.arm}_seed{args.seed}'
    base.fit(ds,x,c,args.arm,args.seed,config,RESULTS/args.dataset/key,CHECKPOINTS/args.dataset/key)
    log(f'{args.dataset} {key} COMPLETE')


@torch.inference_mode()
def inspect(args):
    """What the model learned: gates, hop weights, degree exponents, diagonal curves, and a validation ablation."""
    torch.set_num_threads(args.threads)
    ds,x,c,_,_=base.data(args.dataset);key=f'{args.tag}_{args.arm}_seed{args.seed}'
    choice=json.loads((RESULTS/args.dataset/key/'selected.json').read_text());model=base.load(ds,x,c,choice)
    seeds=[s for s,_ in ds.val_pairs];_,w=model.components(seeds[:256])
    names=['content']+[f'label{i}' for i in range(model.levels)]+['graph']
    diagonal={}
    for i in range(model.levels):
        size=float(np.median(getattr(model,f'label_size{i}').numpy()))
        diagonal[f'label{i}']={str(m):float(model.diagonal[i](torch.tensor([[np.log1p(m),np.log(10.),size]],dtype=torch.float32)).item()) for m in (1,2,4,8)}
    report=dict(gate_mean={k:float(w[:,j].mean()) for j,k in enumerate(names)},
                hops=model.hops.tolist(),exponents=dict(zip(('alpha','beta','gamma'),model.exponents.tolist())),
                diagonal_score_by_matches_at_10_seeds_median_label=diagonal,ablation_ndcg20={})
    for part in [None,'content',*[f'label{i}' for i in range(model.levels)],'graph','hop2','prior','diagonal','lowrank','gate']:
        model.disabled={part} if part else set()
        report['ablation_ndcg20'][part or 'full']=base.evaluate(model,ds.val_pairs)['ndcg20'];log(f'without {part}: {report["ablation_ndcg20"][part or "full"]:.6f}')
    write_json(RESULTS/args.dataset/key/'inspect.json',report);print(json.dumps(report,indent=1))


FROZEN=dict(CONFIG,label_mode='spectral',eval_every=1)
TAG='v31';ARM='loo';DATASETS=base.POG+base.SPOTIFY
PLAN=dict(question='Does the relation-family model v3.1, frozen after development on validation, match or beat BundleFuse on held-out test queries?',
          model='v3.1: content relation, label relations (learned diagonal plus cross-fitted spectral low-rank, 16x16 mixing), cross-fitted co-purchase graph filter, query gate; leave-one-out training without the random mask.',
          development='Designed on seed 0 validation of POG, POG_DENSE and Spotify_sparse (v1 to v3.1). Full Spotify was not used in development. No test query has been scored by any v2, v3 or v3.1 model.',
          training='Seeds 0, 1, 2 on POG, POG_DENSE, Spotify_sparse and Spotify with this configuration; epoch chosen on validation NDCG@20. Seed 0 development runs with this exact configuration are reused.',
          primary='Filtered test queries (complete bundle in neither training nor validation), three-seed per-query means. v3.1 vs BundleFuse on all four datasets; v3.1 vs the earlier category model on POG and POG_DENSE. NDCG@20 and Recall@20. Holm over all of these tests together.',
          references=dict(bundlefuse='Publication recipe predictions where they exist; Spotify_sparse uses the general_lift recipe refit on filtered queries.',category='results/neural_bilinear category arm, three seeds.'),
          decision='Ahead or behind only with Holm p<.05 on NDCG@20. Otherwise not separated, which is not equivalence.',
          scope='Exploratory: POG splits were inspected in earlier work and the design was developed on the same validation queries. Full Spotify is the cleanest check.')


def source_hash():
    import hashlib
    h=hashlib.sha256()
    for path in sorted(Path('experiments/neural_relations').glob('*.py'))+[Path('experiments/neural_lowrank/run.py'),Path('experiments/neural_lowrank/model.py')]:h.update(path.read_bytes())
    return h.hexdigest()


def plan(_):
    path=RESULTS/'plan.json'
    if path.exists():raise RuntimeError('Plan already fixed')
    write_json(path,dict(PLAN,config=FROZEN,tag=TAG,arm=ARM,source_sha256=source_hash()))


def train_frozen(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if json.loads((RESULTS/'plan.json').read_text())['source_sha256']!=source_hash():raise RuntimeError('Source changed after the plan was fixed')
    ds,x,c,_,_=base.data(args.dataset);key=f'{TAG}_{ARM}_seed{args.seed}'
    base.fit(ds,x,c,ARM,args.seed,FROZEN,RESULTS/args.dataset/key,CHECKPOINTS/args.dataset/key)
    log(f'{args.dataset} {key} COMPLETE')


def test(_):
    from experiments.mechanisms.statistics import paired_comparison,holm_adjust
    from experiments.neural_revision.pog import metric_subset
    from experiments.neural_revision.probes import target_frequency_metrics
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    plan_=json.loads((RESULTS/'plan.json').read_text())
    if plan_['source_sha256']!=source_hash():raise RuntimeError('Source changed after the plan was fixed')
    report=dict(plan=plan_['decision'],datasets={});records=[]
    for name in DATASETS:
        keys=[f'{TAG}_{ARM}_seed{s}' for s in (0,1,2)]
        if not all((RESULTS/name/k/'selected.json').exists() for k in keys):raise RuntimeError(f'{name}: not all seeds trained')
        ds,x,c,f,audit=base.data(name);keep=audit['clean_test_indices'];rows=[]
        for k in keys:
            choice=json.loads((RESULTS/name/k/'selected.json').read_text())
            if choice['config']!=FROZEN:raise RuntimeError(f'{name} {k}: configuration differs from the frozen one')
            path=RESULTS/name/k/'test.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                result=base.evaluate(base.load(ds,x,c,choice),ds.test_pairs)
                result['overlap_filtered']=metric_subset(result['queries'],keep);result['by_target_frequency']=target_frequency_metrics(result,ds.test_pairs,f)
                write_json(path,result)
            rows.append(result);log(f'{name} {k} filtered test {result["overlap_filtered"]}')
        mean=lambda runs:np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in runs],axis=0)
        new=mean(rows);others={}
        bundlefuse=base.reference(name,ds.test_pairs)
        if bundlefuse is not None:others['bundlefuse']=bundlefuse
        if name in base.POG:others['category']=mean([json.loads((base.PREVIOUS/name/f'category_seed{s}'/'test.json').read_text()) for s in (0,1,2)])
        ids=base.groups(ds.test_pairs);clean=[ids[i] for i in keep]
        report['datasets'][name]=dict(seeds=[dict(full=[r['recall20'],r['ndcg20']],filtered=r['overlap_filtered'],by_target_frequency=r['by_target_frequency']) for r in rows],
                                      mean={k:dict(full=v.mean(0).tolist() if np.isfinite(v).all() else None,filtered=v[keep].mean(0).tolist()) for k,v in dict(v31=new,**others).items()})
        for second,other in others.items():
            for j,metric in ((1,'ndcg20'),(0,'recall20')):
                records.append(dict(paired_comparison(new[keep,j],other[keep,j],clean,bootstrap=2000,permutations=10000,seed=20261108),dataset=name,metric=metric,first='v31',second=second))
    holm_adjust(records,len(records));report['primary']=records
    write_json(RESULTS/'summary.json',report);log('test COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(required=True)
    for name,fn in (('plan',plan),('test',test)):sub.add_parser(name).set_defaults(run=fn)
    t=sub.add_parser('frozen');t.set_defaults(run=train_frozen);t.add_argument('--dataset',required=True,choices=DATASETS);t.add_argument('--seed',type=int,required=True);t.add_argument('--threads',type=int,default=1)
    for name,fn in (('train',train),('inspect',inspect)):
        t=sub.add_parser(name);t.set_defaults(run=fn)
        t.add_argument('--dataset',required=True,choices=base.POG+base.SPOTIFY);t.add_argument('--arm',default='loo',choices=list(base.ARMS))
        t.add_argument('--tag',required=True);t.add_argument('--seed',type=int,default=0);t.add_argument('--threads',type=int,default=1)
        t.add_argument('--epochs',type=int,default=0);t.add_argument('--disable',nargs='*',default=[])
        t.add_argument('--group_lr',type=float);t.add_argument('--label_mode',choices=['free','spectral']);t.add_argument('--fast_weight_decay',type=float);t.add_argument('--eval_every',type=int)
    args=p.parse_args();args.run(args)
