"""Revision pilot: select architecture on validation, replicate, diagnose, then test."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tarfile
import time
import numpy as np
import torch
from bundlefuse.provenance import write_json
from bundlefuse.splits import file_hash
from experiments.neural.data import load_data
from .training import fit,load_selected,evaluate,construct
from .probes import compare_sources,target_frequency_metrics
from .assays import run_assays


CONDITIONS=[dict(name='attributes',variant='attributes',cold_probability=0.),
            dict(name='pair_no_cold',variant='pair',cold_probability=0.),
            dict(name='pair_cold',variant='pair',cold_probability=.25),
            dict(name='joint_cold',variant='joint',cold_probability=.25),
            dict(name='graph_joint_cold',variant='graph_joint',cold_probability=.25)]
CONSTRAINED=[dict(name='graph_pair_no_category',variant='graph_pair',cold_probability=.25,category_mode='off',project_links=True),
             dict(name='graph_joint_no_category',variant='graph_joint',cold_probability=.25,category_mode='off',project_links=True),
             dict(name='graph_joint_positive_category',variant='graph_joint',cold_probability=.25,category_mode='nonnegative',project_links=True)]


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),message,flush=True)


def source_identity():
    paths=sorted(Path('experiments/neural_revision').glob('*.py'))+sorted(Path('experiments/neural').glob('*.py'))+sorted(Path('bundlefuse').rglob('*.py'))
    paths += [Path('tests/test_neural_revision.py'),Path('scripts/run_neural_revision.sh'),Path('scripts/run_neural_constrained.sh'),Path('scripts/run_neural_transfer.sh')]
    h=hashlib.sha256()
    for p in paths:h.update(str(p).encode());h.update(p.read_bytes())
    return h.hexdigest(),paths


def write_report(out,choices,baseline,status,test=None):
    data=dict(status=status,selected=choices,baseline=baseline,test=test or {})
    write_json(out/'summary.json',data)
    lines=['# Revised independent neural model','',f'Status: {status}','',
           'Architecture selection and checkpoints use validation NDCG only. Test scores are shown only after all selections are frozen.','',
           '| Model | Validation R20 | Validation N20 | Test R20 | Test N20 |','|---|---:|---:|---:|---:|']
    if baseline:
        for name,row in baseline.items():
            val=row['validation'];t=row['test']
            lines.append(f'| {name} | {val["recall20"]:.6f} | {val["ndcg20"]:.6f} | {t["recall20"]:.6f} | {t["ndcg20"]:.6f} |')
    for key,row in choices.items():
        val=row['validation'];t=(test or {}).get(key)
        tail=f'{t["recall20"]:.6f} | {t["ndcg20"]:.6f}' if t else 'pending | pending'
        lines.append(f'| {key} | {val["recall20"]:.6f} | {val["ndcg20"]:.6f} | {tail} |')
    lines += ['', 'All source scores in signal probes are diagnostic targets only. The scorer learns from masked training bundles, with no BundleFuse teacher or score input. Existing splits have been inspected; this revision is exploratory.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=args.output/args.dataset;ckpt=args.checkpoints/args.dataset
    out.mkdir(parents=True,exist_ok=True);ckpt.mkdir(parents=True,exist_ok=True)
    with (out/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ds,features,cats,freq,identity=load_data(args.dataset,args.smoke)
        conditions=CONSTRAINED if args.phase=='constrained' else CONDITIONS
        if args.fixed_condition:
            conditions=[c for c in conditions if c['name']==args.fixed_condition]
            if len(conditions)!=1:raise ValueError('Fixed condition not present in selected phase')
        config=dict(epochs=args.epochs,batch=32,eval_every=2,patience=6,link_lr=.03,shared_lr=.001,weight_decay=.001)
        h,paths=source_identity()
        protocol=dict(data=identity,conditions=conditions,config=config,source_sha256=h,phase=args.phase,
                      fixed_condition=args.fixed_condition,
                      selection='All conditions seed 0; select highest validation NDCG architecture, then repeat it at seeds 1 and 2; freeze before test.',
                      simulation='Hide all direct, joint and graph paths incident on a random 25% of items each batch. No frequency feature or item-ID embeddings.',
                      objective='Full-catalogue per-positive completion softmax, excluding observed and other known positive items.',
                      smoke=args.smoke,torch_version=torch.__version__)
        for p in [out/'protocol.json',ckpt/'protocol.json']:
            if p.exists() and json.loads(p.read_text())!=protocol:raise RuntimeError('Source/protocol changed; use new output AND checkpoint roots')
        write_json(out/'protocol.json',protocol);write_json(ckpt/'protocol.json',protocol)
        if not (ckpt/'source.tar.gz').exists():
            with tarfile.open(ckpt/'source.tar.gz','w:gz') as tar:
                for p in paths:tar.add(p,arcname=str(p))
        log(f'{args.dataset}: revised pilot train={len(ds.train)} val={len(ds.val_pairs)} test={len(ds.test_pairs)}')
        write_json(out/'architecture_assays.json',run_assays())
        baseline={};reference=None
        if not args.smoke:
            from bundlefuse.model import BundleFuse
            from experiments.neural.evaluation import evaluate_classical
            old=json.loads(Path(f'results/neural_small/{args.dataset}/comparisons.json').read_text())
            for name,path in [('bundlefuse_matched',f'checkpoints/neural_small/{args.dataset}/comparisons/bundlefuse_matched.pt'),
                              ('bundlefuse_original_auxiliary',f'checkpoints/reorganized_162dd8c/{args.dataset}/seed_0/model.pt')]:
                m=BundleFuse.load(path)
                for k in ['train_sha256','val_pairs_sha256','test_pairs_sha256','input_sha256']:
                    if ds.metadata[k]!=m.data_spec[k]:raise RuntimeError(f'Comparator mismatch: {k}')
                val=evaluate_classical(m,ds.val_pairs,ds.val_users if 'original' in name else None)
                baseline[name]=dict(validation={k:v for k,v in val.items() if k!='queries'},test=old[name],checkpoint=path,sha256=file_hash(path))
                if name=='bundlefuse_matched':reference=m
        choices={};write_report(out,choices,baseline,'training')
        for condition in conditions:
            key=condition['name']+'_seed0'
            choices[key]=fit(ds,features,cats,condition,0,config,out/key,ckpt/key,log)
            write_report(out,choices,baseline,'training')
        winner=max(choices,key=lambda key:choices[key]['validation']['ndcg20'])
        winning_condition=choices[winner]['condition']
        write_json(out/'architecture_selection.json',dict(selected_condition=winning_condition,criterion='seed-0 validation NDCG only',
                                                       candidates={k:v['validation'] for k,v in choices.items()}))
        log(f'VALIDATION SELECTED {winning_condition["name"]}; repeat training seeds 1 and 2')
        for seed in [1,2]:
            key=winning_condition['name']+f'_seed{seed}'
            choices[key]=fit(ds,features,cats,winning_condition,seed,config,out/key,ckpt/key,log)
            write_report(out,choices,baseline,'training')
        write_json(out/'frozen_selection.json',choices)
        # Diagnostics measure what fixed learned scores retain; they cannot alter training/selection.
        if reference is not None:
            for condition in conditions:
                key=condition['name']+'_seed0';p=out/key/'signal_probe.json'
                if not p.exists():
                    log(f'Diagnostic source probes for {key}')
                    m=load_selected(ds,features,cats,choices[key]);probe=compare_sources(m,reference,ds.val_pairs)
                    write_json(p,probe)
            p=out/'initial_attribute_probe.json'
            if not p.exists():
                initial=construct(ds,features,cats,CONDITIONS[0])
                write_json(p,compare_sources(initial,reference,ds.val_pairs))
        tests={}
        for key,choice in choices.items():
            p=out/key/'test.json'
            if p.exists():result=json.loads(p.read_text())
            else:
                model=load_selected(ds,features,cats,choice);result=evaluate(model,ds.test_pairs)
                result['by_target_training_frequency']=target_frequency_metrics(result,ds.test_pairs,freq)
                write_json(p,result)
            tests[key]={k:v for k,v in result.items() if k!='queries'}
            log(f'{key}: TEST R20={result["recall20"]:.6f} N20={result["ndcg20"]:.6f}')
        for row in baseline.values():
            if file_hash(row['checkpoint'])!=row['sha256']:raise RuntimeError('Comparator checkpoint changed')
        write_report(out,choices,baseline,'smoke_complete' if args.smoke else 'complete',tests)
        log(f'{args.dataset}: COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['electronic','food','clothing'],default='electronic')
    p.add_argument('--phase',choices=['initial','constrained'],default='initial')
    p.add_argument('--fixed-condition',default=None,help='Freeze architecture from an earlier dataset; still select epochs on this validation split')
    p.add_argument('--output',type=Path,default=Path('results/neural_revision'));p.add_argument('--checkpoints',type=Path,default=Path('checkpoints/neural_revision'))
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.epochs<1:p.error('Positive epochs required')
    if a.smoke and (a.output==Path('results/neural_revision') or a.checkpoints==Path('checkpoints/neural_revision')):p.error('Separate smoke output and checkpoint roots required')
    if a.phase=='constrained' and (a.output==Path('results/neural_revision') or a.checkpoints==Path('checkpoints/neural_revision')):p.error('Constrained follow-up needs separate output AND checkpoint roots')
    run(a)
