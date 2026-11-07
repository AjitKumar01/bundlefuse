"""Resumable amplified-neighbor/fallback follow-up on the remaining datasets."""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_key]='1'
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from bundlefuse.metrics import ranking_metrics, top_items
from bundlefuse.provenance import write_json, run_metadata, digest_rows
from experiments.novelty.data import validation_groups
from experiments.novelty.raw import AmplifiedNeighbors
from experiments.mechanisms.statistics import paired_comparison, holm_adjust
from .protocol import PLAN, SEED, DATASETS, CONTRASTS
from .data import load_inputs, source_columns, cache_signals, cache_amplified, ScoreBank
from .fitting import fit, score


def log(message):print(f'{datetime.now(timezone.utc).isoformat()} {message}',flush=True)


def fingerprint():
    paths=sorted(Path('bundlefuse').rglob('*.py'))
    for root in ['experiments/amplified_fallback','experiments/novelty','experiments/mechanisms']:
        paths+=sorted(Path(root).glob('*.py'))
    return hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in paths)).hexdigest()


def canonical_method(recipe):
    weights={('hist' if k=='simp' else k):v for k,v in recipe['weights'].items() if v}
    gamma=recipe['higher_order'][1]
    if gamma:weights['rao']=gamma
    prior={k:recipe['weights'].get(k,.5) for k in recipe['prior']}
    if gamma:prior['rao']=gamma
    if recipe['weights'].get('pop',0)>0:prior['pop']=recipe['weights']['pop']
    return dict(kind='gate',weights=weights,prior_weights=prior,
                kappa=recipe['backoff'][0],scale=recipe['backoff'][1])


def verify_reference(model,pairs,users,bank,recipe,directory):
    cfg=canonical_method(recipe);errors=[]
    for i in sorted(set([0,len(pairs)//2,len(pairs)-1])):
        expected=np.asarray(model.score_batch([pairs[i][0]],None if users is None else users[i:i+1])).ravel()
        derived=score(bank.query(i),cfg,np.asarray(model.freq))
        errors.append(float(np.max(np.abs(expected-derived))))
    write_json(directory/'reference_parity.json',dict(query_indices=sorted(set([0,len(pairs)//2,len(pairs)-1])),
               errors=errors,tolerance=2e-4))
    if max(errors)>2e-4:raise RuntimeError(f'Canonical decomposition differs: {errors}')


def assess(model,pairs,users,banks,methods,directory,chunk):
    directory.mkdir(parents=True,exist_ok=True)
    groups=[str(tuple(sorted(set(s)|set(gt)))) for s,gt in pairs]
    for name,method in methods.items():
        path=directory/f'{name.replace(":","__")}.json'
        if path.exists():continue
        marker=path.with_suffix('.progress.json')
        rows=json.loads(marker.read_text()) if marker.exists() else dict(metrics=[],predictions=[],seconds=0.)
        begin=time.monotonic();initial=rows['seconds']
        for start in range(len(rows['metrics']),len(pairs),chunk):
            stop=min(start+chunk,len(pairs))
            if name=='original_reference':
                values=np.asarray(model.score_batch([s for s,_ in pairs[start:stop]],
                    None if users is None else users[start:stop]))
            else:
                bank=banks[method['bank']]
                values=np.stack([score(bank.query(i),method,np.asarray(model.freq)) for i in range(start,stop)],1)
            for j,i in enumerate(range(start,stop)):
                rows['metrics'].append(list(ranking_metrics(values[:,j:j+1],[pairs[i]],Ks=(20,))[20]))
                rows['predictions'].append(top_items(values[:,j],pairs[i][0],20).tolist())
            rows['seconds']=initial+time.monotonic()-begin
            if stop%100<chunk or stop==len(pairs):
                write_json(marker,rows);log(f'test {name}: {stop}/{len(pairs)}')
        rows['groups']=groups;write_json(path,rows)
        log(f'test {name} COMPLETE metrics={np.mean(rows["metrics"],0).tolist()}')


def report(directory,identity,smoke):
    methods={p.stem.replace('__',':'):json.loads(p.read_text()) for p in (directory/'queries').glob('*.json')
             if not p.name.endswith('.progress.json')}
    summary={k:dict(zip(['recall20','ndcg20'],np.mean(v['metrics'],0).tolist())) for k,v in methods.items()}
    expected=identity.get('original_metrics')
    if expected and not smoke:
        error=max(abs(summary['original_reference'][k]-expected[original])
                  for k,original in [('recall20','recall@20'),('ndcg20','ndcg@20')])
        write_json(directory/'original_metric_parity.json',dict(max_absolute_error=error,tolerance=2e-5))
        if error>2e-5:raise RuntimeError(f'Original checkpoint metrics not reproduced: {error}')
    contrasts=[]
    for first,second in CONTRASTS:
        for j,metric in enumerate(['recall20','ndcg20']):
            value=paired_comparison(np.asarray(methods[first]['metrics'])[:,j],
                np.asarray(methods[second]['metrics'])[:,j],methods[first]['groups'],
                bootstrap=20 if smoke else PLAN['bootstrap'],permutations=50 if smoke else PLAN['permutations'],seed=SEED)
            contrasts.append(dict(value,first=first,second=second,metric=metric))
    holm_adjust(contrasts,PLAN['planned_tests'])
    write_json(directory/'summary.json',dict(status='complete',smoke=smoke,methods=summary,contrasts=contrasts))
    lines=['# Amplified-neighbor and fallback follow-up','',
        'Exploratory results using the recorded input checkpoint and split. Original reference '
        'was selected on all earlier validation queries. Controlled banks fit/tune on identical '
        '70/30 complete-bundle groups. Constant-mixture controls use the same fallback sources.',
        '', '| Method | Recall@20 | NDCG@20 |','|---|---:|---:|']
    lines += [f'| {k} | {v["recall20"]:.6f} | {v["ndcg20"]:.6f} |' for k,v in summary.items()]
    lines += ['', 'Paired statistics reserve 60 tests across the five newly added datasets. '
        'These are fused score banks, not isolated raw operators; the historical column may '
        'receive zero selected weight. See inputs.json for checkpoint/data profile and query scope.']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(args):
    directory=args.output/args.dataset;directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'.runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    artifacts=(directory/'artifacts' if args.smoke else Path('checkpoints/amplified_fallback')/args.dataset)
    artifacts.mkdir(parents=True,exist_ok=True)
    protocol=dict(PLAN,dataset=args.dataset,smoke=args.smoke,source_sha256=fingerprint())
    for root in [directory,artifacts]:
        path=root/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Protocol/source mismatch; choose a new directory')
        write_json(path,protocol)
    log(f'load {args.dataset} retained checkpoint and verify original split')
    model,val,test,val_users,test_users,old,source,identity=load_inputs(args.dataset)
    write_json(directory/'inputs.json',dict(identity,config=vars(model.cfg),source_recipe=source,runtime=run_metadata()))
    if args.smoke:
        val=val[:20];test=test[:8]
        val_users=None if val_users is None else val_users[:20]
        test_users=None if test_users is None else test_users[:8]
    keys=source_columns(source);freq=np.asarray(model.freq)
    fit_ix,tune=validation_groups(val,SEED,PLAN['fit_fraction'])
    write_json(directory/'validation_partition.json',dict(fit=fit_ix.tolist(),tuning=tune.tolist(),
        fit_sha256=digest_rows([val[i] for i in fit_ix]),tuning_sha256=digest_rows([val[i] for i in tune])))
    amp=AmplifiedNeighbors(model.graph);amppath=directory/'amplified_selection.json'
    if amppath.exists():ampcfg=json.loads(amppath.read_text())
    else:
        log('select amplified-neighbor settings on fitting groups only')
        ampcfg=amp.select(val,fit_ix,PLAN['powers'],PLAN['neighbors'],log);write_json(amppath,ampcfg)
    log('cache validation scores')
    valarrays=cache_signals(val,val_users,model,source,keys,old,artifacts/'validation',PLAN['chunk'],log)
    valamp=cache_amplified(val,amp,ampcfg,artifacts/'validation',PLAN['chunk'],log)
    methods={}
    for kind in PLAN['banks']:
        log(f'fit {kind} weighted/gate/constant variants')
        bank=ScoreBank(valarrays,valamp,keys,kind)
        selected=fit(bank,val,fit_ix,tune,freq,source,directory/'selection'/kind,log)
        methods.update({f'{kind}:{key}':dict(value,bank=kind) for key,value in selected.items()})
    methods['original_reference']=dict(kind='original_reference')
    write_json(directory/'frozen_methods.json',methods)
    log('SELECTION FROZEN; generate held-out scores')
    testarrays=cache_signals(test,test_users,model,source,keys,None,artifacts/'test',PLAN['chunk'],log)
    testamp=cache_amplified(test,amp,ampcfg,artifacts/'test',PLAN['chunk'],log)
    banks={kind:ScoreBank(testarrays,testamp,keys,kind) for kind in PLAN['banks']}
    verify_reference(model,test,test_users,banks['reference'],source,directory)
    assess(model,test,test_users,banks,methods,directory/'queries',PLAN['chunk'])
    log('paired comparisons and report');report(directory,identity,args.smoke);log('COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/amplified_fallback'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1),torch.inference_mode():run(args)
