"""Resumable raw-score/fusion attribution study on supplied splits."""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_key]='1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import fcntl
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from bundlefuse.provenance import write_json, run_metadata, digest_rows
from bundlefuse.metrics import ranking_metrics,top_items
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from .protocol import PLAN,SEED,BANKS,FUSIONS,CONTRASTS,PLANNED_TESTS
from .data import load_inputs,validation_groups,selected_columns,cache_extra,Bank
from .raw import AmplifiedNeighbors
from .fusion import fit_bank,score,canonical_weights


def log(message):print(f'{datetime.now(timezone.utc).isoformat()} {message}',flush=True)


def fingerprint():
    paths=sorted(Path('bundlefuse').rglob('*.py'))+sorted(Path('experiments/novelty').glob('*.py'))
    return hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in paths)).hexdigest()


def assess(test,pairs,selected,freq,keys,directory,chunk):
    import xgboost as xgb
    directory.mkdir(parents=True,exist_ok=True)
    groups=[str(tuple(sorted(set(s)|set(gt)))) for s,gt in pairs]
    for name,cfg in selected.items():
        path=directory/f'{name.replace(":","__")}.json'
        if path.exists():continue
        bank=test[cfg['bank']];model=None
        if cfg['kind']=='lambdamart':model=xgb.XGBRanker();model.load_model(cfg['model_path'])
        part=path.with_suffix('.progress.json')
        rows=json.loads(part.read_text()) if part.exists() else dict(metrics=[],predictions=[],seconds=0.)
        begin=time.monotonic()
        for i in range(len(rows['metrics']),len(pairs)):
            values=score(bank.query(i),cfg,keys,freq,model,seeds=pairs[i][0])
            metric=ranking_metrics(values[:,None],[pairs[i]],Ks=(20,))[20]
            rows['metrics'].append(list(metric));rows['predictions'].append(top_items(values,pairs[i][0],20).tolist())
            if (i+1)%100==0:
                write_json(part,rows);log(f'test {name}: {i+1}/{len(pairs)}')
        rows['seconds']+=time.monotonic()-begin;rows['groups']=groups
        write_json(path,rows)
        log(f'test {name} COMPLETE metrics={np.mean(rows["metrics"],axis=0).tolist()}')


def report(directory,smoke=False):
    rows={p.stem.replace('__',':'):json.loads(p.read_text()) for p in (directory/'queries').glob('*.json')
          if not p.name.endswith('.progress.json')}
    summary={k:dict(zip(['recall20','ndcg20'],np.mean(v['metrics'],0).tolist()),seconds=v['seconds'])
             for k,v in rows.items()}
    comparisons=[]
    for first,second in CONTRASTS:
        if first not in rows or second not in rows:continue
        for metric,j in [('recall20',0),('ndcg20',1)]:
            value=paired_comparison(np.asarray(rows[first]['metrics'])[:,j],np.asarray(rows[second]['metrics'])[:,j],
                rows[first]['groups'],bootstrap=20 if smoke else PLAN['bootstrap'],
                permutations=50 if smoke else PLAN['permutations'],seed=SEED)
            comparisons.append(dict(value,first=first,second=second,metric=metric))
    holm_adjust(comparisons,PLANNED_TESTS)
    write_json(directory/'summary.json',dict(status='complete',smoke=smoke,methods=summary,contrasts=comparisons))
    lines=['# Novelty attribution results','',
       'Exploratory supplied-split results conditional on frozen prior source inventory. '
       'All controlled rows use the same fitting/tuning groups; original_reference uses its earlier all-validation recipe.',
       '', '| Method | Recall@20 | NDCG@20 |', '|---|---:|---:|']
    lines.extend(f'| {k} | {v["recall20"]:.6f} | {v["ndcg20"]:.6f} |' for k,v in summary.items())
    lines+=['','Paired comparisons and Holm correction are recorded in summary.json. '
            'This stage is not an exact reproduction of HOSLIM or Kelen kNN; the amplified operator is explicitly defined.']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(args):
    directory=args.output/args.dataset;directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'.runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    artifacts=(directory/'artifacts' if args.smoke else Path('checkpoints/novelty_attribution')/args.dataset)
    artifacts.mkdir(parents=True,exist_ok=True)
    protocol=dict(PLAN,dataset=args.dataset,source_sha256=fingerprint(),smoke=args.smoke)
    for root in [directory,artifacts]:
        path=root/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Protocol/source mismatch; use a new run directory')
        write_json(path,protocol)
    log(f'load retained {args.dataset} artifacts read-only')
    model,val,test,old,source,identity=load_inputs(args.dataset)
    import xgboost
    write_json(directory/'inputs.json',dict(identity,runtime=run_metadata(),xgboost=xgboost.__version__))
    write_json(artifacts/'source_recipe.json',source)
    if args.smoke:
        val=val[:20];test=test[:8];old={k:v[:20] for k,v in old.items()}
    fit,tune=validation_groups(val,SEED,PLAN['fit_fraction'])
    write_json(directory/'validation_partition.json',dict(fit=fit.tolist(),tuning=tune.tolist(),
        fit_sha256=digest_rows([val[i] for i in fit]),tuning_sha256=digest_rows([val[i] for i in tune])))
    keys=selected_columns(source);frequency=np.asarray(model.freq)
    for key in ['pair','simp',*[k for k in keys if k!='hist']]:
        if key not in old:raise ValueError(f'Legacy cache missing {key}')
    # Sampled parity protects reuse of caches under the unchanged production scorer.
    from bundlefuse.selected_signals import SelectedSignals
    verifier=SelectedSignals(model.graph,model.onto,model.FEAT,model.cat_lbl,model.rao_dense,model.music,
                            torch.device('cpu'),beta_clique=model.cfg.beta_clique)
    required={k for k in keys if k!='hist'}|{'pair','simp'}
    _,fresh=verifier.score([val[0][0]],required)
    errors={k:float(np.max(np.abs(fresh[k].numpy().ravel()-old[k][0]))) for k in required}
    if max(errors.values())>1e-4:raise RuntimeError(f'Legacy validation parity failed {errors}')
    write_json(directory/'cache_parity.json',errors)
    amp=AmplifiedNeighbors(model.graph);amppath=directory/'amplified_selection.json'
    if amppath.exists():ampcfg=json.loads(amppath.read_text())
    else:
        log('select amplified neighbors using fitting groups only')
        ampcfg=amp.select(val,fit,PLAN['amplification_powers'],PLAN['amplification_neighbors'],log)
        write_json(amppath,ampcfg)
    valarrays=cache_extra(val,old,model,amp,ampcfg,artifacts/'signals/validation',PLAN['chunk'],log)
    selected={}
    for kind in BANKS:
        log(f'fit/tune fusion methods for {kind} raw bank')
        bank=Bank(valarrays,kind,keys)
        fitted=fit_bank(bank,val,fit,tune,frequency,source,artifacts/'models'/kind,directory/'selection'/kind,log)
        for fusion,cfg in fitted.items():selected[f'{kind}:{fusion}']=dict(cfg,bank=kind)
    # All controlled methods are frozen before any new held-out scores are computed.
    w=canonical_weights(source);prior={k:v for k,v in w.items() if k in source['prior'] or k in ('pop','rao')}
    selected['original_reference']=dict(kind='canonical',weights=w,prior_weights=prior,
          kappa=source['backoff'][0],scale=source['backoff'][1],
          bank='incidence' if source['higher_order'][0]==1 else 'pair')
    write_json(directory/'frozen_methods.json',selected)
    log('SELECTION FROZEN; construct held-out score columns')
    testarrays=cache_extra(test,None,model,amp,ampcfg,artifacts/'signals/test',PLAN['chunk'],log)
    banks={kind:Bank(testarrays,kind,keys) for kind in BANKS}
    # Original-reference parity against production scoring, before accepting any report.
    model.w=source['weights'];model.prior_keys=source['prior'];model.best_eb=source['backoff'];model.best_ho=source['higher_order']
    reference=model.score(test[0][0]);cfg=selected['original_reference']
    derived=score(banks[cfg['bank']].query(0),cfg,keys,frequency,seeds=test[0][0])
    maximum=float(np.max(np.abs(reference-derived)))
    if maximum>2e-4:raise RuntimeError(f'Original scorer parity failed {maximum}')
    write_json(directory/'reference_parity.json',dict(max_absolute_error=maximum))
    assess(banks,test,selected,frequency,keys,directory/'queries',PLAN['chunk'])
    log('paired comparisons and report')
    report(directory,args.smoke);log('COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=PLAN['datasets'],required=True)
    parser.add_argument('--output',type=Path,default=Path('results/novelty_attribution'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1),torch.inference_mode():run(args)
