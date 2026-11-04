"""Run one supplied-split mechanism study; all large artifacts are retained."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from bundlefuse import datasets
from bundlefuse.model import BundleFuse
from bundlefuse.fusion import fit_fusion
from bundlefuse.metrics import top_items
from bundlefuse.provenance import write_json, run_metadata, digest_rows
from bundlefuse.splits import file_hash
from .cache import SignalCache
from .selection import StudyValidation, cluster_bootstrap, recipe, score_recipe
from .protocol import PLAN, SEED, CONTRASTS


def log(message):
    print(f'{datetime.now(timezone.utc).isoformat()} {message}',flush=True)


def fingerprint():
    paths=sorted(Path('bundlefuse').rglob('*.py'))+sorted(Path('experiments/mechanisms').glob('*.py'))
    return hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in paths)).hexdigest()


def select(cache, model, pairs, output):
    selected={}
    history=[k for k in model.SIG if k in PLAN['history']]
    shared=[k for k in model.SIG if k in PLAN['shared']]
    assert set(history+shared)==set(model.SIG),model.SIG
    validation=StudyValidation(cache.blocks,pairs,model.freq,progress=log)
    for name,keys in [('history',history),('shared',shared),('combined',model.SIG),('full',model.SIG)]:
        path=output/f'{name}.json'
        if path.exists():
            selected[name]=json.loads(path.read_text());continue
        log(f'select {name}')
        extra={} if name=='full' else dict(incidence=(0.,),rao=(0.,),shrinkage=False)
        selected[name]=recipe(fit_fusion(validation,keys,shared,rounds=PLAN['rounds'],**extra))
        selected[name]['validation_ndcg20']=validation.evaluate(
            selected[name]['weights'],selected[name]['higher_order'],
            selected[name]['backoff'],selected[name]['prior'])
        write_json(path,selected[name])
    base=selected['combined']
    for label,scale in [('unit',1.),('matched',validation.prior_scale(base['weights'],(0.,0.),base['prior']))]:
        for kind,grid in [('constant',[-u for u in PLAN['mixture_fractions']]),('frequency',PLAN['kappa'])]:
            name=f'{kind}_{label}'
            candidates=[(base['weights'],(0.,0.),(value,scale),base['prior']) for value in grid]
            values=validation.evaluate_many(candidates)
            best=int(np.argmax(values))
            selected[name]=dict(base,backoff=list(candidates[best][2]),validation_ndcg20=values[best],
                                grid_scores=[dict(value=v,ndcg20=s) for v,s in zip(grid,values)])
            write_json(output/f'{name}.json',selected[name])
    rng=np.random.default_rng(SEED)
    for name,order,ix in [
        ('stability_reverse',list(reversed(model.SIG)),None),
        ('stability_permutation',rng.permutation(model.SIG).tolist(),None),
        *[(f'stability_bootstrap_{seed}',model.SIG,cluster_bootstrap(pairs,seed)) for seed in (SEED+1,SEED+2)]]:
        path=output/f'{name}.json'
        if path.exists(): selected[name]=json.loads(path.read_text());continue
        log(f'select {name}')
        val=StudyValidation(cache.blocks,pairs,model.freq,ix,progress=log)
        selected[name]=recipe(fit_fusion(val,order,shared,rounds=PLAN['rounds']))
        selected[name].update(validation_candidates=len(val.memo),order=order,
                              validation_indices_sha256=digest_rows(val.indices.tolist()))
        write_json(path,selected[name])
    write_json(output/'selection_complete.json',dict(methods=list(selected),validation_candidates=len(validation.memo)))
    return selected


def query_metrics(scores, seeds, gt, frequency):
    predicted=top_items(scores,seeds,20)
    hit=np.isin(predicted,gt)
    discount=1/np.log2(np.arange(len(predicted))+2)
    ideal=float((1/np.log2(np.arange(min(len(set(gt)),20))+2)).sum())
    return predicted,[float(hit.sum()/len(set(gt))),float((hit*discount).sum()/ideal)]


def evaluate(model, pairs, selected, directory, chunk):
    target=directory/'queries.json'
    if target.exists():return json.loads(target.read_text())
    rows={k:[] for k in selected}
    predictions={k:[] for k in selected}
    slices={k:{b:[] for b in ('cold','rare','warm')} for k in selected}
    f=model.freq
    buckets={'cold':f==0,'rare':(f>0)&(f<=5),'warm':f>5}
    for start in range(0,len(pairs),chunk):
        block=pairs[start:start+chunk]
        _,z=model._score_all([s for s,_ in block])
        for key,method in selected.items():
            values=score_recipe(z,method,torch.as_tensor(f)[:,None])
            for j,(s,gt) in enumerate(block):
                predicted,metrics=query_metrics(values[:,j],s,gt,f)
                rows[key].append(metrics);predictions[key].append(predicted.tolist())
                for bucket,mask in buckets.items():
                    truth=[i for i in gt if mask[i]]
                    if truth:
                        restricted=values[:,j].copy();restricted[~mask]=-np.inf
                        # Restrict the universe, including when a bucket has <20 items.
                        allowed=np.flatnonzero(mask);allowed=allowed[~np.isin(allowed,s)]
                        best=allowed[np.lexsort((allowed,-restricted[allowed]))[:20]]
                        slices[key][bucket].append(dict(query=start+j,targets=len(truth),
                            global_recall=float(np.isin(predicted,truth).sum()/len(truth)),
                            within_recall=float(np.isin(best,truth).sum()/len(truth))))
        log(f'test {min(start+chunk,len(pairs))}/{len(pairs)} methods={len(selected)}')
    write_json(target,dict(metrics=rows,predictions=predictions,buckets=slices,
        groups=[str(tuple(sorted(set(s)|set(gt)))) for s,gt in pairs],
        seed_counts=[len(s) for s,_ in pairs]))
    return json.loads(target.read_text())


def report(rows, directory):
    from .statistics import paired_comparison, holm_adjust
    comparisons=[]
    for first,second in CONTRASTS:
        for metric,index in [('recall20',0),('ndcg20',1)]:
            result=paired_comparison(np.asarray(rows['metrics'][first])[:,index],
                np.asarray(rows['metrics'][second])[:,index],rows['groups'],seed=SEED)
            result.update(first=first,second=second,metric=metric);comparisons.append(result)
    holm_adjust(comparisons,PLAN['planned_tests'])
    summary={k:dict(zip(('recall20','ndcg20'),np.mean(v,axis=0).tolist())) for k,v in rows['metrics'].items()}
    stability={k:float(np.mean([len(set(a)&set(b))/20 for a,b in zip(rows['predictions']['full'],v)]))
               for k,v in rows['predictions'].items() if k.startswith('stability')}
    bucket_summary={key:{bucket:dict(queries=len(values),
        global_recall=float(np.mean([v['global_recall'] for v in values])) if values else None,
        within_recall=float(np.mean([v['within_recall'] for v in values])) if values else None)
        for bucket,values in by_bucket.items()} for key,by_bucket in rows['buckets'].items()}
    seed_summary={key:{str(size):np.asarray(values)[np.asarray(rows['seed_counts'])==size].mean(0).tolist()
        for size in sorted(set(rows['seed_counts']))} for key,values in rows['metrics'].items()}
    write_json(directory/'summary.json',dict(status='complete',methods=summary,
        stability_top20_overlap=stability,frequency_buckets=bucket_summary,
        by_seed_count=seed_summary,contrasts=comparisons))
    text=['# Mechanism study results','', 'Conditional on the fitted models and supplied split. Holm family: 96 tests.',
          '', '| Method | Recall@20 | NDCG@20 |','|---|---:|---:|']
    text += [f"| {k} | {v['recall20']:.6f} | {v['ndcg20']:.6f} |" for k,v in summary.items()]
    (directory/'REPORT.md').write_text('\n'.join(text)+'\n')


def run(args):
    out=args.output/args.dataset;out.mkdir(parents=True,exist_ok=True)
    artifacts=Path('checkpoints/publication_mechanisms')/args.dataset
    artifacts.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=dict(PLAN,dataset=args.dataset,source_sha256=fingerprint(),smoke=args.smoke)
    planpath=out/'protocol.json'
    if planpath.exists() and json.loads(planpath.read_text())!=plan:
        raise RuntimeError('Protocol/source changed; use a new output and artifact directory')
    write_json(planpath,plan)
    dataset=datasets.load(args.dataset,val_n=8 if args.smoke else None,test_n=8 if args.smoke else None)
    if not args.smoke:
        assert dataset.metadata['full_validation'] and dataset.metadata['full_test']
    identity=dict(metadata=dataset.metadata,source_sha256=fingerprint(),smoke=args.smoke)
    if args.smoke: artifacts=out/'artifacts';artifacts.mkdir(exist_ok=True)
    identity_path=artifacts/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise RuntimeError('Retained artifacts have different data or sources')
    write_json(identity_path,identity)
    write_json(out/'inputs.json',dict(dataset.metadata,runtime=run_metadata()))
    checkpoint=artifacts/'artifacts.pt'
    if checkpoint.exists():model=BundleFuse.load(checkpoint)
    else:
        cfg=datasets.get(args.dataset).config(dataset.n_items)
        # Construction uses training/catalogue inputs. Provisional one-query fusion is
        # discarded; every reported recipe below is selected on the full validation set.
        cfg.fit_rounds=1
        log('construct training-only graph and catalogue artifacts; provisional fusion discarded')
        model=BundleFuse(cfg).fit(dataset.train,dataset.val_pairs[:1],**dataset.fit_kwargs())
        model.cfg.fit_rounds=PLAN['rounds'];model.save(checkpoint)
    write_json(out/'artifact.json',dict(path=str(checkpoint),sha256=file_hash(checkpoint),config=asdict(model.cfg)))
    cache=SignalCache(artifacts/'validation_signals',model,dataset.val_pairs,PLAN['chunk'],log)
    selected=select(cache,model,dataset.val_pairs,out)
    full=selected['full'];model.w=full['weights'];model.prior_keys=full['prior']
    model.best_eb=full['backoff'];model.best_ho=full['higher_order'];model.save(artifacts/'full.pt')
    # This file is the barrier: every recipe is frozen before test signals are computed.
    write_json(out/'frozen_recipes.json',selected)
    log('all selections frozen; beginning held-out evaluation')
    rows=evaluate(model,dataset.test_pairs,selected,out,PLAN['chunk'])
    report(rows,out)
    log('COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=PLAN['datasets'],required=True)
    parser.add_argument('--output',type=Path,default=Path('results/publication_mechanisms'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1),torch.inference_mode():run(args)
