"""Retain a live validation checkpoint or rebuild closed-form models from recorded selections.

Rebuilding uses the immutable source snapshot and original inputs. It never searches for
weights again. Rebuilt checkpoints are labelled and checked against every recorded aggregate
metric; captured checkpoints are copied while their original validation worker is running.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def watch(results, destination):
    roots={Path(tempfile.gettempdir()).resolve(),Path('/private/tmp'),
           Path('/private/var/folders/xf/2s8rjd9n6nj7x91vp1h19qr40000gn/T')}
    deadline=time.time()+8*3600
    while time.time()<deadline:
        manifest=json.loads((results/'manifest.json').read_text())
        running=[j for j in manifest['jobs'] if j['status']=='running']
        for job in running:
            target=destination/job['dataset']/'seed_0/model.pt'
            if target.exists():continue
            for root in roots:
                if not root.exists():continue
                for candidate in root.glob('bundlefuse_verify_*/model.pt'):
                    try:
                        if candidate.stat().st_mtime < job['started']-2:continue
                        target.parent.mkdir(parents=True,exist_ok=True)
                        temporary=target.with_suffix('.capturing')
                        # Holding an open file keeps the bytes readable even if its temp directory is removed.
                        with candidate.open('rb') as src,temporary.open('wb') as dst:
                            shutil.copyfileobj(src,dst,1024*1024)
                        temporary.replace(target)
                        record={'dataset':job['dataset'],'path':str(target.resolve()),'sha256':digest(target),
                                'bytes':target.stat().st_size,'method':'captured_original_checkpoint',
                                'source_tempfile':str(candidate),'captured_at':time.time(),'verified':False}
                        write(target.parent/'retention.json',record)
                        print('PRESERVED',job['dataset'],target,flush=True)
                    except FileNotFoundError:
                        continue
        if all(j['status'] in ('complete','failed') for j in manifest['jobs']):
            print('Original run ended; capture guard complete.',flush=True)
            return
        time.sleep(.2)
    raise RuntimeError('Checkpoint capture guard timed out')


def recover_worker(results,destination,snapshot,dataset):
    sys.path.insert(0,str(snapshot))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import bundlefuse.model as module
    from bundlefuse import datasets
    from bundlefuse.evaluation import evaluate,flat_metrics
    from bundlefuse.provenance import run_metadata
    if Path(module.__file__).resolve().parent.parent != snapshot.resolve():
        raise RuntimeError('Recovery imported the wrong implementation')
    result=json.loads((results/'original'/dataset/'result.json').read_text())
    spec=result['data'];selected=result['model'];target=destination/dataset/'seed_0/model.pt'
    ds=datasets.load(dataset,val_n=spec['val_limit'],test_n=spec['test_limit'],
                     protocol_mode=spec['protocol'],split_seed=spec['split_seed'] or 0,
                     mask_frac=spec['mask_frac'] or .5)
    if ds.metadata!=spec:raise RuntimeError('Original dataset fingerprint/protocol changed')
    if target.exists():
        model=module.load_model(target)
        method='captured_original_checkpoint'
    else:
        class NoSelectionCache:
            def __init__(self,*args,**kwargs):self.keys=selected['signals']+['pair','diff2_pair','rao']
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def blocks(self):raise AssertionError('Recovery must not repeat validation search')
        def recorded_selection(*args,**kwargs):
            return (selected['weights'],selected['prior_keys'],
                    (selected['kappa_f'],selected['prior_scale']),
                    (selected['alpha_inc'],selected['gamma_rao']),selected['variance_components'])
        module.ValidationCache=NoSelectionCache
        module.fit_fusion=recorded_selection
        print('REBUILD',dataset,'from original inputs and recorded selected parameters; no search',flush=True)
        model=module.BundleFuse(module.BundleFuseConfig(**selected['config']))
        model.fit(ds.train,ds.val_pairs,**ds.fit_kwargs())
        model.data_spec=spec
        model.run_info=run_metadata()
        model.recovery_info={'method':'rebuilt_from_recorded_selection','original_commit':'162dd8c',
                             'original_result_sha256':digest(results/'original'/dataset/'result.json')}
        target.parent.mkdir(parents=True,exist_ok=True)
        model.save(target)
        method='rebuilt_from_recorded_selection'
    actual=flat_metrics(evaluate(model,ds.test_pairs,ds.test_users,(10,20,50),64))
    error=max(abs(actual[k]-result['metrics'][k]) for k in actual)
    if error>1e-12:raise RuntimeError(f'Rebuilt metrics differ for {dataset}: {error}')
    loaded=module.load_model(target)
    for start in range(0,len(ds.test_pairs),64):
        seeds=[s for s,_ in ds.test_pairs[start:start+64]]
        users=ds.test_users[start:start+64] if ds.test_users is not None else None
        before=model.score_batch(seeds,users);after=loaded.score_batch(seeds,users)
        if not np.array_equal(before.view(np.uint32),after.view(np.uint32)):
            raise RuntimeError(f'Persistent checkpoint parity failed for {dataset} at {start}')
    record={'dataset':dataset,'path':str(target.resolve()),'sha256':digest(target),
            'bytes':target.stat().st_size,'method':method,'verified':True,
            'original_result_sha256':digest(results/'original'/dataset/'result.json'),
            'metrics':actual,'max_recorded_metric_error':error,'queries_verified':len(ds.test_pairs),
            'checkpoint_score_bit_identity':True,'runtime':run_metadata()}
    write(target.parent/'retention.json',record)
    write(results/'checkpoint_recovery'/f'{dataset}.json',record)
    print('VERIFIED AND RETAINED',dataset,target,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/reorganized_run')
    parser.add_argument('--destination',default='checkpoints/reorganized_162dd8c')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--worker',default=None)
    args=parser.parse_args()
    results=Path(args.results).resolve();destination=Path(args.destination).resolve()
    plan=json.loads((results/'PLAN.json').read_text());original_snapshot=Path(plan['snapshot'])
    snapshot=destination/'source'
    if args.watch:
        watch(results,destination);return
    if args.worker:
        recover_worker(results,destination,snapshot,args.worker);return
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in ('bundlefuse','configs'):
        if not (snapshot/name).exists():
            shutil.copytree(original_snapshot/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('evaluator.py','inference.py','validate.py','requirements.txt'):
        if not (snapshot/name).exists():shutil.copy2(original_snapshot/name,snapshot/name)
    records=[]
    # Rebuild small cases first; defer the second large music model until the original run ends.
    order=['electronic','food','clothing','dunnhumby','pog_dense','pog','spotify','spotify_sparse']
    logdir=results/'checkpoint_recovery';logdir.mkdir(exist_ok=True)
    for dataset in order:
        while True:
            manifest=json.loads((results/'manifest.json').read_text())
            job=next(j for j in manifest['jobs'] if j['dataset']==dataset)
            music_running=any(j['status']=='running' and j['dataset'].startswith('spotify') for j in manifest['jobs'])
            if job['status']=='failed':raise RuntimeError(f'Original {dataset} failed')
            if job['status']=='complete' and not (dataset.startswith('spotify') and music_running):break
            time.sleep(5)
        env=dict(os.environ);env.update(BUNDLEFUSE_DATA_ROOT=str(results.parents[1].parent),
              OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
              VECLIB_MAXIMUM_THREADS='1',PYTHONHASHSEED='0')
        print('RECOVER',dataset,flush=True)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--results',str(results),
                 '--destination',str(destination),'--worker',dataset]
        with (logdir/f'{dataset}.log').open('w') as stream:
            done=subprocess.run(command,env=env,cwd=snapshot,stdout=stream,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError(f'Recovery failed: {logdir/dataset}.log')
        records.append(json.loads((logdir/f'{dataset}.json').read_text()))
        write(logdir/'manifest.json',{'status':'running','checkpoints':records})
    write(logdir/'manifest.json',{'status':'complete','checkpoints':records})
    print('ALL EIGHT CHECKPOINTS RETAINED AND VERIFIED',flush=True)


if __name__=='__main__':main()
