"""Training and inference time of DIALS and the simple baselines on POG, on one machine, one method at a time.

All methods use the same thread count (BLAS and torch). Inference is the wall time to score and rank the whole catalogue
for the 4,000 test queries (metrics included, as in every evaluation here), reported per query.
  simple baselines  fit = building the shared co-occurrence statistics plus the method's own matrix or factors, at the
                    setting chosen on validation (results/simple_baselines/pog); grids are not re-run.
  DIALS             fit = the cross-fitted label factors (computed afresh) and model construction, plus --epochs epochs
                    of training, each with its validation pass, as in dials.train.fit; the total is extrapolated to the
                    epochs the three frozen runs actually ran. Inference uses the frozen seed-0 model.
CLHE and RaMen are timed with experiments/timing/neural.py in their own environment.

THREADS=4; VECLIB_MAXIMUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS PYTHONPATH=.:../dials .venv/bin/python -m experiments.timing.run --threads $THREADS
"""
import dials  # pins BLAS threads before numpy loads (setdefault, so the environment above wins)
import argparse
import json
import tempfile
import time
from pathlib import Path
import torch
from dials import data
from dials import train as T
from dials.evaluate import evaluate as evaluate_dials
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.simple_baselines.methods import Baselines
from experiments.simple_baselines.run import evaluate as evaluate_baseline,features

OUT=Path('results/timing');CACHE=Path(DATA_ROOT)/'dials'/'cache';RUNS=Path(DATA_ROOT)/'dials'/'runs'


def baselines(ds,report):
    chosen=json.loads(Path('results/simple_baselines/pog/summary.json').read_text())['test']
    x=features('pog',ds)
    for name in chosen:
        p=chosen[name]['params'];started=time.monotonic()
        model=Baselines(ds.train,ds.n_items,x if name=='hybrid' else None);model.scores(name,p,[ds.test_pairs[0][0]])  # first call builds the method
        fit=time.monotonic()-started;started=time.monotonic()
        evaluate_baseline(model,name,p,ds.test_pairs);infer=time.monotonic()-started
        report['methods'][name]=dict(params=p,fit_seconds=fit,inference_ms_per_query=1000*infer/len(ds.test_pairs),parameters=None)
        print(name,report['methods'][name],flush=True);del model


def dials_timing(ds,report,epochs):
    with tempfile.TemporaryDirectory() as tmp:
        started=time.monotonic();T.spectral(ds,tmp);setup=time.monotonic()-started  # label factors, not cached
        config=dict(T.CONFIG,epochs=epochs,patience=epochs+1);started=time.monotonic()
        T.fit(ds,0,Path(tmp)/'run',tmp,config);trained=time.monotonic()-started
    val_pass=time.monotonic();model=T.load(ds,RUNS/'pog'/'seed0',CACHE);evaluate_dials(model,ds.val_pairs);val_pass=time.monotonic()-val_pass
    per_epoch=(trained-val_pass)/epochs  # fit replays the selected checkpoint's validation once more at the end
    runs=[json.loads((RUNS/'pog'/f'seed{s}'/'selected.json').read_text()) for s in (0,1,2)]
    ran=[r.get('epochs_run') or len(json.loads((RUNS/'pog'/f'seed{s}'/'history.json').read_text())) for s,r in zip((0,1,2),runs)]
    started=time.monotonic();model=T.load(ds,RUNS/'pog'/'seed0',CACHE);evaluate_dials(model,ds.test_pairs);infer=time.monotonic()-started
    report['methods']['dials']=dict(setup_seconds=setup,seconds_per_epoch_with_validation=per_epoch,validation_pass_seconds=val_pass,
        epochs_run=ran,fit_seconds=setup+per_epoch*sum(ran)/len(ran),inference_ms_per_query=1000*infer/len(ds.test_pairs),parameters=model.parameter_count())
    print('dials',report['methods']['dials'],flush=True)


def main(args):
    torch.set_num_threads(args.threads)
    ds=data.load('pog',CACHE)
    report=dict(note=__doc__,threads=args.threads,dataset='pog',test_queries=len(ds.test_pairs),n_items=ds.n_items,methods={})
    dials_timing(ds,report,args.epochs);baselines(ds,report)
    write_json(OUT/'pog_cpu.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--threads',type=int,default=4);p.add_argument('--epochs',type=int,default=2)
    main(p.parse_args())
