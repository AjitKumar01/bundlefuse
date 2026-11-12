"""Training and inference time of CLHE and RaMen on POG, on CPU, with their own code (companion of run.py).

Runs in the RaMen environment, from the model's repository:
cd ../baselines/CLHE && ../RaMen_repo/.venv/bin/python ../../bundlefuse/experiments/timing/neural.py --model clhe --threads 4
cd ../baselines/RaMen_repo && .venv/bin/python ../../bundlefuse/experiments/timing/neural.py --model ramen --threads 4

Settings: CLHE with the POG preset of its run.sh and the pog block of config.yaml (as in ramen_retail/clhe_runner.py);
RaMen with the pog block of its config.yaml and the settings of its timing_cpu.py (4 GAT layers, gamma 1). --epochs
training epochs are timed, each with a validation pass as in our runners; the reported total multiplies by the epochs
of our CLHE POG runs (fixed 50) and by RaMen's configured 200 (never trained on POG by us). Inference: propagate()
plus scoring and ranking the whole catalogue for every test query, per query.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch
import yaml
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,os.getcwd())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ramen_retail'))
import models
from train import setup_seed
from utility import Datasets

OUT=Path(__file__).resolve().parents[2]/'results'/'timing'


def configure(name):
    conf=yaml.safe_load(open('config.yaml'))['pog']
    if name=='clhe':
        from clhe_runner import DEFAULTS,PRESETS
        conf.update(DEFAULTS,**PRESETS['pog'],dataset='pog',lr=conf['lrs'][0],seed=2023)
    else:
        from ramen_runner import DEFAULTS
        conf.update(DEFAULTS,dataset='pog',lr=conf['lrs'][0],n_layer_gat=4,bundle_alpha=1.,item_alpha=1.,seed=2306)
    conf['device']=torch.device('cpu');return conf


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,choices=['clhe','ramen'])
    p.add_argument('--threads',type=int,default=4);p.add_argument('--epochs',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    from clhe_runner import per_query as clhe_per_query
    from ramen_runner import per_query as ramen_per_query
    per_query=clhe_per_query if a.model=='clhe' else ramen_per_query
    conf=configure(a.model);setup_seed(conf['seed']);started=time.monotonic();ds=Datasets(conf)
    conf.update(num_users=ds.num_users,num_bundles=ds.num_bundles,num_items=ds.num_items,l2_reg=conf['l2_regs'][0],
                embedding_size=64,num_layers=conf['num_layerss'][0])
    model=(models.CLHE if a.model=='clhe' else models.RAMEN)(conf,ds.graphs,ds.features);setup=time.monotonic()-started
    optimizer=torch.optim.Adam(model.parameters(),lr=conf['lr'],weight_decay=conf['l2_reg'])
    epochs=[]
    for _ in range(a.epochs):
        started=time.monotonic();model.train(True)
        for batch in ds.train_loader:
            optimizer.zero_grad();model(batch)['loss'].backward();optimizer.step()
        trained=time.monotonic()-started;per_query(model,ds.val_loader);epochs.append((trained,time.monotonic()-started))
    started=time.monotonic();test=per_query(model,ds.test_loader);infer=time.monotonic()-started
    per_epoch=sum(e[1] for e in epochs)/len(epochs);total_epochs=50 if a.model=='clhe' else 200
    report=dict(note=__doc__,model=a.model,threads=a.threads,setup_seconds=setup,epochs_timed=[dict(train=t,with_validation=v) for t,v in epochs],
                seconds_per_epoch_with_validation=per_epoch,epochs_assumed=total_epochs,fit_seconds=setup+per_epoch*total_epochs,
                inference_ms_per_query=1000*infer/len(test),test_queries=len(test),
                parameters=sum(x.numel() for x in model.parameters() if x.requires_grad))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f'pog_cpu_{a.model}.json').write_text(json.dumps(report,indent=1))
    print({k:v for k,v in report.items() if k!='note'},flush=True)


if __name__=='__main__':
    main()
