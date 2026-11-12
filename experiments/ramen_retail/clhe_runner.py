"""Train CLHE with its own code on a dataset written by export.py, and save per-query test metrics.

Runs in the RaMen environment, from the CLHE repository:
cd ../baselines/CLHE && ../RaMen_repo/.venv/bin/python ../../bundlefuse/experiments/ramen_retail/clhe_runner.py \
    --dataset electronic_clean --preset pog --seed 2023 --out ../../bundlefuse/results/clhe_retail

CLHE ships settings only for POG and Spotify (run.sh). The four presets below are those settings; one is chosen per
dataset on validation NDCG@20. Training settings follow CLHE's pog block in config.yaml (batch 256, lr 1e-3, L2 1e-5)
and train.py defaults. The epoch is chosen as in train.py: a new best needs both validation Recall@20 and NDCG@20 to
improve. Test is scored once, at that epoch.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
sys.path.insert(0,os.getcwd())
import models
from train import setup_seed
from utility import Datasets

DEFAULTS=dict(gpu='0',model='CLHE',info='',reg=1e-5,dropout_rate=.2,noise_weight=.02,attention='',trans_layer=1,num_token=200,epoch=-1)
PRESETS={  # from CLHE's run.sh
    'pog_dense':dict(item_augment='MD',bundle_augment='ID',bundle_ratio=.5,bundle_cl_temp=.05,bundle_cl_alpha=2,cl_temp=.5,cl_alpha=.1),
    'pog':dict(item_augment='FN',bundle_augment='ID',bundle_ratio=.5,bundle_cl_temp=.01,bundle_cl_alpha=.5,cl_temp=.5,cl_alpha=2),
    'spotify':dict(item_augment='MD',bundle_augment='ID',bundle_ratio=.05,bundle_cl_temp=.2,bundle_cl_alpha=.5,cl_temp=.05,cl_alpha=.1),
    'spotify_sparse':dict(item_augment='MD',bundle_augment='ID',bundle_ratio=.05,bundle_cl_temp=1,bundle_cl_alpha=1,cl_temp=.2,cl_alpha=.1)}


def per_query(model,loader,k=20):
    model.eval();out=[];rs=model.propagate()
    with torch.no_grad():
        for index,b_i_input,seq,gt in loader:
            pred=model.evaluate(rs,(index,b_i_input,seq))-1e8*b_i_input
            top=pred.topk(k).indices;hit=gt.gather(1,top)
            pos=gt.sum(1);disc=1/torch.log2(torch.arange(2,k+2,dtype=torch.float))
            idcg=torch.stack([disc[:int(min(p,k))].sum() for p in pos])
            out+=torch.stack([hit.sum(1)/pos,(hit*disc).sum(1)/idcg],1).tolist()
    return np.asarray(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--preset',required=True,choices=sorted(PRESETS))
    p.add_argument('--seed',type=int,default=2023);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(a.threads)
    conf=yaml.safe_load(open('config.yaml'))['pog']
    conf.update(DEFAULTS,**PRESETS[a.preset],dataset=a.dataset,lr=conf['lrs'][0],seed=a.seed,device=torch.device('cpu'))
    setup_seed(a.seed);ds=Datasets(conf)
    conf.update(num_users=ds.num_users,num_bundles=ds.num_bundles,num_items=ds.num_items,l2_reg=conf['l2_regs'][0],
                embedding_size=64,num_layers=conf['num_layerss'][0])
    model=models.CLHE(conf,ds.graphs,ds.features)
    optimizer=torch.optim.Adam(model.parameters(),lr=conf['lr'],weight_decay=conf['l2_reg'])
    setup_seed(a.seed);best=(0,0);state=None;history=[];start=time.time()
    for epoch in range(a.epochs):
        model.train(True)
        for batch in ds.train_loader:
            optimizer.zero_grad();model(batch)['loss'].backward();optimizer.step()
        val=per_query(model,ds.val_loader).mean(0);history.append(val.tolist())
        if val[0]>best[0] and val[1]>best[1]:best=(val[0],val[1]);state=copy.deepcopy(model.state_dict());chosen=epoch
    model.load_state_dict(state);test=per_query(model,ds.test_loader)
    tag=f'{a.dataset}/{a.preset}_seed{a.seed}';path=Path(a.out)/tag;path.mkdir(parents=True,exist_ok=True)
    (path/'result.json').write_text(json.dumps(dict(dataset=a.dataset,preset=a.preset,seed=a.seed,epoch=chosen,validation=list(best),
        test=test.mean(0).tolist(),queries=test.tolist(),history=history,seconds=time.time()-start,
        parameters=sum(x.numel() for x in model.parameters() if x.requires_grad))))
    print(f'{tag} epoch {chosen} val R20 {best[0]:.4f} N20 {best[1]:.4f} test R20 {test[:,0].mean():.4f} N20 {test[:,1].mean():.4f}',flush=True)


if __name__=='__main__':
    main()
