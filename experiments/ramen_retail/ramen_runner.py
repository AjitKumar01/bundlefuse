"""Train RaMen with its own code on a dataset written by export.py, and save per-query test metrics.

Runs in the RaMen environment (torch_geometric), from the RaMen repository:
cd ../baselines/RaMen_repo && .venv/bin/python ../../bundlefuse/experiments/ramen_retail/ramen_runner.py \
    --dataset electronic_clean --lr 1e-3 --n_layer_gat 2 --gamma 0.5 --seed 2306 --out ../../bundlefuse/results/ramen_retail

Settings follow the repository's train.py defaults and its electronic block in config.yaml, with the paper's
H=4 hyperedges, except gamma. The paper's gamma (eqs. 7-8) mixes the characteristic and collaborative (CbSE)
encoders and is tuned in {0.1, ..., 0.9}; the repository default of 1.0 (bundle_alpha = item_alpha = 1) removes CbSE
from the score, so gamma is set here and passed to both. The epoch is chosen as in train.py: a new best needs both validation Recall@20 and NDCG@20 to
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

DEFAULTS=dict(gpu='0',model='RAMEN',info='',reg=1e-5,item_augment='NA',bundle_ratio=.5,bundle_augment='ID',dropout_rate=.2,
              noise_weight=.02,cl_temp=.2,cl_alpha=0,bundle_cl_temp=.2,bundle_cl_alpha=.1,attention='',trans_layer=1,num_token=200,
              epoch=-1,multiGPU=False,wandb=0,custom_log_path='',iui_path='iui_edge_index',bundle_alpha=1.,item_alpha=1.,
              alpha_residual=1.,hyper_num=4,alpha_bundle_loss=.05,alpha_item_loss=.1,residual_type=0)


def per_query(model,loader,k=20):
    model.eval();out=[]
    with torch.no_grad():
        for index,b_i_input,seq,gt in loader:
            pred=model.evaluate(None,(index,b_i_input,seq))-1e8*b_i_input
            top=pred.topk(k).indices;hit=gt.gather(1,top)
            pos=gt.sum(1);disc=1/torch.log2(torch.arange(2,k+2,dtype=torch.float))
            idcg=torch.stack([disc[:int(min(p,k))].sum() for p in pos])
            out+=torch.stack([hit.sum(1)/pos,(hit*disc).sum(1)/idcg],1).tolist()
    return np.asarray(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--lr',type=float,required=True)
    p.add_argument('--n_layer_gat',type=int,required=True);p.add_argument('--gamma',type=float,required=True);p.add_argument('--seed',type=int,default=2306)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--threads',type=int,default=2);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(a.threads)
    conf=yaml.safe_load(open('config.yaml'))['electronic']
    conf.update(DEFAULTS,dataset=a.dataset,lr=a.lr,n_layer_gat=a.n_layer_gat,bundle_alpha=a.gamma,item_alpha=a.gamma,seed=a.seed,device=torch.device('cpu'))
    setup_seed(a.seed);ds=Datasets(conf)
    conf.update(num_users=ds.num_users,num_bundles=ds.num_bundles,num_items=ds.num_items,l2_reg=conf['l2_regs'][0],
                embedding_size=64,num_layers=conf['num_layerss'][0])
    model=models.RAMEN(conf,ds.graphs,ds.features)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=conf['l2_reg'])
    setup_seed(a.seed);best=(0,0);state=None;history=[];start=time.time()
    for epoch in range(a.epochs):
        model.train(True)
        for batch in ds.train_loader:
            optimizer.zero_grad();model(batch)['loss'].backward();optimizer.step()
        val=per_query(model,ds.val_loader).mean(0);history.append(val.tolist())
        if val[0]>best[0] and val[1]>best[1]:best=(val[0],val[1]);state=copy.deepcopy(model.state_dict());chosen=epoch
    model.load_state_dict(state);test=per_query(model,ds.test_loader)
    tag=f'{a.dataset}/lr{a.lr:g}_gat{a.n_layer_gat}_gamma{a.gamma:g}_seed{a.seed}';path=Path(a.out)/tag;path.mkdir(parents=True,exist_ok=True)
    (path/'result.json').write_text(json.dumps(dict(dataset=a.dataset,lr=a.lr,n_layer_gat=a.n_layer_gat,gamma=a.gamma,seed=a.seed,epoch=chosen,
        validation=list(best),test=test.mean(0).tolist(),queries=test.tolist(),history=history,seconds=time.time()-start)))
    print(f'{tag} epoch {chosen} val R20 {best[0]:.4f} N20 {best[1]:.4f} test R20 {test[:,0].mean():.4f} N20 {test[:,1].mean():.4f}',flush=True)


if __name__=='__main__':
    main()
