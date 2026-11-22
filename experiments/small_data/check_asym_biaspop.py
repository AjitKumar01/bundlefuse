"""Candidate preference mu' K z_c: correlation with log(1+f_c), and AUC for validation targets over the whole catalogue, among never-bundled and among bundled items.
Validation only, models of experiments.small_data.readings.

cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python experiments/small_data/check_asym_biaspop.py <names>
"""
import triad, sys, numpy as np, torch
from triad import data
from experiments.small_data.triad_v2 import TRIADv2
from experiments.small_data.readings import CACHE, model_path, blocks
def auc(pos,neg):
    allv=np.concatenate([pos,neg]);r=allv.argsort().argsort()+1.
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
for name in sys.argv[1:]:
    ds=data.load(name,CACHE)
    tg=np.zeros(ds.n_items,bool)
    for _,g in ds.val_pairs:tg[list(g)]=True
    for seed in (0,1,2):
        m=TRIADv2(ds.features,ds.levels,ds.train);m.load_state_dict(torch.load(model_path(name,seed),map_location='cpu'));m.eval()
        with torch.no_grad():
            c=m.cache();z=c['content'];M=blocks(m,c)['zz'];K=(M-M.T)/2;b=((z.mean(0)@K)@z.T).numpy()
        f=m.frequency.numpy();lf=np.log1p(f);cold=f==0
        rho=c['prior'].numpy()
        print(name,seed,'corr(bias,log1p f) %.2f'%np.corrcoef(b,lf)[0,1],
              'AUC target all %.2f'%auc(b[tg],b[~tg]),
              'AUC target among never-bundled %.2f (n_pos %d)'%(auc(b[tg&cold],b[~tg&cold]),(tg&cold).sum()) if (tg&cold).sum()>20 else '',
              'AUC within bundled %.2f'%auc(b[tg&~cold],b[~tg&~cold]))
