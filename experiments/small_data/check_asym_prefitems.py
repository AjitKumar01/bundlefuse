"""Items most and least preferred by the fixed candidate preference mu' K z_c (POG_DENSE), its stability across seeds. Validation-free (parameters only).
cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python experiments/small_data/check_asym_prefitems.py pog_dense
"""
import triad, sys, json, numpy as np, torch
from triad import data
from experiments.small_data.triad_v2 import TRIADv2
from experiments.small_data.readings import CACHE, model_path, blocks
name=sys.argv[1];ds=data.load(name,CACHE)
info=json.load(open(f'../bundlemllm/datasets/{name}/item_info.json'))
B=[]
for seed in (0,1,2):
    m=TRIADv2(ds.features,ds.levels,ds.train);m.load_state_dict(torch.load(model_path(name,seed),map_location='cpu'));m.eval()
    with torch.no_grad():
        c=m.cache();z=c['content'];M=blocks(m,c)['zz'];K=(M-M.T)/2;B.append(((z.mean(0)@K)@z.T).numpy())
B=np.array(B);print('corr across seeds',np.round(np.corrcoef(B),2).tolist())
b=B.mean(0);o=b.argsort()
url=lambda i:info[str(i)].get('pic_url','')
kind=lambda i:'scene' if 'beehive' in url(i) or 'scenes' in url(i) else ('uploaded' if 'uploaded' in url(i) else 'other')
print('TOP');[print(round(b[i],2),kind(i),info[str(i)].get('title','')[:40]) for i in o[::-1][:25]]
print('BOTTOM');[print(round(b[i],2),kind(i),info[str(i)].get('title','')[:40]) for i in o[:25]]
ks=np.array([kind(i) for i in range(len(b))])
for k in set(ks):print(k,(ks==k).sum(),'mean pref %.3f'%b[ks==k].mean())
