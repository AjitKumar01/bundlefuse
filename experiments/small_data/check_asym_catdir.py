"""Outfits: asymmetric score between category centroids (m_a' K m_b) against the validation direction log(#seed a with missing b / #seed b with missing a); and how much of the catalogue scores above an item itself under M_zz.
Validation only, models of experiments.small_data.readings.

cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python experiments/small_data/check_asym_catdir.py <names>
"""
import triad, json, sys, numpy as np, torch
from triad import data
from experiments.small_data.triad_v2 import TRIADv2
from experiments.small_data.readings import CACHE, model_path, blocks
name=sys.argv[1];ds=data.load(name,CACHE);lab=ds.levels[0];L=lab.max()+1
# data direction on validation: seed item in a, target item in b
cnt=np.zeros((L,L))
for s,g in ds.val_pairs:
    for a in set(lab[list(set(s))]):
        for b in set(lab[list(set(g))]):cnt[a,b]+=1
out=[]
for seed in (0,1,2):
    m=TRIADv2(ds.features,ds.levels,ds.train);m.load_state_dict(torch.load(model_path(name,seed),map_location='cpu'));m.eval()
    with torch.no_grad():
        c=m.cache();z=c['content'];M=blocks(m,c)['zz'];K=(M-M.T)/2
        cen=torch.stack([z[torch.as_tensor(lab==l)].mean(0) if (lab==l).any() else torch.zeros(z.shape[1]) for l in range(L)])
        d=(cen@K@cen.T).numpy()
        # self-score rank: for 2000 items, how many catalogue items score above the item itself
        g=torch.Generator().manual_seed(0);q=torch.randint(z.shape[0],(2000,),generator=g)
        sc=(z[q]@M)@z.T;self_=sc[torch.arange(2000),q];above=(sc>self_[:,None]).float().mean(1)
        cos=z[q]@z.T;rank_cos=(cos>cos[torch.arange(2000),q][:,None]).float().mean(1)
    iu=[(a,b) for a in range(L) for b in range(a+1,L) if cnt[a,b]>=20 and cnt[b,a]>=20]
    x=np.array([d[a,b] for a,b in iu]);y=np.array([np.log(cnt[a,b]/cnt[b,a]) for a,b in iu])
    out.append(dict(pairs=len(iu),corr=float(np.corrcoef(x,y)[0,1]),sign_agree=float(np.mean(np.sign(x)==np.sign(y))),
        self_share_above=float(above.mean()),self_median_above=float(above.median()),cos_share_above=float(rank_cos.mean()),
        top=[(int(a),int(b),float(d[a,b]),int(cnt[a,b]),int(cnt[b,a])) for a,b in sorted(iu,key=lambda p:-abs(d[p[0],p[1]]))[:6]]))
    print(seed,{k:v for k,v in out[-1].items() if k!='top'});print('  top',out[-1]['top'])
json.dump(out,open(f'results/small_data/catdir_{name}.json','w'))
