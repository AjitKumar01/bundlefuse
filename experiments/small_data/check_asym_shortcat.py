"""Per category: mean fixed candidate preference against the target rate inside each validation query's top 100.
cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python experiments/small_data/check_asym_shortcat.py pog_dense
"""
import triad, sys, json, numpy as np, torch
from triad import data
from experiments.small_data.triad_v2 import TRIADv2
from experiments.small_data.readings import CACHE, VQ, model_path, blocks, content_from, scaled
name=sys.argv[1];ds=data.load(name,CACHE);lab=ds.levels[0];L=lab.max()+1;pairs=ds.val_pairs[:VQ]
info=json.load(open(f'../bundlemllm/datasets/{name}/item_info.json'))
m=TRIADv2(ds.features,ds.levels,ds.train);m.load_state_dict(torch.load(model_path(name,0),map_location='cpu'));m.eval()
app=np.zeros(L);hit=np.zeros(L)
with torch.no_grad():
    c=m.cache();M=blocks(m,c);z=c['content'];K=(M['zz']-M['zz'].T)/2;Sym=(M['zz']+M['zz'].T)/2;mu=z.mean(0)
    rest={k:v for k,v in M.items() if k!='zz'};base=content_from(dict(rest,zz=Sym),c);bias=((mu@K)@z.T).numpy()
    for a in range(0,len(pairs),128):
        P=pairs[a:a+128];S=[torch.as_tensor(sorted(set(s))) for s,_ in P]
        full=m.content_scores(S,None,c);den=full.abs().amax(1,keepdim=True)
        rs=[m.label_scores(i,S) for i in range(m.levels)]+[m.graph_scores(S,None,c)];w=m.gate_weights(S,None)
        tot=w[:,0,None]*base(S)/den+sum(w[:,j+1,None]*scaled(p) for j,p in enumerate(rs))
        for i,s in enumerate(S):tot[i,s]=-1e9
        top=tot.topk(100,1).indices.numpy()
        for i,(s,g) in enumerate(P):
            np.add.at(app,lab[top[i]],1);np.add.at(hit,lab[top[i][np.isin(top[i],list(g))]],1)
pref=np.array([bias[lab==l].mean() if (lab==l).any() else np.nan for l in range(L)])
keep=app>=300;rate=hit[keep]/app[keep]
print('categories',keep.sum(),'corr(category preference, target rate in shortlist) %.2f'%np.corrcoef(pref[keep],rate)[0,1])
for l in np.where(keep)[0][np.argsort(-pref[keep])]:
    ex=[info[str(i)]['title'][:18] for i in np.where(lab==l)[0][:2]]
    print('%+.2f  rate %.4f  n_short %6d  %s'%(pref[l],hit[l]/app[l],app[l],' | '.join(ex)))
