"""Candidate preference mu' K z_c inside each query's top 100 (ranked by the model without it): AUC for targets, and its spread against the spread of the rest of the score.
Validation only, models of experiments.small_data.readings.

cd ../bundlefuse && PYTHONPATH=.:../dials .venv/bin/python experiments/small_data/check_asym_shortauc.py <names>
"""
import triad, sys, numpy as np, torch
from triad import data
from experiments.small_data.triad_v2 import TRIADv2
from experiments.small_data.readings import CACHE, VQ, model_path, blocks, content_from, scaled
for name in sys.argv[1:]:
    ds=data.load(name,CACHE);pairs=ds.val_pairs[:VQ]
    for seed in (0,1,2):
        m=TRIADv2(ds.features,ds.levels,ds.train);m.load_state_dict(torch.load(model_path(name,seed),map_location='cpu'));m.eval()
        with torch.no_grad():
            c=m.cache();M=blocks(m,c);z=c['content'];K=(M['zz']-M['zz'].T)/2;Sym=(M['zz']+M['zz'].T)/2;mu=z.mean(0)
            rest={k:v for k,v in M.items() if k!='zz'};base=content_from(dict(rest,zz=Sym),c);bias=(mu@K)@z.T
            aucs=[];hits=0;share=[]
            for a in range(0,len(pairs),128):
                P=pairs[a:a+128];S=[torch.as_tensor(sorted(set(s))) for s,_ in P]
                full=m.content_scores(S,None,c);den=full.abs().amax(1,keepdim=True)
                rest_=[m.label_scores(i,S) for i in range(m.levels)]+[m.graph_scores(S,None,c)];w=m.gate_weights(S,None)
                tot=w[:,0,None]*base(S)/den+sum(w[:,j+1,None]*scaled(p) for j,p in enumerate(rest_))
                for i,s in enumerate(S):tot[i,s]=-1e9
                top=tot.topk(100,1).indices
                b=(w[:,0,None]*bias[None,:]/den)
                for i,(s,g) in enumerate(P):
                    t=top[i].numpy();pos=np.isin(t,list(g))
                    if pos.any() and (~pos).any():
                        v=b[i,t].numpy();aucs.append((v[pos][:,None]>v[~pos][None,:]).mean())
                    # spread of the bias in the shortlist vs the spread of the rest of the score
                    share.append(float(b[i,t].std()/tot[i,t].std()))
        print(name,seed,'shortlist AUC %.3f over %d queries; bias std / score std in top100 %.2f'%(np.mean(aucs),len(aucs),np.mean(share)))
