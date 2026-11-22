"""Content knock-outs two ways: (a) renormalized (as in readings.py: the changed content term is divided by its own
per-query max), (b) fixed scale (divided by the trained content term's per-query max: the literal removal)."""
import triad, sys, json, numpy as np, torch
from triad import data
from triad.evaluate import query_metrics
from experiments.small_data.triad_v2 import TRIADv2
from experiments.small_data.readings import CACHE, VQ, model_path, blocks, content_from, scaled
OUT='results/small_data/fixedscale.json'
@torch.no_grad()
def run(m,pairs,variants):
    c=m.cache();res={k:[] for k in ['trained']+[f'{v}_{w}' for v in variants for w in ('renorm','fixed')]}
    for a in range(0,len(pairs),128):
        P=pairs[a:a+128];S=[torch.as_tensor(sorted(set(s))) for s,_ in P]
        full=m.content_scores(S,None,c);den=full.abs().amax(1,keepdim=True).clamp_min(1e-6)
        rest=[m.label_scores(i,S) for i in range(m.levels)]+[m.graph_scores(S,None,c)];w=m.gate_weights(S,None)
        tail=sum(w[:,j+1,None]*scaled(p) for j,p in enumerate(rest))
        def score(cont):
            v=(w[:,0,None]*cont+tail).numpy();return [query_metrics(v[k],s,g)['ndcg20'] for k,(s,g) in enumerate(P)]
        res['trained']+=score(full/den)
        for name,fn in variants.items():
            x=fn(S);res[f'{name}_renorm']+=score(scaled(x));res[f'{name}_fixed']+=score(x/den)
    return {k:float(np.mean(v)) for k,v in res.items()}
rep=json.load(open(OUT)) if len(sys.argv)>1 and False else {}
for name in sys.argv[1:]:
    ds=data.load(name,CACHE);pairs=ds.val_pairs[:VQ];rep[name]=[]
    for seed in (0,1,2):
        m=TRIADv2(ds.features,ds.levels,ds.train);m.load_state_dict(torch.load(model_path(name,seed),map_location='cpu'));m.eval()
        c=m.cache();M=blocks(m,c);z=c['content'];K=(M['zz']-M['zz'].T)/2;Sym=(M['zz']+M['zz'].T)/2;mu=z.mean(0)
        rest={k:v for k,v in M.items() if k!='zz'};base=content_from(dict(rest,zz=Sym),c);bias=(mu@K)@z.T
        def seedpart(S):
            zb=torch.stack([z[s].mean(0) for s in S]);return base(S)+((zb-mu)@K)@z.T
        V=dict(symmetric=base, candidate_only=lambda S:base(S)+bias[None,:], seed_only=seedpart,
               no_history=content_from(dict(zz=M['zz']),c), start=content_from(dict(zz=10*torch.eye(M['zz'].shape[0])),c))
        with torch.no_grad():r=run(m,pairs,V)
        rep[name].append(r);print(name,seed,{k:round(v-r['trained'],4) for k,v in r.items()},flush=True)
    json.dump(rep,open(OUT.replace('.json',f'_{"_".join(sys.argv[1:])}.json'),'w'))
