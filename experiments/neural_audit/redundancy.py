"""Validation-only audit of the frozen neural baseline: component interventions, fitted-weight checks, POG gap."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F
from bundlefuse import datasets
from bundlefuse.metrics import top_items
from bundlefuse.provenance import write_json
from experiments.neural.data import load_data
from experiments.neural_revision.model import RelationNetwork
from experiments.sufficiency.engine import metrics

BETAS=[.1,.25,.5,1.,2.,4.]
ROOTS=dict(pog='results/neural_pog',pog_dense='results/neural_pog',electronic='results/neural_constrained',food='results/neural_transfer',clothing='results/neural_transfer')


def log(message):print(message,flush=True)


def load(name):
    if name in ('pog','pog_dense'):
        ds=datasets.load(name,protocol_mode='provided',data_profile='extended',val_n=0,test_n=0)
        x=np.load(f'checkpoints/neural_pog/{name}/features/projected.npy');ds.features=None
        _,c=np.unique(ds.cat_labels,return_inverse=True)
    else:
        ds,x,c,_,_=load_data(name)
    return ds,x,c


def pair_tables(ds,model):
    """Count and PMI values on the model's own pair support, scaled to the mean learned weight."""
    n=ds.n_items;train=[sorted(set(b)) for b in ds.train]
    rows=np.concatenate([np.repeat(i,len(b)) for i,b in enumerate(train)]);cols=np.concatenate(train)
    B=sp.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(train),n))
    C=(B.T@B).tocsr();C.setdiag(0);C.eliminate_zeros()
    f=np.asarray(B.sum(0)).ravel()
    s,t=model.edges.numpy()
    count=np.asarray(C[s,t]).ravel()
    pmi=np.maximum(np.log(count*len(train)/(f[s]*f[t])),0)
    return f,count,pmi


def fitted(model,count,pmi,f,x):
    with torch.no_grad():
        w=model.pair_weights.numpy();z=model.features+model.text(model.features)
        source,target=model.edges
        degree=torch.zeros(model.n).scatter_add(0,target,torch.ones(len(source))).clamp_min(1)
        adj=torch.sparse_coo_tensor(torch.stack((target,source)),torch.ones(len(source)),(model.n,model.n))
        message=model.graph_projection(torch.sparse.mm(adj,z)/degree[:,None])
        linked=(f>0)&(degree.numpy()>=1)
        sv=torch.linalg.svdvals(model.graph_projection.weight).numpy()
        v=model.item_vectors();reverse={(a,b):i for i,(a,b) in enumerate(zip(source.tolist(),target.tolist()))}
        mirror=np.array([reverse[(b,a)] for a,b in zip(source.tolist(),target.tolist())])
        delta=model.text(model.features)
    nz=w>0
    return dict(pair_weights=len(w),share_zero=float((~nz).mean()),mean_nonzero=float(w[nz].mean()),
                spearman_weight_count=float(spearmanr(w,np.log1p(count)).statistic),spearman_weight_pmi=float(spearmanr(w,pmi).statistic),
                spearman_weight_pmi_nonzero=float(spearmanr(w[nz],pmi[nz]).statistic),symmetry_spearman=float(spearmanr(w,w[mirror]).statistic),
                temperature=float(model.log_text_scale.exp()),encoder_change_ratio=float((delta.norm(dim=1)/model.features.norm(dim=1)).median()),
                graph_change_ratio_linked=float((message[linked].norm(dim=1)/z[linked].norm(dim=1)).median()) if linked.any() else None,
                cosine_final_to_input=float((v*F.normalize(model.features,dim=1)).sum(1).mean()),
                cosine_final_to_input_linked=float((v[linked]*F.normalize(model.features,dim=1)[linked]).sum(1).mean()) if linked.any() else None,
                projection_singular_top5=sv[:5].tolist(),projection_effective_rank=float((sv**2).sum()**2/(sv**4).sum()),
                catalogue_items=model.n,catalogue_share_with_history=float((f>0).mean()))


class Row:
    def __init__(self):self.values=[];self.cold=[0,0];self.warm=[0,0]
    def add(self,score,pair,f):
        self.values.append(metrics(score,pair).tolist())
        top=set(top_items(score,pair[0],20).tolist())
        for g in set(pair[1]):
            bucket=self.cold if f[g]==0 else self.warm
            bucket[0]+=g in top;bucket[1]+=1


def standard(a):return (a-a.mean(0,keepdims=True))/(a.std(0,keepdims=True)+1e-9)


def run(args):
    torch.set_num_threads(1);name=args.dataset
    ds,x,c=load(name)
    pairs=ds.val_pairs[:args.limit] if args.limit else ds.val_pairs
    choice=json.load(open(f'{ROOTS[name]}/{name}/graph_pair_no_category_seed{args.seed}/selected.json'))
    model=RelationNetwork(x,c,ds.train,variant='graph_pair',category_mode='off')
    model.load_state_dict(torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)['model']);model.eval()
    f,count,pmi=pair_tables(ds,model)
    audit=dict(dataset=name,seed=args.seed,checkpoint=choice['checkpoint'],validation_queries=len(pairs),fitted=fitted(model,count,pmi,f,x))
    log(json.dumps(audit['fitted']))
    n=model.n;source,target=[a.numpy() for a in model.edges];w=model.pair_weights.detach().numpy()
    table=lambda v:sp.csr_matrix((v,(target,source)),shape=(n,n))
    nz=w>0;scale=w[nz].mean()
    PW=table(w);Pbin=table(np.full_like(w,scale));Pcount=table(np.log1p(count)*scale/np.log1p(count).mean());Ppmi=table(pmi*scale/max(pmi.mean(),1e-9))
    with torch.no_grad():
        X=F.normalize(model.features,dim=1);z=X+model.text(X)
        deg=torch.zeros(n).scatter_add(0,model.edges[1],torch.ones(len(source))).clamp_min(1)
        adj=torch.sparse_coo_tensor(torch.stack((model.edges[1],model.edges[0])),torch.ones(len(source)),(n,n))
        V=dict(full=model.item_vectors(),no_graph=F.normalize(z,dim=1),no_encoder=F.normalize(X+model.graph_projection(torch.sparse.mm(adj,X)/deg[:,None]),dim=1),raw=X)
    tau=float(model.log_text_scale.exp())
    reference=None
    if name in ('pog','pog_dense'):
        from bundlefuse.model import BundleFuse
        from experiments.mechanisms.selection import score_recipe
        reference=BundleFuse.load(f'checkpoints/publication_mechanisms/{name}/artifacts.pt')
        recipe=json.load(open(f'results/publication_mechanisms/{name}/full.json'))
        reference.w=recipe['weights'];reference.prior_keys=recipe['prior'];reference.best_ho=recipe['higher_order'];reference.best_eb=recipe['backoff']
        frequency=torch.as_tensor(reference.freq)[:,None]
    arms={};rows=lambda k:arms.setdefault(k,Row())
    same_cat=[];keys=None
    for start in range(0,len(pairs),args.chunk):
        chunk=pairs[start:start+args.chunk];y=np.zeros((n,len(chunk)),np.float32)
        for q,(s,_) in enumerate(chunk):y[list(s),q]=1
        size=y.sum(0,keepdims=True);text={k:(tau*(v.numpy()@(v.numpy().T@y))/size) for k,v in V.items()}
        p=PW@y
        base=dict(full=text['full']+p,text_only=text['full'],pair_only=p,no_graph=text['no_graph']+p,no_encoder=text['no_encoder']+p,
                  raw_plus_pair=text['raw']+p,raw_only=text['raw'],binary_pair=text['full']+Pbin@y,count_pair=text['full']+Pcount@y,pmi_pair=text['full']+Ppmi@y,
                  raw_plus_pmi=text['raw']+Ppmi@y)
        grid={}
        if reference is not None:
            raw,sig=reference._score_all([list(s) for s,_ in chunk])
            bf=np.asarray(score_recipe(sig,recipe,frequency));base['bundlefuse']=bf
            if keys is None:keys=sorted(sig)
            neural=standard(base['full']);rawn=standard(base['raw_plus_pair'])
            cat=sig['cat'].numpy() if 'cat' in sig else None;style=sig['style'].numpy() if 'style' in sig else None
            for b in BETAS:
                if cat is not None:grid[f'neural+cat@{b}']=neural+b*cat
                if style is not None:grid[f'neural+style@{b}']=neural+b*style
                grid[f'bundlefuse+neural@{b}']=standard(bf)+b*neural
                if cat is not None and style is not None:
                    for b2 in BETAS:grid[f'neural+cat@{b}+style@{b2}']=neural+b*cat+b2*style
        for k,a in {**base,**grid}.items():
            for q,pair in enumerate(chunk):rows(k).add(np.asarray(a[:,q],np.float32),pair,f)
        for s,g in chunk:
            cats={int(c[i]) for i in s}
            same_cat.extend((f[i]==0,int(c[i]) in cats) for i in set(g))
        log(f'{name} seed {args.seed}: {min(start+args.chunk,len(pairs))}/{len(pairs)}')
    same=np.asarray(same_cat,bool)
    audit['targets']=dict(cold=int(same[:,0].sum()),warm=int((~same[:,0]).sum()),cold_share_same_category_as_seed=float(same[same[:,0],1].mean()) if same[:,0].any() else None,
                          warm_share_same_category_as_seed=float(same[~same[:,0],1].mean()) if (~same[:,0]).any() else None)
    audit['signal_keys']=keys;audit['categories']=int(c.max())+1
    order=np.random.default_rng(20261108).permutation(len(pairs));half=len(order)//2;A,B=order[:half],order[half:]
    result={}
    for k,r in arms.items():
        v=np.asarray(r.values)
        result[k]=dict(recall20=float(v[:,0].mean()),ndcg20=float(v[:,1].mean()),a_ndcg20=float(v[A,1].mean()),b_recall20=float(v[B,0].mean()),b_ndcg20=float(v[B,1].mean()),
                       cold_recall20=r.cold[0]/max(r.cold[1],1),warm_recall20=r.warm[0]/max(r.warm[1],1))
    audit['arms']=result
    selected={}
    for prefix in ('neural+cat@','neural+style@','bundlefuse+neural@'):
        options=[k for k in result if k.startswith(prefix) and '+style' not in k[len(prefix):]]
        if options:selected[prefix[:-1]]=max(options,key=lambda k:result[k]['a_ndcg20'])
    options=[k for k in result if k.startswith('neural+cat@') and '+style@' in k]
    if options:selected['neural+cat+style']=max(options,key=lambda k:result[k]['a_ndcg20'])
    audit['selected_on_half_a']=selected
    write_json(args.output/f'{name}_seed{args.seed}.json',audit);log(f'{name} seed {args.seed} COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=list(ROOTS));parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--chunk',type=int,default=64);parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--output',type=Path,default=Path('results/neural_audit'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():run(args)
