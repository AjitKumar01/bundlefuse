"""Validation-only diagnosis of the low-rank model on Spotify: target structure, single signals, missed cases.

PYTHONPATH=. .venv/bin/python -m experiments.neural_audit.spotify --dataset spotify_sparse
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from bundlefuse import datasets
from bundlefuse.provenance import write_json
from experiments.neural.evaluation import summarize
from experiments.neural_lowrank import run
import experiments.neural_lowrank2.run  # noqa: F401  patches run.construct when --version v2


def incidence(labels,n):
    return sp.csr_matrix((np.ones(n,np.float32),(np.arange(n),labels)),shape=(n,int(labels.max())+1))


def metrics(scores,pairs):
    r=summarize(scores,pairs)
    return [r['recall20'],r['ndcg20']],[set(q['top20']) for q in r['queries']]


@torch.inference_mode()
def main(args):
    torch.set_num_threads(4)
    ds,x,c,f,_=run.data(args.dataset)
    raw=datasets.load(args.dataset,protocol_mode='provided',data_profile='extended',val_n=0,test_n=0)
    album=np.asarray(raw.alb_labels) if raw.alb_labels is not None else None;del raw
    pairs=ds.val_pairs[:args.queries];n=ds.n_items
    model=run.construct(ds,x,c) if args.version=='v2' else __import__('experiments.neural_lowrank.model',fromlist=['x']).LowRankRelation(x,c,ds.train)
    state=torch.load(Path(args.checkpoint),map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);model.eval();keys=model.keys()
    counts=sp.csr_matrix((model.counts.numpy(),(np.repeat(np.arange(n),np.diff(model.pointer.numpy())),model.target.numpy())),shape=(n,n))
    freq=model.frequency.numpy();A=incidence(c,n);B=incidence(album,n) if album is not None else None
    z=model.content().numpy()
    structure=dict(targets=0,never_sold=0,same_artist=0,same_album=0,co_purchased=0,seeds_per_query=float(np.mean([len(set(s)) for s,_ in pairs])))
    for s,g in pairs:
        s=list(set(s));sa=set(c[s]);sb=set(album[s]) if album is not None else set();partners=set(counts[s].indices)
        for t in set(g):
            structure['targets']+=1;structure['never_sold']+=int(freq[t]==0);structure['same_artist']+=int(c[t] in sa)
            structure['same_album']+=int(album is not None and album[t] in sb);structure['co_purchased']+=int(t in partners)
    structure={k:(v/structure['targets'] if k not in ('targets','seeds_per_query') else v) for k,v in structure.items()}
    signals={};tops={}
    def add(name,fn):
        rows=[];
        for start in range(0,len(pairs),64):
            rows.append(fn([list(set(s)) for s,_ in pairs[start:start+64]]))
        signals[name],tops[name]=metrics(np.concatenate(rows),pairs)
    def indicator(batch):
        y=np.zeros((len(batch),n),np.float32)
        for i,s in enumerate(batch):y[i,s]=1
        return y
    add('popularity',lambda b:np.tile(np.log1p(freq),(len(b),1)))
    add('co_purchase_sum',lambda b:np.asarray((sp.csr_matrix(indicator(b))@counts).todense()))
    add('co_purchase_sum + popularity',lambda b:np.asarray((sp.csr_matrix(indicator(b))@counts).todense())+1e-3*np.log1p(freq))
    add('same_artist_count + popularity',lambda b:np.asarray((sp.csr_matrix(indicator(b))@A@A.T).todense())+1e-3*np.log1p(freq))
    if B is not None:add('same_album_count + popularity',lambda b:np.asarray((sp.csr_matrix(indicator(b))@B@B.T).todense())+1e-3*np.log1p(freq))
    total=counts.sum();rowsum=np.asarray(counts.sum(1)).ravel()
    pmi=counts.tocoo();v=np.log(pmi.data*total/(rowsum[pmi.row]*rowsum[pmi.col]));keep=v>0
    ppmi=sp.csr_matrix((v[keep],(pmi.row[keep],pmi.col[keep])),shape=(n,n))
    add('ppmi_sum',lambda b:np.asarray((sp.csr_matrix(indicator(b))@ppmi).todense())+1e-3*np.log1p(freq))
    norm=sp.diags(1/np.maximum(rowsum,1))@counts
    add('two_hop',lambda b:np.asarray(((sp.csr_matrix(indicator(b))@norm)@norm).todense())+1e-3*np.log1p(freq))
    add('content_mean_cosine',lambda b:indicator(b)@z@z.T/np.array([[len(s)] for s in b]))
    for part in (None,'content','history','group','pair')+(('match','popularity') if args.version=='v2' else ()):
        model.disabled={part} if part else set();k=model.keys()
        add('model' if part is None else f'model without {part}',lambda b:model(b,keys=k).numpy())
    model.disabled=set()
    # Which targets each source finds that the model misses
    missed={}
    model_top=tops['model']
    for name,top in tops.items():
        if name.startswith('model'):continue
        found=only=0
        for (s,g),a,m in zip(pairs,top,model_top):
            for t in set(g):found+=int(t in a);only+=int(t in a and t not in m)
        missed[name]=dict(hits=found,hits_model_misses=only)
    hits={k:0 for k in ('never_sold','same_artist','co_purchased','other')};totals=dict(hits)
    for (s,g),m in zip(pairs,model_top):
        s=list(set(s));sa=set(c[s]);partners=set(counts[s].indices)
        for t in set(g):
            k='never_sold' if freq[t]==0 else 'co_purchased' if t in partners else 'same_artist' if c[t] in sa else 'other'
            totals[k]+=1;hits[k]+=int(t in m)
    out=dict(dataset=args.dataset,queries=len(pairs),checkpoint=args.checkpoint,structure=structure,
             single_signal_recall_ndcg=signals,model_recall_by_target_class={k:dict(targets=totals[k],recall=hits[k]/max(totals[k],1)) for k in totals},
             overlap_with_model=missed,
             learned_content_form=dict(symmetric_share=float(((lambda a:torch.linalg.norm((a+a.T)/2)**2/torch.linalg.norm(a)**2)(model.query_content.T@model.key_content)))))
    write_json(Path('results/neural_audit')/f'lowrank_{args.version}_{args.dataset}.json',out)
    import json;print(json.dumps({k:v for k,v in out.items() if k!='checkpoint'},indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--queries',type=int,default=1248);p.add_argument('--version',default='v1',choices=['v1','v2'])
    main(p.parse_args())
