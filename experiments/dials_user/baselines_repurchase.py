"""Personalized baselines on Dunnhumby: every method gets the same household repurchase signal.

    score(c) = base(c) / max_c |base(c)| + lam * r(u, c),   r = log(1 + n_uc)  or  n_uc / B_u,

base: a simple baseline at its validation-chosen setting (results/simple_baselines/dunnhumby), or a frozen DIALS seed
(the same fusion at inference, no retraining). Repurchase only: r(u, c) + 1e-3 * popularity, ties by popularity.
(form, lam) chosen on validation NDCG@20; test scored once, with full training histories and with histories restricted
to baskets bought before the query basket. Metrics: dials.evaluate.query_metrics (seeds excluded), so every row here,
lam = 0 included, is scored by the same code.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_user.baselines_repurchase
"""
import dials  # pins BLAS threads before numpy loads
import json
import numpy as np
import torch
from pathlib import Path
from dials import data as dials_data
from dials.train import load as load_dials
from dials.evaluate import query_metrics
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS,CACHE
from experiments.simple_baselines.methods import Baselines
from experiments.dials_user.repurchase import History

OUT=Path('results/dials_user/repurchase_baselines');LAMBDAS=(0.,.05,.1,.2,.3,.5,1.,2.,5.);FORMS=('log','rate')
METHODS=('hybrid','ease','sgfcf','rp3beta','itemknn')


def signal(hist,n,form):
    """Dense queries x items r(u, c)."""
    out=np.zeros((len(hist),n),np.float32)
    for j,(items,counts,B) in enumerate(hist):
        if len(items):out[j,items.numpy()]=np.log1p(counts.numpy()) if form=='log' else counts.numpy()/max(B,1.)
    return out


def grid(base_batches,pairs,hist,n,popularity=None):
    """{(form, lam): per-query [R, N]} for one base scorer given as an iterator of score batches."""
    out={(f,l):[] for f in FORMS for l in LAMBDAS}
    for start,base in base_batches:
        rows=pairs[start:start+len(base)];h=hist[start:start+len(base)]
        base=base/np.maximum(np.abs(base).max(1,keepdims=True),1e-12)
        for f in FORMS:
            r=signal(h,n,f)
            for l in LAMBDAS:
                if f=='rate' and l==0:out[(f,l)]=out[('log',0.)];continue
                s=(r+1e-3*popularity[None]) if popularity is not None and l==1. else base+l*r
                for v,(q,g) in zip(s,rows):
                    m=query_metrics(v,q,g);out[(f,l)].append([m['recall20'],m['ndcg20']])
    return {k:np.asarray(v) for k,v in out.items() if len(v)}


def baseline_batches(model,name,p,pairs,batch=256):
    for start in range(0,len(pairs),batch):
        yield start,np.asarray(model.scores(name,p,[s for s,_ in pairs[start:start+batch]]),dtype=np.float64)


@torch.inference_mode()
def dials_batches(m,pairs,batch=64):
    cache=m.cache()
    for start in range(0,len(pairs),batch):
        yield start,m([torch.as_tensor(sorted(set(s)),dtype=torch.long) for s,_ in pairs[start:start+batch]],cache=cache).numpy().astype(np.float64)


def choose(val):
    return max(val,key=lambda k:val[k][:,1].mean())


def main():
    torch.set_num_threads(4);ds=dials_data.load('dunnhumby',CACHE);h=History(ds);n=ds.n_items
    hv=[h.query(u) for u,_ in h.split.val];ht=[h.query(u) for u,_ in h.split.test];hb=[h.before(u,d) for u,d in h.split.test]
    params=json.loads(Path('results/simple_baselines/dunnhumby/summary.json').read_text())['test']
    model=Baselines(ds.train,n,ds.features);popularity=np.bincount(np.concatenate([np.unique(b) for b in ds.train]),minlength=n).astype(np.float64)
    popularity/=popularity.max();report={};OUT.mkdir(parents=True,exist_ok=True)
    def record(name,val,test,before,choice):
        report[name]=dict(choice=list(choice),validation_ndcg20=float(val[choice][:,1].mean()),
                          test=test[choice].mean(0).tolist(),test_without_repurchase=test[('log',0.)].mean(0).tolist(),
                          test_before=before[choice].mean(0).tolist(),queries=test[choice].tolist(),queries_before=before[choice].tolist(),
                          queries_without_repurchase=test[('log',0.)].tolist())
        print(f"{name:22s} chosen {choice}  test R {test[choice][:,0].mean():.4f} N {test[choice][:,1].mean():.4f}"
              f"  (without R: N {test[('log',0.)][:,1].mean():.4f})  time-restricted N {before[choice][:,1].mean():.4f}",flush=True)
        write_json(OUT/'summary.json',dict(note=__doc__,methods=report))
    # repurchase only: rows (form, 1.0) of a grid with a zero base
    zero=lambda pairs:((s,np.zeros((min(256,len(pairs)-s),n))) for s in range(0,len(pairs),256))
    val=grid(zero(ds.val_pairs),ds.val_pairs,hv,n,popularity);val={k:v for k,v in val.items() if k[1]==1.}
    choice=choose(val);test=grid(zero(ds.test_pairs),ds.test_pairs,ht,n,popularity);before=grid(zero(ds.test_pairs),ds.test_pairs,hb,n,popularity)
    record('repurchase only',val,test,before,choice)
    for name in METHODS:
        p=params[name]['params']
        val=grid(baseline_batches(model,name,p,ds.val_pairs),ds.val_pairs,hv,n);choice=choose(val)
        test=grid(baseline_batches(model,name,p,ds.test_pairs),ds.test_pairs,ht,n);before=grid(baseline_batches(model,name,p,ds.test_pairs),ds.test_pairs,hb,n)
        record(f'{name} + R',val,test,before,choice)
    seeds=[load_dials(ds,RUNS/'dunnhumby'/f'seed{s}',CACHE).eval() for s in (0,1,2)]
    val=[grid(dials_batches(m,ds.val_pairs),ds.val_pairs,hv,n) for m in seeds];val={k:np.mean([v[k] for v in val],0) for k in val[0]};choice=choose(val)
    test=[grid(dials_batches(m,ds.test_pairs),ds.test_pairs,ht,n) for m in seeds];test={k:np.mean([v[k] for v in test],0) for k in test[0]}
    before=[grid(dials_batches(m,ds.test_pairs),ds.test_pairs,hb,n) for m in seeds];before={k:np.mean([v[k] for v in before],0) for k in before[0]}
    record('DIALS (frozen) + R at inference',val,test,before,choice)


if __name__=='__main__':main()
