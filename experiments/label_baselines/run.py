"""Simple baselines that use the item labels DIALS uses (POG: category; Spotify: artist and album), tuned on
validation NDCG@20 like experiments/simple_baselines; test scored once with the chosen setting.

Labels g_l(c) come from dials.data (the same arrays DIALS reads). With t_l(S) the seeds' label counts:
  samelabel   sum_l a_l log(1 + m_l(S,c)) + gamma log(1 + f_c), m_l = number of seeds sharing c's label at level l.
  transition  sum_l a_l log(eps + t_l(S)/|S| T_l[:, g_l(c)]) + gamma log P(c | g(c)), where T_l is the row-normalized
              label-by-label co-occurrence of training bundles (pairs of distinct items, diagonal kept), and
              P(c | g) = (f_c + 1) / (F_g + |g|) the item's add-one share of its finest label's training frequency.
  label_knn   the best of the two (by validation), min-max scaled per query, blended with ItemKNN (max-abs scaled, at
              the setting simple_baselines chose): (1 - w) knn + w label.
  label_knn_content   the chosen label_knn plus v times the content cosine (released features), max-abs scaled.
f_c is the training frequency. Everything is built from training bundles only.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.label_baselines.run --dataset pog
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.neural_revision.run import log
from experiments.simple_baselines.methods import Baselines
from experiments.simple_baselines.run import load,evaluate,mean,features,POG

RESULTS=Path('results/label_baselines')


def minmax(s):
    lo=s.min(1,keepdims=True);return (s-lo)/np.maximum(s.max(1,keepdims=True)-lo,1e-12)


def maxabs(s):return s/np.maximum(np.abs(s).max(1,keepdims=True),1e-12)


class LabelBaselines(Baselines):
    def __init__(self,train,n,levels,features):
        super().__init__(train,n,features)
        self.levels=[np.asarray(g) for g in levels];self.onehot=[];self.transition=[]
        for g in self.levels:
            k=int(g.max())+1;G=sp.csr_matrix((np.ones(n,np.float32),(np.arange(n),g)),shape=(n,k))
            T=(G.T@self.C@G).tocsr().astype(np.float64);row=np.asarray(T.sum(1)).ravel()
            self.onehot.append(G);self.transition.append((sp.diags(1/np.maximum(row,1e-12))@T).tocsr().astype(np.float32))
        fine=self.levels[-1] if len(self.levels)==1 else max(self.levels,key=lambda g:g.max())
        F=np.bincount(fine,weights=self.f);size=np.bincount(fine)
        self.within=np.log((self.f+1)/(F[fine]+size[fine])).astype(np.float32)

    def label_grid(self):
        L=len(self.levels);weights=[(1.,)] if L==1 else [(1.,0.),(0.,1.),(1.,1.),(1.,.5),(.5,1.)]
        g=[('samelabel',dict(a=a,gamma=c)) for a in weights for c in (0.,.1,.3,1.)]
        g+=[('transition',dict(a=a,gamma=c,eps=e)) for a in weights for c in (0.,.5,1.,2.) for e in (1e-4,1e-2)]
        return g

    def label_scores(self,name,p,seeds):
        q=len(seeds);size=np.array([len(set(s)) for s in seeds],np.float32)[:,None]
        y=sp.csr_matrix((np.ones(int(size.sum()),np.float32),(np.repeat(np.arange(q),size.ravel().astype(int)),np.concatenate([sorted(set(s)) for s in seeds]))),shape=(q,self.n))
        out=np.zeros((q,self.n),np.float32)
        for a,G,T,g in zip(p['a'],self.onehot,self.transition,self.levels):
            if a==0:continue
            t=(y@G)  # q x labels
            if name=='samelabel':out+=a*np.log1p(np.asarray(t.todense())[:,g])
            else:out+=a*np.log(p['eps']+np.asarray((t@T).todense())[:,g]/size)
        prior=np.log1p(self.f).astype(np.float32) if name=='samelabel' else self.within
        return out+p['gamma']*prior[None,:]+self.tie

    def scores(self,name,p,seeds):
        if name in ('samelabel','transition'):return self.label_scores(name,p,seeds)
        if name in ('label_knn','label_knn_content'):
            label=minmax(self.label_scores(p['label'],p['label_params'],seeds))
            q=len(seeds);y=sp.csr_matrix((np.ones(sum(len(set(s)) for s in seeds),np.float32),(np.repeat(np.arange(q),[len(set(s)) for s in seeds]),np.concatenate([sorted(set(s)) for s in seeds]))),shape=(q,self.n))
            knn=maxabs(np.asarray((y@self.matrix('itemknn',p['itemknn'])).todense()))
            out=(1-p['w'])*knn+p['w']*label
            if name=='label_knn_content':out=out+p['v']*maxabs((y@self.features)@self.features.T)
            return out.astype(np.float32)+self.tie
        return super().scores(name,p,seeds)


def run(args):
    from dials import data
    out=RESULTS/args.dataset;out.mkdir(parents=True,exist_ok=True)
    ds,keep=load(args.dataset);other=data.load(args.dataset,Path(DATA_ROOT)/'dials'/'cache')
    if [sorted(set(b)) for b in other.train]!=[sorted(set(b)) for b in ds.train]:raise RuntimeError('Training split differs')
    model=LabelBaselines(ds.train,ds.n_items,other.levels,features(args.dataset,ds))
    itemknn=json.loads((Path('results/simple_baselines')/args.dataset/'summary.json').read_text())['test']['itemknn']['params']
    validation={};report=dict(dataset=args.dataset,selection='Best validation NDCG@20 per method; test scored once with that setting.',
                              levels=[int(g.max())+1 for g in model.levels],itemknn=itemknn,validation=validation,test={})
    def tune(name,grid):
        rows=[]
        for p in grid:
            started=time.monotonic();v=mean(evaluate(model,name,p,ds.val_pairs));rows.append(dict(params=p,**v,seconds=time.monotonic()-started))
            log(f'{args.dataset} {name} {p} VAL N20={v["ndcg20"]:.6f}')
        validation[name]=rows;best=max(rows,key=lambda r:r['ndcg20']);records=evaluate(model,name,best['params'],ds.test_pairs)
        report['test'][name]=dict(params=best['params'],validation_ndcg20=best['ndcg20'],full=mean(records),filtered=mean(records,keep),queries=[[r['recall20'],r['ndcg20']] for r in records])
        log(f'{args.dataset} {name} TEST filtered {report["test"][name]["filtered"]}');return best
    grid=model.label_grid()
    for name in ('samelabel','transition'):tune(name,[p for n,p in grid if n==name])
    label=max(('samelabel','transition'),key=lambda n:report['test'][n]['validation_ndcg20']);lp=report['test'][label]['params']
    best=tune('label_knn',[dict(label=label,label_params=lp,itemknn=itemknn,w=w) for w in (0.,.1,.2,.3,.5,.7,.9,1.)])
    tune('label_knn_content',[dict(best['params'],v=v) for v in (0.,.05,.1,.2,.3,.5)])
    write_json(out/'summary.json',report);log(f'{args.dataset} COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,choices=POG)
    run(p.parse_args())
