"""Personalized DIALS on Dunnhumby: DIALS plus a household repurchase term R, trained jointly. Frozen DIALS is unchanged.

    s(c | S, u) = w_C C + sum_l w_l L_l + w_G G + w_R R,
    R(u, c)     = rho_R(log(1 + n_uc), log(1 + B_u)) - rho_R(0, log(1 + B_u)),

n_uc: how many of household u's training baskets hold c; B_u: u's number of training baskets; rho_R a small MLP. The
subtracted constant is the same for every item of a query, so it changes no ranking and no softmax loss; it makes R
zero on items u never bought, so R is sparse. The gate gains two inputs (log(1 + B_u), share of seeds in u's history)
and one output (w_R).

Leakage. Histories use training baskets only. A training basket b of u is scored with n_uc - [c in b] and B_u - 1
(leave-one-out). Validation and test queries get their household through Context, set by evaluate() batch by batch;
a seed set does not identify its household uniquely (11 of 5,707 test seed sets have two owners).
"""
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dials.model import DIALS,curve
from dials.evaluate import query_metrics
from experiments.dials_user.households import rebuild,households


class History:
    """Training-basket histories per household, and per-query (items, counts, baskets) with optional leave-one-out."""
    def __init__(self,ds):
        self.split=rebuild(ds);counts,self.owner,self.index=households(ds.train,self.split.train)
        self.items=[torch.as_tensor(sorted(c),dtype=torch.long) for c in counts]
        self.counts=[torch.as_tensor([c[i] for i in sorted(c)],dtype=torch.float32) for c in counts]
        self.baskets=np.bincount([self.index[u] for u,_ in self.split.train],minlength=len(counts))
        self.days={}  # household -> list of (day, bundle) for the time-restricted check
        for (u,d),b in zip(self.split.train,ds.train):self.days.setdefault(self.index[u],[]).append((d,sorted(set(b))))

    def query(self,u):
        k=self.index.get(u)
        return (torch.zeros(0,dtype=torch.long),torch.zeros(0),0.) if k is None else (self.items[k],self.counts[k],float(self.baskets[k]))

    def before(self,u,day):
        """History from u's training baskets bought strictly before day."""
        k=self.index.get(u);c={};B=0
        for d,b in (self.days.get(k,[]) if k is not None else []):
            if d<day:
                B+=1
                for i in b:c[i]=c.get(i,0)+1
        keys=sorted(c);return torch.as_tensor(keys,dtype=torch.long),torch.as_tensor([c[i] for i in keys],dtype=torch.float32),float(B)

    def training(self,bundle):
        k=self.owner[tuple(bundle)];items,counts=self.items[k],self.counts[k]
        counts=counts-torch.isin(items,torch.as_tensor(bundle)).float();keep=counts>0
        return items[keep],counts[keep],float(self.baskets[k]-1)


class DIALSRepurchase(DIALS):
    def __init__(self,features,levels,train,spectral,history,rank=64,hidden=32):
        super().__init__(features,levels,train,spectral,rank=rank,hidden=hidden)
        self.households=history;self.context=None  # list of per-query (items, counts, baskets) for evaluation
        self.repurchase=curve(2)
        self.gate=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,3+self.levels))  # content, labels..., graph, repurchase
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,math.log(math.e-1))

    def fast_parameters(self):
        return super().fast_parameters()+list(self.repurchase.parameters())

    def queries(self,seeds,bundles):
        if bundles is not None:return [self.households.training(b.tolist()) for b in bundles]
        if self.context is None or len(self.context)!=len(seeds):raise RuntimeError('Set model.context to the batch households')
        return self.context

    def repurchase_scores(self,hist):
        q=len(hist);out=torch.zeros(q,self.n)
        rows=torch.cat([torch.full((len(i),),j) for j,(i,_,_) in enumerate(hist)]);cols=torch.cat([i for i,_,_ in hist])
        if len(cols)==0:return out
        n=torch.cat([c for _,c,_ in hist]);B=torch.tensor([b for _,_,b in hist])[rows].log1p()
        value=self.repurchase(torch.stack((n.log1p(),B),1)).squeeze(1)-self.repurchase(torch.stack((torch.zeros_like(B),B),1)).squeeze(1)
        return out.index_put((rows,cols),value)

    def gate_weights(self,seeds,bundles,hist):
        stats=[]
        for s,(items,_,b) in zip(seeds,hist):
            f=(self.frequency[s]-(1 if bundles is not None else 0)).clamp_min(0)
            stats.append(torch.stack((torch.tensor(float(len(s))).log(),f.log1p().mean(),(f==0).float().mean(),
                                      torch.tensor(b).log1p(),torch.isin(s,items).float().mean())))
        return F.softplus(self.gate(torch.stack(stats)))

    def forward(self,seeds,bundles=None,cache=None):
        if cache is not None and bundles is not None:raise ValueError('The cache is for evaluation only')
        seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds]
        if bundles is not None:bundles=[torch.as_tensor(sorted(set(b)),dtype=torch.long) for b in bundles]
        hist=self.queries(seeds,bundles)
        parts=[self.content_scores(seeds,bundles,cache)]+[self.label_scores(i,seeds,bundles) for i in range(self.levels)]
        parts+=[self.graph_scores(seeds,bundles,cache),self.repurchase_scores(hist)]
        w=self.gate_weights(seeds,bundles,hist)
        return sum(w[:,j,None]*v for j,v in enumerate(parts))


@torch.inference_mode()
def evaluate(model,pairs,hist,batch=32):
    """dials.evaluate.evaluate with the batch households handed to the model; hist: per-query (items, counts, baskets)."""
    model.eval();cache=model.cache();started=time.monotonic();records=[]
    for start in range(0,len(pairs),batch):
        rows=pairs[start:start+batch];model.context=hist[start:start+batch]
        values=model([s for s,_ in rows],cache=cache).numpy();model.context=None
        if not np.isfinite(values).all():raise RuntimeError('Nonfinite scores')
        records.extend(query_metrics(v,s,g) for v,(s,g) in zip(values,rows))
    return dict(recall20=float(np.mean([r['recall20'] for r in records])),ndcg20=float(np.mean([r['ndcg20'] for r in records])),
                queries=records,seconds=time.monotonic()-started)
