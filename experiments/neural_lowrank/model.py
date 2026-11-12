"""Low-rank relation model scored with leave-one-out history.

score(c | S) = phi(S)^T psi(c) + sum_{s in S} omega(n_sc, f_s, f_c)

phi(S) = mean_s [A z_s + B h_s + U_q e_q[g(s)]]      query: content, history, group
psi(c) =         K z_c + D h_c + U_k e_k[g(c)]       candidate: same three views

z is encoded item content, h the count-weighted mean of an item's training partners' features,
g its group (category or artist). omega is one small network shared by every observed pair, so no
parameter is tied to a pair. During training each bundle's own counts are removed before it is
scored, so its pairs and partners look as they would at test time.
"""
import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


def pair_counts(train,n):
    """Directed co-occurrence counts sorted by source, item frequencies, and the source index pointer."""
    rows=[np.unique(np.asarray(b,np.int64)) for b in train]
    frequency=np.bincount(np.concatenate(rows),minlength=n)
    keys=[]
    for r in rows:
        if len(r)<2:continue
        s=np.repeat(r,len(r));c=np.tile(r,len(r));keep=s!=c
        keys.append(s[keep]*n+c[keep])
    keys,counts=np.unique(np.concatenate(keys),return_counts=True)
    source=keys//n;target=keys%n;del keys
    pointer=np.searchsorted(source,np.arange(n+1))
    return source,target,counts,frequency,pointer


class LowRankRelation(nn.Module):
    def __init__(self,features,groups,train,rank=64,group_rank=16,hidden=32):
        super().__init__()
        x=F.normalize(torch.as_tensor(features,dtype=torch.float32),dim=1)
        self.n,d=x.shape;self.rank=rank
        g=torch.as_tensor(groups,dtype=torch.long);self.ng=int(g.max())+1
        source,target,counts,frequency,pointer=pair_counts(train,self.n)
        matrix=sp.csr_matrix((counts.astype(np.float32),(source,target)),shape=(self.n,self.n))
        self.register_buffer('features',x);self.register_buffer('groups',g)
        self.register_buffer('pointer',torch.as_tensor(pointer,dtype=torch.long))
        self.register_buffer('target',torch.as_tensor(target,dtype=torch.long))
        self.register_buffer('counts',torch.as_tensor(counts,dtype=torch.float32))
        self.register_buffer('frequency',torch.as_tensor(frequency,dtype=torch.float32))
        # Partner feature sums and masses: h_i = history_sum_i / history_mass_i.
        self.register_buffer('history_sum',torch.as_tensor(matrix@x.numpy(),dtype=torch.float32))
        self.register_buffer('history_mass',torch.as_tensor(np.asarray(matrix.sum(1)).ravel(),dtype=torch.float32))
        seen=torch.zeros(self.ng,dtype=torch.bool);seen[g[torch.as_tensor(np.flatnonzero(frequency))]]=True
        # A group absent from training would only ever be pushed down as a negative.
        self.register_buffer('group_seen',seen)
        self.encoder=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,d))
        nn.init.zeros_(self.encoder[-1].weight);nn.init.zeros_(self.encoder[-1].bias)
        # Features come from an SVD, so the first coordinates are the leading directions:
        # at initialization the content term is 10 x cosine on the top `rank` components.
        start=math.sqrt(10.)*torch.eye(rank,d)
        self.query_content=nn.Parameter(start.clone());self.key_content=nn.Parameter(start.clone())
        self.query_history=nn.Parameter(torch.zeros(rank,d));self.key_history=nn.Parameter(torch.zeros(rank,d))
        self.query_group=nn.Parameter(.1*torch.randn(self.ng,group_rank));self.key_group=nn.Parameter(.1*torch.randn(self.ng,group_rank))
        self.query_group_map=nn.Parameter(torch.zeros(rank,group_rank));self.key_group_map=nn.Parameter(.1*torch.randn(rank,group_rank))
        self.pair=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,1))
        self.disabled=set()  # validation diagnostics only: 'content', 'history', 'group', 'pair'

    def content(self):
        return F.normalize(self.features+self.encoder(self.features),dim=1)

    def history(self,keep=None):
        mass=self.history_mass if keep is None else self.history_mass*keep
        return self.history_sum/mass.clamp_min(1)[:,None]*(mass>0)[:,None],mass

    def group_vectors(self,table,mapping):
        return (table*self.group_seen[:,None])[self.groups]@mapping.T

    def keys(self,keep=None,z=None):
        z=self.content() if z is None else z
        out=torch.zeros(self.n,self.rank)
        if 'content' not in self.disabled:out=out+z@self.key_content.T
        if 'history' not in self.disabled:out=out+self.history(keep)[0]@self.key_history.T
        if 'group' not in self.disabled:out=out+self.group_vectors(self.key_group,self.key_group_map)
        return out

    def loo_history(self,items,bundle,keep):
        """History of bundle members with this bundle's own pairs removed."""
        x=self.features[bundle];mass=self.history_mass[items]-(len(bundle)-1)
        if keep is not None:mass=mass*keep[items]
        value=(self.history_sum[items]-(x.sum(0)-self.features[items]))/mass.clamp_min(1)[:,None]
        return value*(mass>0)[:,None]

    def pair_scores(self,seeds,bundles,keep):
        q=len(seeds);ids=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds]
        source=torch.cat(ids);query=torch.cat([torch.full((len(s),),i) for i,s in enumerate(ids)])
        start=self.pointer[source];length=self.pointer[source+1]-start
        offsets=torch.cumsum(length,0)-length
        edge=torch.repeat_interleave(start-offsets,length)+torch.arange(int(length.sum()))
        query=torch.repeat_interleave(query,length);source=torch.repeat_interleave(source,length)
        target=self.target[edge];n=self.counts[edge]
        fs=self.frequency[source];fc=self.frequency[target]
        if bundles is not None:
            member=torch.zeros(q,self.n,dtype=torch.bool)
            for i,b in enumerate(bundles):member[i,b]=True
            inside=member[query,target].float()
            n=n-inside;fs=fs-1;fc=fc-inside
        gate=(n>0).float()
        if keep is not None:
            gate=gate*keep[source]*keep[target];fs=fs*keep[source];fc=fc*keep[target]
        w=F.softplus(self.pair(torch.stack((n.clamp_min(0).log1p(),fs.clamp_min(0).log1p(),fc.clamp_min(0).log1p()),1))).squeeze(1)*gate
        return torch.zeros(q*self.n).scatter_add(0,query*self.n+target,w).view(q,self.n)

    def forward(self,seeds,bundles=None,keep=None,keys=None):
        """bundles: the complete training rows, for leave-one-out; None at evaluation."""
        if keys is not None and (keep is not None or bundles is not None):raise ValueError('Cached keys are evaluation only')
        z=self.content()
        k=self.keys(keep,z) if keys is None else keys
        h,_=self.history(keep);gq=self.group_vectors(self.query_group,self.query_group_map)
        phi=[];fixes=[]
        for i,s in enumerate(seeds):
            s=torch.as_tensor(sorted(set(s)),dtype=torch.long)
            hs=h[s]
            if bundles is not None:
                b=torch.as_tensor(sorted(set(bundles[i])),dtype=torch.long);hs=self.loo_history(s,b,keep)
            v=torch.zeros(len(s),self.rank)
            if 'content' not in self.disabled:v=v+z[s]@self.query_content.T
            if 'history' not in self.disabled:v=v+hs@self.query_history.T
            if 'group' not in self.disabled:v=v+gq[s]
            phi.append(v.mean(0))
            if bundles is not None and 'history' not in self.disabled:
                fixes.append((i,b,(self.loo_history(b,b,keep)-h[b])@self.key_history.T))
        phi=torch.stack(phi);scores=phi@k.T
        if fixes:
            rows=torch.cat([torch.full((len(b),),i) for i,b,_ in fixes]);cols=torch.cat([b for _,b,_ in fixes])
            values=torch.cat([delta@phi[i] for i,_,delta in fixes])
            scores=scores.index_put((rows,cols),values,accumulate=True)
        if 'pair' not in self.disabled:scores=scores+self.pair_scores(seeds,bundles,keep)
        return scores

    def counts_summary(self):
        trainable=sum(p.numel() for p in self.parameters())
        return dict(trainable_parameters=trainable,group_parameters=self.query_group.numel()+self.key_group.numel(),
                    pair_function_parameters=sum(p.numel() for p in self.pair.parameters()),
                    observed_directed_pairs=int(self.counts.numel()),
                    frozen_buffer_bytes=sum(b.numel()*b.element_size() for b in self.buffers()))
