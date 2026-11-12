"""Independent learned relation network on a training-only sparse graph.

No BundleFuse import, scores, weights, PPMI, inverse-frequency rule, or teacher.
The sparse graph indexes where free trainable relationship weights may exist;
all relationship weights start at zero and learn from missing-item labels.
"""
from itertools import combinations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RelationNetwork(nn.Module):
    def __init__(self, features, categories, train, variant='pair', hidden=32, category_mode='free'):
        super().__init__()
        if variant not in ('attributes','pair','joint','graph_pair','graph_joint'):
            raise ValueError(variant)
        if category_mode not in ('free','off','nonnegative'):raise ValueError(category_mode)
        self.category_mode=category_mode
        self.variant=variant; self.hidden=hidden; self.n=len(features)
        x=F.normalize(torch.as_tensor(features,dtype=torch.float32),dim=1)
        cat=torch.as_tensor(categories,dtype=torch.long)
        self.register_buffer('features',x);self.register_buffer('categories',cat)
        self.nc=int(cat.max())+1
        seen=torch.zeros(self.nc,dtype=torch.bool)
        for bundle in train:seen[cat[list(bundle)]]=True
        self.register_buffer('category_seen',seen)
        # Preserve supplied text geometry at initialization, learn a nonlinear correction.
        self.text=nn.Sequential(nn.Linear(x.shape[1],hidden),nn.GELU(),nn.Linear(hidden,x.shape[1]))
        nn.init.zeros_(self.text[-1].weight);nn.init.zeros_(self.text[-1].bias)
        self.log_text_scale=nn.Parameter(torch.tensor(math.log(10.)))
        if category_mode=='off':
            self.register_buffer('category_weights',torch.empty(0))
        else:
            self.category_weights=nn.Parameter(torch.zeros(self.nc,self.nc))
        edges=set();triples=set()
        for bundle in train:
            items=sorted(set(bundle))
            for c in items:
                other=[s for s in items if s!=c]
                edges.update((s,c) for s in other)
                if variant in ('joint','graph_joint'):
                    triples.update((s,t,c) for s,t in combinations(other,2))
        edge_list=sorted(edges)
        edge=torch.tensor(edge_list,dtype=torch.long).T.contiguous()
        self.register_buffer('edges',edge)
        if variant!='attributes':
            self.pair_weights=nn.Parameter(torch.zeros(len(edge_list)))
        if variant in ('joint','graph_joint'):
            keys=sorted({(s,t) for s,t,c in triples})
            self.pair_keys={p:i for i,p in enumerate(keys)}
            group=sorted((self.pair_keys[(s,t)],c,s,t) for s,t,c in triples)
            self.register_buffer('joint_edges',torch.tensor(group,dtype=torch.long).T.contiguous()
                                 if group else torch.empty(4,0,dtype=torch.long))
            self.joint_weights=nn.Parameter(torch.zeros(len(group)))
        if variant in ('graph_pair','graph_joint'):
            self.graph_projection=nn.Linear(x.shape[1],x.shape[1],bias=False)
            nn.init.zeros_(self.graph_projection.weight)

    def item_vectors(self, keep=None):
        x=self.features+self.text(self.features)
        if self.variant in ('graph_pair','graph_joint'):
            source,target=self.edges
            values=torch.ones(len(source),device=x.device)
            if keep is not None:
                values=values*keep[source]*keep[target]
            degree=torch.zeros(self.n,device=x.device).scatter_add(0,target,values).clamp_min(1)
            adjacency=torch.sparse_coo_tensor(torch.stack((target,source)),values,(self.n,self.n))
            neighbours=torch.sparse.mm(adjacency,x)/degree[:,None]
            x=x+self.graph_projection(neighbours)
        return F.normalize(x,dim=1)

    def components(self, seeds, keep=None, item_vectors=None):
        """Return full-catalogue component scores; keep hides all history of selected IDs.

        Attribute/category inputs remain available when history is hidden, just as for a new item.
        No frequency or seen/unseen indicator is supplied to the neural attribute scorer.
        """
        device=self.features.device
        indicator=torch.zeros(len(seeds),self.n,device=device)
        for q,row in enumerate(seeds):
            indicator[q,list(row)]=1
        size=indicator.sum(1,keepdim=True).clamp_min(1)
        if item_vectors is not None and keep is not None:
            raise ValueError('Cached vectors cannot be used with history masking')
        x=self.item_vectors(keep) if item_vectors is None else item_vectors
        text=((indicator@x)/size)@x.T * self.log_text_scale.exp().clamp(max=100)
        cat_counts=torch.zeros(len(seeds),self.nc,device=device)
        cat_counts.scatter_add_(1,self.categories[None].expand(len(seeds),-1),indicator)
        # An unobserved category must not learn exclusively from negative candidates.
        if self.category_mode=='off':
            category=torch.zeros_like(text)
        else:
            category_weights=self.category_weights*(self.category_seen[:,None]&self.category_seen[None,:])
            category=(cat_counts@category_weights)[:,self.categories]
        out={'text':text,'category':category}
        if self.variant!='attributes':
            source,target=self.edges
            values=self.pair_weights
            if keep is not None:
                values=values*keep[source]*keep[target]
            # Avoid an item-by-item dense gradient in sparse-matrix value backward.
            # Memory is O(batch * observed_edges), never O(catalogue**2).
            contributions=indicator[:,source]*values[None,:]
            out['pair']=torch.zeros_like(text).scatter_add(1,target[None,:].expand(len(seeds),-1),contributions)
        if self.variant in ('joint','graph_joint'):
            qpair=torch.zeros(len(seeds),len(self.pair_keys),device=device)
            for q,row in enumerate(seeds):
                ids=[self.pair_keys[p] for p in combinations(sorted(set(row)),2) if p in self.pair_keys]
                if ids:qpair[q,ids]=1
            key,target,s,t=self.joint_edges
            values=self.joint_weights
            if keep is not None:
                values=values*keep[s]*keep[t]*keep[target]
            w=torch.sparse_coo_tensor(torch.stack((target,key)),values,(self.n,len(self.pair_keys)))
            out['joint']=torch.sparse.mm(w,qpair.T).T
        return out

    def forward(self,seeds,keep=None,item_vectors=None):
        return sum(self.components(seeds,keep,item_vectors).values())

    def counts(self):
        return dict(trainable_parameters=sum(p.numel() for p in self.parameters()),
                    pair_parameters=self.pair_weights.numel() if hasattr(self,'pair_weights') else 0,
                    joint_parameters=self.joint_weights.numel() if hasattr(self,'joint_weights') else 0,
                    category_parameters=self.category_weights.numel(),
                    frozen_buffer_bytes=sum(b.numel()*b.element_size() for b in self.buffers()))


def full_catalogue_loss(scores,seeds,truth):
    """Each hidden positive competes against every item outside its complete bundle.

    All observed items and all OTHER known positives are excluded from its denominator.
    No negative sampling, hard mining, teacher, or test labels.
    """
    allowed=torch.ones_like(scores,dtype=torch.bool)
    positive=torch.zeros_like(scores,dtype=torch.bool)
    for q,(s,g) in enumerate(zip(seeds,truth)):
        allowed[q,list(set(s)|set(g))]=False
        positive[q,list(g)]=True
    if not bool(allowed.any(1).all()):raise ValueError('No negatives')
    logneg=scores.masked_fill(~allowed,-torch.inf).logsumexp(1,keepdim=True)
    return ((F.softplus(logneg-scores)*positive).sum(1)/positive.sum(1)).mean()
