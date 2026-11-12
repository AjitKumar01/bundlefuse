"""Baseline relation network with an optional learned bilinear form and category table.

With M=I and a zero category table the scores equal RelationNetwork exactly.
"""
import torch
from torch import nn
from experiments.neural_revision.model import RelationNetwork


class BilinearNetwork(RelationNetwork):
    def __init__(self,features,categories,train,category_mode='off',bilinear=False):
        super().__init__(features,categories,train,variant='graph_pair',category_mode=category_mode)
        self.use_bilinear=bilinear
        if bilinear:self.bilinear=nn.Parameter(torch.eye(self.features.shape[1]))

    def components(self,seeds,keep=None,item_vectors=None):
        if not self.use_bilinear:return super().components(seeds,keep,item_vectors)
        if item_vectors is not None and keep is not None:
            raise ValueError('Cached vectors cannot be used with history masking')
        indicator=torch.zeros(len(seeds),self.n)
        for q,row in enumerate(seeds):indicator[q,list(row)]=1
        size=indicator.sum(1,keepdim=True).clamp_min(1)
        x=self.item_vectors(keep) if item_vectors is None else item_vectors
        text=((indicator@x)/size)@self.bilinear@x.T*self.log_text_scale.exp().clamp(max=100)
        if self.category_mode=='off':
            category=torch.zeros_like(text)
        else:
            counts=torch.zeros(len(seeds),self.nc).scatter_add_(1,self.categories[None].expand(len(seeds),-1),indicator)
            weights=self.category_weights*(self.category_seen[:,None]&self.category_seen[None,:])
            category=(counts@weights)[:,self.categories]
        source,target=self.edges
        values=self.pair_weights if keep is None else self.pair_weights*keep[source]*keep[target]
        pair=torch.zeros_like(text).scatter_add(1,target[None,:].expand(len(seeds),-1),indicator[:,source]*values[None,:])
        return dict(text=text,category=category,pair=pair)
