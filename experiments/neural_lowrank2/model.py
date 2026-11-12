"""Low-rank relation model plus group-match and popularity terms.

score(c | S) = v1 score + sum_l gamma_l(m_l(S,c), |S|) [m_l > 0] + rho(f_c)

m_l(S,c) is the number of seeds sharing c's label at group level l (category; artist and album).
It is an identity relation over labels, which the rank-16 group embeddings cannot express, and it needs
no purchase history. rho is a learned curve over the candidate's training frequency, left out like the rest.
"""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.neural_lowrank.model import LowRankRelation


def curve(inputs):
    return nn.Sequential(nn.Linear(inputs,16),nn.GELU(),nn.Linear(16,1))


class GroupMatchRelation(LowRankRelation):
    def __init__(self,features,levels,train,group_match=True,popularity=True,**kwargs):
        super().__init__(features,levels[0],train,**kwargs)
        self.use_group_match=group_match;self.use_popularity=popularity
        for i,labels in enumerate(levels):self.register_buffer(f'level{i}',torch.as_tensor(labels,dtype=torch.long))
        self.levels=len(levels)
        self.match=nn.ModuleList(curve(2) for _ in levels)
        self.prior=curve(1)

    def fast_parameters(self):
        return [self.query_group,self.key_group,*self.pair.parameters(),*self.match.parameters(),*self.prior.parameters()]

    def group_match(self,seeds):
        q=len(seeds);out=torch.zeros(q,self.n)
        size=torch.tensor([float(len(set(s))) for s in seeds]).log()
        for i in range(self.levels):
            labels=getattr(self,f'level{i}');m=torch.zeros(q,self.n)
            for j,s in enumerate(seeds):
                tally=torch.bincount(labels[sorted(set(s))],minlength=int(labels.max())+1).float()
                m[j]=tally[labels]
            row,col=m.nonzero(as_tuple=True)
            if len(row):
                value=self.match[i](torch.stack((m[row,col].log1p(),size[row]),1)).squeeze(1)
                out=out.index_put((row,col),value,accumulate=True)
        return out

    def popularity(self,q,bundles,keep):
        f=self.frequency if keep is None else self.frequency*keep
        scores=self.prior(f.log1p()[:,None]).squeeze(1).expand(q,-1)
        if bundles is None:return scores
        rows=torch.cat([torch.full((len(b),),i) for i,b in enumerate(bundles)]);cols=torch.cat([torch.as_tensor(sorted(set(b))) for b in bundles])
        left=(self.frequency[cols]-1).clamp_min(0)
        if keep is not None:left=left*keep[cols]
        delta=self.prior(left.log1p()[:,None]).squeeze(1)-scores[rows,cols]
        return scores.index_put((rows,cols),delta,accumulate=True)

    def forward(self,seeds,bundles=None,keep=None,keys=None):
        scores=super().forward(seeds,bundles,keep,keys)
        if self.use_group_match and 'match' not in self.disabled:scores=scores+self.group_match(seeds)
        if self.use_popularity and 'popularity' not in self.disabled:scores=scores+self.popularity(len(seeds),bundles,keep)
        return scores
