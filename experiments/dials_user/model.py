"""DIALS plus a user co-interaction term U, trained jointly. The frozen DIALS code is not modified.

    s(c | S) = w_C C + sum_l w_l L_l + w_G G + w_U U,
    U(S, c)  = theta_U log(1 + d_c^-beta_U sum_{s in S} M_sc d_s^-alpha_U),

with M = X^T W X the user co-interaction matrix, W = diag(1 / log(2 + |history|)), zero diagonal, and d its row sums.
X (users x items) is the release's ui_full.txt (POG), or on Dunnhumby the households' histories from training baskets
only. The gate gains two inputs (share of seeds with any user co-interaction, mean seed log(1 + d)) and one output (w_U).

Leave-one-out. External histories (POG) are not training bundles, so U needs none there. Dunnhumby's histories are
built from the training baskets, so a training basket b of household u is scored with u's history H replaced by
H' = items in another of u's training baskets: M changes by -w 1_H 1_H^T + w' 1_H' 1_H'^T off the diagonal
(w, w' the weights of |H|, |H'|), and d by -w(|H|-1) 1_H + w'(|H'|-1) 1_H'. The correction is exact.
"""
import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from dials.model import DIALS
from experiments.dials_user.audit import histories


def user_households(name,ds):
    """(item -> basket count) per user, and training-bundle key -> user for leave-one-out (None: histories are external)."""
    if name=='dunnhumby':
        from experiments.dials_user.households import training_households
        return training_households(ds)
    return [{i:1 for i in items} for items in histories(name).values()],None


def history_matrix(sets,n):
    """M = X^T W X with W = diag(1 / log(2 + |history|)), zero diagonal."""
    users=[sorted(i for i in items if i<n) for items in sets]
    rows=np.repeat(np.arange(len(users)),[len(v) for v in users]);cols=np.concatenate(users)
    X=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(users),n))
    M=(X.T@sp.diags(1/np.log(2+np.array([len(v) for v in users])))@X).tocsr();M.setdiag(0);M.eliminate_zeros()
    M.sort_indices();return M


def user_matrix(name,ds):
    return history_matrix([set(c) for c in user_households(name,ds)[0]],ds.n_items)


class DIALSUser(DIALS):
    def __init__(self,features,levels,train,spectral,users,rank=64,hidden=32,households=None):
        """users: M (n x n). households: (counts, owner) from user_households when leave-one-out is needed, else None."""
        super().__init__(features,levels,train,spectral,rank=rank,hidden=hidden)
        M=users.tocsr()
        if M.shape!=(self.n,self.n):raise ValueError('User matrix does not match the catalogue')
        degree=np.asarray(M.sum(1)).ravel()
        for k,v in dict(user_pointer=torch.as_tensor(M.indptr,dtype=torch.long),user_target=torch.as_tensor(M.indices,dtype=torch.long),
                        user_weight=torch.as_tensor(M.data,dtype=torch.float32),user_degree=torch.as_tensor(degree,dtype=torch.float32)).items():
            self.register_buffer(k,v,persistent=False)
        self.households=None
        if households is not None:
            counts,owner=households
            self.households=([(torch.as_tensor(sorted(c),dtype=torch.long),torch.as_tensor([c[i] for i in sorted(c)],dtype=torch.long)) for c in counts],owner)
        self.user_exponents=nn.Parameter(torch.zeros(2));self.user_theta=nn.Parameter(torch.tensor(1.))
        self.gate=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,3+self.levels))  # content, labels..., graph, users
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,math.log(math.e-1))

    def fast_parameters(self):
        return super().fast_parameters()+[self.user_exponents,self.user_theta]

    @staticmethod
    def user_log(d):return d.clamp_min(1e-3).log()

    def loo(self,bundles):
        """Per training bundle: (H, H', w, w') of its household, or None; and the degrees d' (queries x n)."""
        q=len(bundles);d=self.user_degree.expand(q,-1).clone();out=[]
        if self.households is None:return [None]*q,d
        sets,owner=self.households
        for i,b in enumerate(bundles):
            k=owner.get(tuple(b.tolist()))
            if k is None:out.append(None);continue
            H,C=sets[k];kept=H[(C-torch.isin(H,b).long())>0]
            w=1/math.log(2+len(H));wk=1/math.log(2+len(kept))
            d[i,H]-=w*(len(H)-1);d[i,kept]+=wk*max(len(kept)-1,0);out.append((H,kept,w,wk))
        return out,d.clamp_min(0)

    def user_scores(self,seeds,bundles=None,loo=None):
        alpha,beta=self.user_exponents;q=len(seeds)
        info,d=loo if loo is not None else ([None]*q,self.user_degree.expand(q,-1))
        rows=torch.cat(seeds);query=torch.cat([torch.full((len(s),),i) for i,s in enumerate(seeds)])
        start=self.user_pointer[rows];length=self.user_pointer[rows+1]-start;offsets=torch.cumsum(length,0)-length
        edge=torch.repeat_interleave(start-offsets,length)+torch.arange(int(length.sum()))
        eq=torch.repeat_interleave(query,length);source=torch.repeat_interleave(rows,length);target=self.user_target[edge]
        ld=self.user_log(d)
        w=self.user_weight[edge]*torch.exp(-alpha*ld[eq,source])
        total=torch.zeros(q*self.n).scatter_add(0,eq*self.n+target,w).view(q,self.n)
        fixes=[]
        for i,(s,x) in enumerate(zip(seeds,info)):
            if x is None:continue
            H,kept,wo,wk=x;fs=torch.exp(-alpha*ld[i,s]);inkept=torch.isin(s,kept).float()
            delta=torch.zeros(self.n);delta=delta.index_add(0,H,(-wo*fs.sum()).expand(len(H)))
            delta=delta.index_add(0,kept,(wk*(fs*inkept).sum()).expand(len(kept)))
            delta=delta.index_add(0,s,-fs*(-wo+wk*inkept))  # the diagonal of M stays zero
            fixes.append((i,delta))
        if fixes:
            idx=torch.tensor([i for i,_ in fixes]);total=total.index_put((idx,),total[idx]+torch.stack([v for _,v in fixes]))
        total=total.clamp_min(0)*torch.exp(-beta*ld)
        return self.user_theta*total.log1p()

    def gate_weights(self,seeds,bundles,loo=None):
        d=loo[1] if loo is not None else None;stats=[]
        for i,s in enumerate(seeds):
            f=(self.frequency[s]-(1 if bundles is not None else 0)).clamp_min(0);u=self.user_degree[s] if d is None else d[i,s]
            stats.append(torch.stack((torch.tensor(float(len(s))).log(),f.log1p().mean(),(f==0).float().mean(),(u>0).float().mean(),u.log1p().mean())))
        return F.softplus(self.gate(torch.stack(stats)))

    def forward(self,seeds,bundles=None,cache=None):
        if cache is not None and bundles is not None:raise ValueError('The cache is for evaluation only')
        seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds]
        if bundles is not None:bundles=[torch.as_tensor(sorted(set(b)),dtype=torch.long) for b in bundles]
        loo=self.loo(bundles) if bundles is not None else None
        parts=[self.content_scores(seeds,bundles,cache)]+[self.label_scores(i,seeds,bundles) for i in range(self.levels)]
        parts+=[self.graph_scores(seeds,bundles,cache),self.user_scores(seeds,bundles,loo)]
        w=self.gate_weights(seeds,bundles,loo)
        return sum(w[:,j,None]*v for j,v in enumerate(parts))
