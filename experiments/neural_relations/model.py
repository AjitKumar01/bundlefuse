"""Relation-family model: the model learns which relations matter instead of being handed them.

score(c | S) = w_content(S) C(S,c) + sum_l w_l(S) L_l(S,c) + w_graph(S) G(S,c)

Content   C = phi(S)^T psi(c) over encoded content z and partner history h (rank 64, asymmetric).
Labels    L_l(S,c) = delta_l(m_l(S,c), |S|, size(label c)) [m_l > 0] + mean_s P[g_l(s)] M_l Q[g_l(c)]^T
          for every label level l (category; artist, album). Diagonal part: is the label shared?
          Low-rank part: which labels go with which. Nothing says which part a dataset needs.
          label_mode='free': P and Q are free embeddings (v3). They memorize rare labels' bundles.
          label_mode='spectral': P and Q come from a cross-fitted SVD of the label-by-label co-purchase
          matrix G^T N G, and only the 16 x 16 matrix M_l is learned (v3.1).
Graph     G = theta_1 log(1 + y A) + theta_2 log(1 + y A A') + rho(f_c), with
          A[s,c] = n_sc deg_s^-alpha deg_c^-beta and A'[p,c] = n_pc deg_p^-gamma deg_c^-beta.
          alpha = beta = gamma = 0 gives raw counts; larger values discount popular items.
          The first hop is exact. The second uses a rank-64 SVD of D^-1/2 N D^-1/2, rescaled back to N,
          with the learned exponents applied exactly around it. The SVD is cross-fitted: training bundles
          are split into folds (duplicates together), and a bundle being trained on uses the SVD built
          without its fold, because an SVD built with it would encode who its items were bought with.
Gate      w(S) = softplus(MLP(log|S|, mean seed log-frequency, share of seeds never sold)).

Training scores each bundle with its own counts removed (leave-one-out). Edge counts, frequencies, partner
histories and first-hop degrees are exact; the second hop is cross-fitted (see Graph above) and its
degree normalizers use full training degrees.
"""
import hashlib
import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from experiments.neural_lowrank.model import pair_counts


def spectral_factors(matrix,rank):
    """N ~ L R^T from a truncated SVD of D^-1/2 N D^-1/2 (better conditioned than N itself)."""
    from sklearn.utils.extmath import randomized_svd
    scale=np.sqrt(np.maximum(np.asarray(matrix.sum(1)).ravel(),1)).astype(np.float32)
    normalized=sp.diags(1/scale)@matrix@sp.diags(1/scale)
    u,sigma,vt=randomized_svd(normalized,rank,n_iter=4,random_state=0)
    return ((u*sigma)*scale[:,None]).astype(np.float32),(vt.T*scale[:,None]).astype(np.float32)


def fold_of(bundle,folds):
    """Deterministic fold of a bundle; identical bundles always share a fold."""
    return int(hashlib.md5(np.asarray(sorted(set(bundle)),dtype='<i8').tobytes()).hexdigest(),16)%folds


def label_factors(matrix,labels,rank):
    """P, Q from a truncated SVD of the normalized label-by-label co-purchase matrix G^T N G."""
    from sklearn.utils.extmath import randomized_svd
    g=sp.csr_matrix((np.ones(len(labels),np.float32),(np.arange(len(labels)),labels)),shape=(len(labels),int(labels.max())+1))
    c=(g.T@matrix@g).tocsr();scale=np.sqrt(np.maximum(np.asarray(c.sum(1)).ravel(),1)).astype(np.float32)
    rank=min(rank,c.shape[0]-1)
    u,sigma,vt=randomized_svd(sp.diags(1/scale)@c@sp.diags(1/scale),rank,n_iter=4,random_state=0)
    root=np.sqrt(sigma)
    return (u*root).astype(np.float32),(vt.T*root).astype(np.float32)


def cross_fitted(train,n,rank,levels=(),label_rank=16,folds=5):
    """Graph and label SVD factors on all training rows (evaluation), plus one set per fold built without it."""
    def factors(rows):
        source,target,counts,_,_=pair_counts(rows,n)
        matrix=sp.csr_matrix((counts.astype(np.float32),(source,target)),shape=(n,n))
        return spectral_factors(matrix,rank),[label_factors(matrix,np.asarray(l),label_rank) for l in levels]
    assignment=[fold_of(b,folds) for b in train]
    full=factors(train);parts=[factors([b for b,f in zip(train,assignment) if f!=k]) for k in range(folds)]
    return dict(graph=(full[0],[p[0] for p in parts]),labels=[(full[1][i],[p[1][i] for p in parts]) for i in range(len(levels))])


def curve(inputs,hidden=16):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.GELU(),nn.Linear(hidden,1))


class RelationFamily(nn.Module):
    def __init__(self,features,levels,train,rank=64,group_rank=16,hidden=32,graph_rank=64,spectral=None,label_mode='free'):
        super().__init__()
        x=F.normalize(torch.as_tensor(features,dtype=torch.float32),dim=1)
        self.n,d=x.shape;self.rank=rank;self.levels=len(levels)
        source,target,counts,frequency,pointer=pair_counts(train,self.n)
        matrix=sp.csr_matrix((counts.astype(np.float32),(source,target)),shape=(self.n,self.n))
        # Data derived from the training split is rebuilt on construction, so checkpoints hold parameters only.
        data=dict(features=x,pointer=torch.as_tensor(pointer,dtype=torch.long),target=torch.as_tensor(target,dtype=torch.long),
                  counts=torch.as_tensor(counts,dtype=torch.float32),frequency=torch.as_tensor(frequency,dtype=torch.float32),
                  history_sum=torch.as_tensor(matrix@x.numpy(),dtype=torch.float32),
                  history_mass=torch.as_tensor(np.asarray(matrix.sum(1)).ravel(),dtype=torch.float32))
        for k,v in data.items():self.register_buffer(k,v,persistent=False)
        self.register_buffer('log_degree',self.history_mass.clamp_min(1).log(),persistent=False)
        if spectral is None:spectral=cross_fitted(train,self.n,graph_rank,levels,group_rank)
        (left,right),folds=spectral['graph'];self.label_mode=label_mode
        self.register_buffer('spectral_left',torch.as_tensor(left),persistent=False)
        self.register_buffer('spectral_right',torch.as_tensor(right),persistent=False)
        self.register_buffer('fold_left',torch.as_tensor(np.stack([f[0] for f in folds])),persistent=False)
        self.register_buffer('fold_right',torch.as_tensor(np.stack([f[1] for f in folds])),persistent=False)
        sold=torch.as_tensor(np.flatnonzero(frequency))
        self.label_query=nn.ParameterList();self.label_key=nn.ParameterList();self.diagonal=nn.ModuleList()
        for i,labels in enumerate(levels):
            g=torch.as_tensor(labels,dtype=torch.long);size=int(g.max())+1
            seen=torch.zeros(size,dtype=torch.bool);seen[g[sold]]=True
            self.register_buffer(f'label{i}',g,persistent=False);self.register_buffer(f'label_seen{i}',seen,persistent=False)
            self.register_buffer(f'label_size{i}',torch.bincount(g,minlength=size).float().log1p(),persistent=False)
            if label_mode=='free':
                self.label_query.append(nn.Parameter(.1*torch.randn(size,group_rank)))
                self.label_key.append(nn.Parameter(torch.zeros(size,group_rank)))
            else:
                (lf,rf),lfolds=spectral['labels'][i]
                self.register_buffer(f'label_left{i}',torch.as_tensor(lf),persistent=False);self.register_buffer(f'label_right{i}',torch.as_tensor(rf),persistent=False)
                self.register_buffer(f'label_fold_left{i}',torch.as_tensor(np.stack([f[0] for f in lfolds])),persistent=False)
                self.register_buffer(f'label_fold_right{i}',torch.as_tensor(np.stack([f[1] for f in lfolds])),persistent=False)
                self.label_query.append(nn.Parameter(torch.zeros(lf.shape[1],lf.shape[1])))  # M_l
            self.diagonal.append(curve(3))
        self.encoder=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,d))
        nn.init.zeros_(self.encoder[-1].weight);nn.init.zeros_(self.encoder[-1].bias)
        start=math.sqrt(10.)*torch.eye(rank,d)
        self.query_content=nn.Parameter(start.clone());self.key_content=nn.Parameter(start.clone())
        self.query_history=nn.Parameter(torch.zeros(rank,d));self.key_history=nn.Parameter(torch.zeros(rank,d))
        self.exponents=nn.Parameter(torch.zeros(3))  # alpha, beta, gamma
        self.hops=nn.Parameter(torch.tensor([1.,0.]))
        self.prior=curve(1)
        self.families=2+self.levels  # content, labels..., graph
        self.gate=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,self.families))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,math.log(math.e-1))  # softplus = 1
        self.disabled=set()  # diagnostics: 'content', 'label0', 'label1', 'graph', 'hop2', 'prior', 'diagonal', 'lowrank', 'gate'

    def fast_parameters(self):
        return [*self.label_query,*self.label_key,*self.diagonal.parameters(),self.exponents,self.hops,*self.prior.parameters(),*self.gate.parameters()]

    # ---- content -------------------------------------------------------------------------------------
    def content(self):
        return F.normalize(self.features+self.encoder(self.features),dim=1)

    def history(self,keep=None):
        mass=self.history_mass if keep is None else self.history_mass*keep
        return self.history_sum/mass.clamp_min(1)[:,None]*(mass>0)[:,None]

    def loo_history(self,items,bundle,keep):
        mass=self.history_mass[items]-(len(bundle)-1)
        if keep is not None:mass=mass*keep[items]
        value=(self.history_sum[items]-(self.features[bundle].sum(0)-self.features[items]))/mass.clamp_min(1)[:,None]
        return value*(mass>0)[:,None]

    def keys(self,keep=None,z=None):
        z=self.content() if z is None else z
        return z@self.key_content.T+self.history(keep)@self.key_history.T

    def content_scores(self,seeds,bundles,keep,keys):
        z=self.content();k=self.keys(keep,z) if keys is None else keys;h=self.history(keep);phi=[];fixes=[]
        for i,s in enumerate(seeds):
            hs=h[s]
            if bundles is not None:
                b=bundles[i];hs=self.loo_history(s,b,keep)
                fixes.append((i,b,(self.loo_history(b,b,keep)-h[b])@self.key_history.T))
            phi.append((z[s]@self.query_content.T+hs@self.query_history.T).mean(0))
        phi=torch.stack(phi);scores=phi@k.T
        if fixes:
            rows=torch.cat([torch.full((len(b),),i) for i,b,_ in fixes]);cols=torch.cat([b for _,b,_ in fixes])
            scores=scores.index_put((rows,cols),torch.cat([delta@phi[i] for i,_,delta in fixes]),accumulate=True)
        return scores

    # ---- labels --------------------------------------------------------------------------------------
    def label_scores(self,i,seeds,bundles=None):
        labels=getattr(self,f'label{i}');seen=getattr(self,f'label_seen{i}')[:,None]
        q=len(seeds);size=int(labels.max())+1;out=torch.zeros(q,self.n)
        tally=torch.zeros(q,size)
        for j,s in enumerate(seeds):tally[j]=torch.bincount(labels[s],minlength=size).float()
        if 'lowrank' not in self.disabled and self.label_mode=='free':
            eq=self.label_query[i]*seen;ek=self.label_key[i]*seen
            out=out+((tally@eq)/tally.sum(1,keepdim=True))@ek[labels].T
        elif 'lowrank' not in self.disabled:
            mean=tally/tally.sum(1,keepdim=True);mix=self.label_query[i]
            if bundles is None:
                out=out+(mean@getattr(self,f'label_left{i}')@mix)@getattr(self,f'label_right{i}')[labels].T
            else:
                fold=torch.tensor([fold_of(b.tolist(),len(self.fold_left)) for b in bundles])
                for k in fold.unique().tolist():
                    rows=(fold==k).nonzero(as_tuple=True)[0]
                    part=(mean[rows]@getattr(self,f'label_fold_left{i}')[k]@mix)@getattr(self,f'label_fold_right{i}')[k][labels].T
                    out=out.index_put((rows,),out[rows]+part)
        if 'diagonal' not in self.disabled:
            m=tally[:,labels];row,col=m.nonzero(as_tuple=True)
            if len(row):
                count=torch.tensor([float(len(s)) for s in seeds]).log()
                x=torch.stack((m[row,col].log1p(),count[row],getattr(self,f'label_size{i}')[labels[col]]),1)
                out=out.index_put((row,col),self.diagonal[i](x).squeeze(1),accumulate=True)
        return out

    # ---- graph ---------------------------------------------------------------------------------------
    def graph_scores(self,seeds,bundles,keep):
        alpha,beta,gamma=self.exponents;q=len(seeds)
        query,source,edge=self.expand(torch.cat([torch.full((len(s),),i) for i,s in enumerate(seeds)]),torch.cat(seeds))
        target=self.target[edge];n=self.counts[edge]
        ds=self.log_degree[source];dt=self.log_degree[target]
        if bundles is not None:
            member=torch.zeros(q,self.n,dtype=torch.bool)
            for i,b in enumerate(bundles):member[i,b]=True
            inside=member[query,target].float();n=n-inside
            others=torch.tensor([float(len(b)-1) for b in bundles])[query]
            ds=(self.history_mass[source]-others).clamp_min(1).log();dt=(self.history_mass[target]-inside*others).clamp_min(1).log()
        if keep is not None:n=n*keep[source]*keep[target]
        w=n*torch.exp(-alpha*ds-beta*dt)
        first=torch.zeros(q*self.n).scatter_add(0,query*self.n+target,w).view(q,self.n)
        out=self.hops[0]*first.log1p()
        if 'hop2' not in self.disabled:
            # A' = D^-gamma N D^-beta (and the history mask is diagonal too): only dense vectors carry gradients
            left=torch.exp(-gamma*self.log_degree);right=torch.exp(-beta*self.log_degree)
            if keep is not None:left=left*keep;right=right*keep
            x=first*left
            if bundles is None:second=(x@self.spectral_left)@self.spectral_right.T*right
            else:
                fold=torch.tensor([fold_of(b.tolist(),len(self.fold_left)) for b in bundles])
                second=torch.zeros(q,self.n)
                for k in fold.unique().tolist():
                    rows=(fold==k).nonzero(as_tuple=True)[0]
                    second=second.index_put((rows,),(x[rows]@self.fold_left[k])@self.fold_right[k].T*right)
            out=out+self.hops[1]*second.clamp_min(0).log1p()
        if 'prior' not in self.disabled:
            f=self.frequency if keep is None else self.frequency*keep
            prior=self.prior(f.log1p()[:,None]).squeeze(1).expand(q,-1)
            if bundles is not None:
                rows=torch.cat([torch.full((len(b),),i) for i,b in enumerate(bundles)]);cols=torch.cat(bundles)
                left=(self.frequency[cols]-1).clamp_min(0)*(1 if keep is None else keep[cols])
                prior=prior.index_put((rows,cols),self.prior(left.log1p()[:,None]).squeeze(1)-prior[rows,cols],accumulate=True)
            out=out+prior
        return out

    def expand(self,query,rows):
        """Every observed pair (row, partner) for the given rows, tagged with its query."""
        start=self.pointer[rows];length=self.pointer[rows+1]-start;offsets=torch.cumsum(length,0)-length
        edge=torch.repeat_interleave(start-offsets,length)+torch.arange(int(length.sum()))
        return torch.repeat_interleave(query,length),torch.repeat_interleave(rows,length),edge

    # ---- gate and total ------------------------------------------------------------------------------
    def gate_weights(self,seeds,bundles,keep):
        stats=[]
        for i,s in enumerate(seeds):
            f=self.frequency[s]-(1 if bundles is not None else 0)
            if keep is not None:f=f*keep[s]
            f=f.clamp_min(0);stats.append(torch.stack((torch.tensor(float(len(s))).log(),f.log1p().mean(),(f==0).float().mean())))
        w=F.softplus(self.gate(torch.stack(stats)))
        return torch.ones_like(w) if 'gate' in self.disabled else w

    def components(self,seeds,bundles=None,keep=None,keys=None):
        if keys is not None and (keep is not None or bundles is not None):raise ValueError('Cached keys are evaluation only')
        seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds]
        if bundles is not None:bundles=[torch.as_tensor(sorted(set(b)),dtype=torch.long) for b in bundles]
        out={}
        if 'content' not in self.disabled:out['content']=self.content_scores(seeds,bundles,keep,keys)
        for i in range(self.levels):
            if f'label{i}' not in self.disabled:out[f'label{i}']=self.label_scores(i,seeds,bundles)
        if 'graph' not in self.disabled:out['graph']=self.graph_scores(seeds,bundles,keep)
        return out,self.gate_weights(seeds,bundles,keep)

    def forward(self,seeds,bundles=None,keep=None,keys=None):
        parts,w=self.components(seeds,bundles,keep,keys)
        order=['content']+[f'label{i}' for i in range(self.levels)]+['graph']
        return sum(w[:,order.index(k),None]*v for k,v in parts.items())

    def counts_summary(self):
        return dict(trainable_parameters=sum(p.numel() for p in self.parameters()),
                    label_parameters=sum(p.numel() for p in [*self.label_query,*self.label_key]),label_mode=self.label_mode,
                    observed_directed_pairs=int(self.counts.numel()),levels=self.levels,
                    frozen_buffer_bytes=sum(b.numel()*b.element_size() for b in self.buffers()))
