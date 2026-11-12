"""Standard non-neural item-to-item baselines for bundle completion.

Every method maps a batch of seed indicators Y (q x n, sparse) to scores (q x n). A query is treated as a
user whose interactions are its seeds. Item statistics come from training bundles only.
Ties are broken by training popularity (1e-6 log(1 + f)) for every method alike.
"""
import numpy as np
import scipy.sparse as sp


def incidence(train,n):
    rows=[np.unique(np.asarray(b,np.int64)) for b in train]
    indptr=np.cumsum([0]+[len(r) for r in rows])
    return sp.csr_matrix((np.ones(indptr[-1],np.float32),np.concatenate(rows),indptr),shape=(len(rows),n))


def without_diagonal(m):
    m=m.tocsr();m.setdiag(0);m.eliminate_zeros();return m


def top_k_rows(m,k):
    """Keep the k largest entries of every row (ItemKNN neighbourhoods)."""
    m=m.tocsr();data,indices,indptr=m.data,m.indices,m.indptr
    keep=np.zeros(len(data),bool)
    for r in range(m.shape[0]):
        a,b=indptr[r],indptr[r+1]
        if b-a<=k:keep[a:b]=True
        else:keep[a+np.argpartition(-data[a:b],k)[:k]]=True
    out=sp.csr_matrix((data*keep,indices,indptr),shape=m.shape);out.eliminate_zeros();return out


class Baselines:
    def __init__(self,train,n,features=None):
        self.n=n;self.X=incidence(train,n)
        self.features=None if features is None else (features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)).astype(np.float32)
        self.f=np.asarray(self.X.sum(0)).ravel()
        self.C=without_diagonal(self.X.T@self.X).astype(np.float32)
        self.tie=1e-6*np.log1p(self.f)[None,:].astype(np.float32)
        self.cache={}

    def grid(self,ease_limit):
        g=[('popularity',{}),('cooccurrence',{})]
        g+=[('itemknn',dict(k=k,shrink=h)) for k in (50,200,1000) for h in (0,10,100)]
        g+=[('rp3beta',dict(alpha=a,beta=b)) for a in (.5,1.) for b in (0.,.25,.5,.75)]
        g+=[('gfcf',dict(rank=r,ideal=w)) for r in (64,256) for w in (0.,.1,.3,1.,3.)]
        if int((self.f>0).sum())<=ease_limit:g+=[('ease',dict(l2=l)) for l in (50.,200.,500.,1000.,3000.)]
        return g

    def matrix(self,name,p):
        key=(name,tuple(sorted(p.items())))
        if key in self.cache:return self.cache[key]
        f=np.maximum(self.f,1)
        if name=='cooccurrence':m=self.C
        elif name=='itemknn':
            s=1/np.sqrt(f)
            cos=sp.diags(s)@self.C@sp.diags(s)
            if p['shrink']:
                c=self.C.tocoo();cos=sp.csr_matrix((c.data/(np.sqrt(f[c.row]*f[c.col])+p['shrink']),(c.row,c.col)),shape=c.shape)
            m=top_k_rows(cos,p['k'])
        elif name=='rp3beta':
            size=np.asarray(self.X.sum(1)).ravel()
            m=without_diagonal(sp.diags(f**-p['alpha'])@(self.X.T@sp.diags(size**-p['alpha'])@self.X)@sp.diags(f**-p['beta']))
        elif name=='gfcf':
            size=np.maximum(np.asarray(self.X.sum(1)).ravel(),1)
            r=sp.diags(size**-.5)@self.X@sp.diags(f**-.5)
            linear=(r.T@r).tocsr()
            ideal=None
            if p['ideal']:
                k=min(p['rank'],min(r.shape)-1)
                from sklearn.utils.extmath import randomized_svd
                _,_,vt=randomized_svd(r,k,n_iter=5,random_state=0)
                ideal=(vt.T*(f**-.5)[:,None]).astype(np.float32),(vt.T*(f**.5)[:,None]).astype(np.float32)
            m=(linear,ideal,p['ideal'])
        elif name=='ease':
            from scipy.linalg import inv
            active=np.flatnonzero(self.f>0)
            g=(self.X[:,active].T@self.X[:,active]).toarray().astype(np.float32)
            g[np.diag_indices_from(g)]+=p['l2']
            B=inv(g,overwrite_a=True,check_finite=False);del g
            B/=-np.diag(B).copy()[None,:];B[np.diag_indices_from(B)]=0  # in place: B = I - P diag(P)^-1
            m=(active,B.astype(np.float32,copy=False))
        else:m=None
        self.cache={key:m} if name in ('ease','gfcf') else {**self.cache,key:m}
        return m

    def extended_grid(self,itemknn):
        """Second suite: a content plus ItemKNN blend (ItemKNN fixed at its validation choice), iALS, SGFCF."""
        g=[('hybrid',dict(itemknn,content=w)) for w in (0.,.1,.2,.3,.5,.7,.9,1.)] if self.features is not None else []
        g+=[('ials',dict(rank=r,alpha=a,l2=l)) for r in (64,128) for a in (5.,20.) for l in (1.,10.)]
        g+=[('sgfcf',dict(rank=r,eps=e,beta=b,gamma=c)) for r in (64,256) for e in (.3,.5) for b in (1.,2.5) for c in (0.,.5,1.)]
        return g

    def ials(self,p,iterations=8,budget=2048):
        """Implicit ALS (Hu et al. 2008) on the bundle-item matrix, confidence 1 + alpha on observed entries.
        Each row solves (V^T V + alpha V_S^T V_S + l2 I) u = (1 + alpha) V_S^T 1, rows batched by at most `budget` nonzeros."""
        k=p['rank'];rng=np.random.default_rng(0)
        V=(rng.standard_normal((self.n,k))*.01).astype(np.float32);X=self.X.tocsr();XT=X.T.tocsr()
        def solve(R,V):
            out=np.zeros((R.shape[0],k),np.float32);base=V.T@V+p['l2']*np.eye(k,dtype=np.float32)
            nonempty=np.flatnonzero(np.diff(R.indptr));a=0
            while a<len(nonempty):
                b=a+1
                while b<len(nonempty) and R.indptr[nonempty[b]+1]-R.indptr[nonempty[a]]<=budget:b+=1
                rows=nonempty[a:b];lo,hi=R.indptr[rows[0]],R.indptr[rows[-1]+1]
                vs=V[R.indices[lo:hi]];starts=R.indptr[rows]-lo
                A=base[None]+p['alpha']*np.add.reduceat(vs[:,:,None]*vs[:,None,:],starts,axis=0)
                rhs=(1+p['alpha'])*np.add.reduceat(vs,starts,axis=0)
                out[rows]=np.linalg.solve(A,rhs[:,:,None])[:,:,0];a=b
            return out
        for _ in range(iterations):U=solve(X,V);V=solve(XT,U)
        return V,solve
    def scores(self,name,p,seeds):
        q=len(seeds);y=sp.csr_matrix((np.ones(sum(len(set(s)) for s in seeds),np.float32),
            (np.repeat(np.arange(q),[len(set(s)) for s in seeds]),np.concatenate([sorted(set(s)) for s in seeds]))),shape=(q,self.n))
        if name=='popularity':return np.tile(np.log1p(self.f).astype(np.float32),(q,1))
        if name=='hybrid':
            def scaled(s):return s/np.maximum(np.abs(s).max(1,keepdims=True),1e-12)
            knn=np.asarray((y@self.matrix('itemknn',dict(k=p['k'],shrink=p['shrink']))).todense())
            content=(y@self.features)@self.features.T
            return ((1-p['content'])*scaled(knn)+p['content']*scaled(content)).astype(np.float32)+self.tie
        if name=='ials':
            key=('ials',tuple(sorted(p.items())))
            if key not in self.cache:self.cache={key:self.ials(p)}
            V,solve=self.cache[key];return (solve(y,V)@V.T).astype(np.float32)+self.tie
        if name=='sgfcf':
            key=('sgfcf',p['rank'],p['eps'])
            if key not in self.cache:
                from sklearn.utils.extmath import randomized_svd
                du=np.asarray(self.X.sum(1)).ravel()**-p['eps'];di=np.maximum(self.f,1)**-p['eps'];di[self.f==0]=0
                N=(sp.diags(du)@self.X@sp.diags(di)).tocsr()
                _,sigma,vt=randomized_svd(N,min(p['rank'],min(N.shape)-1),n_iter=5,random_state=0)
                self.cache={key:(N,di.astype(np.float32),(sigma/sigma.max()).astype(np.float32),vt.T.astype(np.float32),sigma.astype(np.float32))}
            N,di,s,V,sigma=self.cache[key]
            def unit(m):return m/np.maximum(np.linalg.norm(m,axis=1,keepdims=True),1e-12)
            r=y@sp.diags(di)  # the query as a new normalized row; its own degree factor cancels in the row normalization
            low=((r@V)*(s**(2*p['beta'])/sigma))@V.T  # fold-in u = r V / sigma, scores (u w)(V w)^T
            out=unit(np.asarray(low))
            if p['gamma']:out=out+p['gamma']*unit(np.asarray(((r@N.T)@N).todense()))
            return out.astype(np.float32)+self.tie
        m=self.matrix(name,p)
        if name=='gfcf':
            linear,ideal,w=m
            out=np.asarray((y@linear).todense())
            if ideal is not None:out=out+w*((y@ideal[0])@ideal[1].T)
        elif name=='ease':
            active,B=m;out=np.zeros((q,self.n),np.float32)
            out[:,active]=np.asarray(y[:,active].todense())@B
        else:out=np.asarray((y@m).todense())
        return out.astype(np.float32)+self.tie
