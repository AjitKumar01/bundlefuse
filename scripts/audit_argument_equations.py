"""Independent synthetic checks for the technical argument equations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from bundlefuse.operators import SimplicialGraphOperator,DualSpectrumOntology,BilateralContentBridge,zcols
from bundlefuse.fusion import fusion_parts,apply_fusion
rng=np.random.default_rng(20261105)
train=[[0,1,2],[0,3],[1,4,5],[2,3,6],[0,4,6],[1,3,5],[2,6]]
n=9;labels=np.arange(n)%3;queries=[[0],[0,1],[2,4,7]]
g=SimplicialGraphOperator(train,n,svd_dim=3)
B=np.zeros((len(train),n))
for e,s in enumerate(train):B[e,s]=1
f=B.sum(0);d=B.sum(1);D=np.diag((f+5)**-.5);V=np.diag((d+1)**-.5);E=np.diag(1/(d+1))
x=np.zeros((n,len(queries)))
for j,s in enumerate(queries):x[s,j]=1
p=B.T@V@B@D@x;L=D@B.T@E@B@D;np.fill_diagonal(L,0)
checks={}
def check(name,a,b):
 a=np.asarray(a);b=np.asarray(b)
 np.testing.assert_allclose(a,b,rtol=3e-5,atol=3e-6)
 checks[name]=float(np.max(abs(a-b)))
a,ap=g.score_pairwise_and_diff2(queries)
check('weighted_overlap',a,p);check('propagated_overlap',ap,L@p)
counts=B@x;overlap=V@B@D@x
inc=B.T@(overlap*(1.5*np.maximum(counts-1,0)))
ai,aip=g.score_simplicial_incidence_only(queries)
check('incidence',ai,inc);check('propagated_incidence',aip,L@inc)
X=rng.normal(size=(n,4)).astype(np.float32)
res=X.astype(float).copy()
for a in np.unique(labels):res[labels==a]-=X[labels==a].mean(0,dtype=np.float64)
def norm(v):return v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-8)
F=norm(res);actualF=BilateralContentBridge.disentangle(X,labels,3)
check('centroid_residuals',actualF,F)
check('bridge',BilateralContentBridge.build_dense_bridge(actualF,g),norm(F+.5*norm(L@F)))
W=B.T@V@B@D;np.fill_diagonal(W,0)
for music in [False,True]:
 result=np.zeros_like(p)
 for j,s in enumerate(queries):
  q=W[:,s]+.15*np.maximum(F@F[s].T,0)
  G=.5*(labels[s,None]==labels[np.array(s)[None,:]])+.5*np.maximum(F[s]@F[s].T,0)
  K=1+G if music else np.maximum(1-G,0);np.fill_diagonal(K,0)
  result[:,j]=np.einsum('is,st,it->i',q,K,q)/(q.sum(1)+.05)
 check('rao_music' if music else 'rao_fashion',g.score_rao_concordance(queries,actualF,labels,gamma=1. if music else -1.,alpha_dense=.15),result)
onto=DualSpectrumOntology(train,labels,n)
A=np.eye(3)[labels];T=onto.Pcat.toarray().T.astype(float)
penalty=np.where(onto.dup_rate<.15,-.5*(1-onto.dup_rate),0.)
T+=np.diag(penalty)
actual=np.column_stack([np.asarray(onto.Pcat[labels[s]].sum(0)).ravel()[labels] for s in queries])+.5*onto.score_slot_saturation(queries)
check('category_operator',actual,A@T@A.T@x)
check('sample_standardization',zcols(torch.tensor(p)).numpy(),(p-p.mean(0))/(p.std(0,ddof=1)+1e-9))
z={k:torch.tensor(rng.normal(size=(n,3)),dtype=torch.float32) for k in ['pair','simp','diff2_pair','diff2','cat','style','pop','rao']}
for fallback in [False,True]:
 w={'simp':.5,'diff2':.25,'pop':.25} if fallback else {'simp':.5,'diff2':.25,'cat':.5,'style':.25,'pop':.25}
 prior=['cat','style']
 for alpha in [0.,.5,1.]:
  gamma=.2;y,p0=fusion_parts(z,w,prior,(alpha,gamma));y=y.numpy();p0=p0.numpy()
  yz=sum(v*z[{'simp':'pair','diff2':'diff2_pair'}.get(k,k)].numpy() for k,v in w.items())
  yi=sum(v*z[k].numpy() for k,v in w.items())
  expectedY=(1-alpha)*yz+alpha*yi+gamma*z['rao'].numpy()
  expectedP=sum(w.get(k,.5)*z[k].numpy() for k in prior)+gamma*z['rao'].numpy()+w['pop']*z['pop'].numpy()
  check(f'Y_{fallback}_{alpha}',y,expectedY);check(f'P_{fallback}_{alpha}',p0,expectedP)
  for kappa in [0.,.5,5.]:
   scale=1.4
   got=apply_fusion(z,w,prior,(kappa,scale),f,(alpha,gamma)).numpy()
   expect=y if kappa==0 else (f[:,None]*y+kappa*scale*p0)/(f[:,None]+kappa)
   check(f'final_{fallback}_{alpha}_{kappa}',got,expect)
   if kappa:
    check(f'objective_derivative_{fallback}_{alpha}_{kappa}',f[:,None]*(expect-y)+kappa*(expect-scale*p0),np.zeros_like(expect))
# The proposition is exact for real arithmetic; exercise equal scores and distinct scores.
P=np.array([2.,2.,-1.,5.]);Y=P+7
assert np.array_equal(np.lexsort((np.arange(4),-Y)),np.lexsort((np.arange(4),-1.4*P)))
files=['bundlefuse/operators.py','bundlefuse/fusion.py','bundlefuse/model.py']
out={'status':'passed','checks':checks,'cold_order_with_ties':'passed','scope':'Independent NumPy equations on a synthetic catalogue; no proof of statistical optimality or exhaustive dataset reproduction.', 'source_sha256':{s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in files}}
Path('results/argument_audit').mkdir(exist_ok=True)
Path('results/argument_audit/equations.json').write_text(json.dumps(out,indent=2)+'\n')
print(len(checks),'equation checks passed; max absolute error',max(checks.values()))
