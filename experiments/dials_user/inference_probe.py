"""Ceiling probe before retraining: add a user co-interaction term to the frozen POG DIALS models at inference.

M = X^T W X (W = 1 / log(2 + |history|), zero diagonal), U(S, c) = log(1 + sum_s M_sc d_s^-a d_c^-a), d = row sums of M.
Score = frozen DIALS score + lam * U; (lam, a) chosen on validation NDCG@20 (3-seed mean), test scored once.
Nothing is retrained, so this measures what U adds on top of the trained model, not what joint training would reach.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_user.inference_probe
"""
import dials
import numpy as np, scipy.sparse as sp, torch
from pathlib import Path
from dials import data as dials_data
from dials.train import load as load_dials
from dials.evaluate import query_metrics
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS,CACHE,groups
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.dials_user.model import user_matrix

GRID=[(lam,a) for lam in (0.,.3,1.,3.,10.,30.,100.) for a in (0.,.5)]


def u_scores(M,d,seeds,a):
    rows=[];
    for s in seeds:
        s=sorted(set(s));v=np.asarray((sp.diags(d[s]**-a)@M[s]).sum(0)).ravel()*(d**-a);rows.append(np.log1p(v))
    return np.stack(rows)


@torch.inference_mode()
def scores(name,ds,pairs,M,d):
    """Per query: list over seeds of (frozen score, U for each a)."""
    out={(lam,a):np.zeros((len(pairs),2)) for lam,a in GRID}
    for s in (0,1,2):
        m=load_dials(ds,RUNS/name/f'seed{s}',CACHE);m.eval();cache=m.cache()
        for start in range(0,len(pairs),64):
            batch=pairs[start:start+64];sd=[q for q,_ in batch]
            base=m([torch.as_tensor(sorted(set(q)),dtype=torch.long) for q in sd],cache=cache).numpy()
            U={a:u_scores(M,d,sd,a) for a in (0.,.5)}
            for lam,a in GRID:
                sc=base+lam*U[a]
                for j,(q,g) in enumerate(batch):
                    r=query_metrics(sc[j],q,g);out[(lam,a)][start+j]+=np.array([r['recall20'],r['ndcg20']])/3
    return out


def main(name):
    torch.set_num_threads(4);ds=dials_data.load(name,CACHE);M=user_matrix(name,ds);d=np.maximum(np.asarray(M.sum(1)).ravel(),1e-9)
    print('M nnz',M.nnz,flush=True)
    val=scores(name,ds,ds.val_pairs,M,d)
    for k,v in val.items():print('val',k,'R %.4f N %.4f'%tuple(v.mean(0)),flush=True)
    best=max(val,key=lambda k:val[k][:,1].mean());print('chosen',best,flush=True)
    keep=sorted(ds.clean_test);pairs=[ds.test_pairs[i] for i in keep]
    test=scores(name,ds,pairs,M,d);base,new=test[(0.,0.)],test[best];cl=[groups(ds.test_pairs)[i] for i in keep]
    recs=[dict(metric=mt,**paired_comparison(new[:,j],base[:,j],cl,seed=20261109)) for j,mt in enumerate(('recall20','ndcg20'))];holm_adjust(recs)
    print('test clean frozen R %.4f N %.4f'%tuple(base.mean(0)),'| + U R %.4f N %.4f'%tuple(new.mean(0)))
    for r in recs:print(r['metric'],'%+.4f [%+.4f, %+.4f] p_holm %.3f'%(r['delta'],*r['ci95'],r['p_holm']))
    out=Path('results/dials_user');write_json(out/f'inference_probe_{name}.json',dict(note=__doc__,chosen=list(best),
        validation={str(k):v.mean(0).tolist() for k,v in val.items()},test=dict(frozen=base.mean(0).tolist(),with_u=new.mean(0).tolist()),tests=recs))


if __name__=='__main__':
    import sys;main(sys.argv[1] if len(sys.argv)>1 else 'pog')
