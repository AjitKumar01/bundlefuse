"""Which score term carries DIALS in each evidence bucket? Inference-time knockouts of the frozen models.

The score is a gated sum s = w_C C + sum_l w_l (D_l + R_l) + w_G (K + P), with D_l the shared-label diagonal, R_l the
low-rank label relation, K the co-occurrence part theta log(1 + ...) and P the popularity prior rho(log(1 + f_c)).
Each variant drops parts from that sum, keeping the trained weights; nothing is retrained, so a variant measures what
the part contributes inside the trained model, not what a model trained without it would reach.
Buckets as in run.py; clean test queries, 3 seeds, NDCG@20 and Recall@20.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.story_checks.knockout
"""
import dials  # pins BLAS threads before numpy loads
import numpy as np
import torch
from dials import data as dials_data
from dials.train import load as load_dials
from dials.evaluate import query_metrics
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS,CACHE
from experiments.story_checks.run import evidence,buckets,OUT

VARIANTS={'full':set(),'no content':{'C'},'no labels':{'D','R'},'no label diagonal':{'D'},'no label low-rank':{'R'},
          'no co-occurrence':{'K'},'no prior':{'P'},'content only':{'D','R','K','P'},'labels only':{'C','K','P'},
          'co-occurrence and prior only':{'C','D','R'}}


@torch.inference_mode()
def parts(m,seeds):
    """Weighted parts (each queries x items) for a batch of seed lists."""
    seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds];cache=m.cache();w=m.gate_weights(seeds,None)
    out={'C':w[:,0,None]*m.content_scores(seeds,None,cache),'D':0.,'R':0.}
    for i in range(m.levels):
        full=m.label_scores(i,seeds,None);mix=m.mixing[i].data.clone();m.mixing[i].data.zero_()
        diag=m.label_scores(i,seeds,None);m.mixing[i].data.copy_(mix)
        out['D']=out['D']+w[:,1+i,None]*diag;out['R']=out['R']+w[:,1+i,None]*(full-diag)
    graph=m.graph_scores(seeds,None,cache);prior=cache['prior'].expand(len(seeds),-1);g=w[:,-1,None]
    out['K']=g*(graph-prior);out['P']=g*prior
    check=m(seeds,cache=cache);total=sum(out.values())
    if not torch.allclose(total,check,atol=1e-4,rtol=1e-5):raise RuntimeError('parts do not add up to the score')
    return {k:v.numpy() for k,v in out.items()}


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);report={}
    for name in ('pog_dense','pog','spotify_sparse'):
        ds=dials_data.load(name,CACHE);keep=sorted(ds.clean_test);pairs=[ds.test_pairs[i] for i in keep]
        e=evidence(ds);groups,labels,how=buckets(name,e);position={q:i for i,q in enumerate(keep)}
        metric={v:np.zeros((len(pairs),2)) for v in VARIANTS}
        for s in (0,1,2):
            m=load_dials(ds,RUNS/name/f'seed{s}',CACHE);m.eval()
            for start in range(0,len(pairs),32):
                batch=pairs[start:start+32];p=parts(m,[q for q,_ in batch])
                for v,drop in VARIANTS.items():
                    score=sum(p[k] for k in p if k not in drop)
                    for j,(q,g) in enumerate(batch):
                        r=query_metrics(score[j],q,g);metric[v][start+j]+=np.array([r['recall20'],r['ndcg20']])/3
            print(name,'seed',s,'done',flush=True)
        rows=[]
        for label,idx in [('all',keep)]+list(zip(labels,groups)):
            idx=[position[i] for i in idx if i in position]
            if not idx:continue
            rows.append(dict(bucket=label,queries=len(idx),**{v:metric[v][idx].mean(0).tolist() for v in VARIANTS}))
            print(f"{name:15s} {label:12s} n={len(idx):5d} "+'  '.join(f"{v} {metric[v][idx,1].mean():.4f}" for v in VARIANTS),flush=True)
        report[name]=dict(bucketed_by=how,variants={v:sorted(d) for v,d in VARIANTS.items()},buckets=rows)
    write_json(OUT/'knockout.json',dict(note=__doc__,datasets=report))


if __name__=='__main__':main()
