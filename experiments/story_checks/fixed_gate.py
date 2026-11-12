"""Is DIALS just a fixed-weight ensemble of its terms? Replace the per-query gate by constant weights.

The trained model's unweighted terms (content, one label term per level, graph + prior) are kept; only the gate w(S)
is replaced by one constant weight vector per dataset, the mean gate over the validation queries (no test data).
Nothing is retrained, so this measures what query-dependent weighting adds inside the trained model.
Clean test queries, 3 seeds, NDCG@20 and Recall@20; paired cluster bootstrap, Holm over all tests.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.story_checks.fixed_gate
"""
import dials  # pins BLAS threads before numpy loads
import numpy as np
import torch
from pathlib import Path
from dials import data as dials_data
from dials.train import load as load_dials
from dials.evaluate import query_metrics
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS,CACHE,groups
from experiments.mechanisms.statistics import paired_comparison,holm_adjust

DATASETS=('electronic','food','clothing','dunnhumby','pog_dense','pog','spotify_sparse')


@torch.inference_mode()
def terms(m,seeds,cache):
    s=[torch.as_tensor(sorted(set(x)),dtype=torch.long) for x in seeds]
    parts=[m.content_scores(s,None,cache)]+[m.label_scores(i,s,None) for i in range(m.levels)]+[m.graph_scores(s,None,cache)]
    return torch.stack(parts,1),m.gate_weights(s,None)  # queries x terms x items, queries x terms


def main():
    torch.set_num_threads(4);report={};recs=[]
    for name in DATASETS:
        ds=dials_data.load(name,CACHE);keep=sorted(ds.clean_test);pairs=[ds.test_pairs[i] for i in keep]
        cl=[groups(ds.test_pairs)[i] for i in keep];gated=np.zeros((len(pairs),2));fixed=np.zeros((len(pairs),2));consts=[]
        for s in (0,1,2):
            m=load_dials(ds,RUNS/name/f'seed{s}',CACHE).eval();cache=m.cache()
            w=[];
            for start in range(0,len(ds.val_pairs),64):w.append(terms(m,[q for q,_ in ds.val_pairs[start:start+64]],cache)[1])
            const=torch.cat(w).mean(0);consts.append(const.tolist())
            for start in range(0,len(pairs),32):
                batch=pairs[start:start+32];T,g=terms(m,[q for q,_ in batch],cache)
                a=(g[:,:,None]*T).sum(1).numpy();b=(const[None,:,None]*T).sum(1).numpy()
                for j,(q,t) in enumerate(batch):
                    r=query_metrics(a[j],q,t);gated[start+j]+=np.array([r['recall20'],r['ndcg20']])/3
                    r=query_metrics(b[j],q,t);fixed[start+j]+=np.array([r['recall20'],r['ndcg20']])/3
        for j,mt in enumerate(('recall20','ndcg20')):
            recs.append(dict(dataset=name,metric=mt,**paired_comparison(gated[:,j],fixed[:,j],cl,seed=20261109)))
        report[name]=dict(gated=gated.mean(0).tolist(),fixed=fixed.mean(0).tolist(),constant_weights=np.mean(consts,0).tolist())
        print(f"{name:15s} gated N {gated[:,1].mean():.4f} fixed N {fixed[:,1].mean():.4f}  R {gated[:,0].mean():.4f} / {fixed[:,0].mean():.4f}",flush=True)
    holm_adjust(recs)
    for r in recs:print(f"{r['dataset']:15s} {r['metric']:8s} gated - fixed {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p_holm {r['p_holm']:.4f}")
    out=Path('results/story_checks');write_json(out/'fixed_gate.json',dict(note=__doc__,datasets=report,tests=recs))


if __name__=='__main__':main()
