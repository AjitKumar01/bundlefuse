"""Two checks of the DIALS story on POG_DENSE, POG, Spotify_sparse (CLHE three seeds on identical splits).

1. Evidence buckets. For each clean test query and each held-out target c, e_c = max over seeds s of N_sc, the
   training co-occurrence count. Queries are bucketed by their targets' evidence; per-query NDCG@20 and Recall@20 of
   DIALS (3 seeds), CLHE (3 seeds) and the best simple baseline are averaged per bucket.
   POG and POG_DENSE (1-2 targets): bucket by max_c e_c in {0, 1, 2-3, 4+}.
   Spotify_sparse (16 targets): bucket by the share of targets with e_c >= 1.
2. Gate. For every clean test query, the gate weights w(S); ratio w_G / w_C against the share of seeds never seen
   in training and the mean seed log-frequency.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.story_checks.run
"""
import dials  # pins BLAS threads before numpy loads
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from dials import data as dials_data
from dials.train import load as load_dials
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import per_query,clhe_aligned,RUNS,CACHE

OUT=Path('results/story_checks')


def evidence(ds):
    rows=np.concatenate([[i]*len(set(b)) for i,b in enumerate(ds.train)]);cols=np.concatenate([sorted(set(b)) for b in ds.train])
    B=sp.csc_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(ds.train),ds.n_items))
    out=[]
    for s,g in ds.test_pairs:
        s=sorted(set(s));g=sorted(set(g))
        out.append(np.asarray((B[:,s].T@B[:,g]).max(0).todense()).ravel())  # e_c for each target
    return out


def buckets(name,e):
    if name.startswith('spotify'):
        share=np.array([(x>=1).mean() for x in e])
        masks=[share==0,(share>0)&(share<.25),(share>=.25)&(share<.5),(share>=.5)&(share<.75),share>=.75]
        return [list(np.flatnonzero(m)) for m in masks],['0%','(0, 25%)','[25, 50%)','[50, 75%)','[75, 100%]'],'share of targets with a co-occurring seed'
    top=np.array([x.max() for x in e]);names=['0','1','2-3','4+']
    return [list(np.flatnonzero(top==0)),list(np.flatnonzero(top==1)),list(np.flatnonzero((top>=2)&(top<=3))),list(np.flatnonzero(top>=4))],names,'max seed-target co-occurrence count'


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);report={}
    for name in ('pog_dense','pog','spotify_sparse'):
        ds=dials_data.load(name,CACHE);clean=set(ds.clean_test)
        mine=per_query(name);other=clhe_aligned(name)['queries']
        simple=json.loads(Path(f'results/simple_baselines/{name}/summary.json').read_text())['test']
        best=max(simple,key=lambda m:simple[m]['validation_ndcg20']);base=np.asarray(simple[best]['queries'])
        e=evidence(ds);groups,labels,how=buckets(name,e);rows=[]
        for label,idx in zip(labels,groups):
            idx=[i for i in idx if i in clean]
            if not idx:continue
            rows.append(dict(bucket=label,queries=len(idx),share=len(idx)/len(clean),
                             dials=mine[idx].mean(0).tolist(),clhe=other[idx].mean(0).tolist(),simple=base[idx].mean(0).tolist()))
            r=rows[-1];print(f"{name:15s} {label:12s} n={r['queries']:5d} ({100*r['share']:4.1f}%)  N@20 DIALS {r['dials'][1]:.4f} CLHE {r['clhe'][1]:.4f} {best} {r['simple'][1]:.4f}  R@20 DIALS {r['dials'][0]:.4f} CLHE {r['clhe'][0]:.4f}",flush=True)
        # gate
        models=[load_dials(ds,RUNS/name/f'seed{s}',CACHE) for s in (0,1,2)];keep=sorted(clean)
        seeds=[torch.as_tensor(sorted(set(ds.test_pairs[i][0]))) for i in keep]
        with torch.inference_mode():w=np.mean([m.gate_weights(seeds,None).numpy() for m in models],0)
        freq=models[0].frequency.numpy();unseen=np.array([(freq[s.numpy()]==0).mean() for s in seeds]);logf=np.array([np.log1p(freq[s.numpy()]).mean() for s in seeds])
        ratio=w[:,-1]/w[:,0];gate=[]
        for lab,mask in (('0',unseen==0),('(0, 50%)',(unseen>0)&(unseen<.5)),('[50, 100%)',(unseen>=.5)&(unseen<1)),('100%',unseen==1)):
            idx=np.flatnonzero(mask)
            if len(idx)==0:continue
            gate.append(dict(unseen_seeds=lab,queries=int(len(idx)),weights=w[idx].mean(0).tolist(),graph_over_content=float(ratio[idx].mean())))
            print(f"{name:15s} unseen seeds {lab:10s} n={len(idx):5d} weights {np.round(w[idx].mean(0),2).tolist()} w_G/w_C {ratio[idx].mean():.2f}",flush=True)
        corr=dict(unseen=float(np.corrcoef(unseen,ratio)[0,1]),log_frequency=float(np.corrcoef(logf,ratio)[0,1]))
        print(name,'corr(w_G/w_C, share unseen) %.2f, corr(w_G/w_C, mean log f) %.2f'%(corr['unseen'],corr['log_frequency']),'ratio range %.2f-%.2f'%(ratio.min(),ratio.max()))
        report[name]=dict(bucketed_by=how,simple_method=best,buckets=rows,gate=gate,correlation=corr,ratio_quantiles=np.quantile(ratio,[0,.1,.5,.9,1]).tolist(),
                          weight_names=['content']+[f'label{i}' for i in range(models[0].levels)]+['graph'])
    write_json(OUT/'summary.json',dict(note=__doc__,datasets=report))


if __name__=='__main__':main()
