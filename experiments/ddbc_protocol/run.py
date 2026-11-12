"""DIALS (frozen v3.1, three seeds) under DDBC's evaluation protocol (Tu et al., ICLR 2026), on DDBC's own test queries.

DDBC's POG_k=4 and Spotify_k=30/60/90 training files are identical to our POG_DENSE and Spotify training splits, and
each of its test queries is a contiguous window of our test bundle with the same id (checked here). Some of those
test bundles also occur in training (the provided splits' overlap); DDBC keeps them, so they are kept and flagged,
and results are also reported without them. So the models
already trained on those splits are scored as they are, with no retraining or tuning.

Protocol (DDBC section 4.1, Eq. 9, single attempt as in its Table 1): the window is split 1:1 into input b_x and
targets b_y. The candidate pool is b_y plus 99|b_y| items drawn uniformly from the catalogue minus the window. The
prediction is the top-|b_y| pool items by score. With |prediction| = |b_y|, P = R = F1 = hits/|b_y| and Jaccard =
hits/(2|b_y| - hits). Five pool draws per query, averaged. Also reported: the same top-|b_y| set from the whole
catalogue (seeds excluded), which needs no sampling. The best simple baseline (validation NDCG@20 in
results/simple_baselines) is scored the same way for reference.

Data: DDBC's released archive (Google Drive link in its README), text split files only, under
<DATA_ROOT>/ddbc_data/extracted/datasets.

PYTHONPATH=. .venv/bin/python -m experiments.ddbc_protocol.run [--datasets pog_dense spotify]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.neural_lowrank import run as base
from experiments.neural_relations import run as r3  # sets base.construct to the v3.1 model
from experiments.simple_baselines.methods import Baselines

DDBC=Path(DATA_ROOT)/'ddbc_data'/'extracted'/'datasets'
CASES=[('pog_dense','pog_dense',4),('spotify','spotify',30),('spotify','spotify',60),('spotify','spotify',90)]
DRAWS=5;RATIO=100
PUBLISHED={  # DDBC Table 1 (F1, Jaccard): DDBC and the strongest baseline per column
    ('pog_dense',4):dict(DDBC=(.139,.098),best_baseline=('TIGER',.213,.157)),
    ('spotify',30):dict(DDBC=(.282,.176),best_baseline=('BundleNAT',.153,.090)),
    ('spotify',60):dict(DDBC=(.285,.178),best_baseline=('TIGER',.129,.076)),
    ('spotify',90):dict(DDBC=(.287,.177),best_baseline=('TIGER',.123,.070))}


def rows(path):
    out={}
    for line in open(path):
        v=[int(x) for x in line.strip().split(', ') if x];out[v[0]]=v[1:]
    return out


def queries(name,folder,k,ds):
    seeds,targets=rows(DDBC/folder/f'test_input_{k}.txt'),rows(DDBC/folder/f'test_gt_{k}.txt')
    train={frozenset(b) for b in ds.train}
    tests=rows(Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/'bi_test_input.txt');gts=rows(Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/'bi_test_gt.txt')
    out=[]
    for b in sorted(seeds):
        full=set(tests[b])|set(gts.get(b,[]))
        if not (set(seeds[b])|set(targets[b]))<=full:raise RuntimeError(f'{name} k={k}: query {b} is not a window of our test bundle')
        out.append((b,sorted(set(seeds[b])),sorted(set(targets[b])),frozenset(full) in train))
    return out


def score_sets(scores,qs,n,rng_seed):
    """Per-query [pool F1, pool Jaccard, full-catalogue F1, full-catalogue Jaccard]."""
    out=[]
    for row,(b,s,t,_) in zip(scores,qs):
        m=len(t);tset=set(t);window=np.asarray(sorted(set(s)|tset))
        full=row.copy();full[s]=-np.inf;h=len(tset&set(np.argpartition(-full,m)[:m].tolist()))
        pool_scores=[]
        for d in range(DRAWS):
            rng=np.random.default_rng([rng_seed,b,d]);others=np.setdiff1d(np.arange(n),window,assume_unique=True)
            pool=np.concatenate([np.asarray(t),rng.choice(others,(RATIO-1)*m,replace=False)])
            pick=pool[np.argsort(-row[pool],kind='stable')[:m]];hp=len(tset&set(pick.tolist()))
            pool_scores.append((hp/m,hp/(2*m-hp)))
        out.append([*np.mean(pool_scores,0),h/m,h/(2*m-h)])
    return np.asarray(out)


@torch.inference_mode()
def dials_scores(model,qs,batch=32):
    model.eval();cache=model.cache();out=[]
    for a in range(0,len(qs),batch):out.append(model([q[1] for q in qs[a:a+batch]],cache=cache).numpy())
    return np.concatenate(out)


def main(args):
    torch.set_num_threads(4);loaded={};path=Path('results/ddbc_protocol/summary.json')
    report=json.loads(path.read_text())['cases'] if path.exists() else {}
    for name,folder,k in [c for c in CASES if c[0] in args.datasets]:
        if name not in loaded:
            ds,x,c,_,_=base.data(name)
            models=[base.load(ds,x,c,json.loads((Path('results/neural_relations')/name/f'v31_loo_seed{s}'/'selected.json').read_text())) for s in (0,1,2)]
            simple=json.loads(Path(f'results/simple_baselines/{name}/summary.json').read_text())['test']
            best=max(simple,key=lambda m:simple[m]['validation_ndcg20']);loaded={name:(ds,x,models,best,simple[best]['params'])}
        ds,x,models,best,params=loaded[name];qs=queries(name,folder,k,ds);n=ds.n_items
        dials=np.mean([score_sets(dials_scores(m,qs),qs,n,20261109) for m in models],axis=0)
        baseline=Baselines(ds.train,n,x if best=='hybrid' else None)
        other=np.concatenate([baseline.scores(best,params,[q[1] for q in qs[a:a+128]]) for a in range(0,len(qs),128)])
        simple_sets=score_sets(other,qs,n,20261109);clean=np.asarray([not q[3] for q in qs])
        summary=lambda a,i=slice(None):dict(pool_f1=a[i,0].mean(),pool_jaccard=a[i,1].mean(),full_f1=a[i,2].mean(),full_jaccard=a[i,3].mean())
        report[f'{name}_k{k}']=dict(queries=len(qs),queries_whose_bundle_is_in_train=int((~clean).sum()),dials=summary(dials),dials_clean=summary(dials,clean),
            simple=dict(method=best,params=params,**summary(simple_sets)),simple_clean=summary(simple_sets,clean),
            published=PUBLISHED[(name,k)],per_query_dials=dials.tolist(),per_query_simple=simple_sets.tolist(),query_ids=[q[0] for q in qs],in_train=[bool(q[3]) for q in qs])
        r=report[f'{name}_k{k}'];print(f"{name} k={k} n={len(qs)} (in train {int((~clean).sum())}) DIALS F1 {r['dials']['pool_f1']:.3f} Jacc {r['dials']['pool_jaccard']:.3f} | {best} F1 {r['simple']['pool_f1']:.3f} Jacc {r['simple']['pool_jaccard']:.3f} | published DDBC {PUBLISHED[(name,k)]['DDBC']} best baseline {PUBLISHED[(name,k)]['best_baseline']}",flush=True)
        write_json(path,dict(note=__doc__,cases=report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',default=['pog_dense','spotify'],choices=['pog_dense','spotify'])
    main(p.parse_args())
