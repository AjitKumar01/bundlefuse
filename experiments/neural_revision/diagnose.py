"""Validation-only score alignment and frozen-representation probes.

Probe targets are BundleFuse sources, used ONLY in discarded diagnostic regressions.
They never train, initialize, select, or feed the deployed neural scorer.
Linear recoverability is not a proof of general neural representational capacity.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from bundlefuse.model import BundleFuse
from bundlefuse.metrics import top_items
from bundlefuse.provenance import write_json
from experiments.neural.data import load_data, pad_seeds
from experiments.neural.training import load_selected


@torch.inference_mode()
def diagnose(name, output):
    torch.set_num_threads(1)
    ds, features, cats, freq, identity = load_data(name)
    reference = BundleFuse.load(f'checkpoints/neural_small/{name}/comparisons/bundlefuse_matched.pt')
    raw, z = reference._score_all([s for s,_ in ds.val_pairs])
    keys = [k for k in ['pair','simp','cooc','ppmi','cat','homo','content','bridge','csvd','rao','pop'] if k in z]
    targets = {k:z[k].numpy().T for k in keys}
    delta = (raw['simp']-raw['pair']).numpy().T
    targets['joint_surplus'] = (delta-delta.mean(1,keepdims=True))/(delta.std(1,keepdims=True)+1e-9)
    keys.append('joint_surplus')
    rng = np.random.default_rng(712)
    order = rng.permutation(len(ds.val_pairs))
    train_queries = set(order[:len(order)//2].tolist())
    candidate_rows = []
    for q,(seeds,_) in enumerate(ds.val_pairs):
        ids = set(rng.choice(ds.n_items, min(128,ds.n_items),replace=False).tolist())
        for k in keys:
            if np.ptp(targets[k][q]) > 0:
                ids.update(top_items(targets[k][q],seeds,16).tolist())
        candidate_rows.append(np.array(sorted(ids-set(seeds)),dtype=int))
    report = dict(dataset=name,split='validation only; 50/50 query-level probe fit/holdout',
        interpretation='R2 is held-out linear decodability from frozen hidden features, not proof that a source is absent or unlearnable. Correlation is not a mechanism test.',
        diagnostic_training='Only separate Ridge probes fit source targets; neural weights remain frozen. No probe enters inference.',
        selected_bundlefuse_sources=reference.w, source_active_queries={k:int((np.ptp(v,axis=1)>0).sum()) for k,v in targets.items()},
        training_item_frequency={str(v):int((freq==v).sum()) for v in [0,1,2]},
        variants={})
    for variant in ['relations','contextual']:
        selected=json.loads(Path(f'results/neural_small/{name}/{variant}_seed0/selected.json').read_text())
        model=load_selected(selected,features,cats,freq)
        table=model.encode_items(torch.arange(ds.n_items+1))
        captured=[]
        handle=model.decoder[0].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy().reshape(-1,args[0].shape[-1])))
        xx=[];yy=[];split=[];correlations={k:[] for k in keys};overlaps={k:[] for k in keys}
        for q,(seeds,_) in enumerate(ds.val_pairs):
            ids,valid=pad_seeds([seeds],ds.n_items);query=model.encode_query(ids,valid,table)
            scores=[]
            for start in range(0,ds.n_items,256):
                cand=torch.arange(start,min(start+256,ds.n_items))[None]
                scores.append(model.score_candidates(query,cand,table).numpy().ravel())
            score=np.concatenate(scores);hidden=np.concatenate(captured);captured.clear()
            chosen=candidate_rows[q]
            xx.append(hidden[chosen]);yy.append(np.stack([targets[k][q,chosen] for k in keys],1))
            split.extend([q in train_queries]*len(chosen))
            available=np.ones(ds.n_items,bool);available[seeds]=False
            predicted=set(top_items(score,seeds,20))
            for k in keys:
                target=targets[k][q]
                if np.ptp(target[available]) > 0:
                    correlations[k].append(float(spearmanr(score[available],target[available]).statistic))
                    overlaps[k].append(len(predicted & set(top_items(target,seeds,20)))/20)
        handle.remove()
        x=np.concatenate(xx);y=np.concatenate(yy);fit=np.array(split)
        scaler=StandardScaler().fit(x[fit]);x=scaler.transform(x)
        ym=y[fit].mean(0);ys=np.maximum(y[fit].std(0),1e-8)
        probe=Ridge(alpha=100.).fit(x[fit],(y[fit]-ym)/ys)
        prediction=probe.predict(x[~fit])*ys+ym
        probe_r2=r2_score(y[~fit],prediction,multioutput='raw_values')
        report['variants'][variant]=dict(checkpoint=selected['checkpoint'],probe_rows_fit=int(fit.sum()),probe_rows_heldout=int((~fit).sum()),
            sources={k:dict(heldout_probe_r2=float(probe_r2[i]),
                    mean_score_spearman=float(np.mean(correlations[k])) if correlations[k] else None,
                    mean_top20_overlap=float(np.mean(overlaps[k])) if overlaps[k] else None)
                     for i,k in enumerate(keys)})
        print(variant,json.dumps(report['variants'][variant]['sources']),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'signal_diagnostic.json',report)
    lines=['# Frozen neural representation diagnostic','',report['interpretation'],'',
           '| Source | Relations probe R2 | Contextual probe R2 |','|---|---:|---:|']
    for k in keys:
        a=report['variants']['relations']['sources'][k]['heldout_probe_r2']
        b=report['variants']['contextual']['sources'][k]['heldout_probe_r2']
        lines.append(f'| {k} | {a:.4f} | {b:.4f} |')
    lines += ['',report['diagnostic_training'],'', 'All source operators are constructed from training bundles. Probe rows contain a fixed mix of source top candidates and uniform candidates. Split is by query, not candidate row.']
    (output/'SIGNAL_DIAGNOSTIC.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='electronic',choices=['electronic','food'])
    p.add_argument('--output',type=Path,default=Path('results/neural_revision/diagnostics'))
    a=p.parse_args();diagnose(a.dataset,a.output/a.dataset)
