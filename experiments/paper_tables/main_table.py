"""Main table of the DIALS paper: DIALS against the best simple baseline on the 7 bundle datasets.

One pool: the 9 baselines of experiments/simple_baselines (popularity, co-occurrence, ItemKNN, RP3beta, GF-CF, EASE,
content+ItemKNN blend, iALS, SGFCF) plus the 4 label-aware ones of experiments/bundle_budget (same label, transition,
label+ItemKNN, label+ItemKNN+content). The best one per dataset is chosen on validation NDCG@20 (a method present in
both pools keeps its higher-validation setting). Clean test queries, paired cluster bootstrap and sign flips,
Holm over the 14 tests (7 datasets x NDCG@20, Recall@20).

PYTHONPATH=.:../dials .venv/bin/python -m experiments.paper_tables.main_table
"""
import dials  # pins BLAS threads before numpy loads
import json
from pathlib import Path
import numpy as np
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS,groups
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.bundle_budget_retail.conditions import dataset

D=['electronic','food','clothing','pog_dense','pog','spotify_sparse','spotify']
OUT=Path('results/paper_tables')


def pool(name,keep):
    """{method: (validation NDCG@20, clean per-query [R, N] array, params, source)}"""
    out={}
    s=json.loads(Path(f'results/simple_baselines/{name}/summary.json').read_text())['test']
    for k,v in s.items():
        q=np.asarray(v['queries'])
        out[k]=(v['validation_ndcg20'],q[keep] if len(q)>len(keep) else q,v['params'],'simple_baselines')
    for root in ('results/bundle_budget_retail','results/bundle_budget'):
        p=Path(root)/'baselines'/name/'full.json'
        if p.exists():
            for k,v in json.loads(p.read_text())['methods'].items():
                if k not in out or v['validation_ndcg20']>out[k][0]:out[k]=(v['validation_ndcg20'],np.asarray(v['queries']),v['params'],root)
    return out


def main():
    report={};recs=[]
    for name in D:
        ds=dataset(name);keep=sorted(ds.clean_test);cl=[groups(ds.test_pairs)[i] for i in keep]
        d=np.mean([[[q['recall20'],q['ndcg20']] for q in json.loads((RUNS/name/f'seed{s}'/'test.json').read_text())['queries']] for s in (0,1,2)],0)[keep]
        P=pool(name,keep);best=max(P,key=lambda k:P[k][0])
        report[name]=dict(dials=d.mean(0).tolist(),best=best,best_params=P[best][2],best_source=P[best][3],
                          pool={k:dict(validation_ndcg20=v[0],test=v[1].mean(0).tolist()) for k,v in P.items()},queries=len(keep))
        for j,mt in enumerate(('recall20','ndcg20')):
            recs.append(dict(dataset=name,metric=mt,comparison=f'DIALS vs {best}',**paired_comparison(d[:,j],P[best][1][:,j],cl,seed=20261109)))
        print(name,best,'DIALS',np.round(d.mean(0),4),'best',np.round(P[best][1].mean(0),4),flush=True)
    holm_adjust(recs)
    for r in recs:print(f"{r['dataset']:15s} {r['metric']:8s} {r['comparison']:35s} {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p_holm {r['p_holm']:.4f}")
    write_json(OUT/'main_table.json',dict(note=__doc__,datasets=report,tests=recs))


if __name__=='__main__':
    main()
