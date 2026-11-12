"""Frozen DIALS (branch dials, no second hop; runs in ../dials/runs) against every baseline, as before for v3.1.

Families, each Holm-corrected on its own, NDCG@20 and Recall@20, paired cluster bootstrap and sign flips (clusters:
identical complete bundles), clean test queries:
  simple   the best simple baseline per dataset, chosen on validation NDCG@20 (results/simple_baselines); 16 tests.
  neural   RaMen and CLHE rerun on identical splits: RaMen-clean and CLHE-clean on Electronic/Food/Clothing
           (results/ramen_retail, results/clhe_retail), CLHE three seeds on POG_DENSE/POG/Spotify_sparse/Spotify
           (Kaggle, clhe_seeds_out); 20 planned tests, Spotify CLHE counted even while missing.
DDBC protocol (its own queries, pool F1/Jaccard, 5 pool draws as in results/ddbc_protocol) is rescored with the
frozen model; the simple-baseline scores there are reused.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_frozen.compare [--skip-ddbc]
"""
import dials  # pins BLAS threads before numpy loads
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dials import data as dials_data
from dials.train import load as load_dials
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.ramen_retail.compare import ramen
from experiments.ramen_retail.compare_all import clhe as clhe_retail
from experiments.simple_baselines.compare_clhe import clhe as clhe_seeds
from experiments.ddbc_protocol import run as ddbc

RUNS=Path(DATA_ROOT)/'dials'/'runs';CACHE=Path(DATA_ROOT)/'dials'/'cache';OUT=Path('results/dials_frozen')
RETAIL=['electronic','food','clothing'];OTHERS=['pog_dense','pog','spotify_sparse','spotify'];BASKETS=['dunnhumby']  # no neural rerun on BASKETS


def groups(pairs):return [str(tuple(sorted(set(s)|set(g)))) for s,g in pairs]


def per_query(name):
    runs=[json.loads((RUNS/name/f'seed{s}'/'test.json').read_text()) for s in (0,1,2)]
    return np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in runs],axis=0)


def clhe_aligned(name):
    """CLHE three-seed per-query [R, N] in our test order, matched by bundle ID."""
    runs=clhe_seeds(name)
    if runs is None:return None
    order=[int(l.split(', ')[0]) for l in open(Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/'bi_test_input.txt')]
    index={b:i for i,b in enumerate(order)};out=np.zeros((len(order),2))
    for r in runs:
        m=np.full((len(order),2),np.nan)
        for b,rec,nd in r['queries']:m[index[b]]=(rec,nd)
        if np.isnan(m).any():raise RuntimeError(f'{name}: CLHE seed {r["seed"]} misses queries')
        out+=m/len(runs)
    return dict(queries=out,parameters=runs[0]['parameters'],seeds=[r['test'] for r in runs])


def compare(records,family,name,first,second,clusters,label):
    for j,metric in ((1,'ndcg20'),(0,'recall20')):
        records.append(dict(paired_comparison(first[:,j],second[:,j],clusters,bootstrap=2000,permutations=10000,seed=20261109),
                            family=family,dataset=name,metric=metric,first='dials',second=label))


def rescore_ddbc():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    old=json.loads(Path('results/ddbc_protocol/summary.json').read_text())['cases'];out={}
    for name in ('pog_dense','spotify'):
        ds=dials_data.load(name,CACHE);models=[load_dials(ds,RUNS/name/f'seed{s}',CACHE) for s in (0,1,2)]
        for folder,k in [(f,k) for n,f,k in ddbc.CASES if n==name]:
            qs=ddbc.queries(name,folder,k,ds);case=f'{name}_k{k}'
            if [q[0] for q in qs]!=old[case]['query_ids']:raise RuntimeError(f'{case}: query order changed')
            sets=np.mean([ddbc.score_sets(ddbc.dials_scores(m,qs),qs,ds.n_items,20261109) for m in models],axis=0)
            simple=np.asarray(old[case]['per_query_simple']);clean=~np.asarray(old[case]['in_train'])
            summary=lambda a,i=slice(None):dict(pool_f1=float(a[i,0].mean()),pool_jaccard=float(a[i,1].mean()),full_f1=float(a[i,2].mean()),full_jaccard=float(a[i,3].mean()))
            diff=sets[:,0]-simple[:,0];rng=np.random.default_rng(0)
            boot=[diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(2000)]
            out[case]=dict(queries=len(qs),queries_whose_bundle_is_in_train=int((~clean).sum()),dials=summary(sets),dials_clean=summary(sets,clean),
                           simple=dict(method=old[case]['simple']['method'],**summary(simple)),published=old[case]['published'],
                           pool_f1_minus_simple=dict(delta=float(diff.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist()))
            print(case,{k:round(v,3) for k,v in out[case]['dials'].items()},flush=True)
    return out


def main(args):
    report=dict(note=__doc__,datasets={});records=[]
    for name in RETAIL+OTHERS+BASKETS:
        ds=dials_data.load(name,CACHE);keep=ds.clean_test;clusters=[groups(ds.test_pairs)[i] for i in keep]
        mine=per_query(name);entry=report['datasets'][name]=dict(dials=mine[keep].mean(0).tolist(),dials_all=mine.mean(0).tolist(),
                                                                clean_queries=len(keep),queries=len(ds.test_pairs))
        simple=json.loads(Path(f'results/simple_baselines/{name}/summary.json').read_text())['test']
        best=max(simple,key=lambda m:simple[m]['validation_ndcg20']);other=np.asarray(simple[best]['queries'])
        entry['simple']=dict(method=best,params=simple[best]['params'],test=other[keep].mean(0).tolist())
        compare(records,'simple',name,mine[keep],other[keep],clusters,f'simple:{best}')
        if name in RETAIL:
            for method,fn in (('ramen',ramen),('clhe',clhe_retail)):
                for variant in ('clean','published'):
                    r=fn(name,variant);entry[f'{method}_{variant}']=dict(test=r['queries'].mean(0).tolist(),**{k:v for k,v in r.items() if k in ('parameters','preset','gamma','n_layer_gat')})
                compare(records,'neural',name,mine,fn(name,'clean')['queries'],clusters,f'{method}_clean')
        elif name in OTHERS:
            r=clhe_aligned(name)
            if r is not None:
                entry['clhe']=dict(test=r['queries'][keep].mean(0).tolist(),test_all=r['queries'].mean(0).tolist(),parameters=r['parameters'])
                compare(records,'neural',name,mine[keep],r['queries'][keep],clusters,'clhe')
        print(name,'DIALS',np.round(entry['dials'],4),'simple',best,np.round(entry['simple']['test'],4),flush=True)
    for family,planned in (('simple',16),('neural',20)):holm_adjust([r for r in records if r['family']==family],planned)
    report['primary']=records
    hop2=json.loads(Path('results/ablations/hop2.json').read_text())
    report['second_hop']=dict(datasets={k:dict(with_second_hop=v['full'],without=v['ablated']) for k,v in hop2['datasets'].items()},primary=hop2['primary'])
    if not args.skip_ddbc:report['ddbc']=rescore_ddbc()
    elif (OUT/'summary.json').exists():report['ddbc']=json.loads((OUT/'summary.json').read_text()).get('ddbc')
    write_json(OUT/'summary.json',report)
    for r in records:print(r['family'],r['dataset'],r['metric'],r['second'],round(r['delta'],4),[round(x,4) for x in r['ci95']],round(r['p_holm'],4))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skip-ddbc',action='store_true')
    main(p.parse_args())
