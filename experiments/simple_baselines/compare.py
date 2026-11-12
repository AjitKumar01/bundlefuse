"""v3.1 (frozen, three seeds) against the best simple baseline per dataset, on POG, POG_DENSE, Spotify_sparse and Spotify.

The baseline is chosen per method on validation NDCG@20, and across methods also on validation. Filtered test
queries, paired cluster bootstrap and sign flips, Holm over the eight tests. Retail comparisons are in
results/neural_retail/summary.json under their own plan.
"""
import json
from pathlib import Path
import numpy as np
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.neural_lowrank import run as base

DATASETS=['pog_dense','pog','spotify_sparse','spotify']


def main():
    report={};records=[]
    for name in DATASETS:
        ds,_,_,_,audit=base.data(name);keep=audit['clean_test_indices']
        runs=[json.loads((Path('results/neural_relations')/name/f'v31_loo_seed{s}'/'test.json').read_text()) for s in (0,1,2)]
        new=np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in runs],axis=0)
        simple=json.loads(Path(f'results/simple_baselines/{name}/summary.json').read_text())['test']
        best=max(simple,key=lambda k:simple[k]['validation_ndcg20']);other=np.asarray(simple[best]['queries'])
        ids=base.groups(ds.test_pairs);clean=[ids[i] for i in keep]
        report[name]=dict(v31_filtered=new[keep].mean(0).tolist(),v31_full=new.mean(0).tolist(),best_simple=best,params=simple[best]['params'],
                          simple_filtered={k:[v['filtered']['recall20'],v['filtered']['ndcg20']] for k,v in simple.items()},
                          simple_full={k:[v['full']['recall20'],v['full']['ndcg20']] for k,v in simple.items()})
        for j,metric in ((1,'ndcg20'),(0,'recall20')):
            records.append(dict(paired_comparison(new[keep,j],other[keep,j],clean,bootstrap=2000,permutations=10000,seed=20261108),dataset=name,metric=metric,first='v31',second=f'simple:{best}'))
    holm_adjust(records,len(records))
    write_json(Path('results/simple_baselines/comparison.json'),dict(datasets=report,primary=records))
    for r in records:print(r['dataset'],r['metric'],r['second'],round(r['delta'],4),[round(x,4) for x in r['ci95']],r['p_holm'])


if __name__=='__main__':
    main()
