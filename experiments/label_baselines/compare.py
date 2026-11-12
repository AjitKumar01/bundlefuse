"""DIALS (frozen, 3 seeds) against the best label-aware simple baseline per dataset (chosen on validation NDCG@20
among samelabel, transition, label_knn, label_knn_content), clean test queries, paired cluster bootstrap and sign
flips, Holm over the 8 tests (4 datasets x NDCG@20, Recall@20).

PYTHONPATH=.:../dials .venv/bin/python -m experiments.label_baselines.compare
"""
import dials
import json
from pathlib import Path
import numpy as np
from dials import data as dials_data
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import holm_adjust
from experiments.dials_frozen.compare import per_query,groups,compare,CACHE

def main():
    records=[];report=dict(note=__doc__,datasets={})
    for name in ('pog_dense','pog','spotify_sparse','spotify'):
        ds=dials_data.load(name,CACHE);keep=ds.clean_test;clusters=[groups(ds.test_pairs)[i] for i in keep]
        mine=per_query(name);lab=json.loads(Path(f'results/label_baselines/{name}/summary.json').read_text())['test']
        best=max(lab,key=lambda m:lab[m]['validation_ndcg20']);other=np.asarray(lab[best]['queries'])
        report['datasets'][name]=dict(dials=mine[keep].mean(0).tolist(),label_baseline=dict(method=best,params=lab[best]['params'],test=other[keep].mean(0).tolist()))
        compare(records,'label',name,mine[keep],other[keep],clusters,f'label:{best}')
    holm_adjust(records,8);report['primary']=records;write_json(Path('results/label_baselines/comparison.json'),report)
    for r in records:print(r['dataset'],r['metric'],r['second'],round(r['delta'],4),[round(x,4) for x in r['ci95']],round(r['p_holm'],4))

if __name__=='__main__':main()
