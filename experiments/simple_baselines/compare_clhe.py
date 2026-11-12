"""DIALS (v3.1, frozen, three seeds) against CLHE (three seeds, Kaggle GPU) on POG, POG_DENSE, Spotify_sparse, Spotify.

CLHE: its own code and shipped settings, trained on split files byte-identical to ours (SHA-256 recorded by the
kernel and checked here); per-query test metrics keyed by bundle id, from bundle-dpo/clhe_seeds_out. Primary: filtered
test queries, NDCG@20 and Recall@20, paired cluster bootstrap and sign flips, Holm over all available tests.

PYTHONPATH=. .venv/bin/python -m experiments.simple_baselines.compare_clhe
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.neural_lowrank import run as base

DATASETS=['pog_dense','pog','spotify_sparse','spotify']
OUT=Path(DATA_ROOT)/'clhe_seeds_out'
DEVIATIONS={'spotify':'batch_size_train and batch_size_test 128 instead of the preset 256: at 256 the 16 GB T4 runs out of memory at step 1'}


def clhe(name):
    folder=OUT/name.replace('_','-')/'clhe_out'/name
    if not folder.exists():return None
    for f,h in json.loads((folder/'split_sha256.json').read_text()).items():
        local=Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/f
        if any(t in f for t in ('train','valid','test')) and local.exists() and hashlib.sha256(local.read_bytes()).hexdigest()!=h:
            raise RuntimeError(f'{name}: {f} differs from our split')
    runs=[json.loads(p.read_text()) for p in sorted(folder.glob('seed*.json'))]
    if len(runs)!=3:return None
    return runs


def main():
    report={};records=[]
    for name in DATASETS:
        runs=clhe(name)
        if runs is None:continue
        ds,_,_,_,audit=base.data(name);keep=audit['clean_test_indices']
        order=[int(l.split(', ')[0]) for l in open(Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/'bi_test_input.txt')]
        seeds={int(l.split(', ')[0]):sorted(map(int,l.strip().split(', ')[1:])) for l in open(Path(DATA_ROOT)/'bundlemllm'/'datasets'/name/'bi_test_input.txt')}
        if len(order)!=len(ds.test_pairs) or any(sorted(set(s))!=sorted(set(seeds[b])) for b,(s,_) in zip(order,ds.test_pairs)):
            raise RuntimeError(f'{name}: test order differs from the file')
        per={b:i for i,b in enumerate(order)}
        other=np.zeros((len(order),2))
        for r in runs:
            m=np.full((len(order),2),np.nan)
            for b,rec,nd in r['queries']:m[per[b]]=(rec,nd)
            if np.isnan(m).any():raise RuntimeError(f'{name}: CLHE seed {r["seed"]} misses queries')
            other+=m/len(runs)
        dials_runs=[json.loads((Path('results/neural_relations')/name/f'v31_loo_seed{s}'/'test.json').read_text()) for s in (0,1,2)]
        dials=np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in dials_runs],axis=0)
        ids=base.groups(ds.test_pairs);clean=[ids[i] for i in keep]
        report[name]=dict(clhe_seeds=[dict(seed=r['seed'],epoch=r['epoch'],test=r['test']) for r in runs],parameters=runs[0]['parameters'],deviation=DEVIATIONS.get(name),
                          clhe_full=other.mean(0).tolist(),clhe_filtered=other[keep].mean(0).tolist(),
                          dials_full=dials.mean(0).tolist(),dials_filtered=dials[keep].mean(0).tolist())
        for j,metric in ((1,'ndcg20'),(0,'recall20')):
            records.append(dict(paired_comparison(dials[keep,j],other[keep,j],clean,bootstrap=2000,permutations=10000,seed=20261109),
                                dataset=name,metric=metric,first='dials',second='clhe'))
    holm_adjust(records,len(records))
    write_json(Path('results/simple_baselines/clhe_comparison.json'),dict(note=__doc__,datasets=report,primary=records))
    for name,r in report.items():
        print(name,'filtered DIALS',np.round(r['dials_filtered'],4),'CLHE',np.round(r['clhe_filtered'],4),'| full DIALS',np.round(r['dials_full'],4),'CLHE',np.round(r['clhe_full'],4))
    for r in records:print(r['dataset'],r['metric'],round(r['delta'],4),[round(x,4) for x in r['ci95']],r['p_holm'])


if __name__=='__main__':
    main()
