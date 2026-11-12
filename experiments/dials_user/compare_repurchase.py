"""Personalized Dunnhumby: DIALS + R (trained) against every method given the same repurchase signal.

Families, each Holm-corrected over its own tests (Recall@20 and NDCG@20, paired cluster bootstrap and sign flips):
  full histories:            DIALS + R vs best baseline + R (by validation), repurchase only, frozen DIALS + R at
                             inference, and non-personalized DIALS;
  time-restricted histories: the same comparisons, with histories from baskets bought before the query basket.
Also: the learned curve rho_R(log(1 + n), log(1 + B)) - rho_R(0, log(1 + B)) and the mean gate weights on test.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_user.compare_repurchase
"""
import dials  # pins BLAS threads before numpy loads
import json
import numpy as np
import torch
from pathlib import Path
from dials import data as dials_data
from dials import train as dials_train
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS,CACHE,groups
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.dials_user.train_repurchase import RUNS as R_RUNS,setup
from experiments.dials_user.baselines_repurchase import OUT as BASE_OUT


def queries(path):return np.array([[r['recall20'],r['ndcg20']] for r in json.loads(path.read_text())['queries']])


def main():
    ds=dials_data.load('dunnhumby',CACHE);cl=groups(ds.test_pairs);history=setup(ds)
    runs=[R_RUNS/f'seed{s}' for s in (0,1,2)]
    ours={'full':np.mean([queries(r/'test.json') for r in runs],0),'before':np.mean([queries(r/'test_before.json') for r in runs],0)}
    base=json.loads((BASE_OUT/'summary.json').read_text())['methods']
    plain=np.mean([queries(RUNS/'dunnhumby'/f'seed{s}'/'test.json') for s in (0,1,2)],0)
    simple=[k for k in base if k.endswith(' + R') and not k.startswith('DIALS')];best=max(simple,key=lambda k:base[k]['validation_ndcg20'])
    others={'best baseline + R ('+best+')':best,'repurchase only':'repurchase only','frozen DIALS + R at inference':'DIALS (frozen) + R at inference'}
    report=dict(note=__doc__,means={},tests={})
    for split,key in (('full','queries'),('before','queries_before')):
        recs=[]
        report['means'][split]={'DIALS + R (trained)':ours[split].mean(0).tolist(),**{k:np.mean(base[v][key],0).tolist() for k,v in others.items()}}
        comparisons=[(k,np.asarray(base[v][key])) for k,v in others.items()]+([('non-personalized DIALS',plain)] if split=='full' else [])
        for name,other in comparisons:
            for j,mt in enumerate(('recall20','ndcg20')):
                recs.append(dict(comparison='DIALS + R vs '+name,metric=mt,**paired_comparison(ours[split][:,j],other[:,j],cl,seed=20261109)))
        holm_adjust(recs);report['tests'][split]=recs
        print(f'== {split} histories');print(f"  {'DIALS + R (trained)':42s} R {ours[split][:,0].mean():.4f} N {ours[split][:,1].mean():.4f}")
        for k,v in others.items():m=np.mean(base[v][key],0);print(f'  {k:42s} R {m[0]:.4f} N {m[1]:.4f}')
        if split=='full':print(f"  {'non-personalized DIALS':42s} R {plain[:,0].mean():.4f} N {plain[:,1].mean():.4f}")
        for r in recs:print(f"  {r['comparison']:60s} {r['metric']:8s} {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p_holm {r['p_holm']:.4f}")
    curves=[];gates=[]
    with torch.inference_mode():
        for run in runs:
            m=dials_train.load(ds,run,CACHE);grid=torch.tensor([[np.log1p(n),np.log1p(B)] for B in (4,16,64) for n in (1,2,4,8)],dtype=torch.float32)
            zero=grid.clone();zero[:,0]=0;curves.append((m.repurchase(grid)-m.repurchase(zero)).squeeze(1).tolist())
            hist=[history.query(u) for u,_ in history.split.test];seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s,_ in ds.test_pairs]
            gates.append(m.gate_weights(seeds,None,hist).mean(0).tolist())
    report['curve']=dict(points=[dict(baskets=B,count=n) for B in (4,16,64) for n in (1,2,4,8)],value=np.mean(curves,0).tolist())
    report['gate']=np.mean(gates,0).tolist()
    print('learned R(n | B) [B=4: n=1,2,4,8 | B=16 | B=64]',np.round(np.mean(curves,0),2))
    print('mean gate [content, label, graph, repurchase]',np.round(np.mean(gates,0),3))
    print('best epochs',[json.loads((r/'selected.json').read_text())['best_epoch'] for r in runs])
    write_json(Path('results/dials_user')/'comparison_dunnhumby_repurchase.json',report)


if __name__=='__main__':main()
