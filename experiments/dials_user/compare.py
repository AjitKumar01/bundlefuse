"""DIALS + U against frozen DIALS on POG, clean test queries, 3-seed means; paired cluster bootstrap, Holm over 2 tests.
Also reports the learned U parameters and the mean gate weights on clean test queries.

PYTHONPATH=.:../dials .venv/bin/python -m experiments.dials_user.compare
"""
import json,numpy as np,torch
from pathlib import Path
from dials import data as dials_data
from dials import train as dials_train
from bundlefuse.provenance import write_json
from experiments.dials_frozen.compare import RUNS,CACHE,groups
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.dials_user.model import DIALSUser,user_households,history_matrix
from experiments.dials_user.train import RUNS as USER_RUNS


def per_query(runs,keep):
    return np.mean([[[r['recall20'],r['ndcg20']] for r in np.array(json.loads((p/'test.json').read_text())['queries'],dtype=object)[keep]] for p in runs],axis=0)


def main(name):
    ds=dials_data.load(name,CACHE);keep=sorted(ds.clean_test);cl=[groups(ds.test_pairs)[i] for i in keep]
    new=per_query([USER_RUNS/name/f'seed{s}' for s in (0,1,2)],keep);old=per_query([RUNS/name/f'seed{s}' for s in (0,1,2)],keep)
    recs=[dict(metric=mt,**paired_comparison(new[:,j],old[:,j],cl,seed=20261109)) for j,mt in enumerate(('recall20','ndcg20'))];holm_adjust(recs)
    print('clean test  DIALS R %.4f N %.4f | DIALS+U R %.4f N %.4f'%(*old.mean(0),*new.mean(0)))
    for r in recs:print(r['metric'],'%+.4f [%+.4f, %+.4f] p_holm %.4f'%(r['delta'],*r['ci95'],r['p_holm']))
    H=user_households(name,ds);M=history_matrix([set(c) for c in H[0]],ds.n_items);H=H if H[1] is not None else None
    def build(ds,cache,config=dials_train.CONFIG):
        return DIALSUser(ds.features,ds.levels,ds.train,dials_train.spectral(ds,cache,config),M,rank=config['rank'],hidden=config['hidden'],households=H)
    dials_train.build=build;learned=[];gates=[]
    with torch.inference_mode():
        for s in (0,1,2):
            m=dials_train.load(ds,USER_RUNS/name/f'seed{s}',CACHE)
            learned.append(dict(theta_u=float(m.user_theta),alpha_u=float(m.user_exponents[0]),beta_u=float(m.user_exponents[1])))
            seeds=[torch.as_tensor(sorted(set(ds.test_pairs[i][0])),dtype=torch.long) for i in keep]
            gates.append(m.gate_weights(seeds,None).mean(0).tolist())
    print('learned',learned);print('mean gate [content, label, graph, users]',np.round(np.mean(gates,0),3))
    sel=[json.loads((USER_RUNS/name/f'seed{s}'/'selected.json').read_text())['best_epoch'] for s in (0,1,2)];print('best epochs',sel)
    write_json(Path('results/dials_user')/f'comparison_{name}.json',dict(note=__doc__,dials=old.mean(0).tolist(),dials_user=new.mean(0).tolist(),
               tests=recs,learned=learned,gate=np.mean(gates,0).tolist(),best_epochs=sel))


if __name__=='__main__':
    import sys;main(sys.argv[1] if len(sys.argv)>1 else 'pog')
