"""Small matched weighted, frequency-dependent and constant-mixture fits."""
import json
import numpy as np
from bundlefuse.fusion import fit_fusion
from bundlefuse.metrics import ranking_metrics
from bundlefuse.provenance import write_json
from experiments.novelty.fusion import BankValidation
from .protocol import PLAN


def score(z, method, freq):
    y=sum(v*z[k] for k,v in method['weights'].items())
    if method['kind']=='weighted': return y
    p=sum(v*z[k] for k,v in method['prior_weights'].items())*method['scale']
    if method['kind']=='constant':
        u=method['mixture'];return (1-u)*y+u*p
    kappa=method['kappa']
    if kappa==0:return y
    rho=freq/(freq+kappa)
    return rho*y+(1-rho)*p


def fit(bank,pairs,fit_indices,tune,freq,source,directory,log):
    directory.mkdir(parents=True,exist_ok=True)
    final=directory/'selected.json'
    if final.exists():return json.loads(final.read_text())
    validation=BankValidation(bank,pairs,fit_indices,freq,PLAN['chunk'],log)
    basepath=directory/'base_weights.json'
    if basepath.exists():base=json.loads(basepath.read_text());w,prior=base['weights'],base['prior']
    else:
        w,prior,_,_,_=fit_fusion(validation,bank.keys,[k for k in source['prior'] if k in bank.keys],
            incidence=(0.,),rao=(0.,),shrinkage=False,rounds=PLAN['rounds'])
        write_json(basepath,dict(weights=w,prior=prior,coordinate_candidates=len(validation.memo)))
    prior=list(prior)
    for k in ('pop','rao'):
        if w.get(k,0)>0 and k not in prior:prior.append(k)
    pw={k:w.get(k,.5) for k in prior if w.get(k,.5)}
    if not pw:raise RuntimeError('No fallback score is available')
    moments=np.zeros(4);count=0
    for _,z in validation.blocks():
        y=sum(v*z[k].numpy() for k,v in w.items());p=sum(v*z[k].numpy() for k,v in pw.items())
        moments+=np.array([y.sum(dtype=np.float64),np.square(y,dtype=np.float64).sum(),
                           p.sum(dtype=np.float64),np.square(p,dtype=np.float64).sum()]);count+=y.size
    std_y=np.sqrt(max(0,(moments[1]-moments[0]**2/count)/(count-1)))
    std_p=np.sqrt(max(0,(moments[3]-moments[2]**2/count)/(count-1)))
    scales=list(dict.fromkeys([1.,float((std_y+1e-9)/(std_p+1e-9))]))
    candidates=[dict(kind='weighted',weights=w)]
    for scale in scales:
        for kappa in PLAN['kappa']:
            candidates.append(dict(kind='gate',weights=w,prior_weights=pw,scale=scale,kappa=kappa))
        for mixture in PLAN['mixture']:
            candidates.append(dict(kind='constant',weights=w,prior_weights=pw,scale=scale,mixture=mixture))
    progress=directory/'tuning_progress.json'
    records=json.loads(progress.read_text()) if progress.exists() else []
    selected={}
    for j,method in enumerate(candidates):
        if j<len(records):
            if records[j]['method']!=method:raise RuntimeError('Resumed tuning method differs')
            metrics=records[j]['metrics']
        else:
            totals=np.zeros(2)
            for i in tune:
                values=score(bank.query(i),method,freq)
                totals+=ranking_metrics(values[:,None],[pairs[i]],Ks=(20,))[20]
            metrics=dict(zip(['recall20','ndcg20'],(totals/len(tune)).tolist()))
            records.append(dict(method=method,metrics=metrics));write_json(progress,records)
        if method['kind'] not in selected or metrics['ndcg20']>selected[method['kind']]['tuning_metrics']['ndcg20']:
            selected[method['kind']]=dict(method,tuning_metrics=metrics)
        log(f'{bank.kind} tuning {j+1}/{len(candidates)} {method["kind"]}: {metrics}')
    write_json(final,selected)
    return selected
