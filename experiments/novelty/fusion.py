"""Matched combiners: validation-selected sums/gates, rank fusion and trained rankers."""
import json
import time
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from bundlefuse.fusion import ValidationSignals, fit_fusion
from bundlefuse.metrics import ranking_metrics, top_items
from bundlefuse.provenance import write_json
from .protocol import PLAN, SEED


class BankValidation(ValidationSignals):
    def __init__(self,bank,pairs,indices,freq,chunk,log):
        self.bank,self.indices,self.chunk=bank,indices,chunk
        super().__init__(self.blocks,[pairs[i] for i in indices],freq,progress=log)
    def blocks(self):
        for offset in range(0,len(self.indices),self.chunk):
            ix=self.indices[offset:offset+self.chunk]
            data={k:torch.from_numpy(np.stack([self.bank.query(i)[k] for i in ix],1)) for k in self.bank.keys}
            yield offset,data


def canonical_weights(recipe):
    w={('hist' if k=='simp' else k):v for k,v in recipe['weights'].items() if v}
    if recipe['higher_order'][1]:w['rao']=recipe['higher_order'][1]
    return w


def prior_weights(weights,prior):
    return {k:weights.get(k,.5) for k in prior if weights.get(k,.5)}


def features(z,keys,frequency,mode,kappa):
    x=np.stack([z[k] for k in keys],1).astype(np.float32)
    if mode=='frequency_linear':
        rho=frequency/(frequency+kappa)
        x=np.concatenate([x,x*rho[:,None],np.log1p(frequency)[:,None]],1)
    elif mode=='lambdamart':
        x=np.concatenate([x,np.log1p(frequency)[:,None]],1)
    return x.astype(np.float32)


def sample_training(bank,pairs,indices,freq,seed):
    """All positives, same uniform/hard negatives, one grouped row per candidate."""
    rng=np.random.default_rng(seed);xs=[];ys=[];groups=[]
    for ix in indices:
        seeds,gt=pairs[ix];z=bank.query(ix)
        forbidden=set(seeds)|set(gt)
        negatives=set()
        for key in bank.keys:
            negatives.update(int(i) for i in top_items(z[key],seeds,PLAN['hard_per_source']) if i not in forbidden)
        pool=np.setdiff1d(np.arange(bank.n),np.asarray(sorted(forbidden)),assume_unique=True)
        negatives.update(rng.choice(pool,min(PLAN['random_negatives'],len(pool)),replace=False).tolist())
        ids=np.asarray(sorted(set(gt))+sorted(negatives),int)
        xs.append(np.concatenate([np.stack([z[k][ids] for k in bank.keys],1),freq[ids,None]],1))
        ys.append(np.r_[np.ones(len(set(gt))),np.zeros(len(negatives))]);groups.append(len(ids))
    return np.concatenate(xs).astype(np.float32),np.concatenate(ys).astype(np.float32),np.asarray(groups)


def train_linear(x,y,groups,regularization,kappa,seed):
    """Pairwise logistic ranker on [z; rho*z; log1p(f)], no target in inputs."""
    z=x[:,:-1];f=x[:,-1];rho=f/(f+kappa)
    phi=np.concatenate([z,rho[:,None]*z,np.log1p(f)[:,None]],1).astype(np.float64)
    differences=[];offset=0;rng=np.random.default_rng(seed)
    for count in groups:
        rows=phi[offset:offset+count];labels=y[offset:offset+count]
        pos=rows[labels==1];neg=rows[labels==0]
        for p in pos:
            choose=rng.choice(len(neg),min(16,len(neg)),replace=False)
            differences.append(p-neg[choose])
        offset+=count
    d=np.concatenate(differences);scale=np.maximum(d.std(0),1.)
    d/=scale
    def objective(w):
        margin=d@w
        value=np.logaddexp(0,-margin).mean()+.5*regularization*np.dot(w,w)
        grad=-(d.T@expit(-margin))/len(d)+regularization*w
        return value,grad
    fitted=minimize(objective,np.zeros(d.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':200})
    if not fitted.success:raise RuntimeError(f'Linear ranker failed convergence: {fitted.message}')
    return dict(kind='frequency_linear',coefficients=(fitted.x/scale).tolist(),kappa=kappa,
                regularization=regularization,iterations=int(fitted.nit),parameters=len(fitted.x),
                training_pairs=len(d),objective=float(fitted.fun))


def score(z,method,keys,freq,model=None,seeds=()):
    kind=method['kind']
    if kind in ('equal','weighted','bundlefuse_gate','canonical'):
        weights=method['weights'];y=sum(v*z[k] for k,v in weights.items())
        if kind in ('bundlefuse_gate','canonical') and method['kappa']>0:
            p=sum(v*z[k] for k,v in method['prior_weights'].items())
            rho=freq/(freq+method['kappa'])
            return rho*y+(1-rho)*method['scale']*p
        return np.asarray(y)
    if kind=='rrf':
        result=np.zeros(len(freq),np.float64);ids=np.arange(len(freq))
        for k in keys:
            eligible=ids[~np.isin(ids,seeds)]
            order=eligible[np.lexsort((eligible,-z[k][eligible]))]
            result[order]+=1/(method['offset']+np.arange(1,len(order)+1))
        return result
    phi=features(z,keys,freq,kind,method.get('kappa',1.))
    if kind=='frequency_linear':return phi@np.asarray(method['coefficients'])
    if kind=='lambdamart':return model.predict(phi)
    raise ValueError(kind)


def evaluate_method(bank,pairs,indices,method,freq,model=None):
    totals=np.zeros(2)
    for i in indices:
        values=score(bank.query(i),method,bank.keys,freq,model,seeds=pairs[i][0])
        totals+=ranking_metrics(values[:,None],[pairs[i]],Ks=(20,))[20]
    return dict(zip(['recall20','ndcg20'],(totals/len(indices)).tolist()))


def fit_bank(bank,pairs,fit,tune,freq,source_recipe,artifacts,output,log):
    import xgboost as xgb
    artifacts.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    done=output/'selected.json'
    if done.exists():return json.loads(done.read_text())
    begun=time.monotonic();candidates=[]
    val=BankValidation(bank,pairs,fit,freq,PLAN['chunk'],log)
    basepath=output/'base_weights.json'
    if basepath.exists():
        base=json.loads(basepath.read_text());w,prior=base['weights'],base['prior']
    else:
        w,prior,_,_,_=fit_fusion(val,bank.keys,[k for k in source_recipe['prior'] if k in bank.keys],
                                 incidence=(0.,),rao=(0.,),shrinkage=False,rounds=4)
        write_json(basepath,dict(weights=w,prior=prior,coordinate_candidates=len(val.memo)))
    candidates.append(dict(kind='equal',weights={k:1. for k in bank.keys}))
    candidates.append(dict(kind='weighted',weights=w,parameters=len(w)))
    prior=list(prior)
    if w.get('pop',0)>0 and 'pop' not in prior:prior.append('pop')
    if w.get('rao',0)>0 and 'rao' not in prior:prior.append('rao')
    pw=prior_weights(w,prior)
    if not pw:raise RuntimeError('No fallback source available')
    sums=np.zeros(4);count=0
    for _,z in val.blocks():
        y=sum(v*z[k].numpy() for k,v in w.items());p=sum(v*z[k].numpy() for k,v in pw.items())
        sums+=np.array([y.sum(dtype=np.float64),np.square(y,dtype=np.float64).sum(),
                       p.sum(dtype=np.float64),np.square(p,dtype=np.float64).sum()]);count+=y.size
    std_y=np.sqrt(max(0,(sums[1]-sums[0]**2/count)/(count-1)))
    std_p=np.sqrt(max(0,(sums[3]-sums[2]**2/count)/(count-1)))
    matched=float((std_y+1e-9)/(std_p+1e-9))
    for scale in dict.fromkeys([1.,matched]):
        for kappa in PLAN['kappa']:
            candidates.append(dict(kind='bundlefuse_gate',weights=w,prior_weights=pw,kappa=kappa,
                                   scale=scale,parameters=0,base_weight_count=len(w)))
    for offset in PLAN['rrf_offsets']:candidates.append(dict(kind='rrf',offset=offset,parameters=0))
    samplepath=artifacts/'training_sample.npz'
    if samplepath.exists():
        saved=np.load(samplepath);x,y,groups=saved['x'],saved['y'],saved['groups']
    else:
        x,y,groups=sample_training(bank,pairs,fit,freq,SEED)
        np.savez_compressed(samplepath,x=x,y=y,groups=groups)
    base_kappa=max(source_recipe['backoff'][0],.05)
    log(f'{bank.kind}: training sampled rows={len(y)} groups={len(groups)} columns={len(bank.keys)}')
    for regularization in PLAN['linear_regularization']:
        method=train_linear(x,y,groups,regularization,base_kappa,SEED);candidates.append(method)
    xf=np.concatenate([x[:,:-1],np.log1p(x[:,-1,None])],1)
    for trees in PLAN['ranker_trees']:
        for depth in PLAN['ranker_depth']:
            path=artifacts/f'lambdamart_{trees}_{depth}.json'
            model=xgb.XGBRanker(objective='rank:ndcg',eval_metric='ndcg@20',n_estimators=trees,
                  max_depth=depth,learning_rate=.05,reg_lambda=1.,n_jobs=1,tree_method='hist',
                  lambdarank_pair_method='topk',lambdarank_num_pair_per_sample=20,
                  random_state=SEED)
            if path.exists():model.load_model(path)
            else:model.fit(xf,y,group=groups);model.save_model(path)
            candidates.append(dict(kind='lambdamart',trees=trees,depth=depth,model_path=str(path),
                                   trained_trees=len(model.get_booster().get_dump())))
    records=[];selected={}
    progress=output/'tuning_progress.json'
    previous=json.loads(progress.read_text())['candidates'] if progress.exists() else []
    for j,method in enumerate(candidates):
        model=None
        if method['kind']=='lambdamart':
            model=xgb.XGBRanker();model.load_model(method['model_path'])
        if j<len(previous):
            if previous[j]['method']!=method:raise RuntimeError('Resumed candidate differs')
            metrics=previous[j]['metrics']
        else:
            metrics=evaluate_method(bank,pairs,tune,method,freq,model)
        record=dict(method=method,metrics=metrics);records.append(record)
        old=selected.get(method['kind'])
        if old is None or metrics['ndcg20']>old['tuning_metrics']['ndcg20']:
            selected[method['kind']]=dict(method,tuning_metrics=metrics)
        log(f'{bank.kind} tuning candidate {j+1}/{len(candidates)} {method["kind"]}: {metrics}')
        write_json(output/'tuning_progress.json',dict(candidates=records))
    write_json(output/'fit.json',dict(fit_queries=len(fit),tuning_queries=len(tune),sampled_rows=len(y),
          positives=int(y.sum()),keys=bank.keys,coordinate_candidates=len(val.memo),
          seconds=time.monotonic()-begun,source_weight_grid=[0,.25,.5,1.,1.5]))
    write_json(done,selected)
    return selected
