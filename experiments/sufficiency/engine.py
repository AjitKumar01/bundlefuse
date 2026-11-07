"""Explicit banks, legal fallbacks, scalar searches and support corrections."""
import json
import time
from pathlib import Path
import numpy as np
from bundlefuse.metrics import ranking_metrics, top_items
from bundlefuse.provenance import write_json

GRID = (0., .25, .5, 1., 1.5)
KAPPAS = (0., .05, .5, 1., 5., 20.)
MIX = (0., .05, .25, .5, .75, 1.)


class Bank:
    def __init__(self, arrays, keys, history='simp', alpha=1.):
        self.arrays, self.keys, self.history, self.alpha = arrays, list(keys), history, alpha

    def query(self, i):
        z = {}
        for k in self.keys:
            if k == 'hist':
                if self.history!='canonical':z[k]=self.arrays[self.history][i]
                elif self.alpha==0:z[k]=self.arrays['pair'][i]
                elif self.alpha==1:z[k]=self.arrays['simp'][i]
                else:z[k]=(1-self.alpha)*self.arrays['pair'][i]+self.alpha*self.arrays['simp'][i]
            else:
                z[k] = self.arrays[k][i]
        return z

    def aux(self, i):
        prefix='amp_' if self.history=='amp_selected' else ''
        return {k:self.arrays[prefix+k][i] for k in ('ess','count') if prefix+k in self.arrays}


def rho(support, kappa):
    return np.ones_like(support) if kappa == 0 else support/(support+kappa)


def score(z, cfg, freq, aux=None):
    y = sum(v*z[k] for k, v in cfg['weights'].items() if v)
    if cfg['kind'] == 'weighted':
        return np.asarray(y)
    p = sum(v*z[k] for k, v in cfg['prior_weights'].items() if v)*cfg['scale']
    if cfg['kind'] == 'constant_full':
        return (1-cfg['parameter'])*y+cfg['parameter']*p
    rf = rho(freq, cfg['kappa'])
    if cfg['kind'] == 'gate':
        return p+rf*(y-p)
    h = cfg['weights'].get('hist', 0.)*z['hist']
    mode = cfg['kind']
    support = freq if mode in ('history_frequency','history_constant') else aux[mode]
    rq = np.full_like(freq, cfg['parameter']) if mode == 'history_constant' else rho(support, cfg['parameter'])
    return p+rf*(y-p-h)+rq*h


def metrics(values, pair):
    return np.asarray(ranking_metrics(values[:, None], [pair], Ks=(20,))[20])


class Search:
    """Persist evaluations; replay each query once for a coordinate candidate batch."""
    def __init__(self, bank, pairs, indices, freq, path, log):
        self.bank, self.pairs, self.indices, self.freq = bank, pairs, indices, freq
        self.path, self.log = Path(path), log
        self.memo = json.loads(self.path.read_text()) if self.path.exists() else {}

    def many(self, configs):
        keys = [json.dumps(c, sort_keys=True) for c in configs]
        pending = dict((k,c) for k,c in zip(keys, configs) if k not in self.memo)
        sums = {k: np.zeros(2) for k in pending}
        if pending:
            for i in self.indices:
                z = self.bank.query(i)
                aux = self.bank.aux(i)
                for key, cfg in pending.items():
                    sums[key] += metrics(score(z,cfg,self.freq,aux),self.pairs[i])
            for key in pending:
                self.memo[key] = (sums[key]/len(self.indices)).tolist()
            write_json(self.path, self.memo)
            self.log(f'{self.path.parent.name}: evaluations retained={len(self.memo)}')
        return [self.memo[k] for k in keys]


def fit_weights(search, initial=None, fixed=None, variable=None, rounds=4):
    keys = search.bank.keys
    w = {k: 0. for k in keys}; w.update(fixed or {})
    starts = [dict(w, **{k:v for k,v in x.items() if k in keys}) for x in (initial or [])]
    starts += [dict(w, **{keys[0]: .25})] if not any(w.values()) else [w]
    configs = [dict(kind='weighted',weights=x) for x in starts if any(x.values())]
    best_cfg = configs[int(np.argmax([v[1] for v in search.many(configs)]))]
    w = best_cfg['weights'].copy()
    best = search.many([best_cfg])[0][1]
    for _ in range(rounds):
        changed = False
        for key in (keys if variable is None else variable):
            grid = [dict(kind='weighted',weights=dict(w, **{key:v})) for v in GRID]
            grid = [c for c in grid if any(c['weights'].values())]
            vals = search.many(grid)
            j = int(np.argmax([x[1] for x in vals]))
            if vals[j][1] > best+2e-4:
                w, best, changed = grid[j]['weights'], vals[j][1], True
        if not changed:
            break
    return {k:v for k,v in w.items() if v}


def prior_weights(w, source):
    allowed = set(source['prior']) | {'pop','rao'}
    return {k:v for k,v in w.items() if k in allowed and v}


def matched_scale(bank, indices, weights, prior):
    moments = np.zeros(4); n=0
    for i in indices:
        z=bank.query(i)
        y=sum(v*z[k] for k,v in weights.items())
        p=sum(v*z[k] for k,v in prior.items())
        moments += [y.sum(dtype=np.float64),np.square(y,dtype=np.float64).sum(),
                    p.sum(dtype=np.float64),np.square(p,dtype=np.float64).sum()]
        n += len(y)
    vy=max(0,moments[1]/n-(moments[0]/n)**2)
    vp=max(0,moments[3]/n-(moments[2]/n)**2)
    return float((np.sqrt(vy)+1e-9)/(np.sqrt(vp)+1e-9))


def fit(bank, pairs, fitting, tuning, freq, source, directory, log, initial=None, fixed=None, variable=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    final=directory/'selected.json'
    if final.exists():
        return json.loads(final.read_text())
    start=time.monotonic()
    search=Search(bank,pairs,fitting,freq,directory/'fitting_evaluations.json',log)
    w=fit_weights(search,initial,fixed,variable)
    pw=prior_weights(w,source)
    configs=[dict(kind='weighted',weights=w)]
    if pw:
        for scale in dict.fromkeys([1.,matched_scale(bank,fitting,w,pw)]):
            configs += [dict(kind='gate',weights=w,prior_weights=pw,scale=scale,kappa=k) for k in KAPPAS]
    tune=Search(bank,pairs,tuning,freq,directory/'tuning_evaluations.json',log)
    vals=tune.many(configs); j=int(np.argmax([x[1] for x in vals]))
    result=dict(config=configs[j],tuning_metrics=vals[j],keys=bank.keys,history=bank.history,
                alpha=bank.alpha,seconds=time.monotonic()-start,fitting_candidates=len(search.memo))
    write_json(final,result)
    return result


def assess(bank,pairs,cfg,freq,path,log):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return json.loads(path.read_text())
    progress=path.with_suffix('.progress.json')
    rows=json.loads(progress.read_text()) if progress.exists() else dict(metrics=[],predictions=[],groups=[],seed_counts=[])
    for i in range(len(rows['metrics']),len(pairs)):
        z=bank.query(i);aux=bank.aux(i)
        values=score(z,cfg,freq,aux)
        rows['metrics'].append(metrics(values,pairs[i]).tolist())
        rows['predictions'].append(top_items(values,pairs[i][0],20).tolist())
        rows['groups'].append(str(tuple(sorted(set(pairs[i][0])|set(pairs[i][1])))))
        rows['seed_counts'].append(len(pairs[i][0]))
        if (i+1)%100==0:
            write_json(progress,rows);log(f'{path.stem}: {i+1}/{len(pairs)}')
    rows['mean']=np.mean(rows['metrics'],axis=0).tolist()
    write_json(path,rows);log(f'{path.stem} COMPLETE {rows["mean"]}')
    return rows
