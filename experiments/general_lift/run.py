"""One estimator on every dataset: signed group lift from bundles, similarity within groups.

No dataset flags: menu, constants and selection are shared, and only fitted values differ.
Exploratory on already inspected splits. Every arm is frozen before test scores are computed.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from bundlefuse import datasets
from bundlefuse.metrics import top_items
from bundlefuse.model import BundleFuse
from bundlefuse.operators import SimplicialGraphOperator
from bundlefuse.provenance import write_json
from experiments.block_controls.run import curveball
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.novelty.data import validation_groups
from experiments.novelty.raw import AmplifiedNeighbors
from experiments.sufficiency.engine import Bank,fit,score,metrics

DATASETS=('pog','pog_dense','spotify','spotify_sparse','electronic','food','clothing','dunnhumby')
SEED=20261107
PARTITION_SEED=20261119
SHUFFLES=3
PSEUDOCOUNT=1.
HISTORY=['pair','simp','amp']
PLAN=dict(
    datasets=DATASETS,seed=SEED,partition_seed=PARTITION_SEED,shuffles=SHUFFLES,pseudocount=PSEUDOCOUNT,
    scope='Exploratory. Filtered queries: complete bundles overlapping training are excluded, and test bundles overlapping '
          'validation. Splits were inspected earlier. Retail and Dunnhumby use one seed-0 resplit with few test queries.',
    estimator='L_gh=log((O_gh+1)/(E_gh+1)), O ordered distinct-item pairs across bundles, '
              'E=P(F_g F_h-delta_gh sum_i_in_g f_i^2)/(F^2-sum_i f_i^2), P=sum_b d_b(d_b-1). Residual e_i=x_i-n_g/(n_g+1) mu_g on the coarsest level.',
    menu='lift per group level, residual per modality, popularity, pair, incidence and amplified (power 2, top 100) history',
    primary=['general vs raw','general vs shuffled mean'],metrics=['ndcg20','recall20'],holm_family=32,
    decision=dict(
        dataset='Passes if general beats raw and beats the shuffled mean on NDCG with Holm p<.05, and beats every shuffled seed.',
        theory='General only if every dataset passes. Any failure is reported as a counterexample.',
        cost='The gap to the current dataset-specific recipe is reported and does not decide the pass.'),
    recipe_note='Current recipes are refitted separately on this source with their own settings on all validation queries, not under this partition.')


def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)


def z(values):
    values=np.asarray(values,np.float32)
    return (values-values.mean())/(values.std(ddof=1)+1e-9)


def unit(x):
    return (x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)).astype(np.float32)


class GroupLift:
    """Signed log lift between groups, read one seed group at a time."""
    def __init__(self,rows,labels,n):
        self.labels=labels;self.groups=int(labels.max())+1
        data=[(b,i) for b,items in enumerate(rows) for i in items]
        bundle,item=np.asarray(data).T
        incidence=sp.csr_matrix((np.ones(len(item),np.float32),(bundle,item)),shape=(len(rows),n))
        membership=sp.csr_matrix((np.ones(n,np.float32),(np.arange(n),labels)),shape=(n,self.groups))
        counts=(incidence@membership).tocsr()
        self.F=np.asarray(counts.sum(0)).ravel().astype(np.float64)
        f=np.asarray(incidence.sum(0)).ravel().astype(np.float64)
        self.same=np.bincount(labels,weights=f*f,minlength=self.groups)
        O=(counts.T@counts).tocsr().astype(np.float64)
        O=(O-sp.diags(self.F)).tocsr();O.eliminate_zeros()
        self.O=O
        sizes=np.asarray(incidence.sum(1)).ravel().astype(np.float64)
        self.rate=float((sizes*(sizes-1)).sum())/max(float(self.F.sum()**2-(f*f).sum()),1.)

    def row(self,h):
        observed=np.zeros(self.groups);start,stop=self.O.indptr[h],self.O.indptr[h+1]
        observed[self.O.indices[start:stop]]=self.O.data[start:stop]
        expected=self.rate*self.F[h]*self.F
        expected[h]=self.rate*(self.F[h]**2-self.same[h])
        return np.log((observed+PSEUDOCOUNT)/(expected+PSEUDOCOUNT))

    def score(self,seeds):
        total=np.zeros(self.groups);off=np.zeros(self.groups);same=np.zeros(self.groups,np.float32)
        for h in self.labels[seeds]:
            row=self.row(h);total+=row;row[h]=0.;off+=row;same[h]+=1.
        return total[self.labels],off[self.labels],same[self.labels]

    def diagonal_audit(self):
        d=np.asarray([self.row(h)[h] for h in np.where(self.F>0)[0]]);F=self.F[self.F>0]
        return dict(weighted_mean=float((d*F).sum()/F.sum()),negative_share=float((d*F)[d<0].sum()/(d*F)[d!=0].sum()) if np.any(d!=0) else 0.,
                    occurrence_share_negative=float(F[d<0].sum()/F.sum()),occurrence_share_positive=float(F[d>0].sum()/F.sum()))


class Columns:
    def __init__(self,ds,train):
        n=ds.n_items;self.n=n
        self.graph=SimplicialGraphOperator(train,n_items=n,svd_dim=1)
        self.amp=AmplifiedNeighbors(self.graph)
        self.freq=np.asarray(self.graph.freq)
        levels=[np.asarray(ds.cat_labels,int)]
        if getattr(ds,'alb_labels',None) is not None:levels.append(np.asarray(ds.alb_labels,int))
        levels.sort(key=lambda labels:labels.max())
        self.levels=levels
        rows=[sorted(set(b)) for b in train]
        self.lift=[GroupLift(rows,labels,n) for labels in levels]
        self.nulls=[]
        for s in range(SHUFFLES):
            shuffled=curveball(train,SEED+s)
            if not np.array_equal(np.bincount(np.concatenate(shuffled),minlength=n),self.freq.astype(int)):
                raise RuntimeError('Curveball changed an item frequency')
            self.nulls.append([GroupLift(shuffled,labels,n) for labels in levels])
        coarse=levels[0];counts=np.bincount(coarse,minlength=int(coarse.max())+1).astype(np.float32)
        self.raw,self.residual={},{}
        for name,values in ds.features.items():
            x=unit(np.asarray(values,np.float32));self.raw[name]=x
            means=np.zeros((len(counts),x.shape[1]),np.float32);np.add.at(means,coarse,x)
            means/=(counts+1.)[:,None]              # n/(n+1) * group mean
            self.residual[name]=unit(x-means[coarse])
        self.pop=z(np.log1p(self.freq))
        self.keys=[f'lift{l}' for l in range(len(levels))]+[f'off{l}' for l in range(len(levels))]+[f'same{l}' for l in range(len(levels))]
        self.keys+=[f'residual_{m}' for m in self.residual]+[f'raw_{m}' for m in self.raw]+['pop']+HISTORY
        self.keys+=[f'shuf{s}_lift{l}' for s in range(SHUFFLES) for l in range(len(levels))]

    def query(self,seeds):
        seeds=np.asarray(list(seeds),int);out={'pop':self.pop}
        for l,lift in enumerate(self.lift):
            total,off,same=lift.score(seeds)
            out[f'lift{l}']=z(total);out[f'off{l}']=z(off);out[f'same{l}']=z(same)
            for s,null in enumerate(self.nulls):out[f'shuf{s}_lift{l}']=z(null[l].score(seeds)[0])
        for m in self.raw:
            out[f'raw_{m}']=z(self.raw[m]@self.raw[m][seeds].mean(0))
            out[f'residual_{m}']=z(self.residual[m]@self.residual[m][seeds].mean(0))
        g=self.graph;x=g.build_indicator([seeds.tolist()])
        overlap=np.asarray((g.B@x.multiply(g.inv_sqrt_f[:,None])).todense()).ravel()*g.inv_vol_pow
        count=np.asarray((g.B_bin@x).todense()).ravel()
        out['pair']=z(np.asarray(g.B.T@overlap).ravel())
        out['simp']=z(np.asarray(g.B.T@(overlap*(1+g.beta_clique*np.maximum(count-1,0)))).ravel())
        out['amp']=z(self.amp.score(seeds.tolist(),2,100))
        return out


def arms(columns):
    levels=range(len(columns.levels));base=['pop']+HISTORY
    lift=[f'lift{l}' for l in levels];residual=[f'residual_{m}' for m in columns.residual];raw=[f'raw_{m}' for m in columns.raw]
    out=dict(general=lift+residual+base,raw=raw+base,base=list(base))
    out.update({f'shuf{s}':[f'shuf{s}_lift{l}' for l in levels]+residual+base for s in range(SHUFFLES)})
    out.update(lift_only=lift+base,residual_only=residual+base,identity=[f'same{l}' for l in levels]+residual+base,
               off_diagonal=[f'off{l}' for l in levels]+residual+base,
               general_no_history=lift+residual+['pop'],raw_no_history=raw+['pop'])
    return out


def population(train,val,test):
    digest=lambda x:hashlib.sha256(np.asarray(sorted(set(x)),dtype='<i8').tobytes()).hexdigest()
    seen={digest(b) for b in train}
    vs=[digest(list(s)+list(t)) for s,t in val];ts=[digest(list(s)+list(t)) for s,t in test]
    held=set(vs)
    return [i for i,x in enumerate(vs) if x not in seen],[i for i,x in enumerate(ts) if x not in seen and x not in held]


def cache(columns,pairs,directory):
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'progress.json';done=json.loads(marker.read_text())['queries'] if marker.exists() else 0
    keys=[k for k in columns.keys if k!='pop']
    arrays={k:np.lib.format.open_memmap(directory/f'{k}.npy',mode='r+' if (directory/f'{k}.npy').exists() else 'w+',
                                        dtype=np.float32,shape=(len(pairs),columns.n)) for k in keys}
    for i in range(done,len(pairs)):
        row=columns.query(pairs[i][0])
        for k,a in arrays.items():a[i]=row[k]
        if (i+1)%50==0 or i+1==len(pairs):
            for a in arrays.values():a.flush()
            write_json(marker,dict(queries=i+1));log(f'validation columns {i+1}/{len(pairs)}')
    del arrays
    loaded={k:np.load(directory/f'{k}.npy',mmap_mode='r') for k in keys}
    loaded['pop']=np.broadcast_to(columns.pop,(len(pairs),columns.n))
    return loaded


def run(args):
    name=args.dataset;out=args.output/name;out.mkdir(parents=True,exist_ok=True);store=args.store/name
    protocol=dict(PLAN,dataset=name,smoke=args.smoke)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=json.loads(json.dumps(protocol)):
        raise RuntimeError('Protocol changed; use another output root')
    write_json(out/'protocol.json',protocol)
    ds=datasets.load(name,data_profile='research')
    train=[list(dict.fromkeys(b)) for b in (ds.train.values() if isinstance(ds.train,dict) else ds.train)]
    vi,ti=population(train,ds.val_pairs,ds.test_pairs)
    val=[ds.val_pairs[i] for i in vi];test=[ds.test_pairs[i] for i in ti]
    prior=Path('results/sufficiency')/name/'evaluation_population.json'
    if prior.exists():
        recorded=json.loads(prior.read_text())
        if recorded['validation_original_indices']!=vi or recorded['test_original_indices']!=ti:
            raise RuntimeError('Population differs from the sufficiency study')
    fitting,tuning=validation_groups(val,PARTITION_SEED,.7)
    if args.smoke:
        keep=sorted(set(fitting[:16].tolist())|set(tuning[:8].tolist()));where={v:i for i,v in enumerate(keep)}
        fitting=np.asarray([where[i] for i in fitting[:16]]);tuning=np.asarray([where[i] for i in tuning[:8]])
        val=[val[i] for i in keep];test=test[:8]
    log(f'{name}: {len(val)} validation, {len(test)} test queries; building estimators')
    columns=Columns(ds,train)
    freq=columns.freq
    write_json(out/'lift_audit.json',dict(
        levels=[int(l.max())+1 for l in columns.levels],
        real=[lift.diagonal_audit() for lift in columns.lift],
        shuffled=[[lift.diagonal_audit() for lift in null] for null in columns.nulls]))
    log(f'lift audit written')
    arrays=cache(columns,val,store/'validation')
    source=dict(prior=[k for k in columns.keys if k not in HISTORY and k!='pop'])
    menu=arms(columns);methods={}
    for arm,keys in menu.items():
        methods[arm]=fit(Bank(arrays,keys),val,fitting,tuning,freq,source,out/'selection'/arm,log)
        log(f'{arm}: {methods[arm]["config"]} tuning {methods[arm]["tuning_metrics"]}')
    write_json(out/'frozen_methods.json',methods)
    log('SELECTION FROZEN; scoring test queries')
    rows={arm:dict(metrics=[],predictions=[]) for arm in methods}
    groups=[]
    for i,pair in enumerate(test):
        row=columns.query(pair[0])
        for arm,method in methods.items():
            values=score(row,method['config'],freq)
            rows[arm]['metrics'].append(metrics(values,pair).tolist())
            rows[arm]['predictions'].append(top_items(values,pair[0],20).tolist())
        groups.append(str(tuple(sorted(set(pair[0])|set(pair[1])))))
        if (i+1)%200==0:log(f'test {i+1}/{len(test)}')
    (out/'queries').mkdir(exist_ok=True)
    for arm,row in rows.items():
        row['mean']=np.mean(row['metrics'],axis=0).tolist();row['groups']=groups
        write_json(out/'queries'/f'{arm}.json',row)
    write_json(store/'test_pairs.json',[[list(map(int,s)),list(map(int,t))] for s,t in test])
    write_json(store/'freq.json',freq.tolist())
    log('arms COMPLETE '+json.dumps({a:[round(x,4) for x in r['mean']] for a,r in rows.items()}))
    log('COMPLETE')


def report(args):
    records=[];tables={};secondary=[]
    for name in DATASETS:
        out=args.output/name
        rows={p.stem:json.loads(p.read_text()) for p in (out/'queries').glob('*.json')}
        pairs=json.loads((args.store/name/'test_pairs.json').read_text());freq=np.asarray(json.loads((args.store/name/'freq.json').read_text()))
        table={}
        for arm,row in rows.items():
            hits={True:[0,0],False:[0,0]}
            for predicted,(_,targets) in zip(row['predictions'],pairs):
                for t in targets:
                    cold=bool(freq[t]==0);hits[cold][1]+=1;hits[cold][0]+=int(t in predicted)
            table[arm]=dict(recall20=row['mean'][0],ndcg20=row['mean'][1],cold_recall20=hits[True][0]/max(hits[True][1],1),
                            warm_recall20=hits[False][0]/max(hits[False][1],1),cold_targets=hits[True][1],warm_targets=hits[False][1])
        metric=lambda arm:np.asarray(rows[arm]['metrics'])
        groups=rows['general']['groups']
        null=np.mean([metric(f'shuf{s}') for s in range(SHUFFLES)],axis=0)
        for j,label in [(1,'ndcg20'),(0,'recall20')]:
            for against,second in [('raw',metric('raw')),('shuffled_mean',null)]:
                records.append(dict(paired_comparison(metric('general')[:,j],second[:,j],groups,bootstrap=2000,permutations=10000,seed=SEED),
                                    dataset=name,metric=label,first='general',second=against))
        pairs_of=[('general','base'),('general','identity'),('general','lift_only'),('general','residual_only'),('general','off_diagonal'),('general_no_history','raw_no_history')]
        if 'recipe' in rows:pairs_of.append(('general','recipe'))
        for first,second in pairs_of:
            secondary.append(dict(paired_comparison(metric(first)[:,1],metric(second)[:,1],groups,bootstrap=2000,permutations=10000,seed=SEED),
                                  dataset=name,metric='ndcg20',first=first,second=second))
        tables[name]=dict(arms=table,gain_over_each_shuffle_ndcg=[float(metric('general')[:,1].mean()-metric(f'shuf{s}')[:,1].mean()) for s in range(SHUFFLES)],
                          audit=json.loads((out/'lift_audit.json').read_text()))
    holm_adjust(records,PLAN['holm_family'])
    verdict={}
    for name in DATASETS:
        get=lambda second:next(r for r in records if r['dataset']==name and r['metric']=='ndcg20' and r['second']==second)
        raw,shuffled=get('raw'),get('shuffled_mean')
        verdict[name]=dict(beats_raw=bool(raw['delta']>0 and raw['reject_05']),beats_shuffled=bool(shuffled['delta']>0 and shuffled['reject_05']),
                           beats_every_seed=bool(min(tables[name]['gain_over_each_shuffle_ndcg'])>0))
        verdict[name]['passes']=all(verdict[name].values())
    write_json(args.output/'summary.json',dict(protocol=PLAN,verdict=verdict,tables=tables,primary=records,secondary_unadjusted=secondary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--output',type=Path,default=Path('results/general_lift'))
    parser.add_argument('--store',type=Path,default=Path('checkpoints/general_lift'))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    with torch.inference_mode():
        report(args) if args.report else run(args)
