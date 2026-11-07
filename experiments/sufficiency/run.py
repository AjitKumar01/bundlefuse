"""Primary E1--E4 studies. All outputs are exploratory on inspected splits."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from bundlefuse.provenance import write_json,run_metadata,file_hash
from bundlefuse.selected_signals import SelectedSignals
from bundlefuse.metrics import top_items
from experiments.novelty.data import load_inputs,validation_groups,selected_columns
from experiments.novelty.raw import standardize,AmplifiedNeighbors
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from .engine import Bank,Search,fit,score,assess,prior_weights,GRID,KAPPAS,MIX

SEED=20261119
PLAN=dict(seed=SEED,rounds=4,weight_grid=GRID,kappa_grid=KAPPAS,constant_grid=MIX,
          fit_fraction=.7,primary_studies=['E1','E2','E3','E4'],reserved_tests=256,
          scope='Exploratory supplied splits; no untouched confirmation or sufficiency certificate.',
          conditional_studies='E5 probes/bridge and thinning require primary findings; not automatically launched.')


def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)


def overlap_audit(model,val,test):
    b=model.graph.B_bin
    digest=lambda x:hashlib.sha256(np.asarray(sorted(x),dtype='<i8').tobytes()).hexdigest()
    train={digest(b.indices[b.indptr[i]:b.indptr[i+1]]) for i in range(b.shape[0])}
    sets={name:[digest(set(s)|set(gt)) for s,gt in pairs] for name,pairs in [('validation',val),('test',test)]}
    return dict(training_unique_bundles=len(train),
        validation_train_overlap=sum(x in train for x in sets['validation']),
        test_train_overlap=sum(x in train for x in sets['test']),
        validation_test_overlap=len(set(sets['validation'])&set(sets['test'])),
        keep_validation=[i for i,x in enumerate(sets['validation']) if x not in train],
        keep_test=[i for i,x in enumerate(sets['test']) if x not in train and x not in set(sets['validation'])])


class IndexedArray:
    def __init__(self,base,indices):self.base,self.indices=base,np.asarray(indices,dtype=int)
    def __len__(self):return len(self.indices)
    def __getitem__(self,key):return self.base[self.indices[key]]


def fingerprint():
    paths=sorted(Path('bundlefuse').glob('*.py'))+sorted(Path('experiments/sufficiency').glob('*.py'))
    paths+=sorted(Path('experiments/novelty').glob('*.py'))
    return hashlib.sha256(b''.join(str(p).encode()+p.read_bytes() for p in paths)).hexdigest()


def extra_arrays(model,pairs,directory,smoke=False,alpha=1.):
    """Each raw contribution is computed from B, never from standardized differences."""
    directory.mkdir(parents=True,exist_ok=True);g=model.graph;n=g.n
    keys=['joint','ess','count']+(['third','amp2','amp3','amp_selected','amp_ess','amp_count'] if model.music else [])
    marker=directory/'progress.json'
    done=json.loads(marker.read_text())['queries'] if marker.exists() else 0
    arrays={k:np.lib.format.open_memmap(directory/f'{k}.npy',mode='r+' if (directory/f'{k}.npy').exists() else 'w+',
                                        dtype=np.float32,shape=(len(pairs),n)) for k in keys}
    amp=AmplifiedNeighbors(g) if model.music else None
    for i in range(done,len(pairs)):
        seeds=pairs[i][0];x=g.build_indicator([seeds])
        overlap=np.asarray((g.B @ x.multiply(g.inv_sqrt_f[:,None])).todense()).ravel()*g.inv_vol_pow
        count=np.asarray((g.B_bin @ x).todense()).ravel()
        delta=overlap*g.beta_clique*np.maximum(count-1,0)
        arrays['joint'][i]=standardize(np.asarray(g.B.T@delta).ravel()[:,None]).ravel()
        enhancement=1+g.beta_clique*np.maximum(count-1,0)
        if alpha in (0.,1.):h=overlap*((1-alpha)+alpha*enhancement)
        else:
            rawp=np.asarray(g.B.T@overlap).ravel();rawg=np.asarray(g.B.T@(overlap*enhancement)).ravel()
            h=overlap*((1-alpha)/(rawp.std(ddof=1)+1e-9)+alpha*enhancement/(rawg.std(ddof=1)+1e-9))
        a=np.asarray(g.B_bin.T@h).ravel();q=np.asarray(g.B_bin.T@(h*h)).ravel()
        arrays['ess'][i]=np.divide(a*a,q,out=np.zeros_like(a),where=q>0)
        arrays['count'][i]=np.asarray(g.B_bin.T@(h>0).astype(np.float32)).ravel()
        if amp:
            third=overlap*np.maximum(count-1,0)*np.maximum(count-2,0)/2
            arrays['third'][i]=standardize(np.asarray(g.B.T@third).ravel()[:,None]).ravel()
            for key,power,neighbors in [('amp2',2,None),('amp3',3,None),('amp_selected',2,100)]:
                arrays[key][i]=standardize(amp.score(seeds,power,neighbors)[:,None]).ravel()
            h=amp.overlaps(seeds)
            order=np.lexsort((np.arange(len(h)),-h));h[order[100:]]=0;h=h*h
            a=np.asarray(g.B_bin.T@h).ravel();q=np.asarray(g.B_bin.T@(h*h)).ravel()
            arrays['amp_ess'][i]=np.divide(a*a,q,out=np.zeros_like(a),where=q>0)
            arrays['amp_count'][i]=np.asarray(g.B_bin.T@(h>0).astype(np.float32)).ravel()
        for a in arrays.values():a.flush()
        write_json(marker,dict(queries=i+1,keys=keys))
        if (i+1)%100==0 or i+1==len(pairs):log(f'extra columns {directory}: {i+1}/{len(pairs)}')
    del arrays
    return {k:np.load(directory/f'{k}.npy',mmap_mode='r') for k in keys}


def shared_test(model,pairs,keys,directory):
    directory.mkdir(parents=True,exist_ok=True)
    needed=set(keys)-{'hist'}|{'pair','simp'}
    marker=directory/'progress.json';done=json.loads(marker.read_text())['queries'] if marker.exists() else 0
    arrays={k:np.lib.format.open_memmap(directory/f'{k}.npy',mode='r+' if (directory/f'{k}.npy').exists() else 'w+',
                     dtype=np.float32,shape=(len(pairs),model.graph.n)) for k in needed}
    provider=SelectedSignals(model.graph,model.onto,model.FEAT,model.cat_lbl,model.rao_dense,model.music,
                             torch.device('cpu'),beta_clique=model.cfg.beta_clique)
    for start in range(done,len(pairs),8):
        block=[s for s,_ in pairs[start:start+8]];_,z=provider.score(block,needed)
        for k,a in arrays.items():a[start:start+len(block)]=z[k].numpy().T;a.flush()
        write_json(marker,dict(queries=start+len(block),keys=sorted(needed)))
        if start%104==0:log(f'shared test cache {start+len(block)}/{len(pairs)}')
    del arrays
    return {k:np.load(directory/f'{k}.npy',mmap_mode='r') for k in needed}


def subset_name(keys):return 'subset_'+'_'.join(keys)


def timing(model,pairs,arrays,methods,out,smoke):
    """Exact source computation through retrieval, without reading score caches in the timed region."""
    if model.music:return
    path=out/'inference_timing.json';records=json.loads(path.read_text()) if path.exists() else {}
    provider=SelectedSignals(model.graph,model.onto,model.FEAT,model.cat_lbl,model.rao_dense,False,
                             torch.device('cpu'),beta_clique=model.cfg.beta_clique)
    indices=np.linspace(0,len(pairs)-1,min(4 if smoke else 64,len(pairs)),dtype=int)
    for name,method in methods.items():
        if name!='reference' and not name.startswith('E1_subset_'):continue
        if name in records:continue
        required=set(method['keys'])-{'hist'}
        if 'hist' in method['keys']:
            required.add('pair')
            if method['alpha']:required.add('simp')
        def compute(ix):
            seeds=[pairs[i][0] for i in ix];_,z=provider.score(seeds,required)
            z={k:v.numpy().T for k,v in z.items()}
            bank=Bank(z,method['keys'],method['history'],method['alpha'])
            result=[]
            for j,s in enumerate(seeds):
                values=score(bank.query(j),method['config'],np.asarray(model.freq))
                result.append(top_items(values,s,20))
            return result
        expected=top_items(score(Bank(arrays,method['keys'],method['history'],method['alpha']).query(int(indices[0])),
                        method['config'],np.asarray(model.freq)),pairs[int(indices[0])][0],20)
        np.testing.assert_array_equal(compute([int(indices[0])])[0],expected)
        measurements={}
        for batch in (1,8):
            elapsed=[]
            for repeat in range(1 if smoke else 3):
                for start in range(0,len(indices),batch):
                    ix=indices[start:start+batch];begin=time.perf_counter();compute(ix)
                    elapsed.append((time.perf_counter()-begin)*1000/len(ix))
            measurements[str(batch)]=dict(median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)))
        records[name]=dict(batch_query_latency=measurements,required_sources=sorted(required),
               query_sample=indices.tolist(),repeats=1 if smoke else 3,
               scope='Includes sources, standardization, fusion and top20; warm CPU, one thread. '
                     'No compact artifact storage or per-model resident-memory claim.')
        write_json(path,records);log(f'timing COMPLETE {name}')


def select(model,val,arrays,source,fitting,tuning,out,smoke):
    freq=np.asarray(model.freq);keys=selected_columns(source);alpha=source['higher_order'][0]
    methods={};subsets=[]
    if not model.music:
        for length in range(1,len(keys)+1):
            subsets += list(itertools.combinations(keys,length))
        if smoke:subsets=[tuple(keys),tuple(keys[:1])]
        retained=[]
        for subset in subsets:
            name=subset_name(subset);bank=Bank(arrays,subset,'canonical',alpha)
            starts=[r['config']['weights'] for r in retained if set(r['keys'])<=set(subset)]
            result=fit(bank,val,fitting,tuning,freq,source,out/'selection'/name,log,initial=starts)
            retained.append(result);methods['E1_'+name]=result
    full=fit(Bank(arrays,keys,'canonical',alpha),val,fitting,tuning,freq,source,out/'selection'/'full',log)
    methods['reference']=full
    if not model.music:
        fullcfg=full['config'];w=fullcfg['weights']
        for removed in keys:
            newkeys=[k for k in keys if k!=removed]
            neww={k:v for k,v in w.items() if k!=removed}
            if not neww:continue
            cfg=dict(fullcfg,weights=neww)
            if cfg['kind']!='weighted':
                cfg['prior_weights']={k:v for k,v in cfg['prior_weights'].items() if k!=removed}
                if not cfg['prior_weights']:cfg=dict(kind='weighted',weights=neww)
            methods['E1_fixed_remove_'+removed]=dict(full,keys=newkeys,config=cfg)
        shared=[k for k in keys if k!='hist']
        for mode,hkeys,history,ha in [('pair',['pair'],'canonical',alpha),
                   ('joint',['joint'],'canonical',alpha),('separate',['pair','joint'],'canonical',alpha),
                   *[(f'coupled_{a}',['hist'],'canonical',a) for a in (0.,.5,1.)]]:
            for policy in ('fixed','refit'):
                bk=Bank(arrays,shared+hkeys,history,ha)
                fixed={k:w.get(k,0.) for k in shared} if policy=='fixed' else None
                result=fit(bk,val,fitting,tuning,freq,source,out/'selection'/f'E2_{mode}_{policy}',log,
                           fixed=fixed,variable=hkeys if policy=='fixed' else None)
                if policy=='fixed':
                    # Historical weights fit on fitting labels; common fallback freezes the final intervention.
                    cfg=dict(fullcfg,weights=result['config']['weights'])
                    if cfg['kind']!='weighted':cfg['prior_weights']=fullcfg['prior_weights']
                    result['config']=cfg
                    finalsearch=Search(bk,val,tuning,freq,out/'selection'/f'E2_{mode}_{policy}'/'fixed_tuning.json',log)
                    result['tuning_metrics']=finalsearch.many([cfg])[0]
                methods[f'E2_{mode}_{policy}']=result
        for policy in ('fixed','refit'):
            eligible=[methods[f'E2_coupled_{a}_{policy}'] for a in (0.,.5,1.)]
            methods[f'E2_coupled_selected_{policy}']=max(eligible,key=lambda r:r['tuning_metrics'][1])
    # E3: exact reference data/scale; one free scalar per term-specific control.
    supportfull=full
    if model.music:
        supportfull=fit(Bank(arrays,keys,'amp_selected',alpha),val,fitting,tuning,freq,source,
                        out/'selection'/'E4_amp_selected',log)
        methods['E4_amp_selected']=supportfull
    bank=Bank(arrays,keys,supportfull['history'],alpha);cfg=supportfull['config']
    if cfg['kind']=='weighted':
        cfg=dict(kind='gate',weights=cfg['weights'],prior_weights=prior_weights(cfg['weights'],source),scale=1.,kappa=0.)
    if cfg.get('prior_weights') and cfg['weights'].get('hist',0):
        candidates=[]
        for kind in ('history_frequency','history_constant','ess','count'):
            for parameter in (MIX if kind=='history_constant' else KAPPAS):
                candidates.append(dict(cfg,kind=kind,parameter=parameter))
        candidates += [dict(cfg,kind='constant_full',parameter=v) for v in MIX]
        search=Search(bank,val,tuning,freq,out/'selection'/'E3_tuning.json',log)
        vals=search.many(candidates)
        for kind in dict.fromkeys(c['kind'] for c in candidates):
            eligible=[i for i,c in enumerate(candidates) if c['kind']==kind]
            j=max(eligible,key=lambda j:vals[j][1])
            methods['E3_'+kind]=dict(supportfull,config=candidates[j],tuning_metrics=vals[j])
    else:write_json(out/'E3_inactive.json',dict(reason='No explicit history or no retained fallback'))
    if model.music:
        # All predecessor controls receive the same other source columns and fitting budget.
        for hist in ('amp2','amp3','amp_selected'):
            bk=Bank(arrays,keys,hist,alpha)
            methods['E4_'+hist]=fit(bk,val,fitting,tuning,freq,source,out/'selection'/f'E4_{hist}',log)
        bk=Bank(arrays,keys+['third'],'canonical',alpha)
        methods['E4_third_refit']=fit(bk,val,fitting,tuning,freq,source,out/'selection'/'E4_third_refit',log)
        # Fixed-coefficient augmentation selects lambda on fitting groups only.
        candidates=[dict(full['config'],weights=dict(full['config']['weights'],third=v)) for v in GRID]
        search=Search(bk,val,fitting,freq,out/'selection'/'E4_third_fixed.json',log)
        vals=search.many(candidates);j=int(np.argmax([v[1] for v in vals]))
        methods['E4_third_fixed']=dict(full,keys=keys+['third'],config=candidates[j])
    write_json(out/'frozen_methods.json',methods)
    return methods


def report(out,smoke):
    paths=sorted((out/'queries').glob('*.json'))
    rows={p.stem:json.loads(p.read_text()) for p in paths if not p.name.endswith('.progress.json')}
    ref=rows['reference'];contrasts=[]
    for name,row in rows.items():
        if name=='reference':continue
        for metric,j in [('recall20',0),('ndcg20',1)]:
            comp=paired_comparison(np.array(row['metrics'])[:,j],np.array(ref['metrics'])[:,j],row['groups'],
                    bootstrap=20 if smoke else 2000,permutations=50 if smoke else 10000,seed=SEED)
            contrasts.append(dict(comp,first=name,second='reference',metric=metric))
    matched=[('E3_ess','E3_history_frequency'),('E3_ess','E3_history_constant'),
             ('E3_ess','E3_count'),('E4_third_refit','E4_amp_selected'),('E4_third_refit','E4_amp3')]
    matched += [(f'E2_separate_{p}',f'E2_coupled_selected_{p}') for p in ('fixed','refit')]
    for first,second in matched:
        if first not in rows or second not in rows:continue
        for metric,j in [('recall20',0),('ndcg20',1)]:
            comp=paired_comparison(np.array(rows[first]['metrics'])[:,j],np.array(rows[second]['metrics'])[:,j],
                 rows[first]['groups'],bootstrap=20 if smoke else 2000,permutations=50 if smoke else 10000,seed=SEED)
            contrasts.append(dict(comp,first=first,second=second,metric=metric))
    holm_adjust(contrasts,256)
    write_json(out/'summary.json',dict(status='complete',smoke=smoke,methods={k:v['mean'] for k,v in rows.items()},
               contrasts=contrasts,scope=PLAN['scope'],certified_sufficient=False))
    lines=['# Sufficiency study results','',PLAN['scope'],'','| Method | Recall@20 | NDCG@20 |','|---|---:|---:|']
    lines += [f'| {k} | {v["mean"][0]:.6f} | {v["mean"][1]:.6f} |' for k,v in rows.items()]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(args):
    out=args.output/args.dataset;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    artifacts=out/'artifacts' if args.smoke else Path('checkpoints/sufficiency')/args.dataset
    artifacts.mkdir(parents=True,exist_ok=True)
    protocol=dict(PLAN,dataset=args.dataset,smoke=args.smoke,source_sha256=fingerprint())
    for folder in (out,artifacts):
        p=folder/'protocol.json'
        if p.exists() and json.loads(p.read_text())!=json.loads(json.dumps(protocol)):
            raise RuntimeError('Protocol/source mismatch; retain files and use another output root')
        write_json(p,protocol)
    log('Read retained model and validate input digests')
    model,val,test,old,source,identity=load_inputs(args.dataset)
    write_json(out/'inputs.json',dict(identity,runtime=run_metadata()))
    audit=overlap_audit(model,val,test);write_json(out/'complete_bundle_overlap.json',audit)
    log(f'Complete-bundle overlap exclusions: validation/train={audit["validation_train_overlap"]}, '
        f'test/train={audit["test_train_overlap"]}, validation/test={audit["validation_test_overlap"]}')
    vi=audit['keep_validation'];ti=audit['keep_test']
    if args.smoke:vi=vi[:16];ti=ti[:6]
    val=[val[i] for i in vi];test=[test[i] for i in ti]
    old={k:IndexedArray(v,vi) for k,v in old.items()}
    write_json(out/'evaluation_population.json',dict(validation_original_indices=vi,test_original_indices=ti,
               policy='Exclude complete bundles overlapping training; exclude test bundles overlapping validation. '
                      'Original artifacts retained. New metrics are not the unfiltered supplied-split metrics.'))
    fitting,tuning=validation_groups(val,SEED,.7)
    write_json(out/'partition.json',dict(fitting=fitting.tolist(),tuning=tuning.tolist()))
    verifier=SelectedSignals(model.graph,model.onto,model.FEAT,model.cat_lbl,model.rao_dense,model.music,torch.device('cpu'))
    needed=set(selected_columns(source))-{'hist'}|{'pair','simp'}
    _,fresh=verifier.score([val[0][0]],needed)
    errors={k:float(np.max(np.abs(fresh[k].numpy().ravel()-old[k][0]))) for k in needed}
    if max(errors.values())>1e-4:raise RuntimeError(f'Cache parity failed: {errors}')
    write_json(out/'cache_parity.json',errors)
    arrays=dict(old,**extra_arrays(model,val,artifacts/'extra/validation',args.smoke,source['higher_order'][0]))
    methods=select(model,val,arrays,source,fitting,tuning,out,args.smoke)
    log(f'SELECTION FROZEN: {len(methods)} methods; construct/reuse held-out columns')
    prior=Path('checkpoints/novelty_attribution')/args.dataset/'signals/test'
    count=len(test)
    testarrays={}
    marker=prior/'signals.progress.json'
    reusable=marker.exists() and json.loads(marker.read_text()).get('queries',0)>=count
    for k in needed:
        p=prior/f'{k}.npy'
        if reusable and p.exists():
            a=np.load(p,mmap_mode='r')
            if len(a)<=max(ti):raise RuntimeError('Prior cache too short')
            testarrays[k]=IndexedArray(a,ti)
    if len(testarrays)!=len(needed):testarrays=shared_test(model,test,selected_columns(source),artifacts/'shared/test')
    _,check=verifier.score([test[0][0]],needed)
    if max(float(np.max(np.abs(check[k].numpy().ravel()-testarrays[k][0]))) for k in needed)>1e-4:
        raise RuntimeError('Held-out shared cache parity failed')
    testarrays.update(extra_arrays(model,test,artifacts/'extra/test',args.smoke,source['higher_order'][0]))
    for name,method in methods.items():
        bank=Bank(testarrays,method['keys'],method['history'],method['alpha'])
        assess(bank,test,method['config'],np.asarray(model.freq),out/'queries'/f'{name}.json',log)
    timing(model,test,testarrays,methods,out,args.smoke)
    if file_hash(identity['artifact_path'])!=identity['artifact_sha256']:raise RuntimeError('Input checkpoint changed')
    report(out,args.smoke);log('COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['pog_dense','pog','spotify'])
    parser.add_argument('--output',type=Path,default=Path('results/sufficiency'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1),torch.inference_mode():run(args)
