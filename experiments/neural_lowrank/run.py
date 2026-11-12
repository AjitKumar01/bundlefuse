"""Low-rank relation model on POG, POG_DENSE and Spotify under a plan fixed before training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from bundlefuse.provenance import write_json
from experiments.neural.evaluation import summarize
from experiments.neural.training import save_checkpoint
from experiments.neural_revision.model import full_catalogue_loss
from experiments.neural_revision.training import masked_bundles
from experiments.neural_revision.pog import prepare,metric_subset
from experiments.neural_revision.probes import target_frequency_metrics
from experiments.neural_revision.run import log
from .model import LowRankRelation

POG=['pog_dense','pog'];SPOTIFY=['spotify_sparse','spotify']
ARMS=dict(loo=dict(cold_probability=0.),loo_mask=dict(cold_probability=.25))
CONFIG=dict(epochs=100,batch=32,eval_every=2,patience=6,shared_lr=.001,group_lr=.01,weight_decay=.001,rank=64,group_rank=16,hidden=32)
RESULTS=Path('results/neural_lowrank');CHECKPOINTS=Path('checkpoints/neural_lowrank')
PREVIOUS=Path('results/neural_bilinear')  # best earlier neural model: the category arm
PLAN=dict(question='Can a low-rank model with shared pair weights and leave-one-out training replace the per-pair table, symmetric similarity, graph matrix and full category table without losing accuracy?',
          arms={'loo':'leave-one-out history only','loo_mask':'leave-one-out plus the earlier 25% random history mask'},
          selection='On seed 0 validation NDCG@20, pick the arm with the higher mean ratio to the earlier category model across POG and POG_DENSE. Test is not used. The pick is tested whichever way it compares.',
          replication='Train the pick at seeds 1 and 2 on POG and POG_DENSE, and at seeds 0, 1 and 2 on Spotify_sparse and Spotify with no further choices.',
          primary='Filtered test queries, three-seed per-query means. POG and POG_DENSE: pick vs the earlier category model and vs BundleFuse. Spotify datasets: pick vs BundleFuse where a BundleFuse test reference exists. NDCG@20 and Recall@20. Holm over all of these tests together.',
          decision=dict(replaces='Pick not significantly behind the category model on NDCG on either POG dataset (Holm p<.05 would count as behind).',
                        versus_bundlefuse='Ahead or behind only with Holm p<.05 on NDCG. Otherwise not separated, which is not equivalence.'),
          scope='Exploratory: POG splits were inspected earlier, and the design follows an audit of the same validation data. Spotify has not been used by any neural model before.')


def source_hash():
    h=hashlib.sha256()
    for path in sorted(Path('experiments/neural_lowrank').glob('*.py')):h.update(path.read_bytes())
    return h.hexdigest()


def construct(ds,features,cats,config=CONFIG):
    return LowRankRelation(features,cats,ds.train,rank=config['rank'],group_rank=config['group_rank'],hidden=config['hidden'])


def data(name,smoke=False):
    out=RESULTS/name;out.mkdir(parents=True,exist_ok=True)
    root=Path('checkpoints/neural_pog')/name if name in POG else CHECKPOINTS/name
    ds,x,c,f,_,audit=prepare(name,root,out,False)
    if smoke:ds.train=ds.train[:64];ds.val_pairs=ds.val_pairs[:8];ds.test_pairs=ds.test_pairs[:8]
    return ds,x,c,f,audit


@torch.inference_mode()
def evaluate(model,pairs,batch=32):
    model.eval();started=time.monotonic();records=[];keys=model.keys()
    for start in range(0,len(pairs),batch):
        rows=pairs[start:start+batch]
        values=model([s for s,_ in rows],keys=keys).numpy()
        if not np.isfinite(values).all():raise RuntimeError('Nonfinite scores')
        records.extend(summarize(values,rows)['queries'])
    return dict(recall20=float(np.mean([r['recall20'] for r in records])),ndcg20=float(np.mean([r['ndcg20'] for r in records])),
                queries=records,seconds=time.monotonic()-started)


def load(ds,x,c,choice):
    model=construct(ds,x,c,choice['config'])
    model.load_state_dict(torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)['model'])
    return model.eval()


def fit(ds,x,c,arm,seed,config,out,ckpt):
    out.mkdir(parents=True,exist_ok=True);ckpt.mkdir(parents=True,exist_ok=True)
    if (out/'selected.json').exists():return json.loads((out/'selected.json').read_text())
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=construct(ds,x,c,config);cold=ARMS[arm]['cold_probability']
    train=[b for b in ds.train if len(set(b))>=2]  # a single-item row has nothing to hide
    tables=model.fast_parameters() if hasattr(model,'fast_parameters') else [model.query_group,model.key_group];shared=[p for p in model.parameters() if all(p is not t for t in tables)]
    optimizer=torch.optim.AdamW([dict(params=tables,lr=config['group_lr'],weight_decay=config.get('fast_weight_decay',config['weight_decay'])),dict(params=shared,lr=config['shared_lr'])],weight_decay=config['weight_decay'])
    start=1;best=-float('inf');best_epoch=0;stale=0;history=[];elapsed=0
    saved=sorted(ckpt.glob('epoch_*.pt'))
    if saved:
        state=torch.load(saved[-1],map_location='cpu',weights_only=False)
        if state['config']!=config or state['arm']!=arm:raise RuntimeError('Resume mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);rng.bit_generator.state=state['numpy_rng']
        start=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch'];stale=state['stale'];history=state['history'];elapsed=state['elapsed']
    log(f'{ds.name} {arm} seed={seed} {model.counts_summary()} train_rows={len(train)} start_epoch={start}')
    started=time.monotonic()
    for epoch in range(start,config['epochs']+1):
        if stale>=config['patience']:break
        model.train();order=rng.permutation(len(train));losses=[]
        for offset in range(0,len(order),config['batch']):
            rows=[train[i] for i in order[offset:offset+config['batch']]]
            seeds,truth=masked_bundles(rows,rng)
            keep=(torch.rand(model.n)>=cold).float() if cold else None
            optimizer.zero_grad(set_to_none=True)
            loss=full_catalogue_loss(model(seeds,bundles=[list(set(b)) for b in rows],keep=keep),seeds,truth)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            losses.append((float(loss.detach()),len(rows)))
        record=dict(epoch=epoch,loss=sum(v*n for v,n in losses)/sum(n for _,n in losses),seconds=time.monotonic()-started)
        if epoch%config['eval_every']==0 or epoch==config['epochs']:
            val=evaluate(model,ds.val_pairs)
            record['validation']={k:v for k,v in val.items() if k!='queries'}
            if val['ndcg20']>best:best=val['ndcg20'];best_epoch=epoch;stale=0
            else:stale+=1
            history.append(record)
            for old in ckpt.glob('epoch_*.pt'):
                if old.name!=f'epoch_{best_epoch:04d}.pt':old.unlink()
            save_checkpoint(ckpt/f'epoch_{epoch:04d}.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,arm=arm,config=config,
                epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,history=history,elapsed=elapsed+time.monotonic()-started))
            log(f'{ds.name} {arm} seed={seed} epoch={epoch} loss={record["loss"]:.5f} VAL R20={val["recall20"]:.6f} N20={val["ndcg20"]:.6f} best_epoch={best_epoch} patience={stale}/{config["patience"]}')
        else:
            history.append(record);log(f'{ds.name} {arm} seed={seed} epoch={epoch} loss={record["loss"]:.5f}')
        write_json(out/'history.json',history)
    path=ckpt/f'epoch_{best_epoch:04d}.pt'
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    val=evaluate(model,ds.val_pairs)
    if abs(val['ndcg20']-best)>1e-9:raise RuntimeError('Checkpoint replay mismatch')
    choice=dict(arm=arm,config=config,seed=seed,checkpoint=str(path),best_epoch=best_epoch,epochs_run=history[-1]['epoch'],
                validation={k:v for k,v in val.items() if k!='queries'},parameters=model.counts_summary(),elapsed=elapsed+time.monotonic()-started)
    write_json(out/'validation_predictions.json',val);write_json(out/'selected.json',choice)
    return choice


def plan(_):
    path=RESULTS/'plan.json'
    if path.exists():raise RuntimeError('Plan already fixed')
    RESULTS.mkdir(parents=True,exist_ok=True);write_json(path,dict(PLAN,config=CONFIG,source_sha256=source_hash()))


def train(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if not args.smoke and json.loads((RESULTS/'plan.json').read_text())['source_sha256']!=source_hash():raise RuntimeError('Source changed after the plan was fixed')
    ds,x,c,_,_=data(args.dataset,args.smoke)
    config=CONFIG if not args.epochs else dict(CONFIG,epochs=args.epochs)
    key=f'{args.arm}_seed{args.seed}'
    fit(ds,x,c,args.arm,args.seed,config,args.results/args.dataset/key,args.checkpoints/args.dataset/key)
    log(f'{args.dataset} {key} COMPLETE')


def chosen(dataset,arm,seed,root=RESULTS):
    return json.loads((root/dataset/f'{arm}_seed{seed}'/'selected.json').read_text())


def select(_):
    base={d:chosen(d,'category',0,PREVIOUS)['validation']['ndcg20'] for d in POG}
    table={a:{d:chosen(d,a,0)['validation']['ndcg20'] for d in POG} for a in ARMS}
    ratio={a:float(np.mean([table[a][d]/base[d] for d in POG])) for a in ARMS}
    winner=max(ratio,key=ratio.get)
    write_json(RESULTS/'selection.json',dict(category_model=base,validation_ndcg20=table,mean_ratio=ratio,selected=winner));log(f'selected {winner} {ratio}')


@torch.inference_mode()
def ablate(_):
    """Validation only: switch off one view at a time in the seed 0 pick."""
    torch.set_num_threads(4)
    winner=json.loads((RESULTS/'selection.json').read_text())['selected'];report={}
    for name in POG+SPOTIFY:
        if not (RESULTS/name/f'{winner}_seed0'/'selected.json').exists():continue
        ds,x,c,_,_=data(name);model=load(ds,x,c,chosen(name,winner,0));row={}
        for part in (None,'content','history','group','pair'):
            model.disabled={part} if part else set()
            row[part or 'full']=evaluate(model,ds.val_pairs)['ndcg20'];log(f'{name} without {part}: {row[part or "full"]:.6f}')
        model.disabled=set()
        a=(model.query_content.T@model.key_content).detach();s=torch.linalg.svdvals(a)
        row['content_form']=dict(symmetric_share=float(torch.linalg.norm((a+a.T)/2)**2/torch.linalg.norm(a)**2),
                                 singular_values_for_90_percent=int((torch.cumsum(s**2,0)/(s**2).sum()<.9).sum()+1))
        report[name]=row
    write_json(RESULTS/'ablation.json',report)


def groups(pairs):return [str(tuple(sorted(set(s)|set(g)))) for s,g in pairs]


def reference(name,pairs):
    """Per-query BundleFuse metrics on every test query; NaN where a query was not scored."""
    path=Path(f'results/publication_mechanisms/{name}/queries.json')
    if path.exists():return np.asarray(json.loads(path.read_text())['metrics']['full'],float)
    # Spotify_sparse: the publication run stopped before test, so use the general_lift recipe refit, filtered queries only.
    path=Path(f'results/general_lift/{name}/queries/recipe.json')
    if not path.exists():return None
    saved=json.loads(path.read_text());where={}
    for i,g in enumerate(groups(pairs)):where.setdefault(g,[]).append(i)
    out=np.full((len(pairs),2),np.nan)
    # rows were written in test order, so a duplicated bundle's copies are matched in that order
    for g,m in zip(saved['groups'],saved['metrics']):out[where[g].pop(0)]=m
    return out


def test(_):
    from experiments.mechanisms.statistics import paired_comparison,holm_adjust
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    winner=json.loads((RESULTS/'selection.json').read_text())['selected']
    report=dict(selected=winner,plan=json.loads((RESULTS/'plan.json').read_text())['decision'],datasets={});records=[]
    for name in POG+SPOTIFY:
        if not all((RESULTS/name/f'{winner}_seed{s}'/'selected.json').exists() for s in (0,1,2)):log(f'{name}: not trained, skipped');continue
        ds,x,c,f,audit=data(name);keep=audit['clean_test_indices'];rows=[]
        for seed in (0,1,2):
            path=RESULTS/name/f'{winner}_seed{seed}'/'test.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                result=evaluate(load(ds,x,c,chosen(name,winner,seed)),ds.test_pairs)
                result['overlap_filtered']=metric_subset(result['queries'],keep);result['by_target_frequency']=target_frequency_metrics(result,ds.test_pairs,f)
                write_json(path,result)
            rows.append(result)
        mean=lambda runs:np.mean([[[q['recall20'],q['ndcg20']] for q in r['queries']] for r in runs],axis=0)
        new=mean(rows);others={}
        if name in POG:others['category']=mean([json.loads((PREVIOUS/name/f'category_seed{s}'/'test.json').read_text()) for s in (0,1,2)])
        bundlefuse=reference(name,ds.test_pairs)
        if bundlefuse is not None:others['bundlefuse']=bundlefuse
        ids=groups(ds.test_pairs);clean=[ids[i] for i in keep]
        report['datasets'][name]=dict(seeds=[dict(full=[r['recall20'],r['ndcg20']],filtered=r['overlap_filtered'],by_target_frequency=r['by_target_frequency']) for r in rows],
                                      mean={k:dict(full=v.mean(0).tolist(),filtered=v[keep].mean(0).tolist()) for k,v in dict(selected=new,**others).items()})
        for second,other in others.items():
            for j,metric in ((1,'ndcg20'),(0,'recall20')):
                records.append(dict(paired_comparison(new[keep,j],other[keep,j],clean,bootstrap=2000,permutations=10000,seed=20261108),dataset=name,metric=metric,first='selected',second=second))
    holm_adjust(records,len(records));report['primary']=records
    write_json(RESULTS/'summary.json',report);log('test COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(required=True)
    for name,fn in (('plan',plan),('select',select),('ablate',ablate),('test',test)):sub.add_parser(name).set_defaults(run=fn)
    t=sub.add_parser('train');t.set_defaults(run=train);t.add_argument('--dataset',required=True,choices=POG+SPOTIFY);t.add_argument('--arm',required=True,choices=list(ARMS));t.add_argument('--seed',type=int,default=0)
    t.add_argument('--threads',type=int,default=1);t.add_argument('--smoke',action='store_true');t.add_argument('--epochs',type=int,default=0)
    t.add_argument('--results',type=Path,default=RESULTS);t.add_argument('--checkpoints',type=Path,default=CHECKPOINTS)
    args=parser.parse_args();args.run(args)
