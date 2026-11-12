"""Full-catalogue completion training with simulated cold items and retained epochs."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from bundlefuse.provenance import write_json
from experiments.neural.evaluation import summarize
from experiments.neural.training import save_checkpoint
from .model import RelationNetwork,full_catalogue_loss


def masked_bundles(bundles,rng):
    seeds=[];truth=[]
    for b in bundles:
        row=rng.permutation(sorted(set(b)))
        cut=max(1,len(row)//2) if rng.random()<.5 else int(rng.integers(1,len(row)))
        seeds.append(row[:cut].tolist());truth.append(row[cut:].tolist())
    return seeds,truth


@torch.inference_mode()
def evaluate(model,pairs,batch=32):
    model.eval();started=time.monotonic();records=[]
    vectors=model.item_vectors()
    for start in range(0,len(pairs),batch):
        rows=pairs[start:start+batch]
        values=model([s for s,_ in rows],item_vectors=vectors).numpy()
        if not np.isfinite(values).all():raise RuntimeError('Nonfinite scores')
        records.extend(summarize(values,rows)['queries'])
    result=dict(recall20=float(np.mean([r['recall20'] for r in records])),
                ndcg20=float(np.mean([r['ndcg20'] for r in records])),queries=records)
    result.update(seconds=time.monotonic()-started,full_catalogue=True)
    return result


def construct(ds,features,cats,condition):
    return RelationNetwork(features,cats,ds.train,variant=condition['variant'],category_mode=condition.get('category_mode','free'))


@torch.no_grad()
def project_parameters(model,condition):
    if condition.get('project_links',False):
        for key in ('pair_weights','joint_weights'):
            if hasattr(model,key):getattr(model,key).clamp_(min=0)
    if model.category_mode=='nonnegative':model.category_weights.clamp_(min=0)


def load_selected(ds,features,cats,choice):
    m=construct(ds,features,cats,choice['condition'])
    m.load_state_dict(torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)['model'])
    return m.eval()


def fit(ds,features,cats,condition,seed,config,out,ckpt,log):
    out.mkdir(parents=True,exist_ok=True);ckpt.mkdir(parents=True,exist_ok=True)
    if (out/'selected.json').exists():return json.loads((out/'selected.json').read_text())
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=construct(ds,features,cats,condition)
    links=[];shared=[]
    for name,param in model.named_parameters():
        (links if name in ('pair_weights','joint_weights','category_weights') else shared).append(param)
    optimizer=torch.optim.AdamW([dict(params=links,lr=config['link_lr']),dict(params=shared,lr=config['shared_lr'])],weight_decay=config['weight_decay'])
    start=1;best=-float('inf');best_epoch=0;stale=0;history=[];elapsed=0
    saved=sorted(ckpt.glob('epoch_*.pt'))
    if saved:
        state=torch.load(saved[-1],map_location='cpu',weights_only=False)
        if state['config']!=config or state['condition']!=condition:raise RuntimeError('Resume mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);rng.bit_generator.state=state['numpy_rng']
        start=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch'];stale=state['stale'];history=state['history'];elapsed=state['elapsed']
    log(f'{condition["name"]} seed={seed} parameters={model.counts()} start_epoch={start}')
    started=time.monotonic()
    for epoch in range(start,config['epochs']+1):
        if stale>=config['patience']:break
        model.train();order=rng.permutation(len(ds.train));losses=[]
        for offset in range(0,len(order),config['batch']):
            rows=[ds.train[i] for i in order[offset:offset+config['batch']]]
            seeds,truth=masked_bundles(rows,rng)
            keep=(torch.rand(ds.n_items)>=condition['cold_probability']).float() if condition['cold_probability'] else None
            optimizer.zero_grad(set_to_none=True)
            loss=full_catalogue_loss(model(seeds,keep),seeds,truth)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            project_parameters(model,condition)
            losses.append((float(loss.detach()),len(rows)))
            if config.get('log_every_batches') and (offset//config['batch']+1)%config['log_every_batches']==0:
                log(f'{condition["name"]} seed={seed} epoch={epoch} batches={offset//config["batch"]+1}/{int(np.ceil(len(order)/config["batch"]))} loss={float(loss.detach()):.5f}')
        record=dict(epoch=epoch,loss=sum(v*n for v,n in losses)/sum(n for _,n in losses))
        if epoch%config['eval_every']==0 or epoch==config['epochs']:
            val=evaluate(model,ds.val_pairs)
            record['validation']={k:v for k,v in val.items() if k!='queries'}
            if val['ndcg20']>best:best=val['ndcg20'];best_epoch=epoch;stale=0
            else:stale+=1
            history.append(record)
            save_checkpoint(ckpt/f'epoch_{epoch:04d}.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,condition=condition,config=config,
                epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,history=history,elapsed=elapsed+time.monotonic()-started))
            log(f'{condition["name"]} seed={seed} epoch={epoch} loss={record["loss"]:.5f} VAL R20={val["recall20"]:.6f} N20={val["ndcg20"]:.6f} best_epoch={best_epoch} patience={stale}/{config["patience"]}')
        else:
            history.append(record);log(f'{condition["name"]} seed={seed} epoch={epoch} loss={record["loss"]:.5f}')
        write_json(out/'history.json',history)
        write_json(out/'status.json',dict(status='training',epoch=epoch,best_epoch=best_epoch,best_validation_ndcg20=best if np.isfinite(best) else None))
    path=ckpt/f'epoch_{best_epoch:04d}.pt'
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    val=evaluate(model,ds.val_pairs)
    if abs(val['ndcg20']-best)>1e-12:raise RuntimeError('Checkpoint replay mismatch')
    choice=dict(condition=condition,seed=seed,checkpoint=str(path),best_epoch=best_epoch,epochs_run=history[-1]['epoch'],
                validation={k:v for k,v in val.items() if k!='queries'},parameters=model.counts(),elapsed=elapsed+time.monotonic()-started)
    write_json(out/'validation_predictions.json',val);write_json(out/'selected.json',choice)
    write_json(out/'status.json',dict(status='selected',**choice))
    return choice
