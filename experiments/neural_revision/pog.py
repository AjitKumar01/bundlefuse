"""Fixed independent neural architecture on the supplied POG/POG_DENSE splits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from bundlefuse import datasets
from bundlefuse.provenance import digest_rows,write_json
from bundlefuse.splits import file_hash
from bundlefuse.metrics import top_items
from .run import log,source_identity
from .training import fit,load_selected,evaluate
from .probes import compare_sources,target_frequency_metrics


CONDITION=dict(name='graph_pair_no_category',variant='graph_pair',cold_probability=.25,
               category_mode='off',project_links=True)


def overlap_audit(train,validation,test):
    train_groups={tuple(sorted(set(b))) for b in train}
    val_groups={tuple(sorted(set(s)|set(g))) for s,g in validation}
    val_keep=[i for i,(s,g) in enumerate(validation) if tuple(sorted(set(s)|set(g))) not in train_groups]
    test_keep=[i for i,(s,g) in enumerate(test) if tuple(sorted(set(s)|set(g))) not in train_groups|val_groups]
    return dict(primary='Full supplied train/validation/test files, unmodified. Duplicate training rows retained.',
                secondary='Same fitted model evaluated on test queries whose complete bundle occurs in neither training nor validation; no separate refitting.',
                validation_train_overlap_queries=len(validation)-len(val_keep),
                test_train_or_validation_overlap_queries=len(test)-len(test_keep),
                clean_validation_indices=val_keep,clean_test_indices=test_keep,
                limitation='Primary selection uses full supplied validation, including overlaps. Secondary test filtering does not retroactively remove that selection overlap.')


def prepare(name,ckpt,output,smoke):
    log(f'{name}: loading supplied splits and released item features')
    ds=datasets.load(name,protocol_mode='provided',data_profile='extended',val_n=0,test_n=0)
    audit=overlap_audit(ds.train,ds.val_pairs,ds.test_pairs)
    write_json(output/'overlap_audit.json',audit)
    fingerprint=dict(metadata=ds.metadata,modalities={k:list(v.shape) for k,v in ds.features.items()},
                     feature_projection='Normalize each released modality; concatenate; rank-128 randomized SVD; row-normalize. Catalogue attributes only, no bundle labels.',
                     projection_seed=0,smoke=smoke)
    feature_dir=ckpt/'features';feature_dir.mkdir(parents=True,exist_ok=True)
    marker=feature_dir/'projection.json'
    if marker.exists():
        meta=json.loads(marker.read_text())
        if meta['fingerprint']!=fingerprint:raise RuntimeError('Feature input identity changed')
        if file_hash(feature_dir/'projected.npy')!=meta['projected_sha256']:raise RuntimeError('Projected feature cache changed')
        features=np.load(feature_dir/'projected.npy')
        log(f'{name}: reusing retained attribute projection {features.shape}')
    else:
        arrays=[normalize(np.asarray(ds.features[k],np.float32),axis=1) for k in sorted(ds.features)]
        x=normalize(np.concatenate(arrays,axis=1),axis=1);del arrays
        log(f'{name}: projecting released features {x.shape} to 128 dimensions (no bundle labels)')
        svd=TruncatedSVD(n_components=128,n_iter=5,random_state=0)
        features=normalize(svd.fit_transform(x).astype(np.float32),axis=1);del x
        np.save(feature_dir/'projected.npy',features)
        np.save(feature_dir/'projection_components.npy',svd.components_.astype(np.float32))
        meta=dict(fingerprint=fingerprint,shape=list(features.shape),explained_variance_ratio_sum=float(svd.explained_variance_ratio_.sum()),
                  projected_sha256=file_hash(feature_dir/'projected.npy'),components_sha256=file_hash(feature_dir/'projection_components.npy'))
        write_json(marker,meta);log(f'{name}: attribute projection complete')
    ds.features=None  # Large released arrays are no longer needed by the neural scorer.
    _,cats=np.unique(ds.cat_labels,return_inverse=True)
    if smoke:
        ds.train=ds.train[:64];ds.val_pairs=ds.val_pairs[:8];ds.test_pairs=ds.test_pairs[:8]
    frequency=np.zeros(ds.n_items,np.float32)
    for bundle in ds.train:frequency[list(set(bundle))]+=1
    fingerprint.update(actual_train_sha256=digest_rows(ds.train),actual_val_sha256=digest_rows(ds.val_pairs),
                       actual_test_sha256=digest_rows(ds.test_pairs),projection={k:v for k,v in meta.items() if k!='fingerprint'},
                       uses_frequency_as_input=False,condition=CONDITION)
    return ds,features,cats,frequency,fingerprint,audit


def metric_subset(records,indices):
    selected=[records[i] for i in indices]
    if not selected:return dict(queries=0,recall20=None,ndcg20=None)
    return dict(queries=len(selected),recall20=float(np.mean([r['recall20'] for r in selected])),
                ndcg20=float(np.mean([r['ndcg20'] for r in selected])))


@torch.inference_mode()
def reference_predictions(ds,audit,out):
    """Use the canonical publication recipe/predictions, not a weaker arbitrary checkpoint.

    Verify every input hash and query order; replay fixed query probes against saved top-20s.
    Scores are comparison evidence only and are never exposed to the optimizer.
    """
    from bundlefuse.model import BundleFuse
    from experiments.mechanisms.selection import score_recipe
    root=Path('checkpoints/publication_mechanisms')/ds.name
    reports=Path('results/publication_mechanisms')/ds.name
    identity=json.loads((root/'identity.json').read_text())['metadata']
    for key in ('n_items','train_sha256','val_pairs_sha256','test_pairs_sha256','input_sha256'):
        if identity[key]!=ds.metadata[key]:raise RuntimeError(f'Canonical reference data mismatch: {key}')
    path=root/'artifacts.pt';model=BundleFuse.load(path)
    recipe=json.loads((reports/'full.json').read_text())
    saved=json.loads((reports/'queries.json').read_text())
    values=saved['metrics']['full'];predictions=saved['predictions']['full']
    if len(values)!=len(ds.test_pairs):raise RuntimeError('Reference query count mismatch')
    rng=np.random.default_rng(90)
    probes=sorted(set(range(4))|set(rng.choice(len(values),12,replace=False).tolist()))
    _,z=model._score_all([ds.test_pairs[i][0] for i in probes])
    scores=score_recipe(z,recipe,torch.as_tensor(model.freq)[:,None])
    for j,i in enumerate(probes):
        if top_items(np.asarray(scores)[:,j],ds.test_pairs[i][0],20).tolist()!=predictions[i]:
            raise RuntimeError(f'Canonical reference replay mismatch on query {i}')
    records=[dict(recall20=float(r),ndcg20=float(n),top20=p) for (r,n),p in zip(values,predictions)]
    result=dict(full=metric_subset(records,range(len(records))),
                overlap_filtered=metric_subset(records,audit['clean_test_indices']),
                checkpoint=str(path),sha256=file_hash(path),recipe=recipe,
                query_predictions_sha256=file_hash(reports/'queries.json'),replayed_query_indices=probes,
                source='Retained canonical BundleFuse predictions on identical supplied queries; input hashes and 16 deterministic probe rankings checked.')
    write_json(out/'bundlefuse_reference.json',result)
    model.w=recipe['weights'];model.prior_keys=recipe['prior'];model.best_ho=recipe['higher_order'];model.best_eb=recipe['backoff']
    log(f'{ds.name}: canonical BundleFuse full R20={result["full"]["recall20"]:.6f} N20={result["full"]["ndcg20"]:.6f}; replay verified')
    return model,result


def report(out,selected,baseline,status,tests=None):
    write_json(out/'summary.json',dict(status=status,selected=selected,bundlefuse=baseline,tests=tests or {}))
    lines=['# Independent neural POG experiment','',f'Status: {status}','',
           'Fixed architecture from Electronic; three seeds; full supplied splits and full-catalogue ranking. Secondary test results exclude exact complete-bundle overlaps.','',
           '| Model | Full Recall@20 | Full NDCG@20 | Filtered Recall@20 | Filtered NDCG@20 |',
           '|---|---:|---:|---:|---:|']
    if baseline:
        b=baseline['full'];f=baseline['overlap_filtered']
        lines.append(f'| Canonical BundleFuse | {b["recall20"]:.6f} | {b["ndcg20"]:.6f} | {f["recall20"]:.6f} | {f["ndcg20"]:.6f} |')
    for key,row in (tests or {}).items():
        f=row['overlap_filtered']
        lines.append(f'| {key} | {row["recall20"]:.6f} | {row["ndcg20"]:.6f} | {f["recall20"]:.6f} | {f["ndcg20"]:.6f} |')
    lines+=['','Neural model inputs are a catalogue-only rank-128 projection of released title, image and description vectors. BundleFuse uses its own frozen feature processing; identical raw inputs do not imply identical representations. No BundleFuse teacher or score input is used.','',
            'All neural epochs/architectures are selected without test scores. Full validation includes supplied overlaps; secondary test filtering is a sensitivity check, not a fresh independently selected protocol.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=args.output/args.dataset;ckpt=args.checkpoints/args.dataset
    out.mkdir(parents=True,exist_ok=True);ckpt.mkdir(parents=True,exist_ok=True)
    with (out/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ds,x,c,f,identity,audit=prepare(args.dataset,ckpt,out,args.smoke)
        config=dict(epochs=args.epochs,batch=32,eval_every=2,patience=6,link_lr=.03,shared_lr=.001,weight_decay=.001,log_every_batches=50)
        h,paths=source_identity();paths=paths+[Path('scripts/run_neural_pog.sh')]
        h=hashlib.sha256((h+file_hash(paths[-1])).encode()).hexdigest()
        protocol=dict(data=identity,config=config,condition=CONDITION,seeds=args.seeds,source_sha256=h,
                      torch_version=torch.__version__,projection_dimension=128,
                      selection='Fixed Electronic-selected architecture; select epochs on full supplied validation NDCG; no test selection.',
                      smoke=args.smoke,probe_validation_queries=128)
        for path in (out/'protocol.json',ckpt/'protocol.json'):
            if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Source/protocol changed: use separate roots')
        write_json(out/'protocol.json',protocol);write_json(ckpt/'protocol.json',protocol)
        if not (ckpt/'source.tar.gz').exists():
            with tarfile.open(ckpt/'source.tar.gz','w:gz') as tar:
                for p in paths:tar.add(p,arcname=str(p))
        model=baseline=None
        if not args.smoke:model,baseline=reference_predictions(ds,audit,out)
        selected={};report(out,selected,baseline,'training')
        for seed in args.seeds:
            key=f'graph_pair_no_category_seed{seed}'
            selected[key]=fit(ds,x,c,CONDITION,seed,config,out/key,ckpt/key,log)
            report(out,selected,baseline,'training')
        write_json(out/'frozen_selection.json',selected)
        if model is not None:
            index=sorted(np.random.default_rng(712).choice(len(ds.val_pairs),min(128,len(ds.val_pairs)),replace=False).tolist())
            neural=load_selected(ds,x,c,next(iter(selected.values())))
            log(f'{ds.name}: signal diagnostics on {len(index)} fixed validation queries')
            probe=compare_sources(neural,model,[ds.val_pairs[i] for i in index]);probe['validation_indices']=index
            write_json(out/'signal_probe.json',probe)
        tests={}
        for key,choice in selected.items():
            path=out/key/'test.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                neural=load_selected(ds,x,c,choice);result=evaluate(neural,ds.test_pairs)
                keep=list(range(len(ds.test_pairs))) if args.smoke else audit['clean_test_indices']
                result['overlap_filtered']=metric_subset(result['queries'],keep)
                result['by_target_frequency']=target_frequency_metrics(result,ds.test_pairs,f)
                write_json(path,result)
            tests[key]={k:v for k,v in result.items() if k!='queries'}
            log(f'{key}: TEST R20={result["recall20"]:.6f} N20={result["ndcg20"]:.6f}')
        if baseline and file_hash(baseline['checkpoint'])!=baseline['sha256']:raise RuntimeError('Reference checkpoint changed')
        report(out,selected,baseline,'smoke_complete' if args.smoke else 'complete',tests)
        log(f'{ds.name}: COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['pog','pog_dense'])
    parser.add_argument('--output',type=Path,default=Path('results/neural_pog'))
    parser.add_argument('--checkpoints',type=Path,default=Path('checkpoints/neural_pog'))
    parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke and (args.output==Path('results/neural_pog') or args.checkpoints==Path('checkpoints/neural_pog')):parser.error('Smoke needs isolated output and checkpoint roots')
    run(args)
