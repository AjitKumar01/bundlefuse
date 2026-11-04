"""Check inference invariance to hidden targets on an existing POG checkpoint."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from bundlefuse.evaluation import evaluate, flat_metrics, describe_model
from bundlefuse.metrics import top_items
from bundlefuse.model import load_model
from bundlefuse.operators import zcols
from bundlefuse.paths import dataset_dir
from bundlefuse.provenance import digest_rows, run_metadata, write_json
from bundlefuse.splits import ProvidedSplits, file_hash


def audit(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint_sha=file_hash(args.checkpoint)
    model=load_model(args.checkpoint)
    if model.data_spec['dataset']!='pog':
        raise ValueError('This audit requires a POG checkpoint')
    source=ProvidedSplits(dataset_dir('bundlemllm','pog'),model.cfg.n_items)
    for name,value in source.hashes.items():
        if value!=model.data_spec['input_sha256'][name]:
            raise RuntimeError(f'Input changed: {name}')
    if digest_rows(list(source.train.values()))!=model.data_spec['train_sha256']:
        raise RuntimeError('Training fold differs from checkpoint')
    pairs=source.pairs('test')
    if model.data_spec.get('test_limit'):pairs=pairs[:model.data_spec['test_limit']]
    if digest_rows(pairs)!=model.data_spec['test_pairs_sha256']:
        raise RuntimeError('Test queries differ from checkpoint')
    indices=np.random.default_rng(args.seed).choice(len(pairs),min(args.queries,len(pairs)),replace=False)
    ids=sorted(source.splits['test'][0])
    sample=[pairs[i] for i in indices]
    parameters_before=describe_model(model)
    baseline=[]
    class Recorder:
        def score_batch(self,seeds,users):
            values=model.score_batch(seeds,users)
            baseline.append(values.copy())
            return values
    original=flat_metrics(evaluate(Recorder(),sample,Ks=(20,),chunk=args.chunk))
    baseline_scores=np.concatenate(baseline,axis=1)
    predicted=[top_items(baseline_scores[:,j],s,20).tolist() for j,(s,_) in enumerate(sample)]
    positive=[(s,[predicted[j][0]]) for j,(s,_) in enumerate(sample)]
    negative=[]
    for j,(seeds,_) in enumerate(sample):
        forbidden=set(seeds)|set(predicted[j])
        wrong=next(i for i in range(model.cfg.n_items) if i not in forbidden)
        negative.append((seeds,[wrong]))
    variants={}
    for label,changed in [('top_prediction_as_target',positive),('non_top20_item_as_target',negative)]:
        records=[]
        class Verify:
            def score_batch(self,seeds,users):
                values=model.score_batch(seeds,users);records.append(values.copy());return values
        metrics=flat_metrics(evaluate(Verify(),changed,Ks=(20,),chunk=args.chunk))
        actual=np.concatenate(records,axis=1)
        np.testing.assert_array_equal(baseline_scores.view(np.uint32),actual.view(np.uint32))
        changed_predictions=[top_items(actual[:,j],s,20).tolist() for j,(s,_) in enumerate(changed)]
        assert changed_predictions==predicted
        variants[label]={'metrics':metrics,'score_bits_identical':True,'top20_identical':True}
    assert variants['top_prediction_as_target']['metrics']['recall@20']==1.
    assert variants['non_top20_item_as_target']['metrics']['recall@20']==0.

    # Isolate normalization from matrix-multiplication rounding in the raw scorer.
    raw,_=model._score_all([s for s,_ in sample[:min(16,len(sample))]])
    scaling={}
    for key,column_matrix in raw.items():
        if column_matrix.shape[1]<2:raise ValueError('Need at least two queries')
        expected=zcols(column_matrix)[:,0].clone()
        changed=column_matrix.clone()
        changed[:,1:]=-1e6*changed[:,1:]+12345.
        np.testing.assert_array_equal(expected.numpy().view(np.uint32),zcols(changed)[:,0].numpy().view(np.uint32))
        alone=zcols(column_matrix[:,:1])[:,0]
        error=float(torch.max(torch.abs(expected-alone)))
        torch.testing.assert_close(expected,alone,rtol=2e-5,atol=2e-5)
        scaling[key]={'other_queries_changed_score_bits_identical':True,
                      'alone_vs_batch_max_abs_error':error}
    assert describe_model(model)==parameters_before
    assert file_hash(args.checkpoint)==checkpoint_sha
    result={'dataset':'pog','status':'passed','checkpoint':str(args.checkpoint.resolve()),
            'checkpoint_sha256':checkpoint_sha,'checkpoint_unchanged':True,
            'model':parameters_before,'fitted_settings_unchanged':True,
            'sample_queries':len(sample),'query_ids':[ids[i] for i in indices],
            'seed':args.seed,'chunk':args.chunk,'original_metrics':original,
            'target_mutations':variants,'normalization_by_signal':scaling,
            'scope':'Inference and normalization invariance, not an audit of feature pretraining or historical model selection.',
            'runtime':run_metadata()}
    write_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','sample_queries','original_metrics','target_mutations','normalization_by_signal','checkpoint_unchanged')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/reorganized_162dd8c/pog/seed_0/model.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'results/standardization_audit/pog.json')
    parser.add_argument('--queries',type=int,default=128)
    parser.add_argument('--chunk',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261104)
    args=parser.parse_args()
    if args.queries<2 or args.chunk<1:parser.error('At least two queries and a positive chunk are needed')
    with threadpool_limits(limits=1),torch.inference_mode():audit(args)
