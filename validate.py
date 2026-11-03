#!/usr/bin/env python3
"""Verify fit/checkpoint/inference score parity on explicit dataset protocols.

This records the current implementation's metrics. It does not compare unlike split protocols
or treat historical rounded table values as golden numbers for a changed model/configuration.
"""
import argparse
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from bundlefuse import datasets
from bundlefuse.model import BundleFuse
from bundlefuse.metrics import ranking_metrics
from bundlefuse.evaluation import flat_metrics, describe_model
from bundlefuse.provenance import run_metadata, write_json


def run_once(name, split_seed=0, test_n=None, val_n=None, *, protocol_mode=None, chunk=32, config_file=None):
    mode = protocol_mode or ('provided' if name in ('pog','pog_dense','spotify','spotify_sparse') else 'resplit')
    ds = datasets.load(name, split_seed=split_seed, test_n=test_n, val_n=val_n, protocol_mode=mode)
    cfg = datasets.get(name).config(ds.n_items, config_file=config_file)
    cfg.validation_chunk = chunk
    model = BundleFuse(cfg).fit(ds.train, ds.val_pairs, **ds.fit_kwargs())
    model.data_spec = ds.metadata
    totals = {k: np.zeros(2) for k in (10,20,50)}
    with tempfile.TemporaryDirectory(prefix='bundlefuse_verify_') as directory:
        model.save(Path(directory)/'model.pt')
        loaded = BundleFuse.load(Path(directory)/'model.pt')
        for start in range(0, len(ds.test_pairs), chunk):
            pairs = ds.test_pairs[start:start+chunk]
            users = ds.test_users[start:start+chunk] if ds.test_users is not None else None
            before = model.score_batch([s for s,_ in pairs],users)
            after = loaded.score_batch([s for s,_ in pairs],users)
            if not np.array_equal(before.view(np.uint32), after.view(np.uint32)):
                raise RuntimeError(f'Checkpoint parity failed for {name} at query {start}')
            metrics = ranking_metrics(after,pairs,(10,20,50))
            for k in totals:
                totals[k] += np.asarray(metrics[k])*len(pairs)
    return {'data':ds.metadata, 'test_query_ids':ds.test_ids, 'model':describe_model(model),
            'metrics':flat_metrics({k:v/len(ds.test_pairs) for k,v in totals.items()}),
            'checkpoint_score_bit_identity':True, 'queries_verified':len(ds.test_pairs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=sorted(datasets.REGISTRY),default=sorted(datasets.REGISTRY))
    parser.add_argument('--protocol',choices=('provided','resplit'),default=None)
    parser.add_argument('--split-seeds',nargs='+',type=int,default=[0])
    parser.add_argument('--val-n',type=int,default=None)
    parser.add_argument('--test-n',type=int,default=None)
    parser.add_argument('--chunk',type=int,default=32)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--config',default=None)
    parser.add_argument('--out',default='reports/pipeline_validation.json')
    args=parser.parse_args(argv)
    if min(args.threads,args.chunk)<1:
        parser.error('Threads and chunk must be positive')
    torch.set_num_threads(args.threads)
    output={'runtime':run_metadata(),'status':'running','runs':[]}
    write_json(args.out,output)
    try:
        for name in args.datasets:
            mode=args.protocol or ('provided' if name in ('pog','pog_dense','spotify','spotify_sparse') else 'resplit')
            if mode=='provided' and args.split_seeds != [0]:
                raise ValueError('Provided protocol has one fixed split; use --split-seeds 0')
            for seed in args.split_seeds:
                start=time.perf_counter()
                result=run_once(name,seed,args.test_n,args.val_n,protocol_mode=mode,chunk=args.chunk,config_file=args.config)
                result.update(dataset=name,seconds=time.perf_counter()-start)
                output['runs'].append(result);write_json(args.out,output)
                print(name,mode,seed,result['metrics'],'checkpoint parity OK',flush=True)
    except Exception as error:
        output.update(status='failed',error=str(error));write_json(args.out,output)
        raise
    output['status']='complete';write_json(args.out,output)
    return output


if __name__=='__main__':
    main()
