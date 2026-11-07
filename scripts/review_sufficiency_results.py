from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.novelty.data import load_inputs
from experiments.sufficiency.engine import Bank,score
from bundlefuse.selected_signals import SelectedSignals
from bundlefuse.metrics import top_items
from bundlefuse.provenance import write_json
root=Path('results/sufficiency');combined={}
torch.set_num_threads(1)
for ds in ['pog_dense','pog','spotify']:
 d=root/ds;s=json.loads((d/'summary.json').read_text());f=json.loads((d/'frozen_methods.json').read_text());pop=json.loads((d/'evaluation_population.json').read_text())
 for k,cfg in f.items():
  a=json.loads((d/'queries'/f'{k}.json').read_text());assert len(a['metrics'])==len(pop['test_original_indices']);np.testing.assert_allclose(np.mean(a['metrics'],0),s['methods'][k],atol=1e-14)
 legacy=json.loads((Path('results/publication_mechanisms')/ds/'queries.json').read_text())['metrics']['full']
 original=np.asarray(legacy)
 combined[ds]=dict(test_queries=len(pop['test_original_indices']),methods=len(f),reference=s['methods']['reference'],
 original_full_population=original.mean(0).tolist(),original_filtered_population=original[pop['test_original_indices']].mean(0).tolist())
 if ds=='spotify':continue
 eligible=[k for k,v in f.items() if k.startswith('E1_subset') and all(a>=.99*b for a,b in zip(v['tuning_metrics'],f['reference']['tuning_metrics']))]
 eligible.sort(key=lambda k:(len(f[k]['config']['weights']),-f[k]['tuning_metrics'][1],k))
 compact=eligible[0] if eligible else None
 combined[ds]['compact_selected_on_tuning']=compact
 names=['reference']+([compact] if compact else [])
 zero=[k for k,v in s['methods'].items() if k.startswith('E1_fixed_remove') and v==s['methods']['reference']]
 names+=zero
 model,val,test,old,source,identity=load_inputs(ds)
 test=[test[i] for i in pop['test_original_indices']]
 provider=SelectedSignals(model.graph,model.onto,model.FEAT,model.cat_lbl,model.rao_dense,False,torch.device('cpu'),beta_clique=model.cfg.beta_clique)
 timings={}
 with threadpool_limits(limits=1),torch.inference_mode():
  for name in names:
   m=f[name];cfg=m['config'];active={k for k,v in cfg['weights'].items() if v}
   if cfg['kind']!='weighted' and cfg.get('kappa',0)>0:active|={k for k,v in cfg['prior_weights'].items() if v}
   required=active-{'hist'}
   if 'hist' in active:
    if m['alpha']<1:required.add('pair')
    if m['alpha']>0:required.add('simp')
   def compute(ix):
    seeds=[test[i][0] for i in ix];_,z=provider.score(seeds,required);arrays={k:v.numpy().T for k,v in z.items()};b=Bank(arrays,sorted(active),m['history'],m['alpha'])
    return [top_items(score(b.query(j),cfg,np.asarray(model.freq)),seed,20) for j,seed in enumerate(seeds)]
   expected=json.loads((d/'queries'/f'{name}.json').read_text())['predictions'][0]
   np.testing.assert_array_equal(compute([0])[0],expected)
   ix=np.linspace(0,len(test)-1,min(64,len(test)),dtype=int)
   values={}
   for batch in [1,8]:
    times=[]
    for repeat in range(3):
     for start in range(0,len(ix),batch):
      part=ix[start:start+batch];begin=time.perf_counter();compute(part);times.append((time.perf_counter()-begin)*1000/len(part))
    values[str(batch)]=dict(median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)))
   timings[name]=dict(active_columns=sorted(active),required_sources=sorted(required),batch_query_latency=values)
  write_json(d/'inference_timing_active_dependencies.json',dict(scope='Corrected measurement excludes zero-weight sources; original timing file retained.',methods=timings))
 combined[ds]['active_dependency_timing']=timings
 print(ds,'verified and rebenchmarked',flush=True)
write_json(root/'review_summary.json',dict(status='all_primary_runs_complete',methods_verified=sum(x['methods'] for x in combined.values()),datasets=combined))
print(json.dumps(combined,indent=2),flush=True)
