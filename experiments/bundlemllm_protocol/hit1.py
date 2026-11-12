"""DIALS HitRate@1 under the Bundle-MLLM protocol (KDD'25, utils.BundleTestDataset): one question per (test bundle,
held-out item); candidates = that item + 9 items drawn uniformly (with replacement, as np.random.randint does) from
items in neither the seeds nor the held-out set; hit if the true item has the highest score. 3 model seeds x 5 draws.
Checked against the paper (section 4.1.2), their code (github.com/Xiaohao-Liu/Bundle-MLLM, commit 7818c05) and their
data release (huggingface.co/datasets/xhLiu/BundleConstruction), 2026-11-11:
- the paper fixes 10 candidates with one positive and HitRate@1, but not how negatives are drawn; the code draws them
  uniformly (with replacement) from items outside the bundle, afresh at each access, when no bi_test_cans10.txt
  exists, and none is released (data or code repository);
- the code scores one pass over the first 2,048 (bundle, held-out item) pairs of a random permutation (toy_eval=2048);
  here every pair is scored, which estimates the same quantity with less noise;
- the code shows the model at most 5 seed items (num_token=5); no POG or POG_dense test query has more than 4;
- the paper states an 8:1:1 split, but the released files (the only ones; the copies under .ipynb_checkpoints hold
  identical bundles in another line order) are 14,000 / 2,000 / 4,000 on POG, 7:1:2. These are the files used here;
- Bundle-MLLM checkpoints (LoRA adapter, fusion, projector over Llama-2-7b-hf) are released at
  huggingface.co/xhLiu/Bundle-MLLM, so it can be rerun for inference.
Published comparison (Table 2, arXiv 2407.11712): CLHE .3477 / .6367, Bundle-MLLM .4131 / .6675 on POG / POG_dense.

cd ../dials && PYTHONPATH=. ../bundlefuse/.venv/bin/python ../bundlefuse/experiments/bundlemllm_protocol/hit1.py
"""
import dials,json,numpy as np,torch
from pathlib import Path
from dials import data
from dials.train import load
from dials.evaluate import score_batches
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
out={}
for name in ('pog_dense','pog'):
    ds=data.load(name,Path('cache'));clean=set(ds.clean_test);pairs=ds.test_pairs;n=ds.n_items
    res=dict(all=[],clean=[],full_hit1_all=[],full_hit1_clean=[])
    for seed in (0,1,2):
        m=load(ds,Path(f'runs/{name}/seed{seed}'),'cache')
        hits={d:[] for d in range(5)};hclean={d:[] for d in range(5)}
        rngs=[np.random.default_rng(1000*seed+d) for d in range(5)]
        q=json.loads(Path(f'runs/{name}/seed{seed}/test.json').read_text())['queries']
        res['full_hit1_all'].append(float(np.mean([q[i]['top20'][0] in set(g) for i,(s,g) in enumerate(pairs)])))
        res['full_hit1_clean'].append(float(np.mean([q[i]['top20'][0] in set(pairs[i][1]) for i in clean])))
        start=0
        for values in score_batches(m,[s for s,_ in pairs]):
            for k,v in enumerate(values):
                i=start+k;s,g=pairs[i];inside=set(s)|set(g)
                allowed=np.setdiff1d(np.arange(n),np.fromiter(inside,int),assume_unique=False)
                for d in range(5):
                    for t in g:
                        neg=allowed[rngs[d].integers(0,len(allowed),9)]
                        h=float(v[t]>v[neg].max())
                        hits[d].append(h)
                        if i in clean:hclean[d].append(h)
            start+=len(values)
        res['all'].append([float(np.mean(hits[d])) for d in range(5)]);res['clean'].append([float(np.mean(hclean[d])) for d in range(5)])
        print(name,'seed',seed,'HitRate@1 all',np.round(res['all'][-1],4),'full-catalogue Hit@1',round(res['full_hit1_all'][-1],4),flush=True)
    res['questions']=sum(len(g) for _,g in pairs)
    out[name]=res
    a=np.array(res['all']);c=np.array(res['clean'])
    print(f"{name}: HitRate@1 (10 candidates) all {a.mean():.4f} (seed sd {a.mean(1).std(ddof=1):.4f}, draw sd {a.std(1,ddof=1).mean():.4f}); clean {c.mean():.4f}; full-catalogue Hit@1 all {np.mean(res['full_hit1_all']):.4f} clean {np.mean(res['full_hit1_clean']):.4f}; questions {res['questions']}")
json.dump(out,open('../bundlefuse/results/bundlemllm_protocol/hit1.json','w'),indent=1)
