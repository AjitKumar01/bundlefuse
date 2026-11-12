"""Controls for the claim that bundles replace the between-category block of description similarity.

Exploratory on already inspected splits. Every arm is frozen before its test columns exist.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import torch
from bundlefuse.model import BundleFuse
from bundlefuse.operators import DualSpectrumOntology,zcols
from bundlefuse.paths import DATA_ROOT
from bundlefuse.provenance import write_json,file_hash
from bundlefuse.selected_signals import SelectedSignals
from bundlefuse.splits import ProvidedSplits
from experiments.mechanisms.statistics import paired_comparison,holm_adjust
from experiments.sufficiency.engine import Bank,fit,score,assess,Search

SEED=20261107
SHUFFLES=5
SLOT_THRESHOLD=.15
BASE={'pog':['hist'],'pog_dense':['pop','hist']}
ALPHA={'pog':0.,'pog_dense':1.}
PLAN=dict(
    seed=SEED,shuffles=SHUFFLES,base=BASE,
    scope='Exploratory: same filtered population as results/sufficiency, whose test split is already inspected.',
    primary=['cat_style vs content','cat_style vs per-query mean of shuffled arms'],
    metrics=['ndcg20','recall20'],holm_family=8,
    null='Curveball trades on training bundles: item frequencies and bundle sizes kept, co-occurrence destroyed. '
         'Only cat is rebuilt from them, including its slot penalty; hist stays real.',
    decision=dict(
        raw_description='Supported on a dataset if cat_style beats content on NDCG with Holm p<.05.',
        shuffled_history='Supported on a dataset if cat_style beats the shuffled mean on NDCG with Holm p<.05 and beats every seed. '
                         'If not, the gain does not depend on real bundle lift and the claim fails there.',
        spotify='Not tested: its selected recipes do not use cat, so these controls cannot speak to it.'),
    note='style uses title features only, while content joins title, image and description, so the primary raw-description arm has more input. '
         'allstyle_cat and title are the same-input secondary arms.')


def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)


def arms(dataset):
    base=BASE[dataset]
    out={'cat_style':['cat','style']+base,'content':['content']+base}
    out.update({f'shuf{s}':[f'cat_shuf{s}','style']+base for s in range(SHUFFLES)})
    out.update(base_only=list(base),style_only=['style']+base,samecat_style=['samecat','style']+base,
               catpmi_style=['cat_pmi','style']+base,catslot_style=['cat_slot','style']+base,
               content_cat=['content','cat']+base,allstyle_cat=['cat','style','imgstyle','descstyle']+base,
               title=['title']+base,nohist_cat_style=['cat','style']+[k for k in base if k!='hist'],
               nohist_content=['content']+[k for k in base if k!='hist'])
    return out


def curveball(bundles,seed):
    """Random trades between bundle pairs; each item's frequency and each bundle's size stay fixed."""
    rng=np.random.default_rng(seed)
    rows=[sorted(set(b)) for b in bundles];m=len(rows)
    for a,b in rng.integers(0,m,size=(5*m,2)):
        if a==b:continue
        A,B=set(rows[a]),set(rows[b]);only_a=sorted(A-B);only_b=sorted(B-A)
        if not only_a or not only_b:continue
        pool=rng.permutation(only_a+only_b).tolist();cut=len(only_a)
        rows[a]=sorted((A&B)|set(pool[:cut]));rows[b]=sorted((A&B)|set(pool[cut:]))
    return rows


def category_columns(onto,labels,seeds):
    """Equal to the production cat column, computed once per category."""
    pmi=np.asarray(onto.Pcat[[labels[i] for i in seeds]].sum(0)).ravel()
    slot=np.zeros(onto.NCc,np.float32)
    for i in seeds:
        k=labels[i]
        if onto.dup_rate[k]<SLOT_THRESHOLD:slot[k]-=1.-onto.dup_rate[k]
    return pmi[labels].astype(np.float32),slot[labels]


class Columns:
    def __init__(self,model,train,dataset):
        self.model,self.labels=model,model.cat_lbl
        self.provider=SelectedSignals(model.graph,model.onto,model.FEAT,model.cat_lbl,model.rao_dense,False,
                                      torch.device('cpu'),beta_clique=model.cfg.beta_clique)
        self.direct={'pair','simp','pop','content','style','imgstyle','descstyle','bridge','cat'}
        if dataset=='pog_dense':self.direct.add('csvd')  # listed by its recorded reference for the parity replay
        title=model.FEAT['content'][:,:512]
        self.title=title/(title.norm(dim=1,keepdim=True)+1e-8)
        self.nulls=[];self.audit=[]
        f=np.asarray(model.freq)
        n=model.cfg.n_items
        for s in range(SHUFFLES):
            rows=curveball(train,SEED+s)
            check=np.bincount(np.concatenate(rows),minlength=n)
            if not np.array_equal(check,f.astype(int)) or [len(r) for r in rows]!=[len(set(b)) for b in train]:
                raise RuntimeError('Curveball changed a margin')
            onto=DualSpectrumOntology(rows,cat_lbl=self.labels,n_items=n,device=torch.device('cpu'))
            self.nulls.append(onto)
            real=model.onto.Pcat.toarray();null=onto.Pcat.toarray()
            self.audit.append(dict(seed=SEED+s,pmi_abs_real=float(np.abs(real).mean()),pmi_abs_null=float(np.abs(null).mean()),
                pmi_correlation=float(np.corrcoef(real.ravel(),null.ravel())[0,1]),
                slot_categories_real=int((model.onto.dup_rate<SLOT_THRESHOLD).sum()),
                slot_categories_null=int((onto.dup_rate<SLOT_THRESHOLD).sum())))
        self.keys=sorted(self.direct|{'title','samecat','cat_pmi','cat_slot'}|{f'cat_shuf{s}' for s in range(SHUFFLES)})

    def block(self,bundles):
        _,z=self.provider.score(bundles,self.direct)
        out={k:v.numpy().T for k,v in z.items()}
        n=self.model.cfg.n_items;nb=len(bundles)
        extra={'samecat':np.zeros((nb,n),np.float32),'cat_pmi':np.zeros((nb,n),np.float32),'cat_slot':np.zeros((nb,n),np.float32),
               **{f'cat_shuf{s}':np.zeros((nb,n),np.float32) for s in range(SHUFFLES)}}
        counts=np.zeros(self.model.onto.NCc,np.float32)
        for j,seeds in enumerate(bundles):
            pmi,slot=category_columns(self.model.onto,self.labels,seeds)
            parity=torch.tensor(pmi+.5*slot)
            extra['cat_pmi'][j]=pmi;extra['cat_slot'][j]=slot
            counts[:]=0;np.add.at(counts,self.labels[seeds],1.);extra['samecat'][j]=counts[self.labels]
            if j==0:
                check=zcols(parity[:,None])[:,0].numpy()
                if float(np.max(np.abs(check-out['cat'][0])))>1e-3:raise RuntimeError('cat parity failed')
            for s,onto in enumerate(self.nulls):
                a,b=category_columns(onto,self.labels,seeds);extra[f'cat_shuf{s}'][j]=a+.5*b
        for k,v in extra.items():out[k]=zcols(torch.tensor(v.T)).numpy().T
        means=torch.stack([self.title[s].mean(0) for s in bundles],1)
        out['title']=zcols(self.title@means).numpy().T
        return out


def cache(columns,pairs,directory,chunk=16):
    directory.mkdir(parents=True,exist_ok=True);n=columns.model.cfg.n_items
    marker=directory/'progress.json';done=json.loads(marker.read_text())['queries'] if marker.exists() else 0
    arrays={k:np.lib.format.open_memmap(directory/f'{k}.npy',mode='r+' if (directory/f'{k}.npy').exists() else 'w+',
                                        dtype=np.float32,shape=(len(pairs),n)) for k in columns.keys}
    for start in range(done,len(pairs),chunk):
        block=columns.block([list(s) for s,_ in pairs[start:start+chunk]])
        for k,a in arrays.items():a[start:start+chunk]=block[k];a.flush()
        write_json(marker,dict(queries=min(start+chunk,len(pairs))))
        if start%(chunk*25)==0:log(f'{directory.name} columns {min(start+chunk,len(pairs))}/{len(pairs)}')
    del arrays
    return {k:np.load(directory/f'{k}.npy',mmap_mode='r') for k in columns.keys}


def run(args):
    dataset=args.dataset;out=args.output/dataset;out.mkdir(parents=True,exist_ok=True)
    store=args.store/dataset
    protocol=dict(PLAN,dataset=dataset,smoke=args.smoke)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=json.loads(json.dumps(protocol)):
        raise RuntimeError('Protocol changed; use another output root')
    write_json(out/'protocol.json',protocol)
    root=Path('results/sufficiency')/dataset
    population=json.loads((root/'evaluation_population.json').read_text())
    partition=json.loads((root/'partition.json').read_text())
    directory=Path(DATA_ROOT)/'bundlemllm/datasets'/dataset
    n=json.loads((directory/'count.json').read_text())['#I']
    splits=ProvidedSplits(directory,n)
    train=list(splits.train.values());val=splits.pairs('valid');test=splits.pairs('test')
    val=[val[i] for i in population['validation_original_indices']];test=[test[i] for i in population['test_original_indices']]
    fitting=np.asarray(partition['fitting']);tuning=np.asarray(partition['tuning'])
    log('load retained model')
    path=Path('checkpoints/publication_mechanisms')/dataset/'artifacts.pt'
    before=file_hash(path)
    model=BundleFuse.load(path);freq=np.asarray(model.freq)
    if args.smoke:
        keep=sorted(set(fitting[:24].tolist())|set(tuning[:12].tolist()));where={v:i for i,v in enumerate(keep)}
        fitting=np.asarray([where[i] for i in fitting[:24]]);tuning=np.asarray([where[i] for i in tuning[:12]])
        val=[val[i] for i in keep];test=test[:12]
    columns=Columns(model,train,dataset)
    write_json(out/'null_audit.json',columns.audit)
    log(f'null audit {columns.audit[0]}')
    title=columns.title.numpy()[:2000];style=model.FEAT['style'].numpy()[:2000]
    write_json(out/'title_slice_check.json',dict(mean_abs_cosine_title_style=float(np.abs((title*style).sum(1)).mean())))
    arrays=cache(columns,val,store/'validation')
    alpha=ALPHA[dataset]
    frozen=json.loads((root/'frozen_methods.json').read_text())['reference']
    if not args.smoke:
        replay=Search(Bank(arrays,frozen['keys'],'canonical',alpha),val,tuning,freq,out/'reference_tuning.json',log).many([frozen['config']])[0]
        if np.max(np.abs(np.asarray(replay)-frozen['tuning_metrics']))>1e-6:
            raise RuntimeError(f'Reference tuning parity failed: {replay} vs {frozen["tuning_metrics"]}')
        log(f'reference tuning parity ok {replay}')
    menu=arms(dataset);methods={}
    source=dict(prior=[k for k in columns.keys if k not in ('pop','pair','simp')])
    for name,keys in menu.items():
        methods[name]=fit(Bank(arrays,keys,'canonical',alpha),val,fitting,tuning,freq,source,out/'selection'/name,log)
        log(f'{name}: {methods[name]["config"]} tuning {methods[name]["tuning_metrics"]}')
    write_json(out/'frozen_methods.json',methods)
    log('SELECTION FROZEN; building test columns')
    tests=cache(columns,test,store/'test')
    methods=dict(methods,reference=dict(frozen,keys=frozen['keys']))
    for name,method in methods.items():
        assess(Bank(tests,method['keys'],'canonical',alpha),test,method['config'],freq,out/'queries'/f'{name}.json',log)
    if not args.smoke:
        recorded=json.loads((root/'summary.json').read_text())['methods']['reference']
        got=json.loads((out/'queries/reference.json').read_text())['mean']
        if np.max(np.abs(np.asarray(got)-recorded))>1e-6:raise RuntimeError(f'Reference test parity failed: {got} vs {recorded}')
        log(f'reference test parity ok {got}')
    if file_hash(path)!=before:raise RuntimeError('Input checkpoint changed')
    write_json(store/'test_pairs.json',[[list(map(int,s)),list(map(int,t))] for s,t in test])
    log('COMPLETE')


def report(args):
    records=[];tables={}
    for dataset in BASE:
        out=args.output/dataset
        rows={p.stem:json.loads(p.read_text()) for p in (out/'queries').glob('*.json') if not p.name.endswith('.progress.json')}
        pairs=json.loads((args.store/dataset/'test_pairs.json').read_text())
        model=BundleFuse.load(Path('checkpoints/publication_mechanisms')/dataset/'artifacts.pt')
        frequency=np.asarray(model.freq);del model
        table={}
        for name,row in rows.items():
            hits={True:[0,0],False:[0,0]}
            for predicted,(_,targets) in zip(row['predictions'],pairs):
                for t in targets:
                    cold=bool(frequency[t]==0);hits[cold][1]+=1;hits[cold][0]+=int(t in predicted)
            table[name]=dict(recall20=row['mean'][0],ndcg20=row['mean'][1],
                             cold_target_recall20=hits[True][0]/max(hits[True][1],1),warm_target_recall20=hits[False][0]/max(hits[False][1],1),
                             cold_targets=hits[True][1],warm_targets=hits[False][1])
        tables[dataset]=table
        groups=rows['cat_style']['groups']
        metric=lambda name:np.asarray(rows[name]['metrics'])
        null=np.mean([metric(f'shuf{s}') for s in range(SHUFFLES)],axis=0)
        for j,label in [(1,'ndcg20'),(0,'recall20')]:
            for against,second in [('content',metric('content')),('shuffled_mean',null)]:
                comparison=paired_comparison(metric('cat_style')[:,j],second[:,j],groups,bootstrap=2000,permutations=10000,seed=SEED)
                records.append(dict(comparison,dataset=dataset,metric=label,first='cat_style',second=against))
        table['cat_style_minus_each_shuffle_ndcg']=[float(metric('cat_style')[:,1].mean()-metric(f'shuf{s}')[:,1].mean()) for s in range(SHUFFLES)]
    holm_adjust(records,PLAN['holm_family'])
    secondary=[]
    for dataset in BASE:
        out=args.output/dataset
        rows={p.stem:json.loads(p.read_text()) for p in (out/'queries').glob('*.json') if not p.name.endswith('.progress.json')}
        metric=lambda name:np.asarray(rows[name]['metrics'])[:,1]
        for first,second in [('allstyle_cat','content'),('cat_style','title'),('content_cat','content'),('cat_style','samecat_style'),
                             ('cat_style','style_only'),('catpmi_style','style_only'),('catslot_style','style_only'),('nohist_cat_style','nohist_content'),
                             # Added after reading the first report, so post hoc.
                             ('cat_style','content_cat'),('style_only','title')]:
            secondary.append(dict(paired_comparison(metric(first),metric(second),rows['cat_style']['groups'],bootstrap=2000,permutations=10000,seed=SEED),
                                  dataset=dataset,metric='ndcg20',first=first,second=second))
    write_json(args.output/'summary.json',dict(protocol=PLAN,tables=tables,primary=records,secondary_unadjusted=secondary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=list(BASE))
    parser.add_argument('--output',type=Path,default=Path('results/block_controls'))
    parser.add_argument('--store',type=Path,default=Path('checkpoints/block_controls'))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    with torch.inference_mode():
        report(args) if args.report else run(args)
