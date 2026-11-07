"""Read-only legacy caches, independent study storage and grouped validation splits."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from bundlefuse.model import BundleFuse
from bundlefuse.paths import DATA_ROOT
from bundlefuse.splits import ProvidedSplits, file_hash
from bundlefuse.provenance import write_json, digest_rows
from bundlefuse.selected_signals import SelectedSignals
from experiments.mechanisms.run import fingerprint as legacy_fingerprint
from .raw import standardize


def load_inputs(dataset):
    root=Path('checkpoints/publication_mechanisms')/dataset
    identity=json.loads((root/'identity.json').read_text())
    if identity['smoke'] or identity['source_sha256']!=legacy_fingerprint():
        raise RuntimeError('Legacy source identity changed; cannot reuse retained caches')
    directory=Path(DATA_ROOT)/'bundlemllm/datasets'/dataset
    n=json.loads((directory/'count.json').read_text())['#I']
    splits=ProvidedSplits(directory,n)
    train=list(splits.train.values());val=splits.pairs('valid');test=splits.pairs('test')
    for key,rows in [('train',train),('val_pairs',val),('test_pairs',test)]:
        if digest_rows(rows)!=identity['metadata'][key+'_sha256']:
            raise RuntimeError(f'{dataset} {key} differs from retained artifacts')
    progress=json.loads((root/'validation_signals/progress.json').read_text())
    if progress['queries']!=len(val):raise RuntimeError('Legacy validation cache incomplete')
    model=BundleFuse.load(root/'artifacts.pt')
    frozen=json.loads((Path('results/publication_mechanisms')/dataset/'full.json').read_text())
    old={k:np.load(root/'validation_signals'/f'{k}.npy',mmap_mode='r') for k in progress['keys']}
    return model,val,test,old,frozen,dict(identity=identity,split_manifest=splits.manifest(),
            artifact_path=str(root/'artifacts.pt'),artifact_sha256=file_hash(root/'artifacts.pt'))


def validation_groups(pairs, seed, fraction):
    groups={}
    for i,(s,gt) in enumerate(pairs):groups.setdefault(tuple(sorted(set(s)|set(gt))),[]).append(i)
    keys=list(groups);order=np.random.default_rng(seed).permutation(len(keys))
    split=max(1,min(len(keys)-1,int(len(keys)*fraction)))
    fit=sorted(i for j in order[:split] for i in groups[keys[j]])
    tune=sorted(i for j in order[split:] for i in groups[keys[j]])
    return np.asarray(fit),np.asarray(tune)


def retained_array(directory,key,count,n,compute,chunk,log):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{key}.npy';marker=directory/f'{key}.progress.json'
    done=json.loads(marker.read_text())['queries'] if marker.exists() else 0
    array=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',
                                   shape=(count,n),dtype=np.float32)
    for start in range(done,count,chunk):
        stop=min(start+chunk,count);block=compute(start,stop)
        if not np.all(np.isfinite(block)):raise ValueError(f'Nonfinite {key}')
        array[start:stop]=block;array.flush();write_json(marker,dict(queries=stop))
        if stop%100<chunk or stop==count:log(f'cache {key}: {stop}/{count}')
    del array
    return np.load(path,mmap_mode='r')


def selected_columns(frozen):
    # Structural variants are explicit columns, not hidden changes inside fusion.
    keys=[k for k,v in frozen['weights'].items() if v and k!='simp']
    if frozen['higher_order'][1]:keys.append('rao')
    return keys+['hist']


def cache_extra(pairs,old,model,amplifier,amplification,directory,chunk,log):
    n=model.cfg.n_items
    if old is None:
        needed=set(selected_columns(json.loads((directory.parent.parent/'source_recipe.json').read_text())))
        needed.discard('hist');needed.update(['pair','simp'])
        scorer=SelectedSignals(model.graph,model.onto,model.FEAT,model.cat_lbl,model.rao_dense,
                               model.music,torch.device('cpu'),beta_clique=model.cfg.beta_clique)
        # Compute all requested columns once per block, with independent flush barriers.
        metadata=directory/'signals.progress.json';directory.mkdir(parents=True,exist_ok=True)
        done=json.loads(metadata.read_text())['queries'] if metadata.exists() else 0
        arrays={k:np.lib.format.open_memmap(directory/f'{k}.npy',mode='r+' if (directory/f'{k}.npy').exists() else 'w+',
                          shape=(len(pairs),n),dtype=np.float32) for k in sorted(needed)}
        for start in range(done,len(pairs),chunk):
            block=[s for s,_ in pairs[start:start+chunk]]
            _,z=scorer.score(block,needed)
            for k,a in arrays.items():a[start:start+len(block)]=z[k].numpy().T;a.flush()
            write_json(metadata,dict(queries=min(start+chunk,len(pairs)),keys=sorted(needed)))
            if (start+len(block))%100<chunk:log(f'test shared signals {start+len(block)}/{len(pairs)}')
        old={k:np.load(directory/f'{k}.npy',mmap_mode='r') for k in needed}
    def amp_block(a,b):
        return np.stack([standardize(amplifier.score(pairs[i][0],amplification['power'],
                            amplification['neighbors'])[:,None]).ravel() for i in range(a,b)])
    amp=retained_array(directory,'amplified',len(pairs),n,amp_block,chunk,log)
    return dict(old,amplified=amp)


class Bank:
    def __init__(self,arrays,kind,keys):
        self.arrays,self.kind,self.keys=arrays,kind,list(keys)
        self.n=next(iter(arrays.values())).shape[1]
    def block(self,start,stop):
        history={'pair':'pair','incidence':'simp','amplified':'amplified'}[self.kind]
        return {k:np.asarray(self.arrays[history if k=='hist' else k][start:stop]) for k in self.keys}
    def query(self,i):return {k:v[0] for k,v in self.block(i,i+1).items()}
