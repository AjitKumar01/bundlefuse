"""Read retained checkpoints, verify split identity, and cache only required columns."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from bundlefuse import datasets
from bundlefuse.model import BundleFuse
from bundlefuse.provenance import digest_rows, write_json
from bundlefuse.splits import file_hash
from bundlefuse.selected_signals import SelectedSignals
from experiments.novelty.data import load_inputs as load_supplied, retained_array
from experiments.novelty.raw import standardize


def load_inputs(name):
    if name == 'spotify_sparse':
        model, val, test, old, recipe, identity = load_supplied(name)
        model.w = recipe['weights']; model.prior_keys = recipe['prior']
        model.best_ho = recipe['higher_order']; model.best_eb = recipe['backoff']
        return model, val, test, None, None, old, recipe, identity
    path = Path('checkpoints/reorganized_162dd8c')/name/'seed_0/model.pt'
    expected = json.loads((Path('results/reorganized_run/original')/name/'result.json').read_text())
    model = BundleFuse.load(path)
    ds = datasets.load(name, val_n=800, test_n=2000, split_seed=0,
                       protocol_mode='resplit', data_profile='extended')
    spec = model.data_spec
    for key, rows in [('train_sha256', ds.train), ('val_pairs_sha256', ds.val_pairs),
                      ('test_pairs_sha256', ds.test_pairs)]:
        if digest_rows(rows) != spec[key]:
            raise RuntimeError(f'{name}: reconstructed {key} differs from checkpoint')
    if ds.metadata['input_sha256'] != spec['input_sha256']:
        raise RuntimeError(f'{name}: source data files changed')
    recipe = dict(weights=model.w, prior=model.prior_keys,
                  higher_order=list(model.best_ho), backoff=list(model.best_eb))
    identity = dict(artifact_path=str(path), artifact_sha256=file_hash(path),
                    metadata=ds.metadata, original_metrics=expected['metrics'],
                    val_users_sha256=digest_rows(ds.val_users or []),
                    test_users_sha256=digest_rows(ds.test_users or []))
    return model, ds.val_pairs, ds.test_pairs, ds.val_users, ds.test_users, None, recipe, identity


def source_columns(recipe):
    keys = [k for k,v in recipe['weights'].items() if v and k != 'simp']
    keys += [k for k in recipe['prior'] if k not in keys and k != 'simp']
    if recipe['higher_order'][1] and 'rao' not in keys: keys.append('rao')
    return keys+['hist']


def scorer_for(model):
    return SelectedSignals(model.graph, model.onto, model.FEAT, model.cat_lbl,
        model.rao_dense, model.music, torch.device('cpu'), beta_clique=model.cfg.beta_clique,
        aux=model.aux, groups=model.group_inc, category_homo=model.Pcat_homo,
        user_incidence=model.UI, user_items=model.user_items)


def effective(signals, key, alpha):
    if key in ('hist', 'diff2'):
        pair, inc = ('pair', 'simp') if key == 'hist' else ('diff2_pair', 'diff2')
        if alpha == 0: return signals[pair]
        if alpha == 1: return signals[inc]
        return (1-alpha)*signals[pair]+alpha*signals[inc]
    return signals[key]


def cache_signals(pairs, users, model, recipe, keys, old, directory, chunk, log):
    """Query-major standardized scores; preserve optional personalized inputs."""
    directory.mkdir(parents=True, exist_ok=True)
    alpha = recipe['higher_order'][0]
    needed = set(keys)-{'hist'}
    needed.update(['pair', 'simp'] if alpha else ['pair'])
    if 'diff2' in keys: needed.update(['diff2_pair', 'diff2'] if alpha else ['diff2_pair'])
    if old is not None:
        missing = needed-set(old)
        if missing: raise RuntimeError(f'Retained cache lacks {missing}')
        _, fresh = scorer_for(model).score([pairs[0][0]], needed,
                       None if users is None else users[:1])
        errors = {k:float(np.max(np.abs(fresh[k].numpy().ravel()-old[k][0]))) for k in needed}
        if max(errors.values()) > 1e-4: raise RuntimeError(f'Input cache parity failed: {errors}')
        write_json(directory/'input_cache_parity.json', errors)
        signals = dict(old)
        signals['hist'] = retained_array(directory,'hist',len(pairs),model.cfg.n_items,
            lambda a,b:effective({k:np.asarray(v[a:b]) for k,v in old.items()},'hist',alpha),chunk,log)
        if 'diff2' in keys:
            signals['diff2'] = retained_array(directory,'effective_diff2',len(pairs),model.cfg.n_items,
                lambda a,b:effective({k:np.asarray(v[a:b]) for k,v in old.items()},'diff2',alpha),chunk,log)
        return {k:signals[k] for k in keys}
    marker = directory/'progress.json'
    done = json.loads(marker.read_text())['queries'] if marker.exists() else 0
    if not 0 <= done <= len(pairs): raise RuntimeError('Invalid cache progress')
    required_bytes=(len(pairs)-done)*model.cfg.n_items*4*len(keys)
    if shutil.disk_usage(directory).free < required_bytes+5*1024**3:
        raise RuntimeError(f'Insufficient disk for retained cache ({required_bytes} bytes)')
    arrays={}
    for k in keys:
        path=directory/f'{k}.npy'
        if path.exists():
            existing=np.load(path,mmap_mode='r')
            if existing.shape != (len(pairs),model.cfg.n_items) or existing.dtype != np.float32:
                raise RuntimeError('Cache shape/dtype mismatch')
            del existing
        arrays[k]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',
            shape=(len(pairs),model.cfg.n_items),dtype=np.float32)
    scorer=scorer_for(model)
    for start in range(done,len(pairs),chunk):
        stop=min(start+chunk,len(pairs))
        _,z=scorer.score([s for s,_ in pairs[start:stop]],needed,
                           None if users is None else users[start:stop])
        z={k:v.numpy() for k,v in z.items()}
        for k,array in arrays.items():
            values=effective(z,k,alpha).T
            if not np.all(np.isfinite(values)): raise RuntimeError(f'Nonfinite source {k}')
            array[start:stop]=values;array.flush()
        write_json(marker,dict(queries=stop,keys=keys))
        if stop%100<chunk or stop==len(pairs):log(f'shared signals {directory.name} {stop}/{len(pairs)}')
    del arrays
    return {k:np.load(directory/f'{k}.npy',mmap_mode='r') for k in keys}


def cache_amplified(pairs, amplifier, selected, directory, chunk, log):
    return retained_array(directory,'amplified',len(pairs),amplifier.B.shape[1],
        lambda a,b:np.stack([standardize(amplifier.score(pairs[i][0],selected['power'],
                          selected['neighbors'])[:,None]).ravel() for i in range(a,b)]),chunk,log)


class ScoreBank:
    def __init__(self, arrays, amplified, keys, kind):
        self.arrays,self.amplified,self.keys,self.kind=arrays,amplified,list(keys),kind
        self.n=arrays['hist'].shape[1]
    def query(self,i):
        return {k:np.asarray(self.amplified[i] if k=='hist' and self.kind=='amplified'
                             else self.arrays[k][i]) for k in self.keys}
