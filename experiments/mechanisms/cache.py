"""Persistent query-major signal caches. Completed blocks survive interruption."""
import numpy as np
import torch
from bundlefuse.provenance import write_json


class SignalCache:
    def __init__(self, directory, model, pairs, chunk, log):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.chunk, self.count = directory, chunk, len(pairs)
        self.keys = list(dict.fromkeys(model.SIG + ['pair','diff2_pair','rao']))
        marker = directory/'progress.json'
        import json
        import shutil
        done = json.loads(marker.read_text())['queries'] if marker.exists() else 0
        required = (len(pairs)-done)*model.cfg.n_items*4*len(self.keys)
        if shutil.disk_usage(directory).free < required + 5*1024**3:
            raise RuntimeError(f'Insufficient disk for retained validation cache: {required} bytes needed')
        self.arrays = {key: np.lib.format.open_memmap(directory/f'{key}.npy',
                      mode='r+' if done else 'w+', dtype=np.float32,
                      shape=(len(pairs),model.cfg.n_items)) for key in self.keys}
        for start in range(done,len(pairs),chunk):
            _, z = model._score_all([s for s,_ in pairs[start:start+chunk]])
            for key, array in self.arrays.items():
                array[start:start+chunk] = np.asarray(z[key].numpy().T)
                array.flush()
            write_json(marker,dict(queries=min(start+chunk,len(pairs)),keys=self.keys))
            log(f'validation signals {min(start+chunk,len(pairs))}/{len(pairs)}')

    def blocks(self):
        for start in range(0,self.count,self.chunk):
            yield start,{k:torch.from_numpy(v[start:start+self.chunk].T)
                         for k,v in self.arrays.items()}
