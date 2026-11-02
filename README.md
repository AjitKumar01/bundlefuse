# BundleFuse

A zero-parameter, config-driven scorer for **bundle completion**: given a partial bundle (seed items),
rank the catalogue for the items that complete it. BundleFuse fuses a handful of closed-form signals —
organised into families — with validation-selected weights and empirical-Bayes cold-start shrinkage.
No gradient training; one model object per dataset.

## Signal families
Switched on per dataset via the config:

| family | signals | needs |
|---|---|---|
| `collaborative` | cooc, ppmi, simp, diff2, csvd | train bundles (always) |
| `content` | content, bridge | a content embedding array |
| `taxonomy` | cat | item category labels |
| `popularity` | pop | train bundles (always) |
| `personalization` | user_freq, user_cooc | per-user histories |

## Use

```python
from bundlefuse import datasets

model, ds = datasets.build("pog_dense")        # object configured for this dataset + its data
model.fit(ds.train, ds.val_pairs, content=ds.content, cat_labels=ds.cat_labels,
          user_items=ds.user_items, val_users=ds.val_users)   # all processing, once

model.save("pog_dense.pt")                 # checkpoint: artifacts + fitted weights
model = BundleFuse.load("pog_dense.pt")
items = model.predict(seeds, k=20, user=None)   # fast: one pass over the seed set
```

Each dataset has its own file under `bundlefuse/datasets/` (`pog_dense.py`, `dunnhumby.py`, …) that
owns its loader, its families, and the BundleFuse object configured for it.

- **`fit`** builds every artifact (hypergraph operator, content bridge, taxonomy co-occurrence,
  user histories) and selects the fusion weights + shrinkage on the validation pairs.
- **`predict` / `score`** only compute the active signals for the given seed set, fuse, shrink and
  rank — no training at predict time.
- **`save` / `load`** is a self-contained checkpoint.

### Command line

```bash
python -m bundlefuse.run pog_dense                       # train + evaluate (Recall@20 / NDCG@20)
python -m bundlefuse.run dunnhumby --save ckpt/dh.pt     # with a checkpoint
python -m bundlefuse.run electronic --families collaborative,content,taxonomy,popularity
```

Datasets: `pog`, `pog_dense`, `spotify`, `spotify_sparse` (bundlemllm); `dunnhumby`; `electronic`,
`food`, `clothing` (BundleRec). Data is read from `../bundlemllm`, `../dunnhumby`, `../BundleRec`.

## Layout

```
bundlefuse/
  model.py         # the BundleFuse class (fit / predict / score / save / load) + signal families
  operators.py     # the closed-form math (hypergraph operator, content bridge, EB shrinkage)
  run.py           # CLI entry point: build dataset object -> fit -> evaluate -> optional checkpoint
  datasets/        # ONE FILE PER DATASET -- each builds the object configured for it
    base.py        #   shared family loaders (bundlemllm / dunnhumby / BundleRec) + Dataset
    pog.py pog_dense.py spotify.py spotify_sparse.py
    dunnhumby.py electronic.py food.py clothing.py
  metrics.py       # Recall / NDCG
  provided_splits.py, protocol.py, paths.py   # split I/O, reveal-fraction masking, data roots
tests/test_model.py  # fit / predict / checkpoint / family-switch
```
