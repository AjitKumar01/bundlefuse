# BundleFuse

A zero-parameter, config-driven scorer for **bundle completion**: given a partial bundle (seed items),
rank the catalogue for the items that complete it. BundleFuse fuses a handful of closed-form signals —
organised into families — with validation-selected weights and empirical-Bayes cold-start shrinkage.
No gradient training; one model object per dataset; compressed checkpoints.

```python
from bundlefuse import datasets
model, ds = datasets.build("pog_dense")        # object configured for this dataset + its data
model.fit(ds.train, ds.val_pairs, content=ds.content, cat_labels=ds.cat_labels,
          user_items=ds.user_items, val_users=ds.val_users)
model.save("checkpoints/pog_dense.pt")
items = model.predict(seeds=[7917, 15610], k=10)     # fast: one pass over the seed set
```

## Install

```bash
git clone https://github.com/AjitKumar01/bundlefuse.git
cd bundlefuse
pip install -r requirements.txt          # numpy, scipy, scikit-learn, torch, tqdm
python -m pytest tests/ -q               # or: python tests/test_model.py
```

## Get the data

The datasets are **not vendored**. By default BundleFuse looks for them **one level above the repo**
(so the repo and the data are siblings); point `BUNDLEFUSE_DATA_ROOT` somewhere else if you prefer:

```
<DATA_ROOT>/                         # default: the parent of this repo
  bundlemllm/datasets/<name>/        # pog, pog_dense, spotify, spotify_sparse
  BundleRec/dataset/<name>/          # electronic, food, clothing
  dunnhumby/                         # transaction_data.csv, product.csv
```

```bash
export BUNDLEFUSE_DATA_ROOT=/path/to/data     # optional; defaults to the repo's parent dir
```

### 1. bundlemllm — `pog`, `pog_dense`, `spotify`, `spotify_sparse`
Pre-extracted features from **BundleMLLM** (Liu et al., KDD'25):
➡ **https://huggingface.co/datasets/xhLiu/BundleConstruction**

Download the four BundleConstruction datasets and place each under
`<DATA_ROOT>/bundlemllm/datasets/<name>/` so the folder name matches the value you pass to
`--dataset` (`pog`, `pog_dense`, `spotify`, `spotify_sparse` — rename if the archive uses a different
case/suffix; match by the counts in each `count.json` if unsure). Each folder must contain:
`bi_train.txt`, `bi_{valid,test}_{input,gt}.txt`, `count.json`, `item_info.json`,
`content_feature.pt` (and, for the POG/fashion sets, `title_emb.npy`, `description_feature.pt`).

### 2. BundleRec — `electronic`, `food`, `clothing`
Crowd-sourced bundles from **"Revisiting Bundle Recommendation"** (Sun et al., SIGIR'22):
➡ **https://github.com/BundleRec/bundle_recommendation**

Copy its `dataset/<name>/` folders to `<DATA_ROOT>/BundleRec/dataset/<name>/`. Files used:
`bundle_item.csv`, `user_bundle.csv`, `item_titles.csv`, `item_categories.csv`, `user_item_pretrain.csv`.

### 3. dunnhumby — `dunnhumby`
**"The Complete Journey"** grocery transactions:
➡ **https://www.dunnhumby.com/source-files/** (form-gated) — or the Mendeley mirror
**https://data.mendeley.com/datasets/7myy93ym6k/1**

Put **`transaction_data.csv`** and **`product.csv`** in `<DATA_ROOT>/dunnhumby/`
(override the location with `DUNN_DIR=/path/to/dunnhumby`).

## Train — `evaluator.py`

Reads the dataset and fit-config from the command line, trains the per-dataset model, evaluates, and
writes a checkpoint to `checkpoints/<dataset>.pt`:

```bash
python evaluator.py pog_dense                              # BundleFuse (default)
python evaluator.py dunnhumby --test-n 2000 --ckpt-dir checkpoints
python evaluator.py spotify --beta-clique 1.0 --seed-cap 20 --no-eval
python evaluator.py electronic --model itemknn             # a baseline instead
python evaluator.py --help                                 # all flags
```

Each dataset has a **default config** in `configs/<dataset>.json` (families, `svd_dim`, `content_rank`,
operator hyper-parameters, compression) tuned to match the research results; `evaluator.py` loads it
automatically. Override it with `--config <path>`, ignore it with `--no-config`, and any explicit flag
below wins over the file.

Fit flags: `--families`, `--svd-dim`, `--content-rank`, `--no-compress`, `--val-n/--test-n`,
`--alpha-deg/--gamma-vol/--beta-clique/--alpha-bridge/--fit-rounds/--seed-cap`.
Checkpoint flags: `--ckpt-dir/--save/--no-save/--no-eval`.
Checkpoints are **compressed by default** (incidence `B` + low-rank content; pog 834 MB → 50 MB). The
per-dataset `content_rank` is set so compression stays ~lossless: 64 for the high-dim multimodal
bundlemllm sets, 128 (= full, since their content is 128-dim) for dunnhumby / BundleRec.

## Use a checkpoint — `inference.py`

```bash
# complete a seed bundle -> top-k
python inference.py predict pog_dense --seeds 7917,15610 --k 5
python inference.py predict dunnhumby --from-test 0 --k 10 --user 42

# test set under one metric
python inference.py evaluate spotify --metric ndcg@50 --test-n 2000

# full evaluation (R/N @10/20/50, overall + by seed count) -> JSON
python inference.py report dunnhumby --out reports/dunnhumby.json
```

Add `--model <baseline>` to use a baseline checkpoint instead of BundleFuse.

## Models & signal families

BundleFuse's signals, switched on per dataset by the config:

| family | signals | needs |
|---|---|---|
| `collaborative` | cooc, ppmi, simp, diff2, csvd | train bundles (always) |
| `content` | content, bridge | a content embedding array |
| `taxonomy` | cat | item category labels |
| `popularity` | pop | train bundles (always) |
| `personalization` | user_freq, user_cooc | per-user histories |

Baselines (same interface, selectable with `--model`): `popularity`, `cooccurrence`, `itemknn`,
`p3alpha`, `rp3beta`, `contentknn`.

## Layout

```
evaluator.py        # CLI: train a model -> checkpoint
inference.py        # CLI: predict / evaluate / report from a checkpoint
configs/            # one JSON per dataset: the research-matched default config
bundlefuse/
  model.py          # the BundleFuse class (fit/predict/score/save/load) + signal families
  baselines.py      # the baseline models
  operators.py      # the closed-form math (hypergraph operator, content bridge, EB shrinkage)
  config.py         # load configs/<dataset>.json -> BundleFuseConfig
  datasets/         # one file per dataset (each builds its configured object) + base.py loaders
  metrics.py  provided_splits.py  protocol.py  paths.py  __init__.py
tests/test_model.py
checkpoints/        # written by evaluator.py (gitignored)
```

Datasets are read from `<DATA_ROOT>/{bundlemllm,BundleRec,dunnhumby}`; see **Get the data** above.
