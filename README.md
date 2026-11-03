# BundleFuse

A config-driven closed-form scorer for **bundle completion**: given a partial bundle (seed items),
rank the catalogue for the items that complete it. BundleFuse fuses a handful of closed-form signals —
organised into families — with validation-selected weights and empirical-Bayes cold-start shrinkage.
No gradient-trained encoder; fusion weights and shrinkage are fitted on validation. One model
object per dataset supports fitting, exact checkpoint round trips, batched scoring and reporting.

```python
from bundlefuse import datasets
model, ds = datasets.build("pog_dense")        # object configured for this dataset + its data
model.fit(ds.train, ds.val_pairs, **ds.fit_kwargs())
model.save("checkpoints/pog_dense.pt")
items = model.predict(seeds=[7917, 15610], k=10)
```

## Install

```bash
git clone https://github.com/AjitKumar01/bundlefuse.git
cd bundlefuse
pip install -r requirements-dev.txt      # runtime dependencies + pytest
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
- **Data (HuggingFace):** https://huggingface.co/datasets/xhLiu/BundleConstruction
- **Code (GitHub):** https://github.com/Xiaohao-Liu/Bundle-MLLM

Download the four BundleConstruction datasets and place each under
`<DATA_ROOT>/bundlemllm/datasets/<name>/` so the folder name matches the value you pass to
`--dataset` (`pog`, `pog_dense`, `spotify`, `spotify_sparse` — rename if the archive uses a different
case/suffix; match by the counts in each `count.json` if unsure). Each folder must contain:
`bi_train.txt`, `bi_{valid,test}_{input,gt}.txt`, `count.json`, `item_info.json`,
`content_feature.pt` (and, for the POG/fashion sets, `title_emb.npy`, `description_feature.pt`).

**Content-feature provenance.** BundleFuse consumes these features as-is (it never re-extracts them).
They are the released encoder outputs: **BLIP** (768-d) for POG image/description
(`content_feature.pt`, `description_feature.pt`), **BGE** (`bge-large-zh-v1.5`, 1024-d) for POG titles
(`title_emb.npy`), and **CLAP** (512-d) for Spotify audio (`content_feature.pt`). The `content` /
`bridge` signal families use these; for dunnhumby / BundleRec the content is TF-IDF over product text.

### 2. BundleRec — `electronic`, `food`, `clothing`
Crowd-sourced bundles from **"Revisiting Bundle Recommendation"** (Sun et al., SIGIR'22):
- **Data + code (GitHub):** https://github.com/BundleRec/bundle_recommendation

Copy its `dataset/<name>/` folders to `<DATA_ROOT>/BundleRec/dataset/<name>/`. Files used:
`bundle_item.csv`, `user_bundle.csv`, `item_titles.csv`, `item_categories.csv`, `user_item_pretrain.csv`.

### 3. dunnhumby — `dunnhumby`
**"The Complete Journey"** grocery transactions:
- **Kaggle (easiest):** https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey
- **GitHub mirror:** https://github.com/omerfarukeker/The-Complete-Journey
- **Official (form-gated):** https://www.dunnhumby.com/source-files/ · **Mendeley:** https://data.mendeley.com/datasets/7myy93ym6k/1

Put **`transaction_data.csv`** and **`product.csv`** in `<DATA_ROOT>/dunnhumby/`
(override the location with `DUNN_DIR=/path/to/dunnhumby`).

## Train — `evaluator.py`

Reads the dataset and fit-config from the command line, trains the per-dataset model, evaluates, and
writes a checkpoint to `checkpoints/<dataset>.pt`:

```bash
python evaluator.py pog_dense                              # BundleFuse (default)
python evaluator.py pog_dense --protocol provided --val-n 0 --test-n 0
python evaluator.py pog_dense --protocol resplit --split-seed 2 --mask-frac 0.5
python evaluator.py dunnhumby --test-n 2000 --ckpt-dir checkpoints
python evaluator.py spotify --beta-clique 1.0 --seed-cap 20 --no-eval
python evaluator.py electronic --model itemknn             # a baseline instead
python evaluator.py --help                                 # all flags
```

Each dataset has a **default config** in `configs/<dataset>.json` (families, `svd_dim`, `content_rank`,
operator hyper-parameters, compression); `evaluator.py` loads it
automatically. Override it with `--config <path>`, ignore it with `--no-config`, and any explicit flag
below wins over the file.

Fit flags: `--families`, `--svd-dim`, `--content-rank`, `--no-compress`, `--val-n/--test-n`,
`--alpha-deg/--gamma-vol/--beta-clique/--alpha-bridge/--fit-rounds/--seed-cap`.
Checkpoint flags: `--ckpt-dir/--save/--no-save/--no-eval`.
`--fit-rounds` controls the validation coordinate-search passes. `--seed-cap` limits the seeds used
to compute signals (0 means all); every revealed item is still excluded from predictions.
`--chunk` bounds validation/scoring batches and `--threads` controls CPU computation threads.

**Protocols are explicit.** POG/Spotify default to the supplied train/validation/test files, retaining
duplicate training bundles, original query IDs and masks. Their validation/test defaults are the
full supplied splits. `--val-n 0 --test-n 0` also means full; positive limits produce a labelled
partial run. `--protocol resplit` explicitly enables deduplication, new splits and reveal masks.
BundleRec/Dunnhumby support their existing random-resplit completion task; their CLI defaults
remain 800 validation / 2,000 test queries. This retail loader is not a chronological benchmark.

Checkpoints omit derived graph matrices without mutating the fitted object. Required matrices
are rebuilt lazily using the original zero-diagonal formulas, so storage compression preserves
scores; rebuilding can add first-query time and memory. Separately, `content_rank` applies a
**lossy projection during fitting**, before validation selection. It is not guaranteed lossless:
defaults are 256 for POG, 64 for the other POG/Spotify datasets and 128 for retail/BundleRec.
`--no-compress` disables both that projection and derived-matrix omission.

Every fitted CLI run writes a JSON sidecar (`<checkpoint>.json`, or `--report PATH`) with effective
configuration, actual selected weights/routing, query IDs, protocol, data/code hashes, versions,
metrics and checkpoint hash. Saving includes a validation-query score parity probe. `--no-eval`
does not score the test set. Validation signals use a temporary disk cache rather than retaining
all catalogue-by-query matrices in RAM; full-catalogue validation still needs temporary disk space.

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
For an explicit checkpoint, put its global option before the subcommand:

```bash
python inference.py --ckpt checkpoints/pog_dense.pt report pog_dense --out reports/pog_dense.json
```

Evaluation reconstructs the checkpoint's recorded protocol/seed/mask and verifies source-data
and training-fold hashes. A different protocol or changed input files are rejected. `--test-n`
can select a prefix of the same test split. Legacy checkpoints without dataset provenance can
still predict supplied item indices, but must be refitted with this pipeline for dataset evaluation.
Reports keep full-precision metrics and true fitted model metadata. `by_seed_count` groups the
original queries; `seed_stress` reduces context and moves removed seeds into the hidden targets.

## Verification

```bash
python -m pytest tests/ -q
# Explicit dataset integration check; fit, save, reload, compare every requested test score:
python validate.py --datasets pog_dense --protocol provided --val-n 0 --test-n 0
# Random-split repetitions are a separate experiment:
python validate.py --datasets food --protocol resplit --split-seeds 0 1 2
```

The mandatory golden test uses a committed synthetic reference captured from `15bd250` before
the fixes. It requires no external datasets. Tests cover pre-save versus loaded score bits,
selected versus full scoring, failed writes, dataset masks/IDs, ties, all baseline interfaces and
the CLI fit/save/report path. `validate.py` writes a protocol-bound, fail-closed JSON report; it
does not silently compare different protocols or invent missing historical benchmark goldens.

The reorganized signal pool and content compression need their own full benchmark measurements.
Successful integration tests do not establish equality with an earlier paper score or transfer
the earlier branch's measured inference speedups to this implementation.

Pipeline fix verification is recorded in [`results/pipeline_fixes/README.md`](results/pipeline_fixes/README.md),
including the real-data smoke commands and their deliberately limited scope.

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
  selected_signals.py # compute only dependencies of the fitted fusion
  graph_state.py    # exact reconstruction of matrices omitted from checkpoints
  validation_cache.py # bounded-RAM, temporary disk-backed validation signals
  evaluation.py  cli_data.py  provenance.py # shared CLI contracts, metrics and fingerprints
  datasets/         # one file per dataset (each builds its configured object) + base.py loaders
  metrics.py  provided_splits.py  protocol.py  paths.py  __init__.py
tests/test_model.py
checkpoints/        # written by evaluator.py (gitignored)
```

Datasets are read from `<DATA_ROOT>/{bundlemllm,BundleRec,dunnhumby}`; see **Get the data** above.
