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

All eight dataset directories are available in the public
[BundleFuse Project Datasets on Kaggle](https://www.kaggle.com/datasets/ajit1729/bundlefuse-project-datasets).
The collection includes 90 original files, supplied splits, feature arrays and auxiliary inputs.
Inside Kaggle, set `BUNDLEFUSE_DATA_ROOT` to the input mount's `bundlefuse-data/` subdirectory
before importing the package. [Upload verification and setup](results/kaggle_upload/README.md)
include the original hashes and a data-root discovery cell.

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

`validate.py` retains each fitted checkpoint permanently under `checkpoints/validation/`.
Use `--checkpoint-dir PATH` to choose another location. Each fit gets a unique directory;
the JSON result records its path, SHA-256 and size. Verification never deletes the checkpoint.

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
  operators.py      # closed-form math + exact checkpoint graph reconstruction
  config.py         # load configs/<dataset>.json -> BundleFuseConfig
  selected_signals.py # selected-signal scoring for POG/Spotify and extended profiles
  basket_signals.py  # retail/commerce signals: full reference + selected dependencies
  fusion.py         # validation objectives, weight selection and final score fusion
  validation_cache.py # bounded-RAM validation signal cache
  splits.py         # supplied split files + deterministic resplit masking
  evaluation.py     # shared evaluation and seed-stress measurements
  cli_data.py       # checkpoint-bound dataset arguments
  provenance.py     # input/code fingerprints and run metadata
  metrics.py  paths.py  __init__.py
  datasets/
    __init__.py     # all eight datasets in one registry; shared load/config/build API
    base.py         # Dataset contract, full-ranking and extended-profile loaders
    baskets.py      # retail/commerce loaders for the original experiment recipe
tests/              # unit, pipeline and independent historical recipe checks
checkpoints/        # retained fitted models (gitignored)
```

Datasets are read from `<DATA_ROOT>/{bundlemllm,BundleRec,dunnhumby}`; see **Get the data** above.

The package has 18 Python files. Dataset names and their loader families live in one registry;
dataset model settings remain in `configs/<name>.json`. Use `datasets.load(name)` or
`datasets.build(name)`; `datasets.get(name)` returns the shared dataset definition with
`load`, `config` and `build` methods. Separate per-dataset Python wrappers are no longer needed.

`research` selects the original experiment's preprocessing and signal recipe; `extended` keeps
the expanded feature path for ablations and older checkpoints. Both recipes share the public
fit/predict/evaluate pipeline. Their normalization and validation tie rules remain distinct
where necessary to reproduce results. Older saved model checkpoints remain loadable.

The simplified pipeline passed all 29 tests, including loading the previous basket-signal
checkpoint class name. An Electronic seed-0 fit/save/reload check reproduced the historical
R@20 **0.517919** and NDCG@20 **0.359460**, with identical scores across all 346 test queries
before and after reloading; its checkpoint is retained. See the
[verification result](results/reorganized_run/simplified_verification.json). Full corrected
benchmark reruns remain pending; the table below records the original reorganized snapshot.

## Full reorganized pipeline run

Inference timing for the retained eight-dataset checkpoints is measured separately from fitting:

```bash
python scripts/benchmark_inference.py --queries 256 --repeats 3 --batch-size 32
```

The [timing report](results/inference_timing/REPORT.md) records warm top-20 prediction latency,
batch throughput, checkpoint loading and the first request separately. Measurements use one CPU
thread and the simplified runtime, with the projected-feature checkpoints from the original
reorganized run. They do not time the pending corrected research-profile fits. Per-dataset JSON
files retain every timing sample, query IDs, fitted settings, checkpoint hashes and source hashes;
per-dataset logs are in the same directory. Benchmarking never fits or rewrites a checkpoint.

On the Apple M5 Pro (24 GiB), the measured warm single-query medians were: Electronic 4.89 ms,
Food 1.31 ms, Clothing 1.68 ms, Dunnhumby 8.08 ms, POG Dense 5.25 ms, POG 8.22 ms,
Spotify 559.59 ms and Spotify Sparse 264.75 ms. Each uses 256 test queries repeated three times.
Spotify's first request took 2.62 s; this is excluded from its warm timing. A separate
[three-query diagnostic profile](results/inference_timing/spotify_profile.json) spent 58.6% of
its time in sparse format conversion, identifying an optimization target for subsequent work.

<!-- BEGIN REORGANIZED RUN -->
| Dataset | Status | R@20 | NDCG@20 | Earlier R@20 | Comparison |
|---|---|---:|---:|---:|---|
| pog_dense | complete | 0.391939 | 0.286365 | 0.395223 | supplied files |
| pog | complete | 0.053542 | 0.036320 | 0.054458 | supplied files |
| electronic | complete | 0.486802 | 0.343218 | separate protocol | seed 0 only |
| food | complete | 0.594854 | 0.462355 | separate protocol | seed 0 only |
| clothing | complete | 0.516936 | 0.404863 | separate protocol | seed 0 only |
| dunnhumby | complete | 0.207513 | 0.153609 | separate protocol | seed 0 only |
| spotify | complete | 0.154316 | 0.255460 | 0.155646 | supplied files |
| spotify_sparse | complete | 0.162959 | 0.228914 | 0.154288 | supplied files |

[Protocol, results and logs](results/reorganized_run/REPORT.md). Original snapshot and repaired runs are distinguished.
<!-- END REORGANIZED RUN -->

## Code-derived pipeline theory

The [POG standardization audit](results/standardization_audit/README.md) checks that changing
hidden targets leaves fitted-model scores and predictions unchanged, and that normalizing one
query does not pool information from other query columns. Reproduce it with
`python scripts/audit_standardization.py`; the audit retains full numerical results and identifies
the exact fitted checkpoint used.

The [standalone LaTeX source](docs/pipeline_theory.tex) and
[compiled PDF](output/pdf/pipeline_theory.pdf) explain the implemented pipeline step by step:
predefined notation, data construction, graph/content/context artifacts, signal equations,
validation search, checkpoint storage, prediction and evaluation. Dataset-specific and extended
paths are stated explicitly. The document was derived from the current code independently of
earlier notes; its [source manifest](docs/pipeline_theory_sources.json) identifies the exact
working-tree files and records compilation/equation checks.
