# Experiments and reproducibility

The verified refresh is driven by one command:

```bash
python -m bundlefuse.experiments --output results/evaluation_refresh
```

Its live status and paired results are in [../README.md](../README.md), with all commands,
environments, split IDs, source fingerprints, library versions, logs, and selected configurations
under `results/evaluation_refresh/`. The run is resumable with the same command and source.
Changed Python source requires a new output directory. Failed or incomplete jobs prevent promotion
of canonical tables. Subset runs never replace the complete eight-dataset canonical matrix.

## Supplied neural benchmark splits

```bash
python -m bundlefuse.provided_experiments --output results/provided_splits
```

This separate run executes **only BundleFuse**, using `bi_train.txt`, `bi_valid_input.txt`,
`bi_valid_gt.txt`, `bi_test_input.txt`, and `bi_test_gt.txt` verbatim. It preserves original bundle
IDs and repeated bundles and never reads `bi_full.txt`, regenerates masks, or samples queries.
The dataset populations are:

| Dataset | Training bundles | Validation queries | Test queries |
|---|---:|---:|---:|
| POG | 14,000 | 2,000 | 4,000 |
| POG_dense | 20,780 | 2,968 | 5,938 |
| Spotify | 14,000 | 2,000 | 4,000 |
| Spotify_sparse | 8,740 | 1,248 | 2,498 |

All supplied validation queries select the same fusion/higher-order/shrinkage grids as the
in-memory evaluator. Float32 validation signals are cached on disk in bounded query blocks;
the cache is deleted after selection. The music jobs run sequentially. The output directory
contains split and feature hashes, equality checks against available saved neural-run split
copies, commands, source/version fingerprints, full logs, selected settings, and per-query metrics.
The root README updates after each job. These artifacts do not replace the earlier resplit matrix.

`--data-root` changes the feature/dataset root, `--split-root` optionally changes only the provided
split root, and `--chunk` bounds the number of catalogue score columns held in memory. Resuming
requires identical source, configuration, and input hashes. A leftover validation cache from an
interrupted process is reported explicitly rather than silently reused.

The direct evaluator equivalent is:

```bash
FULLRANK_PROTOCOL=provided BUNDLE_DS=pog BUNDLEFUSE_RESULTS_DIR=results/provided_pog python -m bundlefuse.eval.fullrank
```

`SPLIT_DIR` optionally points to a directory with the five split files and `count.json`; features
continue to come from `DATA_DIR`. Provided mode rejects query caps, training subsampling, the
optional encoder, and the legacy fold diagnostic. The default `FULLRANK_PROTOCOL=resplit` retains
the earlier deduplication, generated splits/masks, and configured query sampling.

Published references are taken from [RaMen Table 2](https://arxiv.org/html/2507.14361v1) for
POG/Spotify and [CLHE Table 2](https://arxiv.org/html/2310.18770) for dense/sparse variants.
The CLHE preprint describes 80/10/10 whereas the supplied files use 70/10/20; its variant scores
remain published context until this discrepancy is resolved. The original CLHE paper already
contains both variants; the historical claim that these were newly constructed local variants
was incorrect. No neural model is trained or evaluated by this command.

## Current experiment plan

| Experiment | Datasets | Split seeds | Output |
|---|---|---|---|
| Main matrix + SimpleHybrid / HardSwitchHybrid | All eight | POG/Spotify families: 0,1,2; e-commerce/retail: 0–4 | `matrix_<dataset>_seed*/selcmp_*.json` |
| Corrected matched baseline suite | All eight | Identical to the main matrix | `baselines_<dataset>_seed*/baselines_*.json` |
| Selection objective: NDCG@10/20/50, Recall@20, MRR, full NDCG, standardized softmax NLL | POG, Electronic, Dunnhumby | 0 (sensitivity probe, not a multi-seed claim) | `selection_<dataset>_<objective>/selcmp_*.json` |

The selection probe uses 1,500 POG test queries; matrix/baseline runs use the full configured test
sample (POG/Spotify up to 3,000; Dunnhumby up to 2,000; e-commerce all held-out bundles). No tuning
uses test metrics. NDCG@20 on a held-out validation split is the primary selection objective; using
the same metric on validation and test is normal model selection, not test leakage.

`SimpleHybrid` combines raw co-occurrence, raw content similarity, category PMI and popularity,
plus available training-only user/session/pretraining interactions. `HardSwitchHybrid` selects a
frequency threshold and prior multiplier on validation. It can retain the plain hybrid when
switching does not improve the validation score. Neither control uses category residualization,
content propagation, simplicial surplus, or Rao concordance.

The baseline suite shares one selection/chunking implementation across all dataset families.
P3α/RP3β exponentiate transition probabilities before multiplying them, remove self-transitions,
and retain deterministic top-k neighbours. BundleKNN uses exactly its selected neighbour count.
SLIM has a size gate, explicit solver seed, and convergence warnings; its omission on large
catalogues is a limitation, not evidence that EASE or another sparse-linear method was evaluated.

## Ranking and statistics

Every current evaluator uses `bundlefuse.metrics`: seeds excluded, scores descending, exact ties
broken by ascending item ID. No random jitter is added. Full NDCG uses distinct ranked positions;
MRR uses the first relevant result per query. Definitions are shared across NumPy and torch inputs.
Metrics are macro averages over queries. Reported standard deviations describe variation over split
seeds and are population standard deviations; they are not confidence intervals or significance tests.
Canonical CSV and JSON are produced from the same unrounded per-seed records.

Changing tie handling changes baseline and occasionally selected BundleFuse configurations. Historical
numbers in `RESULTS.md`, legacy logs, and the paper should therefore not be substituted into current
comparisons. Published CLHE/RaMen/LLM numbers have not been rerun on this refresh's exact splits.

## Individual runs and data

```bash
BUNDLE_DS=pog RAMEN_SEED=0 SELDUMP=1 python -m bundlefuse.eval.fullrank
BRDS=electronic SEEDS=5 SELDUMP=1 python -m bundlefuse.eval.bundlerec
SEEDS=5 SELDUMP=1 python -m bundlefuse.eval.dunnhumby
BASELINES=1 BRDS=electronic SEEDS=5 python -m bundlefuse.eval.bundlerec
```

Set `BUNDLEFUSE_RESULTS_DIR` to isolate evaluator outputs. The experiment runner sets it separately
for every job. Evaluators default to the sibling data layout:

```
<parent>/bundlemllm/datasets/<ds>/   POG / POG_dense / Spotify / Spotify_sparse
<parent>/BundleRec/dataset/<ds>/     electronic / food / clothing
<parent>/dunnhumby/                  The Complete Journey transactions
```

Override the parent with `BUNDLEFUSE_DATA_ROOT`, or an individual family with `DATA_DIR`, `BR_DIR`,
`DUNN_DIR`. POG/Spotify use externally extracted content features; retail/e-commerce use TF-IDF/SVD.
All collaborative structures are rebuilt from the training fold. Catalogue metadata is available
transductively, and optional pretraining interactions are assumed available to both the main scorer
and same-information hybrid controls.

## Tests and historical diagnostics

```bash
python3 tests/test_core.py
python3 -m pytest -q tests
python scripts/add_retail_stds.py  # consistency check after canonical tables are regenerated
```

`ABLATE`, `VALGAP`, `WSWEEP`, `CASE_STUDY`, `ENCODER`, and `TIMING` remain evaluator diagnostics.
They are not automatically refreshed by the 53-job plan. The legacy `KFOLD` mode is not genuine
cross-validation and supports no CV claim. DDBC candidate-pool F1 and triple2vec+adaLoyal tables
still have no reproduction path in this checkout; those paper claims remain unverified here.

The compatibility command `python scripts/regen_zeroparam.py` runs the same fail-closed engine in
matrix-only mode. Error bars are generated alongside means, so `add_retail_stds.py` now validates
CSV/JSON agreement instead of patching results from a separate run.
