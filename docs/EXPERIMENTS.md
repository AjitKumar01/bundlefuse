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

## Retuned component study on supplied splits

```bash
python -m bundlefuse.provided_experiments --study components --output results/component_study
python scripts/verify_components.py
```

This study runs eight methods on each of the four released POG/Spotify splits: full BundleFuse,
NoIncidence, NoRao, NoHigherOrder, NoShrinkage, NoBridge, SimpleHybrid, and HardSwitchHybrid.
The full-model weights and every ablation's remaining settings are selected independently using
all supplied validation queries. Ablations restrict the incidence/Rao grids to zero, disable
posterior shrinkage, or remove the bridging signal from both the fusion and prior candidates.
SimpleHybrid uses raw content, co-occurrence, category/artist/album evidence, and popularity.
HardSwitchHybrid uses the same selected blend and a validation-selected frequency gate.

All methods' settings are frozen before test evaluation. They share query IDs, candidates,
seed exclusion, and metrics. Validation candidates within each coordinate are evaluated in one
disk-cache pass without changing their order, thresholds, or tie policy. Completed full-model
per-query metrics must agree with the preceding supplied-split run to within 1e-9.

The report uses 2,000 cluster-bootstrap replicates for pointwise 95% CIs and 10,000 paired
cluster sign flips for two-sided p-values. Identical complete held-out bundles form clusters;
statistics retain original query multiplicities. Holm adjustment covers the prespecified family
of 56 comparisons (seven contrasts × two metrics × four datasets). This measures uncertainty
conditional on fitted models and these fixed splits. It does not include training-seed/split
variation or infer significance against published neural summary numbers.

The README and report update after each dataset completes. This phase tests component necessity
and the existing simple controls; a newly trained metadata ranker, alternative masks, and temporal
retail evaluation remain separate studies.

## Learned baseline, sparse inputs, chronological retail, and costs

```bash
python -m bundlefuse.generalization_experiments --output results/generalization_study
python scripts/verify_generalization.py
```

Five sequential CPU jobs cover POG, POG_dense, Spotify, Spotify_sparse, and a separate chronological
Dunnhumby experiment. Each job evaluates six methods under four input conditions: frozen BundleFuse,
SimpleHybrid, HardSwitchHybrid, and three training seeds of MetadataRanker. Supplied-split settings
for the first three methods are reused from the verified component study; their original-mask
per-query metrics must reproduce that reference. Temporal-retail settings are selected on its own
validation fold. No old canonical tables are replaced.

MetadataRanker is a shared item encoder with a trainable content projection, categorical embeddings,
and collaborative item factors; a nonlinear residual encoder pools the revealed item set. Dense
source embeddings are consumed directly. Retail uses the same raw TF-IDF metadata through a weighted
embedding bag. Cold item-ID factors/biases are masked out; metadata still scores those items. Training
uses only training bundles, sampled-softmax loss with 32 negatives excluding known positives, and
training-only household histories for retail (20% auxiliary two-item samples). Sparse embedding
gradients use SparseAdam; dense parameters use AdamW, with representation regularization.

The prespecified grid is learning rate {0.001, 0.003} × regularization {1e-5, 1e-3}, 64 latent
dimensions, batch size 128, at most 100 epochs, validation every five epochs, and patience of four
checks. Hyperparameters are selected using seed-0 validation NDCG@20. Seeds 1 and 2 use those
hyperparameters and select their own checkpoints on the same validation fold. All learning curves,
stop reasons, and best-at-final-check flags are retained. A run that peaks at its epoch limit is
reported as budget-limited rather than evidence of convergence. No best-test-seed or ensemble is used.
Checkpoints remain local `.pt` files, with hashes recorded; they are excluded from Git.

Sparse-input cases retain all supplied seeds, half of them, one quarter, or exactly one seed.
A deterministic permutation makes the retained sets nested. Removed seed items join the hidden
targets, preserving complete bundle membership and keeping at least one revealed item. All model
settings are frozen using the original validation queries before scoring any test condition.

The temporal experiment first chooses whole-day 70/10/20 boundaries from all eligible 3–20-item
baskets, then samples at most 21,000 training, 3,000 validation, and 6,000 test baskets with seed 0.
It does not deduplicate recurring baskets or filter catalogue items by future interaction frequency.
Preflight day ranges are train 1–515, validation 516–580, and test 581–711. All 92,353 metadata-catalogue
items remain candidates, including cold targets. This explicitly assumes a fixed, transductively
available catalogue: product introduction dates are unavailable. Every interaction-derived graph,
household history, and item-ID training signal uses only the sampled training period.

The full default study prespecifies 200 paired contrasts: five datasets × four cases × five
alternatives to BundleFuse × two metrics. The seed-0 learned model is the primary learned competitor;
other seeds show training variability. Query-level inference conditions on each fitted configuration;
cluster bootstrap/sign flips use complete-bundle clusters for POG/Spotify and household clusters for
retail. Holm correction covers the whole family. Pointwise CIs use 2,000 bootstrap replicates and
two-sided tests use 10,000 sign flips.

Costs are measured separately: feature/graph preprocessing, learned training and validation search,
catalogue-vector encoding, warm-cache scoring, scoring plus metrics, and whole-worker peak RSS.
Inference uses the same 32 sampled queries and three repeats per method. External embedding
extraction/pretraining is excluded; whole-worker RSS is not a per-method memory measurement.
For the supplied POG/Spotify folds, `baseline_selection_seconds` measures loading the frozen
reference settings, not their earlier grid-search cost. That earlier search is outside this run's
cost totals. The chronological job includes its newly performed validation search. Warm-cache
latency therefore supports inference comparisons, not a claim about total optimization cost.

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
