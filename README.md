# BundleFuse

BundleFuse completes a partial bundle by ranking the remaining catalogue items. It combines
training-fold collaborative statistics, content/category evidence, and exposure-dependent backoff:

```
score(c | S) = (1 - u_c) * Y(c | S) + u_c * prior(c | S)
u_c = kappa / (frequency_c + kappa)
```

The default scorer has no gradient-trained encoder. Fusion weights and routing settings **are fit
on validation**; SVD factors, taxonomy statistics, and content preprocessing are also estimated.
POG/Spotify consume externally pretrained embeddings, whose extraction cost is outside scoring.
An optional set-attention encoder is enabled with `ENCODER=1`. A selected `kappa=0` disables
backoff explicitly, including for zero-frequency candidates.

## Verified experiment refresh

Branch `evaluation-reliability-experiments` starts from main commit `039e07c`. This refresh fixes
P3α/RP3β transition powers, exact BundleKNN neighbour counts, NDCG/MRR definitions, deterministic
ranking ties, and experiment failure handling. It adds raw-content/category hybrid controls.

<!-- BEGIN VERIFIED EXPERIMENTS -->
Verified refresh: **53/53 jobs complete**, 0 failed.
Run manifest and full logs: [`results/evaluation_refresh/manifest.json`](results/evaluation_refresh/manifest.json).
CPU, one thread per process; deterministic item-ID tie breaks. Fullrank: 3 split seeds; retail/e-commerce: 5 splits.

| Dataset | BundleFuse R@20 | SimpleHybrid | HardSwitchHybrid | Best other matched baseline |
|---|---:|---:|---:|---|
| pog | 0.0681 ± 0.0026 | 0.0378 ± 0.0010 | 0.0443 ± 0.0095 | 0.0243 (P3alpha) |
| pog_dense | 0.3752 ± 0.0065 | 0.3637 ± 0.0082 | 0.3663 ± 0.0082 | 0.2857 (LinearCombiner) |
| spotify | 0.1567 ± 0.0018 | 0.0988 ± 0.0021 | 0.0991 ± 0.0024 | 0.1305 (LinearCombiner) |
| spotify_sparse | 0.1680 ± 0.0044 | 0.1134 ± 0.0012 | 0.1135 ± 0.0014 | 0.1169 (LinearCombiner) |
| electronic | 0.5163 ± 0.0212 | 0.4745 ± 0.0470 | 0.4878 ± 0.0447 | 0.3343 (iALS) |
| food | 0.5939 ± 0.0281 | 0.5488 ± 0.0186 | 0.5559 ± 0.0324 | 0.3389 (iALS) |
| clothing | 0.5881 ± 0.0326 | 0.5412 ± 0.0315 | 0.5518 ± 0.0441 | 0.2657 (BundleKNN) |
| dunnhumby | 0.1504 ± 0.0044 | 0.1422 ± 0.0034 | 0.1422 ± 0.0034 | 0.1405 (LinearCombiner) |

These are matched split/candidate comparisons. The simple hybrids also receive the available raw
content, taxonomy, and auxiliary interactions. Heavy published models are literature context,
not paired results from this refresh. Historical sensitivity/ablation tables have not been regenerated
unless explicitly listed in this run. Missing DDBC/triple2vec reproduction paths remain open.
<!-- END VERIFIED EXPERIMENTS -->

BundleFuse has the highest **mean** Recall@20 among the local methods tested here on all eight
datasets. The strongest controls narrow the gaps to about **0.90 percentage points on dense POG**
and **0.81 points on Dunnhumby**; seed dispersion is not a significance test. These results support
an effective hybrid reference method, not a claim of superiority over every published model.

The [corrected selection study](results/evaluation_refresh/SELECTION.md) replaces the old MRR/full-NDCG
conclusions. The [verification record](results/evaluation_refresh/VALIDATION.md) records completed jobs,
paired checks, unchanged inputs, tests, and artifact consistency.

Reproduce or resume the complete run:

```bash
python -m bundlefuse.experiments --output results/evaluation_refresh
```

The runner records commands, split seeds, selected hyperparameters, package versions, a source
fingerprint, and full logs. It updates this section after completed jobs. Music jobs run alone to
limit RAM; other jobs can run two at a time, with one CPU thread per process. It refuses to publish
canonical tables when a required job fails. A changed source fingerprint requires a new output
directory. The full run includes paired matrix/baseline experiments on all eight datasets and a
selection-objective study on POG, Electronic, and Dunnhumby. The selection study uses one split
seed per dataset; POG uses 1,500 test queries, and the matrix uses its full configured test sample.

After the complete run, verify paired outputs, canonical tables, and unchanged input files:

```bash
python scripts/report_refresh.py
python scripts/fingerprint_data.py --check
python scripts/add_retail_stds.py
```

The measured top-level runtime versions are saved in
`results/evaluation_refresh/requirements.txt`; the manifest also records Python and the platform.
Recorded durations are wall times for entire evaluator/grid-search jobs, including comparison
methods; they are not dedicated per-query latency benchmarks.

For a subset or an individual experiment:

```bash
python -m bundlefuse.experiments --datasets electronic --modes matrix baselines --output results/electronic_refresh
BUNDLE_DS=pog MATRIX=1 SELDUMP=1 python -m bundlefuse.eval.fullrank
BRDS=electronic SEEDS=5 BASELINES=1 python -m bundlefuse.eval.bundlerec
SEL_OBJ=mrr BRDS=electronic SEEDS=1 SELDUMP=1 python -m bundlefuse.eval.bundlerec
```

Subset runs retain their own results and do not replace the eight-dataset canonical tables.
`BUNDLEFUSE_RESULTS_DIR=/path` isolates direct evaluator outputs from committed results.

## Baselines and comparison scope

The shared suite contains ItemKNN, P3α, RP3β, SLIM, iALS, BundleKNN, and a collaborative
LinearCombiner. SLIM is explicitly skipped above its implementation's size limit; dense EASE is
also gated and is **not** a substitute for a scalable sparse-linear baseline on those catalogues.

`SimpleHybrid` blends raw co-occurrence, unbridged content similarity, category PMI, popularity,
and available training-only user/session/pretraining interactions. `HardSwitchHybrid` tests a
validation-selected frequency threshold for backing off to its metadata prior. These controls
receive the same source modalities and auxiliary interactions as the main scorer, without its
style residualization, content bridging, or higher-order operators.

Within each matched experiment, local methods share splits, candidate catalogues, seed exclusion,
and deterministic ties (descending score, then ascending item ID). The resplit refresh reports
seed dispersion; the supplied-split component study below adds paired uncertainty estimates.
Published neural-score comparisons are documented in the supplied-split section with their
release/protocol and feature-attribution limits. RaMen's published visual-removal ablation rules out the earlier claim that
images alone explain its advantage; the precise cause of our cross-model gap remains unproven.

## Install and data

```bash
git clone https://github.com/AjitKumar01/bundlefuse
cd bundlefuse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 tests/test_core.py
# Additional regression tests:
pip install pytest
python3 -m pytest -q tests
```

Datasets are external. The default layout is:

```
<parent>/bundlefuse/                 this repository
<parent>/bundlemllm/datasets/<ds>/   POG / POG_dense / Spotify / Spotify_sparse
<parent>/BundleRec/dataset/<ds>/     electronic / food / clothing
<parent>/dunnhumby/                  The Complete Journey transactions
```

Set `BUNDLEFUSE_DATA_ROOT` to change the parent, or `DATA_DIR`, `BR_DIR`, and `DUNN_DIR` for
individual families. POG/Spotify require their released/precomputed content features. Retail and
e-commerce use TF-IDF/SVD. CPU is the default; `FORCE_DEVICE` overrides it. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for loaders and historical commands.

## Code map

| Module | Responsibility |
|---|---|
| `operators.py` | Collaborative, taxonomic, content, and shrinkage operators |
| `metrics.py`, `selection.py` | Shared ranking/tie definitions and validation objectives |
| `baselines/` | One implementation per baseline; shared selection and chunked evaluation |
| `protocol.py`, `paths.py` | Masking protocol and configurable data/result paths |
| `artifacts.py`, `experiment_io.py` | Atomic result writes, aggregation, and fail-closed execution |
| `experiments.py` | Resumable experiment jobs, provenance, and README progress |
| `provided_splits.py`, `provided_experiments.py` | Supplied neural benchmark split loading and BundleFuse-only runs |
| `fusion_selection.py`, `eval/provided.py` | Shared fusion selection and bounded-memory full-validation evaluation |
| `eval/components.py`, `paired_stats.py`, `component_report.py` | Retuned ablations, clustered paired inference, and live component-study reports |
| `baselines/metadata_ranker.py`, `eval/generalization.py` | Train-fold metadata/collaborative ranker and fixed-selection sparse-input tests |
| `temporal_data.py`, `retail_signals.py`, `eval/temporal.py` | Whole-day retail folds and training-only retail graph scoring |
| `generalization_experiments.py`, `generalization_report.py`, `profiling.py` | Generalization jobs, live results, and separated cost measurements |
| `eval/` | Dataset loaders and evaluator entry points |
| `tests/` | Core math and evaluation/runner regression checks |

The method derivation is in [notes/methodology.tex](notes/methodology.tex). The
[paper](paper/) and [historical results](RESULTS.md) are retained for comparison; DDBC F1 and
triple2vec+adaLoyal claims still lack reproduction paths in this checkout. Current results are
limited to the completed experiments in the verified refresh and the supplied-split runs below.

## Supplied neural benchmark splits

<!-- BEGIN PROVIDED SPLITS -->
BundleFuse only; supplied training bundles and every supplied validation/test query.
No bundle deduplication, resplitting, remasking, or query sampling.

| Dataset | Status | BundleFuse R@20 | CLHE reference | RaMen reference | Reference source |
|---|---|---:|---:|---:|---|
| pog | complete | 0.054458 | 0.0284 | 0.0375 | RaMen Table 2 |
| pog_dense | complete | 0.395223 | 0.3811 | — | CLHE Table 2 |
| spotify | complete | 0.155646 | 0.1081 | 0.1091 | RaMen Table 2 |
| spotify_sparse | complete | 0.154288 | 0.098 | — | CLHE Table 2 |

Reference scores are taken directly from [RaMen Table 2](https://arxiv.org/html/2507.14361v1)
and [CLHE Table 2](https://arxiv.org/html/2310.18770); neural models are not run here.
Both dense/sparse variants already appear in the CLHE paper; earlier local documentation incorrectly
called them newly constructed variants and substituted local reproduction scores.
The supplied files use 70/10/20, consistent with RaMen. The CLHE preprint describes 80/10/10;
its dense/sparse published scores are therefore reference context until that release/paper discrepancy is resolved.
The manifest records hashes of the supplied split files and equality with saved neural-run copies.
Catalogue candidates, seed exclusion, macro Recall/NDCG, and selected settings are recorded per dataset.
Neural scores retain their original tie handling; BundleFuse resolves exact ties by ascending item ID.
This controls the split/mask mismatch. BundleFuse retains its own metadata/features; this is not a
same-feature architecture ablation. Published-score comparisons do not constitute a local neural reproduction.
Electronic/Food are excluded because their public BundleRec data do not establish the published neural split.
Full logs, protocol hashes, and per-query metrics: [results/provided_splits/REPORT.md](results/provided_splits/REPORT.md).

```bash
python -m bundlefuse.provided_experiments --output results/provided_splits
```
<!-- END PROVIDED SPLITS -->

<!-- BEGIN PROVIDED CONCLUSION -->
All **4/4 supplied-split runs are complete and verified**, covering 16,436 test queries. BundleFuse exceeds the listed published R@20 and NDCG@20 point estimates on every available comparison. On POG and Spotify, Recall@20 is respectively **45.2% and 42.7% higher than RaMen’s published score**. These are point-estimate comparisons, not significance tests.

The released split/mask mismatch is resolved. The CLHE preprint’s different split-ratio description still limits strict paper-equivalence claims for dense/sparse variants. See the [full metric comparison and interpretation](results/provided_splits/ANALYSIS.md) and [verification record](results/provided_splits/VALIDATION.md). **32 regression tests passed**; all recorded inputs remain unchanged.
<!-- END PROVIDED CONCLUSION -->

## Supplied-split controls and ablations

<!-- BEGIN COMPONENT STUDY -->
**4/4 datasets complete.** Full BundleFuse, five retuned ablations, SimpleHybrid, and HardSwitchHybrid use all supplied queries.

| Dataset | Status | BundleFuse | SimpleHybrid | HardSwitchHybrid |
|---|---|---:|---:|---:|
| pog | complete | 0.054458 | 0.034083 | 0.041042 |
| pog_dense | complete | 0.395223 | 0.363647 | 0.364741 |
| spotify | complete | 0.155646 | 0.099754 | 0.099961 |
| spotify_sparse | complete | 0.154288 | 0.115149 | 0.115458 |

[Live results, paired uncertainty estimates, and logs](results/component_study/REPORT.md).

```bash
python -m bundlefuse.provided_experiments --study components --output results/component_study
```
<!-- END COMPONENT STUDY -->

<!-- BEGIN COMPONENT CONCLUSION -->
All **32 method/dataset evaluations** are complete and verified. BundleFuse beats both raw-information hybrid controls in R@20 and NDCG@20 on all four supplied splits, with significance after Holm adjustment over all 56 planned contrasts. R@20 gains over HardSwitchHybrid are **+1.342 pp (POG), +3.048 pp (POG_dense), +5.569 pp (Spotify), and +3.883 pp (Spotify_sparse)**.

The ablations support dataset-dependent benefits. Higher-order terms help music but do nothing on POG; shrinkage improves three datasets reliably; removing bridging improves POG NDCG while bridging helps Spotify. Removing all higher-order terms still retains roughly **92–100%** of the R@20 advantage over HardSwitchHybrid, so they do not explain most of the baseline gap.

[Full interpretation, effect sizes, and remaining limitations](results/component_study/ANALYSIS.md). [Verification](results/component_study/VALIDATION.md) recomputes all 56 paired contrasts from per-query metrics. **42 tests passed**, and every full-model per-query metric exactly reproduced the preceding verified run. Uncertainty estimates condition on these fitted models and supplied splits. The subsequent learned-baseline, mask and temporal study below tests those remaining questions.
<!-- END COMPONENT CONCLUSION -->

## Learned baseline and generalization

<!-- BEGIN GENERALIZATION STUDY -->
**5/5 dataset jobs complete.** A trained metadata/collaborative ranker (three seeds), sparse-input stress tests, chronological retail evaluation, and measured costs.

| Dataset | Status | BundleFuse R@20 | Learned seed 0 | Learned mean ± SD (3 seeds) |
|---|---|---:|---:|---:|
| pog | complete | 0.054458 | 0.023167 | 0.023625 ± 0.000863 |
| pog_dense | complete | 0.395223 | 0.358566 | 0.358360 ± 0.001269 |
| dunnhumby_chrono | complete | 0.118867 | 0.096630 | 0.096970 ± 0.001112 |
| spotify | complete | 0.155646 | 0.120616 | 0.118988 ± 0.001154 |
| spotify_sparse | complete | 0.154288 | 0.109306 | 0.111407 ± 0.001769 |

[Live results and logs](results/generalization_study/REPORT.md).

```bash
python -m bundlefuse.generalization_experiments --output results/generalization_study
```
<!-- END GENERALIZATION STUDY -->

<!-- BEGIN GENERALIZATION CONCLUSION -->
All **five jobs, 120 method/case evaluations, and 200 paired contrasts are verified**. BundleFuse beats both simple controls and all three learned-ranker seeds in R@20 and NDCG@20 under the original input condition on all five datasets, including chronological retail, after Holm correction.

Sparse-input limitations are confirmed. With one seed, **POG_dense SimpleHybrid scores 0.36102 versus BundleFuse 0.35449**; on chronological retail, **learned seed 0 scores 0.07577 versus BundleFuse 0.06562** in R@20. Both Recall reversals survive correction. These are separate stress conditions; original POG_dense remains **0.39522**. Spotify one-seed NDCG also favors HardSwitch.

The cached learned ranker is faster in the measured warm-cache scoring pipeline. Report preprocessing, fitting, catalogue encoding, scoring/metric time and whole-worker RSS separately; the earlier supplied-fold baseline search and external feature extraction are excluded from the new totals. Two Spotify confirmation seeds reached the epoch budget, so universal convergence is unestablished.

[Full findings, cost measurements, and operating limits](results/generalization_study/ANALYSIS.md). [Verification](results/generalization_study/VALIDATION.md) checks masks, chronology, hashes, checkpoints, reference agreement, and all 200 statistics. **51 regression tests passed.**
<!-- END GENERALIZATION CONCLUSION -->

## Compact-model feasibility pilot

<!-- BEGIN COMPACT PILOT -->
**2/2 dataset jobs complete.** Exact selected-signal optimization plus a raw-versus-engineered compact two-tower pilot.

| Dataset | Status | BundleFuse R@20 | Raw mean ± SD | Engineered mean ± SD | Validation gate |
|---|---|---:|---:|---:|---|
| pog_dense | complete | 0.395223 | 0.344869 ± 0.002084 | 0.354422 ± 0.001444 | False |
| spotify | complete | 0.155646 | 0.075347 ± 0.001446 | 0.105349 ± 0.000078 | False |

[Protocol, live results and logs](results/compact_pilot/REPORT.md). No distillation or sparse-input adaptation in this stage.
<!-- END COMPACT PILOT -->

<!-- BEGIN COMPACT CONCLUSION -->
Both pilot jobs are verified: **9,938 original test queries**, exact reference/optimized fused-score
bit equality, unchanged historical metrics, and all **12 planned paired comparisons** recomputed.
Skipping unused signals makes BundleFuse scoring **3.92× faster on POG_dense** and **1.46× faster
on Spotify** in the measured warm 32-query CPU batches, with identical accuracy.

The **58,314-parameter** engineered student beats its matched raw-feature student on both metrics
and datasets after Holm correction. It retains only **89.68% / 67.68%** of BundleFuse test Recall
(three-seed means, POG_dense / Spotify). Both validation-only 97.5% retention gates fail. The
student is faster, but this representation does not meet the accuracy-preserving replacement
goal; no distillation or sparse-input adaptation was started. Two raw Spotify fits hit the epoch
budget; all engineered fits stopped by validation patience. These are architecture/budget-specific
findings, not a proof that every compact model fails.

[Full interpretation and measured costs](results/compact_pilot/ANALYSIS.md) ·
[Verification](results/compact_pilot/VALIDATION.md). The implementation passed **57 regression tests**.
<!-- END COMPACT CONCLUSION -->
