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

All local methods share splits, candidate catalogues, seed exclusion, and deterministic ties
(descending score, then ascending item ID). We report seed dispersion; it is not a significance
test. Published CLHE/RaMen/LLM numbers remain literature context until reproduced on identical
splits and features. RaMen's published visual-removal ablation rules out the earlier claim that
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
| `eval/` | Dataset loaders and evaluator entry points |
| `tests/` | Core math and evaluation/runner regression checks |

The method derivation is in [notes/methodology.tex](notes/methodology.tex). The
[paper](paper/) and [historical results](RESULTS.md) are retained for comparison; DDBC F1 and
triple2vec+adaLoyal claims still lack reproduction paths in this checkout. Current results are
limited to experiments explicitly completed in the verified refresh.
