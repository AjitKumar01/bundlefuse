# Raw-score and fusion attribution study

Branch: `novelty-score-fusion-experiments`.

This is the first decisive stage of the [novelty assessment](../../docs/novelty_assessment.md).
It tests whether improvements come from the historical raw score or the fusion of identical score columns. Production BundleFuse is unchanged. The supplied splits have already been inspected; these are exploratory results, not an untouched confirmation.

## Planned run

Datasets run sequentially: **POG_DENSE, Spotify, POG**, with all supplied validation/test queries and full-catalogue test ranking. Each dataset has 18 controlled systems plus one separately labelled canonical reference.

Historical score banks:

1. `pair`: direct historical score `p_c(S)`.
2. `incidence`: complete enhanced score `g_c(S) = p_c(S) + delta_c(S)`; `delta` alone is not substituted for `g`.
3. `amplified`: frequency-weighted cosine historical-bundle overlap raised to a selected power, with optional top-k bundle truncation. Powers are 1/2/3; k is 100/500/all.

The amplified comparator is a **predecessor-inspired controlled implementation**, not an exact reproduction of Kelen et al. It omits their positional and min/max scaling choices. Its complete equation and selected settings are saved. This stage is not HOSLIM/HOKNN reproduction; deciding whether to invest in those larger itemset fits follows these results. Bridge-specific operator comparisons and untouched-split confirmation remain separate experiments.

Every bank has the same nonhistorical columns, selected from the earlier frozen full model. It replaces only the historical-match column. PPMI/SVD, bridge and any selected agreement score remain identical across banks. The experiment therefore tests the historical operator conditional on a prior selected source inventory; it does not compare every possible complete feature-engineering pipeline.

Six fusions receive those same columns:

| Fusion | Fitting/selection |
|---|---|
| `equal` | Sum all standardized columns with weight one |
| `weighted` | Nonnegative scalar weights on a 0/.25/.5/1/1.5 coordinate grid, four-round ceiling |
| `rrf` | Reciprocal rank fusion with offset 10/60/100; observed items excluded from source ranks |
| `frequency_linear` | Pairwise logistic ranker on `[z; rho*z; log1p(f)]`, with regularization .001/.01/.1 |
| `lambdamart` | XGBoost `rank:ndcg`, 50/150 trees x depth 3/5, plus log frequency |
| `bundlefuse_gate` | Same fitted base weights as `weighted`, with scalar frequency interpolation, kappa 0/.05/.5/1/5/20 and unit/matched scale |

`bundlefuse_gate` is a controlled refit of the gate, with optional interaction sources represented as explicit columns; it is not silently called the original complete scorer. Its fallback uses selected shared source weights, popularity when selected, and agreement when selected. No default negative or zero-frequency branch is inferred; kappa zero explicitly returns the combined score.

`original_reference` uses the earlier full frozen recipe, including its original historical and agreement settings. It was selected using all earlier validation queries and is labelled separately. A fresh score-parity check against production scoring must pass before results are accepted.

## Fitting versus tuning

Validation queries are grouped by their complete bundle (the union of seeds and withheld items). Groups are randomly split 70% fitting / 30% tuning with a fixed seed. Duplicate/alternate queries of the same complete bundle cannot cross that boundary within validation.

- Raw amplified-neighbor parameters are selected using fitting groups only, by full-catalogue raw-score NDCG@20.
- Scalar coordinate weights fit on the fitting groups, using full-catalogue scores.
- Trainable rankers fit on all fitting-query positives plus 128 uniform negatives and up to 16 hard negatives per source, deduplicated. Withheld labels are not input features. All positives are retained.
- Hyperparameters are selected on full-catalogue tuning-query NDCG@20. No trained model is refitted after this selection.
- Original-reference selection is not forced into the new split: its earlier all-validation provenance is disclosed rather than treated as an equally refitted competitor.
- Every controlled recipe is frozen to disk before new held-out score caches are generated.

All controlled methods use the same query populations. Fitting procedures and candidate counts differ and are recorded; no equal-compute assertion is made. Direct coordinate-weight selection and supervised negative-sampled training have different objectives. The linear ranker uses the earlier recipe's positive kappa (or .05 if unavailable) for its fixed rho transform; it is an exact basis-class comparator for that transform, not an exhaustive search of every possible gate.

The raw-score definitions use training bundles only. Fitting/tuning completion labels come from independent supplied validation bundles, not masked rows left inside the construction matrix. The same test catalogue, exclusion mask and stable item-ID tie policy apply to all methods.

## Reuse, retention, resume

Previous model artifacts and validation arrays are opened read-only. The unchanged production/mechanism source fingerprint, train/query hashes, cache completion markers and a fresh signal parity check are required before reuse. A checkpoint checksum is recorded for the input model.

New outputs are isolated:

- `results/novelty_attribution/<dataset>/`: protocol, manifests, validation-group split, amplified grid, selected configurations, tuning metrics, query-level predictions/metrics, report and paired statistics.
- `checkpoints/novelty_attribution/<dataset>/`: persistent supplemental validation/test arrays, completion markers, sampled fitting rows, linear recipes and saved LambdaMART models.
- `results/novelty_attribution/queue.log`: sequential queue status.
- `results/novelty_attribution/<dataset>/run.log`: timestamped per-dataset progress.

Models, arrays, previous checkpoints and interrupted partial files are never automatically deleted. Each dataset holds a lock. Completed cache blocks, base coordinate fits, candidate tuning results and query metric blocks are reused on resume. Source/protocol mismatch rejects resume; do not edit experiment source while a worker is running.

The baseline score arrays plus amplified variants require additional disk, especially for Spotify. Prior validation arrays are reused rather than copied. Test arrays contain only needed columns and are shared across all three historical banks.

## Statistical scope

The fixed contrasts compare each bank's gate against weighted/RRF/linear/LambdaMART, incidence versus pair for each fusion, and incidence versus amplified for each fusion. Both Recall@20 and NDCG@20 are recorded. Duplicate complete test bundles are clustered for 2,000 bootstrap draws and 10,000 paired sign flips. Dataset summaries reserve **144 tests** (24 contrasts x 2 metrics x 3 datasets) for Holm correction. The supplied input/test splits remain the unit of generalization; no population-wide or all-domain claim follows.

## Commands

```sh
.venv/bin/python -m unittest experiments.novelty.test_operators -v
.venv/bin/python -m experiments.novelty.run --dataset pog_dense --smoke --output results/novelty_smoke_v2
zsh scripts/run_novelty_studies.sh
```

For a retained detached queue:

```sh
screen -dmS novelty_attribution zsh scripts/run_novelty_studies.sh
```

Follow progress:

```sh
tail -F results/novelty_attribution/queue.log results/novelty_attribution/pog_dense/run.log
# Once Spotify starts:
tail -F results/novelty_attribution/spotify/run.log
```

Resume a specific dataset with its identical command:

```sh
.venv/bin/python -u -m experiments.novelty.run --dataset pog_dense
```

Runtime environment reuses installed scientific packages through the local `.venv`; the added dependency is pinned in `requirements.txt`. Versions and source hashes are saved per run. A smoke report is never a benchmark result.
