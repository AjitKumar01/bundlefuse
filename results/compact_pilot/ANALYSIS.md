# Verified compact-model pilot findings

Both dataset jobs completed successfully. The audit verified all 9,938 original test queries,
69,566 method/query records, input/source/reference hashes, checkpoint selection, validation
gates and all 12 prespecified paired comparisons. The experiment implementation is commit
`29e6ca3`; its unchanged package-source digest is in the manifest. The implementation passed
57 regression tests, including independent legacy scorer/fusion parity and the end-to-end pilot.

## Outcome

The engineered features improve the compact model, but this student does not preserve
BundleFuse accuracy. Both validation feasibility gates fail. Keep the exact scorer optimization;
do not proceed automatically to sparse-input adaptation or distillation for this architecture.
This establishes a result for this representation, architecture and search budget, not an
impossibility result for every compact model.

## Original supplied-input accuracy

These are original benchmark masks, not the separate one-seed stress conditions. Means and
population SDs cover three separately trained models; no ensemble or best-test-seed selection.
Both student arms have 58,314 trainable parameters and the same 128-dimensional scoring space.

| Dataset | Metric | BundleFuse | Raw student mean ± SD | Engineered student mean ± SD |
|---|---|---:|---:|---:|
| POG_dense | R@20 | 0.395223 | 0.344869 ± 0.002084 | 0.354422 ± 0.001444 |
| POG_dense | NDCG@20 | 0.287876 | 0.227853 ± 0.000670 | 0.243653 ± 0.000312 |
| Spotify | R@20 | 0.155646 | 0.075347 ± 0.001446 | 0.105349 ± 0.000078 |
| Spotify | NDCG@20 | 0.257926 | 0.126813 ± 0.002835 | 0.173565 ± 0.000597 |

The engineered mean improves R@20 over the raw mean by 2.77% relatively on POG_dense and
39.82% on Spotify. It retains 89.68% and 67.68% of BundleFuse test Recall respectively;
test NDCG retention is 84.64% and 67.29%. These are descriptive mean ratios, not the
validation-only decision rule.

Primary inference uses seed 0, with independently validation-selected learning rates for each arm:

| Dataset | Engineered minus raw R@20 (percentage points) | Pointwise 95% cluster CI | Holm p |
|---|---:|---:|---:|
| POG_dense | +1.013 | [+0.267, +1.699] | 0.005699 |
| Spotify | +3.141 | [+2.849, +3.443] | 0.001200 |

The engineered-over-raw NDCG gains are also significant after Holm correction over all 12
contrasts. BundleFuse beats both seed-0 student arms on both metrics in both datasets after
the same correction. Its R@20 lead over engineered seed 0 is 4.174 points on POG_dense and
5.039 points on Spotify. Query uncertainty conditions on fitted models and fixed folds;
training-seed dispersion is a separate statistic. Duplicate complete bundles form clusters.

## Validation feasibility decision

The prespecified rule requires engineered seed 0 to beat raw seed 0 in validation NDCG@20
and retain at least 97.5% of BundleFuse validation R@20 and NDCG@20 on both datasets.

| Dataset | Engineered beats raw validation NDCG? | Validation R@20 retention | Validation NDCG retention | Pass |
|---|---|---:|---:|---|
| POG_dense | Yes | 88.28% | 85.36% | No |
| Spotify | Yes | 66.43% | 66.03% | No |

The rule fails before using the student test results. It is an engineering tolerance, not a
statistical equivalence test. The experiments meet the first question (engineering helps the
tested small learner) and fail the second (near-BundleFuse accuracy under this compression).

## Exact scorer optimization

Every candidate's float32 fused score is bit-identical between the legacy and selectively
computed scorer on all 5,938 POG_dense and 4,000 Spotify test queries. All historical
per-query R/N@10/20/50 values are reproduced. The scorer skips unused signals and diffusion
and shares indicator/weighted-overlap calculations. No candidate truncation or operator
approximation is involved. The supplied-split evaluator uses this selected path after fitting;
validation selection and the independent reference path continue to compute the full set.

The following are warm-cache median milliseconds per query, amortized over the same 32-query
batch, one CPU thread, one warmup and three repeats on an Apple M5 Pro with 24 GiB RAM.
They are not single-request/tail-latency or service-throughput measurements.

| Dataset | Reference scoring | Optimized scoring | Optimization speedup | Engineered student scoring |
|---|---:|---:|---:|---:|
| POG_dense | 7.911 | 2.019 | 3.92× | 0.0217 |
| Spotify | 80.441 | 54.949 | 1.46× | 0.1390 |

Including ranking/metric calculation, reference / optimized / engineered-student medians are
8.138 / 2.238 / 0.240 ms on POG_dense and 83.094 / 57.396 / 2.578 ms on Spotify.
The student is 9.34× / 22.26× faster than optimized BundleFuse under that measured pipeline,
with the accuracy losses above. Scoring-only speedups are much larger and should not be
substituted for end-to-end service latency claims. Broader backend/dataset speed claims are
not established by this pilot.

## Offline costs and training diagnostics

Frozen 256-dimensional input storage per student is 30.49 MiB on POG_dense and 248.20 MiB
on Spotify. A fitted student's cached 128-dimensional serving vectors are 15.24 MiB and
124.10 MiB respectively. These are distinct objects: offline training can need both, while
the serving scorer uses cached vectors plus small query weights/frequency data. The 58,314
parameter count does not include these frozen inputs or serving vectors.

Recorded shared preprocessing takes 19.36 / 61.26 seconds, and extra compact-feature construction
0.35 / 14.24 seconds (POG_dense / Spotify). Eight fits plus their validation evaluations take
457.11 / 1,579.94 seconds total per dataset, not the cost of one selected model. The separately
timed post-selection catalogue-encoding pass is about 0.006 / 0.053 seconds for engineered seed 0;
repeated catalogue encodings during validation are already included in the fitting total.
Feature construction still depends on graph/ontology/content preprocessing and does not remove
those offline costs. External embedding extraction/pretraining remains excluded. Whole-worker
peak RSS (4.10 / 8.49 GiB) includes shared data and several models; it cannot establish per-model
memory superiority.

All engineered fits stopped by the configured validation-patience rule. Two raw Spotify fits
hit the 100-epoch budget: the unselected 0.003 seed-0 configuration (best epoch 95) and the
selected-rate confirmation seed 1 (best epoch 90). All four primary seed-0 selected models
stopped by patience. These diagnostics do not prove global optimization or universal convergence.

## Interpretation and next research decision

The positive result is specific: within a matched compact architecture and fixed input budget,
the joint engineered representation is more useful than the tested raw representation. This
does not isolate which engineered block matters or establish methodological novelty by itself.
The previous, larger metadata ranker also has higher mean Recall than this compact engineered
student on both datasets (0.358360 / 0.118988); the smaller model is an accuracy/size trade-off.

The loss of useful full-operator information is a plausible explanation, but this experiment
does not identify it causally. It simultaneously compresses collaborative/content/ontology
representations and changes the scoring/training architecture. The rank-32 collaborative SVD
explained-variance diagnostics (0.561 / 0.311) are not ranking-accuracy guarantees or a proof
that the dot-product output rank alone causes the loss.

Do not treat distillation as an established remedy or proceed to it automatically. If the
student direction is revisited, first measure whether exact sparse collaborative retrieval plus
cheap metadata candidates can build a small shortlist with sufficient candidate coverage, then
test a candidate-specific scorer retaining shrinkage/structured signals. This is a proposed new
experiment, not an implemented or verified result. It would need a new validation-only protocol;
candidate selection and changed normalization must be evaluated explicitly.

The current BundleFuse benchmark results remain intact. The defensible outcome here is an
exact, faster implementation and evidence that engineering helps a small model, alongside a
clear failed feasibility gate for replacing BundleFuse with this particular student.

See [PLAN.md](PLAN.md), [full report](REPORT.md), [paired statistics](paired.json),
[verification](VALIDATION.md), [audit log](verification.log), and [hardware](hardware.json).

Verification note: one historical Spotify BundleFuse NDCG@50 record is 1.0000000000000002
from float64 rounding. The verifier now uses the existing suite's 1e-12 upper-bound tolerance.
No experiment score, model, checkpoint, selection or package source was changed by that fix.
