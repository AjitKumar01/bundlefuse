# Supplied-split controls and component study

Every method uses the same supplied training bundles, all validation queries, and all test queries.
Each ablation is retuned on validation with the removed component disabled. All selections are frozen before test scoring.

| Dataset | Status | BundleFuse | SimpleHybrid | HardSwitchHybrid |
|---|---|---:|---:|---:|
| pog | complete | 0.054458 | 0.034083 | 0.041042 |
| pog_dense | complete | 0.395223 | 0.363647 | 0.364741 |
| spotify | complete | 0.155646 | 0.099754 | 0.099961 |
| spotify_sparse | complete | 0.154288 | 0.115149 | 0.115458 |

R@20 values above. Ablation deltas below are BundleFuse minus the ablated method (percentage points).

| Dataset | No incidence | No Rao | No higher-order | No shrinkage | No bridge |
|---|---:|---:|---:|---:|---:|
| pog | +0.000 | +0.000 | +0.000 | +0.300 | -0.125 |
| pog_dense | +0.233 | +0.000 | +0.233 | +0.132 | +0.379 |
| spotify | +0.290 | +0.339 | +0.385 | +0.841 | +0.094 |
| spotify_sparse | +0.000 | +0.288 | +0.288 | +0.265 | +0.000 |

## Paired control comparisons

| Dataset | Control | R@20 gain (pp) | Pointwise cluster-bootstrap 95% CI (pp) | Holm p |
|---|---|---:|---|---:|
| pog | SimpleHybrid | +2.038 | [+1.679, +2.396] | 0.0056 |
| pog | HardSwitchHybrid | +1.342 | [+0.950, +1.704] | 0.0056 |
| pog_dense | SimpleHybrid | +3.158 | [+2.586, +3.699] | 0.0056 |
| pog_dense | HardSwitchHybrid | +3.048 | [+2.474, +3.592] | 0.0056 |
| spotify | SimpleHybrid | +5.589 | [+5.234, +5.960] | 0.0056 |
| spotify | HardSwitchHybrid | +5.569 | [+5.212, +5.936] | 0.0056 |
| spotify_sparse | SimpleHybrid | +3.914 | [+3.257, +4.577] | 0.0056 |
| spotify_sparse | HardSwitchHybrid | +3.883 | [+3.214, +4.547] | 0.0056 |

Duplicate complete held-out bundles are resampled/flipped as clusters. Means retain every original query.
Holm correction covers the prespecified family of 56 contrasts (both R@20 and NDCG@20).
Confidence intervals are pointwise. Inference is conditional on these fitted configurations and splits;
it does not measure training/split uncertainty or include published neural summary scores.
The same-information controls use raw content, category/artist/album evidence, co-occurrence, and popularity.
This study does not add a new trained metadata ranker or change the supplied masking protocol.
Commands, hashes, selection records, full logs, and per-query metrics are in the manifest and dataset directories.
