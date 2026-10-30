# Learned baseline and generalization study

Selections use only original validation queries and are frozen before every test case.
The learned ranker consumes raw metadata, categories, and training-only collaborative item IDs.

| Dataset | Status | BundleFuse R@20 | Learned seed 0 | Learned mean ± SD (3 seeds) |
|---|---|---:|---:|---:|
| pog | complete | 0.054458 | 0.023167 | 0.023625 ± 0.000863 |
| pog_dense | complete | 0.395223 | 0.358566 | 0.358360 ± 0.001269 |
| dunnhumby_chrono | complete | 0.118867 | 0.096630 | 0.096970 ± 0.001112 |
| spotify | complete | 0.155646 | 0.120616 | 0.118988 ± 0.001154 |
| spotify_sparse | complete | 0.154288 | 0.109306 | 0.111407 ± 0.001769 |

## Sparse-input stress tests

| Dataset | Case | Mean revealed | Changed queries | BundleFuse | HardSwitch | Learned seed 0 | Full − learned gain (pp), 95% CI | Holm p |
|---|---|---:|---:|---:|---:|---:|---|---:|
| pog | supplied | 1.50 | 0 | 0.054458 | 0.041042 | 0.023167 | +3.129, [+2.666, +3.580] | 0.0200 |
| pog | half_seeds | 1.00 | 2010 | 0.044812 | 0.032146 | 0.020667 | +2.415, [+2.024, +2.793] | 0.0200 |
| pog | quarter_seeds | 1.00 | 2010 | 0.044812 | 0.032146 | 0.020667 | +2.415, [+2.024, +2.793] | 0.0200 |
| pog | one_seed | 1.00 | 2010 | 0.044812 | 0.032146 | 0.020667 | +2.415, [+2.024, +2.793] | 0.0200 |
| pog_dense | supplied | 2.00 | 0 | 0.395223 | 0.364741 | 0.358566 | +3.666, [+2.994, +4.321] | 0.0200 |
| pog_dense | half_seeds | 1.00 | 5927 | 0.354471 | 0.358263 | 0.320436 | +3.404, [+2.896, +3.906] | 0.0200 |
| pog_dense | quarter_seeds | 1.00 | 5927 | 0.354494 | 0.358252 | 0.320453 | +3.404, [+2.896, +3.907] | 0.0200 |
| pog_dense | one_seed | 1.00 | 5927 | 0.354494 | 0.358252 | 0.320453 | +3.404, [+2.896, +3.907] | 0.0200 |
| dunnhumby_chrono | supplied | 4.13 | 0 | 0.118867 | 0.113179 | 0.096630 | +2.224, [+1.667, +2.777] | 0.0200 |
| dunnhumby_chrono | half_seeds | 1.97 | 6000 | 0.084758 | 0.086168 | 0.084061 | +0.070, [-0.319, +0.440] | 1.0000 |
| dunnhumby_chrono | quarter_seeds | 1.14 | 6000 | 0.068694 | 0.069732 | 0.077852 | -0.916, [-1.268, -0.570] | 0.0200 |
| dunnhumby_chrono | one_seed | 1.00 | 6000 | 0.065616 | 0.066488 | 0.075765 | -1.015, [-1.372, -0.658] | 0.0200 |
| spotify | supplied | 31.19 | 0 | 0.155646 | 0.099961 | 0.120616 | +3.503, [+3.200, +3.809] | 0.0200 |
| spotify | half_seeds | 15.34 | 4000 | 0.121762 | 0.090683 | 0.100857 | +2.090, [+1.852, +2.324] | 0.0200 |
| spotify | quarter_seeds | 7.44 | 4000 | 0.094927 | 0.081966 | 0.082597 | +1.233, [+1.032, +1.436] | 0.0200 |
| spotify | one_seed | 1.00 | 4000 | 0.061574 | 0.062468 | 0.052999 | +0.857, [+0.699, +1.020] | 0.0200 |
| spotify_sparse | supplied | 21.84 | 0 | 0.154288 | 0.115458 | 0.109306 | +4.498, [+3.991, +5.031] | 0.0200 |
| spotify_sparse | half_seeds | 10.80 | 2498 | 0.117162 | 0.095966 | 0.093748 | +2.341, [+1.955, +2.734] | 0.0200 |
| spotify_sparse | quarter_seeds | 5.17 | 2498 | 0.094877 | 0.079065 | 0.077724 | +1.715, [+1.412, +2.057] | 0.0200 |
| spotify_sparse | one_seed | 1.00 | 2498 | 0.065089 | 0.055489 | 0.056514 | +0.857, [+0.626, +1.096] | 0.0200 |

## Measured costs

| Dataset | Preprocess (s) | Learned training + validation search (s) | Worker peak RSS (GiB) | Full scoring (ms/query) | Learned warm scoring (ms/query) |
|---|---:|---:|---:|---:|---:|
| pog | 19.90 | 347.37 | 4.97 | 9.807 | 0.020 |
| pog_dense | 19.10 | 454.42 | 4.04 | 8.550 | 0.016 |
| dunnhumby_chrono | 6.84 | 554.19 | 10.50 | 10.540 | 0.034 |
| spotify | 60.03 | 2064.00 | 7.93 | 79.094 | 0.087 |
| spotify_sparse | 30.89 | 865.48 | 10.05 | 56.536 | 0.073 |

## Learned-training diagnostics

| Dataset | Seed | Best epoch | Epochs run | Stop reason | Best at final check | Validation NDCG@20 |
|---|---:|---:|---:|---|---|---:|
| pog | 0 | 20 | 40 | patience | False | 0.015705 |
| pog | 1 | 10 | 30 | patience | False | 0.015711 |
| pog | 2 | 10 | 30 | patience | False | 0.016025 |
| pog_dense | 0 | 25 | 45 | patience | False | 0.242188 |
| pog_dense | 1 | 20 | 40 | patience | False | 0.246160 |
| pog_dense | 2 | 10 | 30 | patience | False | 0.244175 |
| dunnhumby_chrono | 0 | 10 | 30 | patience | False | 0.062141 |
| dunnhumby_chrono | 1 | 25 | 45 | patience | False | 0.062529 |
| dunnhumby_chrono | 2 | 5 | 25 | patience | False | 0.062199 |
| spotify | 0 | 80 | 100 | patience | False | 0.197388 |
| spotify | 1 | 90 | 100 | epoch_budget | False | 0.197590 |
| spotify | 2 | 90 | 100 | epoch_budget | False | 0.198354 |
| spotify_sparse | 0 | 30 | 50 | patience | False | 0.134415 |
| spotify_sparse | 1 | 50 | 70 | patience | False | 0.136175 |
| spotify_sparse | 2 | 75 | 95 | patience | False | 0.138913 |

The metadata ranker uses a shared content/category/item-ID encoder and a nonlinear set-query encoder,
sampled-softmax training, a four-configuration learning-rate/regularization grid, and three training seeds.
Hyperparameters are chosen on seed-0 validation; confirmation seeds use those hyperparameters and their own validation-selected checkpoints.
Learning curves and epoch-budget/early-stopping diagnostics are retained; check those before calling a capped run converged.
Dropped seeds move into hidden targets. All cases retain original folds and complete bundle membership, with at least one revealed item.
Half/quarter cases retain those fractions of the supplied seeds; one-seed queries are unchanged when already minimal.
The temporal retail case uses strictly separated whole-day folds, training-only interactions, and an explicit fixed metadata catalogue.
Its base case uses the original half-hidden mask on the new chronological future fold.
Catalogue introduction dates are unavailable; this is not a historical product-launch simulation.
The prespecified inference family contains 200 contrasts: all configured datasets/cases, five alternatives, and both R@20/NDCG@20.
Pointwise cluster-bootstrap CIs use 2,000 replicates; 10,000 paired cluster sign flips and Holm adjustment test differences.
Retail clusters are households; POG/Spotify clusters are identical complete bundles. Learned seed 0 is the primary competitor; no best-test-seed selection.
Training-seed dispersion is reported separately and does not make the query-bootstrap CI a training-uncertainty interval.
Warm-cache timings use the same 32 sampled test queries and three repeats. Catalogue encoding, preprocessing, search/training, and full-worker peak RSS are separate.
External embedding extraction/pretraining is excluded; worker RSS is not per-method memory. No latency number includes neural-model training.
