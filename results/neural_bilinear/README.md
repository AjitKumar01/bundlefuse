# Bilinear form and category table for the neural baseline

Code: `experiments/neural_bilinear/`. The plan in `plan.json` was fixed before any training. Results are in `selection.json` and `summary.json`.

Arms added to `graph_pair_no_category`:

- **bilinear**: `V M Vᵀ`, with M starting at the identity.
- **category**: a free signed category table, starting at zero.
- **both**: the two changes together.

Every other setting matches the original model. No BundleFuse score is an input, but the changes were chosen from a validation audit (`results/neural_audit`) that used BundleFuse signals as diagnostics. Exploratory: the splits were inspected earlier.

## Selection: seed 0 validation NDCG@20

| Arm | POG_DENSE | POG | Mean ratio to original |
|---|---:|---:|---:|
| Original | .2701 | .0231 | 1.00 |
| Bilinear | .3015 | .0227 | 1.05 |
| **Category** | .3013 | .0294 | **1.20** |
| Both | .3023 | .0231 | 1.06 |

The category arm was selected and trained at seeds 1 and 2.

## Test, three-seed per-query means

| Dataset | Model | Full R@20 / N@20 | Filtered R@20 / N@20 |
|---|---|---|---|
| POG_DENSE | Category | .4011 / .2992 | .3663 / .2669 |
| POG_DENSE | Original | .3735 / .2683 | .3381 / .2352 |
| POG_DENSE | BundleFuse | .3952 / .2879 | .3600 / .2580 |
| POG | Category | .0442 / .0283 | .0434 / .0275 |
| POG | Original | .0359 / .0228 | .0350 / .0220 |
| POG | BundleFuse | .0545 / .0361 | .0535 / .0352 |

Filtered seeds, NDCG@20: POG_DENSE category .2671 / .2680 / .2658 and original .2357 / .2346 / .2354; POG category .0276 / .0274 / .0275 and original .0220 / .0224 / .0217.

## Primary tests, filtered queries

Holm correction over 8 tests; 95% intervals are cluster-bootstrap.

| Dataset | Metric | Category minus | Difference (95% CI) | Holm p |
|---|---|---|---|---:|
| POG_DENSE | NDCG | Original | +.0317 (+.0292, +.0342) | .0008 |
| POG_DENSE | Recall | Original | +.0282 (+.0245, +.0322) | .0008 |
| POG_DENSE | NDCG | BundleFuse | **+.0089 (+.0060, +.0119)** | .0008 |
| POG_DENSE | Recall | BundleFuse | +.0063 (+.0020, +.0108) | .0043 |
| POG | NDCG | Original | +.0055 (+.0043, +.0067) | .0008 |
| POG | Recall | Original | +.0084 (+.0064, +.0106) | .0008 |
| POG | NDCG | BundleFuse | **−.0078 (−.0103, −.0054)** | .0008 |
| POG | Recall | BundleFuse | −.0101 (−.0140, −.0064) | .0008 |

Against the fixed rule:

- **Helps:** yes, on both datasets.
- **Versus BundleFuse:** ahead on POG_DENSE, behind on POG.

## Recall by target training frequency, full test, micro, three-seed mean

| Dataset | Model | Never sold | Sold 1–2 times | Sold 3+ times |
|---|---|---:|---:|---:|
| POG_DENSE | Category | .0002 (2,127 targets) | .0999 (1,358) | .5942 (5,700) |
| POG_DENSE | Original | .0002 | .0952 | .5542 |
| POG | Category | .0211 (4,934) | .0639 (1,903) | .0904 (1,574) |
| POG | Original | .0172 | .0520 | .0726 |

The new parameter counts are 182,300 on POG_DENSE (4,489 category weights added to 177,811) and 168,195 on POG (5,184 added to 163,011). BundleFuse's POG comparison uses the canonical publication recipe; the later block-control parity replay was higher, at .0400 filtered NDCG.
