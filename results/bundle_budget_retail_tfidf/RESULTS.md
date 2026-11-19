# Bundle-budget test: how much training data does DIALS need?

**Question.** A retailer starting out has few example bundles. At what amount of training data does DIALS stop
beating the simple, training-free baselines?

**Answer so far.** Bundle count does not decide it:

- **Spotify and Spotify_sparse:** DIALS wins at every count, down to 50 playlists. Every difference is significant.
- **Electronic, Food and Clothing:** DIALS wins only at full size (about 1,200 bundles), ties at 600, and loses at
  300 and below.
- **POG_DENSE** (earlier test): DIALS still wins at 208 bundles.

The pattern fits two quantities together: how many item pairs the training bundles contain, and what share of test
targets appear in training. DIALS lost only where both were small (§5). The diagnostics in §4 point to overfitting as
the cause, not too few training steps.

## 1. Setup

- **Datasets:** Electronic, Food, Clothing (family "retail") and Spotify_sparse, Spotify (family "spotify").
- **Conditions:** the full training split (the split of the frozen DIALS runs), plus nested random subsets of 3,000,
  1,000, 600, 300, 100 and 50 training bundles. The subsets are the first k of one fixed permutation per dataset (seed
  20261113). Retail has only about 1,200 bundles, so it uses 600 / 300 / 100 / 50. Validation and test queries are
  the same in every condition.
- **DIALS:** the frozen training recipe, unchanged, with 3 seeds per condition. Full-size numbers are the frozen runs.
  Every statistic (pair counts, degrees, label factors, partner histories) is rebuilt from the condition's bundles
  only.
- **Simple baselines** (training-free; only hyperparameters are chosen on validation): popularity, co-occurrence, ItemKNN, RP3beta, EASE, hybrid, same-label, transition, label_knn
  and label_knn_content. They are fitted on the same bundles. The **best simple** method is picked on validation NDCG@20
  separately for each condition.
- **Metric:** NDCG@20 and Recall@20 on the clean test queries, i.e. those whose bundle is in neither training nor
  validation.
- **Test:** a paired cluster bootstrap (CI) and sign flips (p), where clusters are identical complete bundles. Holm
  correction runs over all tests of one family (retail: 30 tests; spotify: 28). The smallest reachable p after Holm is
  about .003.

## 2. Retail: Electronic, Food, Clothing

**Bold** marks p (Holm) < .05.

### Electronic

| Bundles | DIALS N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | DIALS R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (1,211) | 0.3346 | hybrid | 0.2986 | +0.0360 [+0.0155, +0.0573] | **0.007** | 0.4691 | 0.4320 | +0.0371 | 0.134 |
| 600 | 0.2007 | label_knn_content | 0.2146 | −0.0138 [−0.0321, +0.0050] | 0.923 | 0.3058 | 0.3150 | −0.0092 | 1.000 |
| 300 | 0.1436 | hybrid | 0.1712 | −0.0276 [−0.0457, −0.0087] | 0.056 | 0.2364 | 0.2697 | −0.0333 | 0.103 |
| 100 | 0.0903 | label_knn_content | 0.1226 | −0.0323 [−0.0510, −0.0152] | **0.004** | 0.1843 | 0.2135 | −0.0293 | 0.531 |
| 50 | 0.0732 | label_knn_content | 0.1002 | −0.0270 [−0.0438, −0.0117] | **0.008** | 0.1577 | 0.1947 | −0.0369 | 0.177 |

### Food

| Bundles | DIALS N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | DIALS R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (1,232) | 0.4388 | label_knn_content | 0.4117 | +0.0271 [+0.0078, +0.0483] | 0.128 | 0.5862 | 0.5460 | +0.0402 | 0.102 |
| 600 | 0.2978 | label_knn_content | 0.3016 | −0.0038 [−0.0242, +0.0163] | 1.000 | 0.4294 | 0.4122 | +0.0172 | 1.000 |
| 300 | 0.2258 | label_knn_content | 0.2518 | −0.0260 [−0.0453, −0.0076] | 0.102 | 0.3396 | 0.3553 | −0.0156 | 1.000 |
| 100 | 0.1614 | label_knn_content | 0.2534 | −0.0920 [−0.1160, −0.0684] | **0.003** | 0.2769 | 0.3596 | −0.0828 | **0.003** |
| 50 | 0.1515 | label_knn_content | 0.2314 | −0.0799 [−0.1023, −0.0585] | **0.003** | 0.2676 | 0.3456 | −0.0780 | **0.003** |

### Clothing

| Bundles | DIALS N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | DIALS R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (1,294) | 0.3876 | label_knn_content | 0.3331 | +0.0545 [+0.0333, +0.0777] | **0.003** | 0.5338 | 0.4462 | +0.0876 | **0.003** |
| 600 | 0.2573 | label_knn_content | 0.2651 | −0.0078 [−0.0248, +0.0089] | 1.000 | 0.3747 | 0.3852 | −0.0105 | 1.000 |
| 300 | 0.1858 | label_knn_content | 0.2147 | −0.0289 [−0.0440, −0.0136] | **0.006** | 0.3015 | 0.3362 | −0.0347 | 0.102 |
| 100 | 0.1466 | label_knn_content | 0.1919 | −0.0453 [−0.0621, −0.0294] | **0.003** | 0.2527 | 0.3209 | −0.0682 | **0.003** |
| 50 | 0.1230 | label_knn_content | 0.1899 | −0.0669 [−0.0872, −0.0480] | **0.003** | 0.2316 | 0.3177 | −0.0860 | **0.003** |

**Reading.**
- **Full size:** DIALS is ahead on all three. The lead is significant on Clothing (both metrics) and on Electronic
  NDCG. It is not significant on Food or on Electronic recall.
- **600:** no significant difference anywhere.
- **300 and below:** DIALS is behind on every row. 9 of the 9 NDCG gaps have CIs below 0, and 7 are significant after
  Holm.

## 3. Spotify_sparse and Spotify

### Spotify_sparse

| Bundles | DIALS N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | DIALS R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (8,740) | 0.2562 | label_knn_content | 0.2053 | +0.0509 [+0.0464, +0.0555] | **0.003** | 0.1837 | 0.1467 | +0.0370 | **0.003** |
| 3,000 | 0.2322 | label_knn_content | 0.2026 | +0.0295 [+0.0257, +0.0333] | **0.003** | 0.1655 | 0.1433 | +0.0222 | **0.003** |
| 1,000 | 0.2154 | label_knn_content | 0.1900 | +0.0254 [+0.0221, +0.0289] | **0.003** | 0.1497 | 0.1298 | +0.0199 | **0.003** |
| 600 | 0.2074 | label_knn_content | 0.1861 | +0.0213 [+0.0183, +0.0246] | **0.003** | 0.1448 | 0.1270 | +0.0178 | **0.003** |
| 300 | 0.1999 | label_knn_content | 0.1822 | +0.0177 [+0.0144, +0.0210] | **0.003** | 0.1363 | 0.1236 | +0.0127 | **0.003** |
| 100 | 0.1884 | label_knn_content | 0.1786 | +0.0098 [+0.0071, +0.0124] | **0.003** | 0.1284 | 0.1220 | +0.0064 | **0.003** |
| 50 | 0.1803 | label_knn_content | 0.1763 | +0.0040 [+0.0017, +0.0064] | **0.003** | 0.1235 | 0.1210 | +0.0026 | **0.016** |

### Spotify

| Bundles | DIALS N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | DIALS R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (14,000) | 0.2824 | label_knn_content | 0.2469 | +0.0356 [+0.0314, +0.0398] | **0.003** | 0.1674 | 0.1466 | +0.0208 | **0.003** |
| 3,000 | 0.2596 | label_knn_content | 0.2299 | +0.0298 [+0.0268, +0.0329] | **0.003** | 0.1528 | 0.1335 | +0.0193 | **0.003** |
| 1,000 | 0.2447 | label_knn_content | 0.2193 | +0.0254 [+0.0228, +0.0283] | **0.003** | 0.1419 | 0.1249 | +0.0170 | **0.003** |
| 600 | 0.2362 | label_knn_content | 0.2145 | +0.0217 [+0.0191, +0.0244] | **0.003** | 0.1363 | 0.1215 | +0.0148 | **0.003** |
| 300 | 0.2222 | label_knn_content | 0.2034 | +0.0188 [+0.0163, +0.0215] | **0.003** | 0.1270 | 0.1148 | +0.0122 | **0.003** |
| 100 | 0.2030 | label_knn_content | 0.1836 | +0.0194 [+0.0169, +0.0220] | **0.003** | 0.1159 | 0.1004 | +0.0155 | **0.003** |
| 50 | 0.1892 | label_knn_content | 0.1769 | +0.0122 [+0.0102, +0.0144] | **0.003** | 0.1081 | 0.0993 | +0.0088 | **0.003** |

**Reading.** DIALS is ahead on all 28 tests, and all are significant after Holm. The lead shrinks steadily as bundles
are removed. On Spotify_sparse at 50 it is small (+0.004 NDCG).

## 4. Why retail fails at small counts: two diagnostics

Both diagnostics ran on the retail conditions where DIALS lost (300, 100 and 50 bundles), with 3 seeds each.
Numbers are test NDCG@20 means.

- **Equal steps.** The frozen loop counts patience and the epoch limit in epochs. So 50 bundles give 2 gradient steps
  per epoch, against about 38 at full size. Here the condition's bundles are repeated until an epoch has as many rows
  as the full split, which matches the full-size step count. Statistics still come from the condition's bundles, each
  counted once.
- **Untrained.** DIALS at its initial parameters, with no training step.

| Dataset | Bundles | Trained (frozen recipe) | Equal steps | Untrained | Best simple | Best epoch, frozen | Best epoch, equal steps |
|---|---|---|---|---|---|---|---|
| Electronic | 300 | 0.1436 | 0.1457 | 0.0936 | 0.1712 | 10, 17, 3 | 5, 6, 2 |
| Electronic | 100 | 0.0903 | 0.0833 | 0.0778 | 0.1226 | 21, 22, 18 | 2, 2, 2 |
| Electronic | 50 | 0.0732 | 0.0665 | 0.0727 | 0.1002 | 1, 1, 1 | 2, 2, 1 |
| Food | 300 | 0.2258 | 0.2266 | 0.1412 | 0.2518 | 7, 10, 8 | 2, 4, 2 |
| Food | 100 | 0.1614 | 0.1693 | 0.1317 | 0.2534 | 8, 21, 9 | 2, 2, 2 |
| Food | 50 | 0.1515 | 0.1595 | 0.1262 | 0.2314 | 36, 4, 7 | 2, 2, 1 |
| Clothing | 300 | 0.1858 | 0.1858 | 0.1011 | 0.2147 | 8, 9, 10 | 4, 3, 3 |
| Clothing | 100 | 0.1466 | 0.1476 | 0.0937 | 0.1919 | 26, 26, 17 | 2, 2, 2 |
| Clothing | 50 | 0.1230 | 0.1032 | 0.0925 | 0.1899 | 11, 13, 7 | 5, 5, 1 |

**Reading.**
- **More steps do not help.** Equal steps is within ±0.02 of the frozen recipe everywhere and closes none of the gap
  to the best simple method. With equal steps, validation peaks after 1–6 epochs. After that, the training loss keeps
  falling fast while validation NDCG falls. Example, Food at 50 bundles, seed 0: the loss goes 6.47 → 0.43 over 8
  epochs while validation NDCG goes 0.167 → 0.114. So the model is not undertrained. It memorises the few bundles:
  overfitting.
- **Training helps, but not enough.** Training improves on the untrained model in 8 of 9 cases (Electronic at 50 is
  tied), but the trained model never reaches the best simple method.
- **The starting point is the problem.** Untrained DIALS uses only 64 of the 128 feature coordinates for content, and
  its label low-rank part is switched off (M = 0). label_knn_content uses the full features and the labels,
  and fits only a few hyperparameters on validation. With few bundles, training cannot move far enough from that weaker start before it overfits.

## 5. Across all datasets: what predicts the outcome?

"Pairs" are the item pairs inside the training bundles, Σ_b |b|(|b|−1)/2. "Target coverage" is the share of test
targets that appear in at least one training bundle. Δ is DIALS minus the best simple method, in NDCG@20. POG
and POG_DENSE come from the earlier fraction test (`results/bundle_budget`), where the comparison set also included
CLHE.

| Dataset | Bundles | Mean size | Pairs | Target coverage | Δ N@20 | Outcome |
|---|---|---|---|---|---|---|
| Electronic | 1,211 / 600 / 300 / 100 / 50 | 3.5 | 7,802 / 3,964 / 1,876 / 615 / 331 | .59 / .41 / .28 / .11 / .05 | +.036 / −.014 / −.028 / −.032 / −.027 | win, tie, lose, lose, lose |
| Food | 1,232 / 600 / 300 / 100 / 50 | 3.5 | 7,780 / 3,885 / 1,797 / 638 / 271 | .54 / .34 / .19 / .09 / .03 | +.027 / −.004 / −.026 / −.092 / −.080 | tie*, tie, lose, lose, lose |
| Clothing | 1,294 / 600 / 300 / 100 / 50 | 3.3 | 7,102 / 3,449 / 1,821 / 714 / 369 | .33 / .17 / .08 / .04 / .01 | +.055 / −.008 / −.029 / −.045 / −.067 | win, tie, lose, lose, lose |
| POG_DENSE | 20,780 / 2,078 / 208 | 3.6 | 100,640 / 10,237 / 952 | .76 / .52 / .34 | +.040 / +.026 / +.025 | win, win, win |
| POG | 14,000 / 1,400 / 140 | 3.6 | 69,633 / 7,165 / 664 | .41 / .13 / .02 | +.008 / +.003 / −.001 | win, win, lose |
| Spotify_sparse | 8,740 … 50 | 44 | 14.8 M … 72,510 | .71 … .05 | +.051 … +.004 | win at every count |
| Spotify | 14,000 … 50 | 64 | 45.9 M … 228,648 | .86 … .19 | +.036 … +.012 | win at every count |

\* Food at full size: +.027 NDCG with a CI above 0, but p (Holm) = .128.

**Reading.**
- **Bundle count alone fails.** 50 Spotify playlists win, while 300 retail bundles lose.
- **Pairs alone fail.** POG_DENSE wins with 952 pairs, while retail loses with 1,800.
- **Coverage alone fails.** Spotify_sparse wins at 5% coverage, while Electronic loses at 28%.
- **Together, they fit every row.** DIALS lost only where the training data had both few pairs (under about 2,000)
  and low target coverage (28% or less): retail at 300 and below, and POG at 1%. It won whenever either was large:
  Spotify has many pairs, and POG_DENSE at 1% has 34% coverage. The coverage margin is thin: 28% lost, while 34%
  won.

This is an observation on 7 datasets, not a tested rule. The two quantities are not varied independently, and the
datasets also differ in features, labels and domain.

## 6. What this means for a retailer

- With large bundles (playlist-sized, tens of items), DIALS was ahead of every simple baseline from 50 bundles up.
- With small bundles (3–4 items), use label_knn_content (training-free) until there are at least about 1,000
  bundles. DIALS tied at 600 and won at about 1,200 on Electronic and Clothing. Food was not significant even at full
  size.
- The gap between 600 and 1,200 retail bundles is not sampled.

## 7. Not done

- **A fix for small retail data.** One idea: initialise DIALS from label_knn_content's formula (full 128-dimensional
  content plus the label prior), with a penalty that pulls it toward that start, tuned on validation. Not implemented.
- **Separating pairs from coverage.** For example, truncating Spotify playlists to 3–4 tracks while keeping
  coverage, or matching pair counts across datasets.
- **An uncontended timing of full-size Spotify training.**

## 8. Reproduce

Code: `experiments/bundle_budget_retail/` (`conditions.py`, `train.py`, `baselines.py`, `compare.py`,
`diagnostics.py`). Data is read from the sibling data directories, never from this repository.

```
results/bundle_budget_retail/queue.sh              # retail: DIALS 3 seeds x 4 counts x 3 datasets, plus baselines
results/bundle_budget_retail/queue_diagnostics.sh  # equal-steps and untrained diagnostics
results/bundle_budget_retail/queue_spotify.sh      # Spotify_sparse and Spotify
PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget_retail.compare --family retail    # comparison.json
PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget_retail.compare --family spotify   # comparison_spotify.json
```

| File | Contents |
|---|---|
| `conditions.json` | bundle counts, mean sizes and coverage of every condition |
| `comparison.json`, `comparison_spotify.json` | means, best simple per condition, all tests with CI and Holm p |
| `runs/<dataset>/<condition>/seed<s>/` | DIALS history, selected epoch, test scores (checkpoints are not committed) |
| `runs_steps/…` | the equal-steps diagnostic |
| `init/<dataset>/<condition>.json` | the untrained diagnostic |
| `baselines/<dataset>/<condition>.json` | every simple baseline, with validation and test scores |
| `logs/` | all run logs |
