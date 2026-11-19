# Bundle-budget test: how much training data does TRIAD need?

**Question.** A retailer starting out has few example bundles. At what amount of training data does TRIAD stop
beating the simple, training-free baselines?

**Answer.** Bundle count does not decide it:

- **Spotify and Spotify_sparse:** TRIAD is ahead at every count, down to 50 playlists. Every difference is
  significant.
- **Electronic, Food and Clothing:** TRIAD is ahead at full size (about 1,200 bundles). At 600 and 300 it is level on
  Electronic and Food and ahead on Clothing. At 100 and 50 it is behind on all three.
- **POG_DENSE** (fraction test, `results/bundle_budget`): TRIAD is still ahead at 1% (208 bundles). **POG** at 1% is
  slightly behind.

The pattern fits two quantities together: how many item pairs the training bundles contain, and what share of test
targets appear in training. TRIAD fell behind only where both were small (§5). The diagnostics in §4 point to
overfitting as the cause, not too few training steps.

These numbers are for TRIAD with BLIP title features on e-commerce. The earlier version (DIALS, with TF-IDF title
features) needed about 1,000 e-commerce bundles; its runs are archived in `runs_dials/` and `runs_steps_dials/`, and
this file's earlier version is in the git history.

## 1. Setup

- **Datasets:** Electronic, Food, Clothing (family "retail") and Spotify_sparse, Spotify (family "spotify").
- **Conditions:** the full training split (the split of the frozen TRIAD runs), plus nested random subsets of 3,000,
  1,000, 600, 300, 100 and 50 training bundles. The subsets are the first k of one fixed permutation per dataset (seed
  20261113). Retail has only about 1,200 bundles, so it uses 600 / 300 / 100 / 50. Validation and test queries are
  the same in every condition.
- **TRIAD:** the frozen training recipe, unchanged, with 3 seeds per condition. Full-size numbers are the frozen runs.
  Every statistic (pair counts, degrees, frequencies, partner histories) is rebuilt from the condition's bundles only.
- **Simple baselines** (training-free; only hyperparameters are chosen on validation): popularity, co-occurrence,
  ItemKNN, RP3beta, EASE, hybrid (content+ItemKNN), same-label, transition, label_knn and label_knn_content. They are
  fitted on the same bundles, with the same BLIP title features on e-commerce. The **best simple** method is picked on
  validation NDCG@20 separately for each condition.
- **Metric:** NDCG@20 and Recall@20 on the clean test queries, i.e. those whose bundle is in neither training nor
  validation.
- **Test:** a paired cluster bootstrap (CI) and sign flips (p), where clusters are identical complete bundles. Holm
  correction runs over all tests of one family (retail: 30 tests; spotify: 28). The smallest reachable p after Holm is
  about .003.

In the comparison files TRIAD's series is still keyed "DIALS", so older readers of those files keep working.

## 2. Retail: Electronic, Food, Clothing

**Bold** marks p (Holm) < .05.

### Electronic

| Bundles | TRIAD N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | TRIAD R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (1,211) | 0.3345 | hybrid | 0.3024 | +0.0321 [+0.0128, +0.0534] | 0.066 | 0.4669 | 0.4160 | +0.0509 | 0.066 |
| 600 | 0.2102 | hybrid | 0.2108 | −0.0007 [−0.0172, +0.0158] | 1.000 | 0.3185 | 0.2968 | +0.0217 | 1.000 |
| 300 | 0.1540 | hybrid | 0.1663 | −0.0123 [−0.0251, +0.0003] | 0.856 | 0.2308 | 0.2381 | −0.0073 | 1.000 |
| 100 | 0.1061 | hybrid | 0.1161 | −0.0101 [−0.0209, −0.0008] | 0.776 | 0.1921 | 0.1975 | −0.0054 | 1.000 |
| 50 | 0.0889 | hybrid | 0.0936 | −0.0047 [−0.0136, +0.0034] | 1.000 | 0.1758 | 0.1789 | −0.0031 | 1.000 |

### Food

| Bundles | TRIAD N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | TRIAD R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (1,232) | 0.4738 | hybrid | 0.4450 | +0.0289 [+0.0093, +0.0470] | 0.060 | 0.6178 | 0.5628 | +0.0549 | **0.007** |
| 600 | 0.3579 | hybrid | 0.3462 | +0.0118 [−0.0050, +0.0280] | 1.000 | 0.4948 | 0.4638 | +0.0310 | 0.207 |
| 300 | 0.3001 | hybrid | 0.2984 | +0.0017 [−0.0113, +0.0150] | 1.000 | 0.4270 | 0.4104 | +0.0166 | 1.000 |
| 100 | 0.2496 | hybrid | 0.2807 | −0.0311 [−0.0468, −0.0166] | **0.003** | 0.3875 | 0.3968 | −0.0093 | 1.000 |
| 50 | 0.2352 | hybrid | 0.2574 | −0.0222 [−0.0324, −0.0120] | **0.003** | 0.3689 | 0.3762 | −0.0073 | 1.000 |

### Clothing

| Bundles | TRIAD N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | TRIAD R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (1,294) | 0.4388 | hybrid | 0.3593 | +0.0794 [+0.0588, +0.1002] | **0.003** | 0.5927 | 0.4912 | +0.1014 | **0.003** |
| 600 | 0.3246 | label_knn_content | 0.2704 | +0.0543 [+0.0344, +0.0744] | **0.003** | 0.4777 | 0.4147 | +0.0630 | **0.010** |
| 300 | 0.2500 | label_knn_content | 0.2247 | +0.0253 [+0.0052, +0.0448] | 0.236 | 0.4097 | 0.3765 | +0.0332 | 0.590 |
| 100 | 0.2151 | label_knn_content | 0.2311 | −0.0160 [−0.0339, +0.0001] | 1.000 | 0.3681 | 0.3828 | −0.0147 | 1.000 |
| 50 | 0.2061 | label_knn_content | 0.2266 | −0.0206 [−0.0350, −0.0071] | 0.102 | 0.3737 | 0.3803 | −0.0066 | 1.000 |

**Reading.**
- **Full size:** TRIAD is ahead on all three, and every interval lies above 0. After Holm over 30 tests the lead is
  significant on Clothing and on Food recall; Electronic and Food NDCG stop at p ≈ .06.
- **600 and 300:** level on Electronic and Food (all four NDCG intervals contain 0); ahead on Clothing (+.054,
  significant, and +.025).
- **100 and 50:** TRIAD is behind on every row, by .005–.031 NDCG. Four of the six NDCG intervals lie below 0; Food
  is significant after Holm at both counts.

## 3. Spotify_sparse and Spotify

### Spotify_sparse

| Bundles | TRIAD N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | TRIAD R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (8,740) | 0.2562 | label_knn_content | 0.2053 | +0.0509 [+0.0464, +0.0555] | **0.003** | 0.1837 | 0.1467 | +0.0370 | **0.003** |
| 3,000 | 0.2322 | label_knn_content | 0.2026 | +0.0295 [+0.0257, +0.0333] | **0.003** | 0.1655 | 0.1433 | +0.0222 | **0.003** |
| 1,000 | 0.2154 | label_knn_content | 0.1900 | +0.0254 [+0.0221, +0.0289] | **0.003** | 0.1497 | 0.1298 | +0.0198 | **0.003** |
| 600 | 0.2074 | label_knn_content | 0.1861 | +0.0213 [+0.0183, +0.0246] | **0.003** | 0.1448 | 0.1270 | +0.0178 | **0.003** |
| 300 | 0.1999 | label_knn_content | 0.1822 | +0.0177 [+0.0144, +0.0210] | **0.003** | 0.1364 | 0.1236 | +0.0129 | **0.003** |
| 100 | 0.1883 | label_knn_content | 0.1786 | +0.0097 [+0.0071, +0.0124] | **0.003** | 0.1284 | 0.1220 | +0.0064 | **0.003** |
| 50 | 0.1804 | label_knn_content | 0.1763 | +0.0040 [+0.0017, +0.0064] | **0.003** | 0.1236 | 0.1210 | +0.0026 | **0.016** |

### Spotify

| Bundles | TRIAD N@20 | Best simple | Simple N@20 | Δ N@20 [95% CI] | p (Holm) | TRIAD R@20 | Simple R@20 | Δ R@20 | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| full (14,000) | 0.2823 | label_knn_content | 0.2469 | +0.0354 [+0.0313, +0.0396] | **0.003** | 0.1674 | 0.1466 | +0.0208 | **0.003** |
| 3,000 | 0.2596 | label_knn_content | 0.2299 | +0.0297 [+0.0268, +0.0328] | **0.003** | 0.1528 | 0.1335 | +0.0193 | **0.003** |
| 1,000 | 0.2447 | label_knn_content | 0.2193 | +0.0254 [+0.0228, +0.0283] | **0.003** | 0.1418 | 0.1249 | +0.0170 | **0.003** |
| 600 | 0.2362 | label_knn_content | 0.2145 | +0.0217 [+0.0192, +0.0244] | **0.003** | 0.1364 | 0.1215 | +0.0149 | **0.003** |
| 300 | 0.2222 | label_knn_content | 0.2034 | +0.0188 [+0.0163, +0.0215] | **0.003** | 0.1270 | 0.1148 | +0.0122 | **0.003** |
| 100 | 0.2030 | label_knn_content | 0.1836 | +0.0194 [+0.0169, +0.0220] | **0.003** | 0.1159 | 0.1004 | +0.0155 | **0.003** |
| 50 | 0.1891 | label_knn_content | 0.1769 | +0.0122 [+0.0101, +0.0143] | **0.003** | 0.1080 | 0.0993 | +0.0087 | **0.003** |

**Reading.** TRIAD is ahead on all 28 tests, and all are significant after Holm. The lead shrinks steadily as bundles
are removed. On Spotify_sparse at 50 it is small (+0.004 NDCG).

## 4. Why retail falls behind at small counts: two diagnostics

Both diagnostics ran on the retail conditions of 300, 100 and 50 bundles, with 3 seeds each. Numbers are test
NDCG@20 means.

- **Equal steps.** The frozen loop counts patience and the epoch limit in epochs. So 50 bundles give 2 gradient steps
  per epoch, against about 38 at full size. Here the condition's bundles are repeated until an epoch has as many rows
  as the full split, which matches the full-size step count. Statistics still come from the condition's bundles, each
  counted once.
- **Untrained.** TRIAD at its initial parameters, with no training step.

| Dataset | Bundles | Trained | Equal steps | Untrained | Best simple | Best epoch, trained | Best epoch, equal steps |
|---|---|---|---|---|---|---|---|
| Electronic | 300 | 0.1540 | 0.1488 | 0.1229 | 0.1663 | 3, 8, 3 | 1, 3, 2 |
| Electronic | 100 | 0.1061 | 0.1042 | 0.1005 | 0.1161 | 10, 14, 11 | 1, 1, 1 |
| Electronic | 50 | 0.0889 | 0.0807 | 0.0878 | 0.0936 | 5, 2, 4 | 1, 1, 1 |
| Food | 300 | 0.3001 | 0.2987 | 0.2570 | 0.2984 | 4, 10, 6 | 1, 2, 3 |
| Food | 100 | 0.2496 | 0.2539 | 0.2434 | 0.2807 | 5, 6, 7 | 1, 2, 1 |
| Food | 50 | 0.2352 | 0.2419 | 0.2324 | 0.2574 | 12, 1, 1 | 1, 1, 1 |
| Clothing | 300 | 0.2500 | 0.2493 | 0.2092 | 0.2247 | 15, 12, 11 | 3, 5, 3 |
| Clothing | 100 | 0.2151 | 0.2166 | 0.1987 | 0.2311 | 15, 13, 16 | 1, 1, 1 |
| Clothing | 50 | 0.2061 | 0.1860 | 0.1961 | 0.2266 | 5, 8, 2 | 1, 4, 1 |

**Reading.**
- **More steps do not help.** Equal steps is within ±0.02 of the frozen recipe everywhere and closes none of the gap
  to the best simple method. With equal steps, validation peaks after 1–5 epochs, mostly the first. After that, the
  training loss keeps falling fast while validation NDCG falls. Example, Food at 50 bundles, seed 0: the loss goes
  5.86 → 0.44 over 7 epochs while validation NDCG goes 0.213 → 0.133. So the model is not undertrained. It memorises
  the few bundles: overfitting.
- **Training helps, but not enough at 100 and below.** Training improves on the untrained model in all 9 cases (on
  Electronic at 50 by only .001). At 300 it reaches the best simple method on Food and passes it on Clothing; at 100
  and 50 it does not.
- **The starting point is already strong.** With BLIP title features, the untrained model is within .01–.04 of the
  best simple method at 100 and 50 bundles. Training has little room before it overfits.

## 5. Across all datasets: what predicts the outcome?

"Pairs" are the item pairs inside the training bundles, Σ_b |b|(|b|−1)/2. "Target coverage" is the share of test
targets that appear in at least one training bundle. Δ is TRIAD minus the best simple method, in NDCG@20. POG
and POG_DENSE come from the fraction test (`results/bundle_budget`), where the comparison set also included CLHE.
"Ahead" and "behind" mean the 95% interval excludes 0; "level" means it does not.

| Dataset | Bundles | Mean size | Pairs | Target coverage | Δ N@20 | Outcome |
|---|---|---|---|---|---|---|
| Electronic | 1,211 / 600 / 300 / 100 / 50 | 3.5 | 7,802 / 3,964 / 1,876 / 615 / 331 | .59 / .41 / .28 / .11 / .05 | +.032 / −.001 / −.012 / −.010 / −.005 | ahead, level, level, behind, level |
| Food | 1,232 / 600 / 300 / 100 / 50 | 3.5 | 7,780 / 3,885 / 1,797 / 638 / 271 | .54 / .34 / .19 / .09 / .03 | +.029 / +.012 / +.002 / −.031 / −.022 | ahead, level, level, behind, behind |
| Clothing | 1,294 / 600 / 300 / 100 / 50 | 3.3 | 7,102 / 3,449 / 1,821 / 714 / 369 | .33 / .17 / .08 / .04 / .01 | +.079 / +.054 / +.025 / −.016 / −.021 | ahead, ahead, ahead, level, behind |
| POG_DENSE | 20,780 / 2,078 / 208 | 3.6 | 100,640 / 10,237 / 952 | .76 / .52 / .34 | +.037 / +.026 / +.024 | ahead, ahead, ahead |
| POG | 14,000 / 1,400 / 140 | 3.6 | 69,633 / 7,165 / 664 | .41 / .13 / .02 | +.007 / +.002 / −.001 | ahead, ahead, behind |
| Spotify_sparse | 8,740 … 50 | 44 | 14.8 M … 72,510 | .71 … .05 | +.051 … +.004 | ahead at every count |
| Spotify | 14,000 … 50 | 64 | 45.9 M … 228,648 | .86 … .19 | +.035 … +.012 | ahead at every count |

**Reading.**
- **Bundle count alone fails.** 50 Spotify playlists are ahead, while 100 retail bundles are behind.
- **Pairs alone fail.** POG_DENSE is ahead with 952 pairs; Electronic and Food are level, not ahead, with about
  1,800–1,900 pairs.
- **Coverage alone fails.** Spotify_sparse is ahead at 5% coverage, while Food at 9% is behind.
- **Together, they fit every row.** TRIAD fell behind only where the training data had both few pairs (under about
  2,000) and low target coverage (28% or less): retail at 100 and 50, and POG at 1%. It was ahead whenever either was
  large: Spotify has many pairs, and POG_DENSE at 1% has 34% coverage. Small-and-small does not always mean behind:
  Clothing at 300 (1,821 pairs, 8%) is still ahead.

This is an observation on 7 datasets, not a tested rule. The two quantities are not varied independently, and the
datasets also differ in features, labels and domain.

## 6. What this means for a retailer

- With large bundles (playlist-sized, tens of items), TRIAD was ahead of every simple baseline from 50 bundles up.
- With small bundles (3–4 items), a tuned content+ItemKNN blend (training-free) is as good or better up to a few
  hundred bundles; TRIAD draws level at 300–600 and is ahead at about 1,200.
- The gap between 600 and 1,200 retail bundles is not sampled.

## 7. Not done

- **A fix for small retail data.** One idea: initialise TRIAD from the content+ItemKNN blend, with a penalty that
  pulls it toward that start, tuned on validation. Not implemented.
- **Separating pairs from coverage.** For example, truncating Spotify playlists to 3–4 tracks while keeping
  coverage, or matching pair counts across datasets.
- **An uncontended timing of full-size Spotify training.**

## 8. Reproduce

Code: `experiments/bundle_budget_retail/` (`conditions.py`, `train.py`, `baselines.py`, `compare.py`,
`diagnostics.py`). Data is read from the sibling data directories, never from this repository.

```
results/bundle_budget_retail/queue_blip.sh             # retail: TRIAD 3 seeds x 4 counts x 3 datasets, plus baselines
results/bundle_budget_retail/queue_diagnostics_blip.sh # equal-steps and untrained diagnostics
results/bundle_budget_retail/queue_spotify.sh          # Spotify_sparse and Spotify
PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget_retail.compare --family retail    # comparison.json
PYTHONPATH=.:../dials .venv/bin/python -m experiments.bundle_budget_retail.compare --family spotify   # comparison_spotify.json
```

| File | Contents |
|---|---|
| `conditions.json` | bundle counts, mean sizes and coverage of every condition |
| `comparison.json`, `comparison_spotify.json` | means, best simple per condition, all tests with CI and Holm p |
| `runs/<dataset>/<condition>/seed<s>/` | TRIAD history, selected epoch, test scores (checkpoints are not committed) |
| `runs_steps/…` | the equal-steps diagnostic |
| `init/<dataset>/<condition>.json` | the untrained diagnostic |
| `baselines/<dataset>/<condition>.json` | every simple baseline, with validation and test scores |
| `runs_dials/`, `runs_steps_dials/`, `init_dials/` | the earlier DIALS runs with TF-IDF features |
| `logs/` | all run logs |
