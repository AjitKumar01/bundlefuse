# Neural baseline audit

Validation-only audit of frozen `graph_pair_no_category` checkpoints, from `experiments/neural_audit/redundancy.py`. No test query was scored and nothing was retrained. Seeds 0 and 1 on POG and POG_DENSE, seed 0 on Electronic, Food and Clothing. Raw numbers are in the JSON files.

Limits:

- Interventions change one part of a frozen model, so removals overstate the cost: the remaining parts were trained alongside it.
- Replacement pair tables are rescaled to the mean learned weight, with no tuning or refit.
- Combination weights for BundleFuse signals are picked on a random half of validation queries (A) and reported on the other half (B). These signals are diagnostics only. They are not part of the neural model.
- Cold and warm recall for combination arms cover all validation queries, including half A.
- Nothing here is significance-tested.
- BundleFuse validation NDCG recomputed here matches the stored recipe values (.0404 and .2849).

## Removing or replacing parts: NDCG@20

| Arm | POG s0 / s1 | POG_DENSE s0 / s1 | Electronic | Food | Clothing |
|---|---|---|---:|---:|---:|
| Full | .0231 / .0238 | .2701 / .2710 | .3408 | .3818 | .4437 |
| Pair weights replaced by log counts | .0229 / .0234 | .2646 / .2665 | .3291 | .3825 | .4416 |
| Pair weights replaced by a constant | .0207 / .0215 | .2581 / .2656 | .3312 | .3843 | .4462 |
| No pair term | .0188 / .0192 | .2223 / .2247 | .2919 | .3524 | .4275 |
| No graph layer | .0099 / .0104 | .2338 / .2464 | .2553 | .2576 | .2989 |
| No encoder | .0150 / .0151 | .0918 / .1148 | .3091 | .3453 | .3924 |
| Raw features only | .0002 / .0002 | .0001 / .0001 | .1072 | .1846 | .1809 |

## POG gap, half B

| Arm | Seed 0 R@20 / N@20 | Seed 1 R@20 / N@20 |
|---|---|---|
| Neural | .0367 / .0231 | .0375 / .0229 |
| Neural + style | .0400 / .0252 | .0395 / .0253 |
| Neural + cat | .0458 / .0297 | .0450 / .0297 |
| Neural + cat + style | .0537 / .0367 | .0553 / .0368 |
| BundleFuse | .0545 / .0383 | .0545 / .0383 |

## Target structure, validation

| Dataset | Never-sold targets | Share of never-sold targets in a seed's category | Share of sold targets in a seed's category |
|---|---:|---:|---:|
| POG | 2,495 of 4,203 | .0008 | .0012 |
| POG_DENSE | 1,103 of 4,619 | 0 | 0 |
| Electronic | 129 of 274 | .209 | .166 |
| Food | 138 of 293 | .304 | .219 |
| Clothing | 183 of 302 | .175 | .218 |
