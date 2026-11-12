# General lift estimator

One estimator and menu on all eight datasets, with no dataset-specific flags: signed lift per group level,
residual description per modality, popularity, pair, incidence and amplified history. Only fitted values differ.
Protocol and all numbers are in `summary.json`. Exploratory: filtered queries, on splits inspected earlier.

Rule fixed before the run: a dataset passes if `general` beats `raw` and the shuffled mean on NDCG@20 with Holm p<.05
(family of 32), and beats every shuffled seed. The theory is general only if every dataset passes.

**Result: 4 of 8 pass, so the theory as stated is not general.**

| Dataset | Queries | General R@20/N@20 | Raw N@20 | Shuffled mean N@20 | Recipe R@20/N@20 | vs raw (Holm p) | vs shuffled (Holm p) | Pass |
|---|---:|---|---:|---:|---|---|---|---|
| pog | 3995 | .0530/.0369 | .0198 | .0306 | .0535/.0352 | +.0171 (.003) | +.0064 (.003) | yes |
| pog_dense | 5547 | .3597/.2579 | .2477 | .2468 | .3600/.2580 | +.0102 (.003) | +.0111 (.003) | yes |
| spotify | 3995 | .1318/.2241 | .2201 | .2210 | .1549/.2570 | +.0040 (.003) | +.0031 (.003) | yes |
| spotify_sparse | 2497 | .1237/.1575 | .1532 | .1527 | .1540/.2216 | +.0043 (.006) | +.0048 (.003) | yes |
| electronic | 346 | .4726/.3199 | .3199 | .3208 | .5179/.3595 | .0000 (1) | -.0009 (1) | no |
| food | 353 | .5490/.4140 | .4271 | .4015 | .6182/.4799 | -.0131 (1) | +.0125 (.60) | no |
| clothing | 371 | .4688/.3551 | .3596 | .3186 | .5345/.4222 | -.0045 (1) | +.0365 (.003) | no |
| dunnhumby | 2000 | .1587/.1109 | .1080 | .1046 | .1571/.1103 | +.0029 (.072) | +.0063 (.003) | no |

The gap to the recipe does not decide a pass, and `general` vs `recipe` was not significance-tested. Recipes were refit
separately on all validation queries, not under this study's partition. Retail and Dunnhumby use one seed-0 resplit.

Temporary signal caches under `checkpoints/general_lift/` were deleted after this report. Per-query predictions remain in
`results/general_lift/*/queries/`. Rerunning requires deleting that dataset's checkpoint folder because of its progress marker.
