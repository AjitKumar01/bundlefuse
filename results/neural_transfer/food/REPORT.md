# Revised independent neural model

Status: complete

Architecture selection and checkpoints use validation NDCG only. Test scores are shown only after all selections are frozen.

| Model | Validation R20 | Validation N20 | Test R20 | Test N20 |
|---|---:|---:|---:|---:|
| bundlefuse_matched | 0.536269 | 0.404336 | 0.575071 | 0.449279 |
| bundlefuse_original_auxiliary | 0.568466 | 0.419377 | 0.594854 | 0.462355 |
| graph_pair_no_category_seed0 | 0.531534 | 0.381817 | 0.569783 | 0.443200 |
| graph_pair_no_category_seed1 | 0.526799 | 0.379453 | 0.567186 | 0.445680 |
| graph_pair_no_category_seed2 | 0.550663 | 0.378394 | 0.554202 | 0.433453 |

All source scores in signal probes are diagnostic targets only. The scorer learns from masked training bundles, with no BundleFuse teacher or score input. Existing splits have been inspected; this revision is exploratory.
