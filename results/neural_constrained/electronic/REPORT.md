# Revised independent neural model

Status: complete

Architecture selection and checkpoints use validation NDCG only. Test scores are shown only after all selections are frozen.

| Model | Validation R20 | Validation N20 | Test R20 | Test N20 |
|---|---:|---:|---:|---:|
| bundlefuse_matched | 0.425723 | 0.350196 | 0.460453 | 0.344215 |
| bundlefuse_original_auxiliary | 0.476012 | 0.363504 | 0.486802 | 0.343218 |
| graph_pair_no_category_seed0 | 0.432274 | 0.340799 | 0.456647 | 0.330101 |
| graph_joint_no_category_seed0 | 0.433719 | 0.340126 | 0.456647 | 0.331477 |
| graph_joint_positive_category_seed0 | 0.395472 | 0.311141 | 0.449952 | 0.306888 |
| graph_pair_no_category_seed1 | 0.440751 | 0.328332 | 0.458189 | 0.333792 |
| graph_pair_no_category_seed2 | 0.434393 | 0.325671 | 0.458719 | 0.325368 |

All source scores in signal probes are diagnostic targets only. The scorer learns from masked training bundles, with no BundleFuse teacher or score input. Existing splits have been inspected; this revision is exploratory.
