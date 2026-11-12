# Revised independent neural model

Status: complete

Architecture selection and checkpoints use validation NDCG only. Test scores are shown only after all selections are frozen.

| Model | Validation R20 | Validation N20 | Test R20 | Test N20 |
|---|---:|---:|---:|---:|
| bundlefuse_matched | 0.425723 | 0.350196 | 0.460453 | 0.344215 |
| bundlefuse_original_auxiliary | 0.476012 | 0.363504 | 0.486802 | 0.343218 |
| attributes_seed0 | 0.228324 | 0.117275 | 0.217775 | 0.096569 |
| pair_no_cold_seed0 | 0.335549 | 0.229467 | 0.364692 | 0.248930 |
| pair_cold_seed0 | 0.333622 | 0.222125 | 0.354383 | 0.243445 |
| joint_cold_seed0 | 0.309827 | 0.202310 | 0.328468 | 0.219577 |
| graph_joint_cold_seed0 | 0.379865 | 0.268235 | 0.394316 | 0.260694 |
| graph_joint_cold_seed1 | 0.364451 | 0.262816 | 0.405780 | 0.278184 |
| graph_joint_cold_seed2 | 0.369268 | 0.265333 | 0.396339 | 0.270884 |

All source scores in signal probes are diagnostic targets only. The scorer learns from masked training bundles, with no BundleFuse teacher or score input. Existing splits have been inspected; this revision is exploratory.
