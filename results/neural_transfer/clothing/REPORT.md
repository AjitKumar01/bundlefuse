# Revised independent neural model

Status: complete

Architecture selection and checkpoints use validation NDCG only. Test scores are shown only after all selections are frozen.

| Model | Validation R20 | Validation N20 | Test R20 | Test N20 |
|---|---:|---:|---:|---:|
| bundlefuse_matched | 0.574638 | 0.449072 | 0.474663 | 0.388848 |
| bundlefuse_original_auxiliary | 0.569293 | 0.443812 | 0.516936 | 0.404863 |
| graph_pair_no_category_seed0 | 0.579076 | 0.443717 | 0.502650 | 0.381105 |
| graph_pair_no_category_seed1 | 0.587409 | 0.442988 | 0.494654 | 0.375255 |
| graph_pair_no_category_seed2 | 0.593659 | 0.448647 | 0.513163 | 0.383035 |

All source scores in signal probes are diagnostic targets only. The scorer learns from masked training bundles, with no BundleFuse teacher or score input. Existing splits have been inspected; this revision is exploratory.
