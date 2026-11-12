# Independent neural POG experiment

Status: complete

Fixed architecture from Electronic; three seeds; full supplied splits and full-catalogue ranking. Secondary test results exclude exact complete-bundle overlaps.

| Model | Full Recall@20 | Full NDCG@20 | Filtered Recall@20 | Filtered NDCG@20 |
|---|---:|---:|---:|---:|
| Canonical BundleFuse | 0.395223 | 0.287876 | 0.360014 | 0.258000 |
| graph_pair_no_category_seed0 | 0.371646 | 0.268194 | 0.337329 | 0.235731 |
| graph_pair_no_category_seed1 | 0.373218 | 0.268353 | 0.337810 | 0.234600 |
| graph_pair_no_category_seed2 | 0.375603 | 0.268329 | 0.339192 | 0.235378 |

Neural model inputs are a catalogue-only rank-128 projection of released title, image and description vectors. BundleFuse uses its own frozen feature processing; identical raw inputs do not imply identical representations. No BundleFuse teacher or score input is used.

All neural epochs/architectures are selected without test scores. Full validation includes supplied overlaps; secondary test filtering is a sensitivity check, not a fresh independently selected protocol.
