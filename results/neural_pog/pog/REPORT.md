# Independent neural POG experiment

Status: complete

Fixed architecture from Electronic; three seeds; full supplied splits and full-catalogue ranking. Secondary test results exclude exact complete-bundle overlaps.

| Model | Full Recall@20 | Full NDCG@20 | Filtered Recall@20 | Filtered NDCG@20 |
|---|---:|---:|---:|---:|
| Canonical BundleFuse | 0.054458 | 0.036092 | 0.053525 | 0.035244 |
| graph_pair_no_category_seed0 | 0.036208 | 0.022676 | 0.035503 | 0.021954 |
| graph_pair_no_category_seed1 | 0.036042 | 0.023201 | 0.035086 | 0.022354 |
| graph_pair_no_category_seed2 | 0.035333 | 0.022568 | 0.034376 | 0.021710 |

Neural model inputs are a catalogue-only rank-128 projection of released title, image and description vectors. BundleFuse uses its own frozen feature processing; identical raw inputs do not imply identical representations. No BundleFuse teacher or score input is used.

All neural epochs/architectures are selected without test scores. Full validation includes supplied overlaps; secondary test filtering is a sensitivity check, not a fresh independently selected protocol.
