# Amplified-neighbor and fallback follow-up

Exploratory results using the recorded input checkpoint and split. Original reference was selected on all earlier validation queries. Controlled banks fit/tune on identical 70/30 complete-bundle groups. Constant-mixture controls use the same fallback sources.

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| amplified:gate | 0.525427 | 0.408100 |
| reference:constant | 0.468643 | 0.349496 |
| amplified:weighted | 0.470440 | 0.349862 |
| reference:weighted | 0.470440 | 0.349862 |
| amplified:constant | 0.468643 | 0.349496 |
| original_reference | 0.516936 | 0.404863 |
| reference:gate | 0.525427 | 0.408100 |

Paired statistics reserve 60 tests across the five newly added datasets. These are fused score banks, not isolated raw operators; the historical column may receive zero selected weight. See inputs.json for checkpoint/data profile and query scope.
