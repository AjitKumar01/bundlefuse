# Amplified-neighbor and fallback follow-up

Exploratory results using the recorded input checkpoint and split. Original reference was selected on all earlier validation queries. Controlled banks fit/tune on identical 70/30 complete-bundle groups. Constant-mixture controls use the same fallback sources.

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| amplified:gate | 0.164725 | 0.231332 |
| reference:constant | 0.164725 | 0.231332 |
| amplified:weighted | 0.164725 | 0.231332 |
| reference:weighted | 0.164725 | 0.231332 |
| amplified:constant | 0.164725 | 0.231332 |
| original_reference | 0.154288 | 0.221912 |
| reference:gate | 0.164725 | 0.231332 |

Paired statistics reserve 60 tests across the five newly added datasets. These are fused score banks, not isolated raw operators; the historical column may receive zero selected weight. See inputs.json for checkpoint/data profile and query scope.
