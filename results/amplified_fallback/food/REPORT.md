# Amplified-neighbor and fallback follow-up

Exploratory results using the recorded input checkpoint and split. Original reference was selected on all earlier validation queries. Controlled banks fit/tune on identical 70/30 complete-bundle groups. Constant-mixture controls use the same fallback sources.

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| amplified:gate | 0.594995 | 0.461578 |
| reference:constant | 0.567139 | 0.428837 |
| amplified:weighted | 0.561709 | 0.434705 |
| reference:weighted | 0.566195 | 0.428079 |
| amplified:constant | 0.561709 | 0.434705 |
| original_reference | 0.594854 | 0.462355 |
| reference:gate | 0.607602 | 0.463644 |

Paired statistics reserve 60 tests across the five newly added datasets. These are fused score banks, not isolated raw operators; the historical column may receive zero selected weight. See inputs.json for checkpoint/data profile and query scope.
