# Amplified neighbours with frequency fallback: live results

The first three datasets reuse the completed novelty-attribution study. The remaining five run in a separate follow-up with constant-mixture controls. Input checkpoint profiles and query limits differ by dataset; see the [protocol](../../experiments/amplified_fallback/README.md). Original references used all earlier validation queries; controlled variants use grouped 70/30 fit/tuning splits. Comparisons with the originals are descriptive.

| Dataset | Status | Original R@20 | Original N@20 | Amplified weighted R@20 | Amplified weighted N@20 | Amplified fallback R@20 | Amplified fallback N@20 | Historical weight in amplified fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pog_dense | complete | 0.395223 | 0.287876 | 0.388851 | 0.287204 | 0.390985 | 0.288159 | 0.5 |
| spotify | complete | 0.155646 | 0.257926 | 0.160019 | 0.271723 | 0.163278 | 0.281146 | 0.25 |
| pog | complete | 0.054458 | 0.036092 | 0.051042 | 0.033425 | 0.053542 | 0.035597 | 0.25 |
| electronic | complete | 0.486802 | 0.343218 | 0.454961 | 0.328351 | 0.472495 | 0.331170 | 1 |
| food | complete | 0.594854 | 0.462355 | 0.561709 | 0.434705 | 0.594995 | 0.461578 | 0.5 |
| clothing | complete | 0.516936 | 0.404863 | 0.470440 | 0.349862 | 0.525427 | 0.408100 | 0 |
| dunnhumby | complete | 0.207513 | 0.153609 | 0.208755 | 0.154635 | 0.210508 | 0.153193 | 0.25 |
| spotify_sparse | complete | 0.154288 | 0.221912 | 0.164725 | 0.231332 | 0.164725 | 0.231332 | 0 |

Follow the queue with `tail -n 100 -F results/amplified_fallback/queue.log`.
Per-dataset logs and reports are under `results/amplified_fallback/<dataset>/`. All prior and new checkpoints/caches are retained.

A historical weight of zero means the selected model does not use amplified neighbours. An improved metric in such a row cannot be attributed to amplification.
