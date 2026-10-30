# Compact-model feasibility pilot

Original supplied inputs only. Direct training; no teacher or sparse-input adaptation.
Raw/engineered arms share architecture, raw columns, parameter count and tuning budget.

| Dataset | Status | BundleFuse R@20 | Raw mean ± SD | Engineered mean ± SD | Validation gate |
|---|---|---:|---:|---:|---|
| pog_dense | pending | pending | pending | pending | pending |
| spotify | pending | pending | pending | pending | pending |

## Warm-cache scoring costs

| Dataset | Reference ms/query | Optimized ms/query | Raw ms/query | Engineered ms/query |
|---|---:|---:|---:|---:|

Timings are amortized over the same 32-query CPU batch after warmup; three repeats.
Scoring-plus-metrics, offline preprocessing, encoding, training and cache bytes are in each timing.json.
Per-arm seed-0 validation selects learning rate; seeds 1/2 are independent confirmations, not an ensemble.
The 97.5% validation-retention target is an engineering feasibility gate, not a statistical equivalence test.
Query CIs condition on the fitted model; seed SD describes a separate source of variation.
No later stage starts automatically from test results. This pilot cannot establish that every low-rank model fails.

## Logs

- [pog_dense](pog_dense/run.log) — pending
- [spotify](spotify/run.log) — pending

Protocol and limits: [PLAN.md](PLAN.md).
