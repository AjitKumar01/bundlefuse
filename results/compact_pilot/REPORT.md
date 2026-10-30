# Compact-model feasibility pilot

Original supplied inputs only. Direct training; no teacher or sparse-input adaptation.
Raw/engineered arms share architecture, raw columns, parameter count and tuning budget.

| Dataset | Status | BundleFuse R@20 | Raw mean ± SD | Engineered mean ± SD | Validation gate |
|---|---|---:|---:|---:|---|
| pog_dense | complete | 0.395223 | 0.344869 ± 0.002084 | 0.354422 ± 0.001444 | False |
| spotify | complete | 0.155646 | 0.075347 ± 0.001446 | 0.105349 ± 0.000078 | False |

## Warm-cache scoring costs

| Dataset | Reference ms/query | Optimized ms/query | Raw ms/query | Engineered ms/query |
|---|---:|---:|---:|---:|
| pog_dense | 7.9110 | 2.0191 | 0.0228 | 0.0217 |
| spotify | 80.4412 | 54.9487 | 0.1383 | 0.1390 |

Timings are amortized over the same 32-query CPU batch after warmup; three repeats.
Scoring-plus-metrics, offline preprocessing, encoding, training and cache bytes are in each timing.json.
Per-arm seed-0 validation selects learning rate; seeds 1/2 are independent confirmations, not an ensemble.
The 97.5% validation-retention target is an engineering feasibility gate, not a statistical equivalence test.
Query CIs condition on the fitted model; seed SD describes a separate source of variation.
No later stage starts automatically from test results. This pilot cannot establish that every low-rank model fails.

## Logs

- [pog_dense](pog_dense/run.log) — complete
- [spotify](spotify/run.log) — complete

Protocol and limits: [PLAN.md](PLAN.md).

Both jobs are now independently verified. [Interpretation and decision](ANALYSIS.md) ·
[Verification record](VALIDATION.md). Engineering helps the compact learner, but both
validation feasibility gates fail. The exact BundleFuse scoring optimization is retained.
