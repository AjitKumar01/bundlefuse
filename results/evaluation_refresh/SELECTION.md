# Corrected selection-objective study

Single split seed 0 for each dataset; POG uses 1,500 test queries. These are sensitivity
probes, not multi-seed estimates or evidence that selection cannot overfit. MRR uses the
first relevant item; full NDCG assigns distinct positions to tied items. All objectives
share the same deterministic item-ID tie rule. Values below are test Recall@20.

| Validation objective | POG | Electronic | Dunnhumby |
|---|---:|---:|---:|
| ndcg10 | 0.0509 | 0.5222 | 0.1559 |
| ndcg20 | 0.0636 | 0.5179 | 0.1571 |
| ndcg50 | 0.0586 | 0.5033 | 0.1561 |
| recall20 | 0.0694 | 0.4859 | 0.1534 |
| mrr | 0.0538 | 0.4889 | 0.1509 |
| ndcgfull | 0.0594 | 0.4938 | 0.1565 |
| nll | 0.0442 | 0.4520 | 0.1427 |

The default remains NDCG@20 on held-out validation. Objective sensitivity is reported
directly; no blanket plateau or no-overfitting conclusion is inferred. Older MRR/full-NDCG
tables were generated with incorrect definitions and are superseded.
