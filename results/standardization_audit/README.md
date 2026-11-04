# POG standardization and target-isolation audit

Status: **passed**. Sample: 128 randomly selected supplied test queries,
seed 20261104, scoring batches of 32. The current runtime loaded the retained
POG checkpoint from `checkpoints/reorganized_162dd8c/pog/seed_0/model.pt`.
Its fitted signal profile is `extended` and its content rank is
`256`; this audit does not refit the current default configuration.

The audit verified training/test-query fingerprints against the checkpoint and kept revealed
items fixed. It evaluated the original targets, then replaced every target with the highest-ranked
prediction, then replaced every target with an item outside the top 20. Original sample
Recall@20 was 0.07421875; the two altered-label evaluations were
1.0 and 0.0 respectively. Across these label changes, all full-catalogue score bits and top-20
predictions were identical. Fitted settings and the checkpoint SHA-256 were unchanged.

For each of the 17 full-reference signals, all other query columns
were replaced by extreme values before standardization. The unchanged query retained identical
standardized score bits. Alone-versus-batch normalization differed by at most
9.536743164e-07;
this floating-point reduction difference is recorded separately from statistical query dependence.

This checks that hidden labels and other queries do not supply information to the audited
inference standardization. It is not an audit of external feature pretraining or historical
parameter-selection decisions.

Reproduce from the repository root:

```bash
python scripts/audit_standardization.py
```

The complete query IDs, model settings, checkpoint fingerprint, numerical checks and runtime
fingerprints are recorded in [pog.json](pog.json). Three regression cases cover NumPy/Torch
query isolation and fixed-model target isolation in `tests/test_pipeline.py`.
