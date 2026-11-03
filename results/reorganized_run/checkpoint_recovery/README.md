# Checkpoint restoration

The original eight-dataset validation run used a temporary checkpoint directory and removed
the model files after its save/reload checks. This was an experiment-runner error. The original
result JSON files and frozen implementation snapshot were retained.

Restoration rebuilds the closed-form graph/content state using the original frozen implementation,
original input fingerprints and exact recorded selected weights/routing parameters. It does not
repeat the validation search or choose new settings. Each rebuilt model is checked against all
recorded aggregate test metrics and compared bit-for-bit with its reloaded checkpoint on every
original test query. Rebuilt files are explicitly labelled; they are not claimed to be the original
deleted byte streams.

Permanent models are stored at:

```text
checkpoints/reorganized_162dd8c/<dataset>/seed_0/model.pt
checkpoints/reorganized_162dd8c/<dataset>/seed_0/retention.json
```

`retention.json` records the SHA-256, size, restoration method and verification results. A permanent
copy of the original source/configuration is under `checkpoints/reorganized_162dd8c/source/`.
The checkpoints are intentionally not committed to Git; they remain on disk after verification.

Follow restoration:

```bash
tail -F results/reorganized_run/checkpoint_recovery/runner.log
```

Per-dataset restoration logs and verification JSON files live in this directory. `manifest.json`
records completed restorations and reports `complete` only after all eight are verified.

The working validation runner now saves permanent checkpoints by default and returns their paths,
hashes and sizes. The persistence regression test checks the file still exists and loads after the
validation function returns.
