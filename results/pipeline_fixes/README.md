# Pipeline fix verification

Branch: `reorganized`, based on `organised` commit `15bd250`.

The fixes address checkpoint mutation/scoring drift, hidden split-protocol changes, mismatched
model/dataset/baseline arguments, ignored flags and report attributes, inconsistent tie ranking,
and the disconnected selected-signal optimization. Validation now uses temporary float32 disk
caches with bounded query blocks. Shared helpers hold protocol reconstruction, provenance,
metric evaluation and exact graph restoration rather than duplicating those contracts in CLIs.

## Regression checks

The suite passes 24 tests. [tests.log](tests.log) records the run. Coverage includes:

- A committed, data-independent pre-fix synthetic reference from `15bd250`: unchanged selected
  weights, routing, fused scores and Recall/NDCG. The reference was captured before editing
  the scorer, and is not a replacement for missing historical dataset benchmark goldens.
- Pre-save versus live-after-save and loaded predictions/float32 score bits, with compressed
  and uncompressed checkpoints. Save failures preserve the live model and prior checkpoint.
- Bit-identical full/reference and selected scores across fashion, music, group homophily,
  auxiliary co-occurrence, personalization, non-default incidence coefficient, higher-order
  mixes and shrinkage. Unselected expensive operations are not executed.
- Supplied training rows, masks and original query IDs, including duplicate complete bundles;
  explicit resplit mode, caps/full-query selection and rejected remasking of provided inputs.
- Shared feature arguments for all six baselines; model and baseline prediction tie ordering.
- End-to-end fitting/saving/reporting for BundleFuse, ContentKNN and Popularity. Reports contain
  actual weights/routing and full-precision matching metrics. Mismatched protocols and changed
  data files are rejected; fitted dataset provenance is copied rather than aliased.
- Effective search-round and scoring seed-cap controls; reduced-context targets retain dropped
  seeds; temporary validation files are cleaned up when scoring fails.
- The verification CLI checks every requested test query's checkpoint score parity and writes
  explicit protocol-bound results instead of comparing incompatible historical averages.

Run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest -q
```

## Real-data integration smoke

The smoke uses the real POG_dense catalogue (31,217 items) and all 20,780 supplied training
rows. It deliberately limits validation to 8 queries and test to 16, with collaborative rank 8,
content rank 16 and one fusion-search round. This is an integration check, **not a full accuracy
benchmark or evidence of reproducing the earlier 0.39522 result**.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 evaluator.py pog_dense \
  --protocol provided --val-n 8 --test-n 16 --svd-dim 8 --content-rank 16 \
  --fit-rounds 1 --chunk 8 --save /private/tmp/bundlefuse_pipeline_fixed_pog_dense.pt \
  --report results/pipeline_fixes/real_fit.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 inference.py \
  --ckpt /private/tmp/bundlefuse_pipeline_fixed_pog_dense.pt report pog_dense \
  --chunk 8 --out results/pipeline_fixes/real_inference.json
```

The fit and reloaded-inference metrics, selected weights, shrinkage and test IDs match exactly.
An additional check compares selected/reference score bits and pre-save/reloaded bits for
**all 499,472 candidate scores across those 16 test queries**, with maximum error zero.

Artifacts: [fit report](real_fit.json), [fit log](real_smoke.log),
[inference report](real_inference.json), [inference log](real_inference.log), and
[real score parity](real_parity.json). Earlier smoke snapshots are retained separately as
`real_fit_initial.json` / `real_smoke_initial.log`. Each run records its own executed source and
runtime; the last save-directory handling change is additionally covered by the nested-path
checkpoint regression test. Checkpoints live outside Git and can be regenerated with the commands.

## Operational and scientific scope

POG/Spotify now default to complete supplied splits. Random-resplit completion remains available
explicitly; retail/BundleRec keep that protocol and are not chronological deployment simulations.
Content projection is a lossy fit-time model choice, while checkpoint graph-matrix omission is
exact. Lazy reconstruction can increase first-query memory/latency. The optimization is connected
and tested for parity, but this change does not claim a newly measured speedup or a full
eight-dataset reproduction of historical research results.
