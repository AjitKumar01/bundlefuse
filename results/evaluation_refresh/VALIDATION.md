# Final verification

- All **53/53** planned jobs completed, with **0 failures**; the controller exited successfully.
- **26 tests passed**. The captured output is in [tests.log](tests.log).
- All **52 evaluator input files** match their original SHA-256 hashes in [data_manifest.json](data_manifest.json).
- All `bundlefuse/**/*.py` files match the source fingerprint recorded before execution.
  The matching implementation commit is recorded in [source_revision.json](source_revision.json).
- All **256** paired hybrid metric comparisons agree between the matrix and independent baseline runs.
- All **576** canonical matrix mean/std values exactly match the unrounded seed summary.
- All **8** canonical baseline tables match their seed summaries; recorded seed IDs agree.
- Canonical CSV and JSON agree, including standard deviations. CSV line endings are normalized to LF.
- No experiment log contains a traceback, SLIM nonconvergence message, or runtime warning.

## Recheck

```bash
python scripts/report_refresh.py
python scripts/fingerprint_data.py --check
python scripts/add_retail_stds.py
python3 -m pytest -q tests
```

## Scope

Fullrank matrix/baseline comparisons use split seeds 0, 1, 2. E-commerce and retail use five
held-out splits (0–4). Selection-objective probes use one split seed per dataset; POG uses
1,500 test queries. Standard deviations describe split variability, not significance tests.
SLIM is explicitly gated out on the large catalogues. Published neural/LLM/diffusion figures
were not rerun on these exact splits. Historical ablations, figures, DDBC F1, and triple2vec
claims are not certified by this refresh. Durations are evaluator wall times, not isolated
inference benchmarks.
