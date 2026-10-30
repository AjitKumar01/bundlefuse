# Experiment helpers

The primary runner is `python -m bundlefuse.experiments`; see
[../docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md).

| Script | Purpose |
|---|---|
| `fingerprint_data.py` | Records/checks SHA-256 hashes of the 52 evaluator input files |
| `report_refresh.py` | Verifies paired hybrid outputs, source fingerprint, and canonical consistency; renders selection results |
| `regen_zeroparam.py` | Runs the fail-closed engine in matrix-only mode with paired split seeds |
| `add_retail_stds.py` | Checks canonical CSV/JSON means and standard deviations agree; no longer patches tables |
| `head2head.py` | Historical optional-encoder comparison |
| `kfold_ab.py` | Historical fold-averaging probe; not genuine cross-validation |
| `trainfrac_study.py` | Training-fraction learning curve |
| `harvest_image_urls.py`, `clip_embed.py` | Optional public-image feature preparation |

The historical helpers other than `regen_zeroparam.py` have not been promoted to the new
provenance/paired-seed runner. Use isolated result directories for exploratory runs.
