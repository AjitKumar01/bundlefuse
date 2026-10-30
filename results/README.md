# Result provenance

The current verified refresh is [evaluation_refresh/REPORT.md](evaluation_refresh/REPORT.md).
Its manifest lists each job's status, source fingerprint, command, environment, split seed and
runtime. Each job directory contains its full log and structured result. `summary.json` aggregates
only datasets whose required split seeds completed. The root README is updated from that summary.

Once every planned job succeeds, the runner replaces `matrix_zeroparam.csv` and
`matrix_zeroparam.json` from the same unrounded seed records, writes `matrix_zeroparam.meta.json`,
and replaces `baselines_<dataset>.json` with corrected paired results and selected settings.
Before that promotion, root-level tables are historical, even if a newer subset result exists.

The refresh contains matrix/baseline runs over eight datasets and selection-objective probes over
POG, Electronic and Dunnhumby. It does not regenerate old encoder comparisons, image experiments,
weight sweeps, exposure-bucket ablations, DDBC F1, or triple2vec results. Historical logs and the
paper are retained for provenance and should not be read as corrected-run measurements.

`runner_smoke/` is an untracked end-to-end runner verification on one Electronic split; it is not
a replacement for the complete matrix. Other temporary smoke outputs are kept outside the repo.
