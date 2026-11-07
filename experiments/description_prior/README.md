# Description-prior pilot (stopped, exploratory)

Tests whether descriptions can act as a learned prior for co-purchase rather than as another
fused column: `W = G^-1 (Y'Y - X'FX) G^-1`, with `Y = BX` and `G = X'FX + lam I`. Counts then
update the predicted lift through a candidate-frequency gate.

Status: stopped on 2026-11-07, not in `bundlefuse/`. Scalars were chosen on the fitting split
and metrics reported on the tuning split of the filtered validation queries. No test split was
read, and the first run's tuning scores were seen before the grids were widened.

Files run in order: `cpm.py`, `cpm2.py`, `cpm3.py`, `cpm4.py` (the frozen settings), then the
Spotify diagnostics `spotify_diag.py`, `gamma.py` and `coldprior.py`. Run from the repo root with
`PYTHONPATH=.:experiments/description_prior`. Outputs are in `results/description_prior/`.

Known defects: the `counts only` row in `*_val.json` uses mismatched settings; the link test in
`cpm2.py` includes pairs seen in training (`cpm3.py` excludes them); `coldprior.py` was stopped
after its first variant.

Tuning results, R@20 / NDCG@20, new model against the best frozen recipe: POG_DENSE
0.3887/0.2779 vs 0.3815/0.2654; POG 0.0667/0.0423 vs 0.0617/0.0428; Spotify 0.1561/0.2587 vs
0.1764/0.2979. The Gamma-Poisson form did not close the Spotify gap, and the negative part of `W`
matched a shuffled control.

Source note: these studies read caches tied to a source hash, so re-running them requires commit `008558e`. Later fixes to `bundlefuse/` make the identity check refuse those caches, by design.
