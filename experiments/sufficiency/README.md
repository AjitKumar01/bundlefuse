# Score sufficiency experiments: execution

Branch: `sufficiency-studies`. This implements the primary E1–E4 sequence in
[the proposal](../../output/pdf/score_sufficiency_experiments.pdf).
Production scoring and all earlier artifacts remain unchanged.

**Primary runs completed:** POG_DENSE 56 configurations, POG 39 and Spotify 11.
See the [completed-results review](../../results/sufficiency/README.md), including
scope qualifications, paired comparisons and corrected active-source timing.
Conditional E5 and compact export/memory work remain unrun. The original timing
files included zero-weight menu columns; use the corrected timing files named
in the results review for efficiency comparisons.

## What runs

- POG_DENSE: all 31 nonempty selected-source subsets, fixed source deletions,
  E2 pair/joint/separate/coupled controls with fixed and refitted other weights,
  E3 support gates, full-ranking evaluation and dependency-aware warm timing.
- POG: the corresponding 15 subsets and E2/E3 controls.
- Spotify: canonical and amplified controls, E3 on power-two/top-100 amplified
  history (with support calculated from that exact bank), power-two/all and
  power-three/all controls, and fixed/refitted extra candidate–three-seed support.

E5 bridge/probe studies and nested training thinning are conditional on primary
findings, as specified in the proposal. They are not automatically launched.
There is no new untouched confirmation dataset; these results are exploratory.
No sufficiency/minimality certificate is reported from nonsignificant results.

## Data and controls

Input hashes and earlier source fingerprints must match. The retained validation
cache is checked against fresh source computation. Existing held-out caches are
read only after their completion marker and a fresh parity check pass.
New raw joint/third-order counts are computed from B; standardized arrays are
never subtracted to obtain a raw correction.

The complete-bundle audit discovered some cross-split duplicate item sets.
This study excludes validation queries duplicating training bundles and test
queries duplicating training or validation bundles. Exact original retained
indices, overlap counts and input hashes are saved per dataset. Operators and
old checkpoints are preserved. The new evaluation population therefore differs
slightly from the full supplied splits, and its metrics must not be silently
compared with the earlier unfiltered published-split results.

All remaining validation queries are grouped by the complete bundle and split
70% fitting / 30% tuning with seed 20261119. The earlier source menu remains a
previously selected inventory. Coordinate weights use the established grid and
four-round budget; subset enumeration does not imply globally optimal weights.
Nested subset solutions are allowed as warm starts for supersets. Fallback
membership is restricted to retained, nonzero designated sources; no default
source can resurrect a removed column. An empty fallback disables the gate.

E2 has each coupled alpha in 0/.5/1 and selects a representative using tuning
metrics before test scoring. Fixed-source comparisons inherit the common
reference fallback and refit historical weights only. E3 fixes the source bank
and scale and tunes one scalar each for history-frequency, history-constant,
effective-support and contributing-count controls. Its constant-full reference
tests ordinary interpolation too. Primary support is computed from the actual
selected pair/incidence bank, not unconditionally from incidence.

All configurations are frozen before held-out extra arrays are created.
Full-catalogue Recall/NDCG@20 share exclusion and stable tie rules. Per-query
metrics, top20 lists and complete-bundle clusters are retained. Paired cluster
bootstrap intervals (2,000 draws) are descriptive/unadjusted; 10,000 sign flips
and Holm correction reserve a conservative 256-test family across this queue.
The reports contain matched E2, E3 and E4 contrasts as well as reference contrasts.

## Cost scope

E1 timing recomputes required sources through top20 retrieval, rather than timing
cached score-vector addition. It records one-thread warm CPU batch 1 and batch 8,
64 fixed test queries and three repeats, with median and p95 milliseconds/query.
Timing verifies top20 parity outside the measured region first.
Required source dependencies are recorded. Per-model compact-export storage and
resident-memory claims require separate exports/measurement after selection;
the current timing report does not pretend to supply them.

## Retention and resume

Results and progress: `results/sufficiency/<dataset>/`.
Persistent supplemental arrays: `checkpoints/sufficiency/<dataset>/`.
No existing files or incomplete artifacts are deleted. A dataset lock prevents
concurrent mutations. Source/protocol changes reject resume, and completion
markers allow scoring, fitting evaluations and test-query blocks to resume.
Do not edit experiment code while full workers are running.

```sh
.venv/bin/python -m unittest discover -s tests -p test_sufficiency.py -v
screen -dmS sufficiency_studies zsh scripts/run_sufficiency_studies.sh
tail -F results/sufficiency/queue.log results/sufficiency/pog_dense/run.log
```

The queue is sequential: POG_DENSE, POG, Spotify. It skips completed summaries
on resume and stops on failures so errors cannot be reported as completed runs.
Individual resume command:

```sh
.venv/bin/python -u -m experiments.sufficiency.run --dataset pog_dense
```

Smoke outputs under `results/sufficiency_smoke*/` are separate and never
benchmark evidence. No neural scorer is trained in E1–E4; scalar weights,
gate strengths and source choices are fitted/selected on validation groups.

Source note: these studies read caches tied to a source hash, so re-running them requires commit `008558e`. Later fixes to `bundlefuse/` make the identity check refuse those caches, by design.
