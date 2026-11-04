# Publication mechanism studies

Run from the repository root on `publication-mechanism-studies`:

```bash
zsh scripts/run_mechanism_studies.sh
```

The queue evaluates all supplied validation and test queries for POG_dense, POG,
Spotify and Spotify_sparse. Workers run sequentially with one compute thread to
bound memory. `protocol.py` fixes the plan; every worker records a source fingerprint,
input hashes, configuration and runtime. This is follow-up research informed by
previously inspected benchmark results.

Each dataset has twelve recipes: historical relations (including popularity), shared
category/descriptive evidence, their combination, the original full construction,
four constant/frequency-by-unit/matched-scale controls, and four stability repeats
(reverse order, seeded permutation, two cluster bootstrap validation samples).
The canonical full selection is the fifth stability condition. All recipes are
frozen before test scoring. The research-profile feature dimensions are preserved.

The family fits use the same coordinate grid and round ceiling; actual evaluated
candidates can differ because of convergence, available channels and memoization.
Full construction adds optional multi-item/backoff search. Routing controls share
the exact same combined weights, higher-order setting (off), and fallback; each
constant/frequency grid has six candidates. The disabled mixture is a grid endpoint.
Zero backoff uses the implementation's explicit disabled branch even for cold items.

The graph/catalogue artifacts are built using training data and known catalogue
features. Construction calls the existing fit API on one validation query solely
to initialize its artifacts; that provisional fusion is discarded. Every reported
selection uses all supplied validation queries (or the explicitly recorded bootstrap
population). No fitting or selection uses test labels. This is transductive catalogue
evaluation, not a claim that upstream embeddings have never seen evaluation items.

Retained files:

- `checkpoints/publication_mechanisms/<dataset>/artifacts.pt`: reusable preprocessing;
- `full.pt`: the full selected scorer;
- `validation_signals/*.npy`: all query-major signal caches, with completed-block progress;
- `results/publication_mechanisms/<dataset>/*.json`: every recipe, exact grids, fingerprints,
  per-query metrics, top-20 predictions, seed counts and frequency-bucket diagnostics;
- `run.log`: timestamped cache, selection and evaluation progress;
- root `paired.json` and `REPORT.md`: generated once all four jobs complete.

No checkpoints or caches are automatically deleted. A worker can resume with the
same command; changed source/protocol/input identities are rejected. Completed
selection recipes are reused; interrupted test evaluation restarts that stage.
Do not edit the running experiment source. Dataset-level locking prevents duplicate
workers writing the same outputs.

Twelve contrasts times two metrics times four datasets give 96 planned tests.
Duplicate complete bundles are clusters for 2,000 bootstrap samples and 10,000 sign
flips. Dataset-level summaries conservatively reserve the 96-test family; the final
aggregator pools all 96 tests for their joint Holm correction. Stability repeats,
frequency buckets and seed-size slices are descriptive, not additional confirmed
significance claims. Frequency buckets are 0, 1–5 and >5 training appearances;
bucket recalls condition on queries with targets in that bucket. Both full-catalogue
and within-bucket retrieval are recorded. The randomization utility is reused from
the earlier verified experiment branch.

Smoke check (eight validation/test queries; never a benchmark result):

```bash
python3 -m experiments.mechanisms.run --dataset pog_dense --smoke --output results/mechanism_smoke
```

Current scope is the four supplied splits. Chronological retail and the other native
datasets are not included in these claims. EASE/SLIM provenance reconciliation and
independent new-split confirmation remain separate from these mechanism experiments.
