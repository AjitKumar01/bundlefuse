# Amplified neighbours with frequency fallback on the remaining datasets

This isolated follow-up extends the completed POG_DENSE, Spotify and POG attribution
study to **electronic, food, clothing, dunnhumby and Spotify_sparse**. Production
BundleFuse and the completed experiment code remain unchanged.

## Completed results (6 November 2026)

All five datasets completed successfully. The [eight-dataset results table](../../results/amplified_fallback/README.md)
includes the three earlier completed datasets. Each new dataset has seven fully
evaluated methods, retained per-query results and a paired-statistics report.

No newly added dataset establishes a significant advantage for amplified history
plus its gate over the matched reference-history bank plus its gate, under the
planned correction. The frequency gate improves NDCG over both no fallback and
the constant-mixture control on food (adjusted p about .012). On clothing it
improves Recall and NDCG over both controls (adjusted p about .006), but the
selected amplified history weight is zero: this supports the fallback, not
amplification. Spotify_sparse likewise discards amplified history and selects
kappa zero, so its improved descriptive comparison with the old frozen reference
comes from reweighting the other sources. Electronic and dunnhumby have no
significant planned contrasts. These outcomes do not make amplification a
consistent replacement for the existing historical estimator.

All four retail/commerce original checkpoint metric pairs reproduced with maximum
absolute discrepancy below 5e-16. The reference decompositions passed on three
test queries per dataset; all five input checkpoints were checked unchanged by
SHA-256. Eight targeted unit checks and four end-to-end smoke runs passed before
launch. A retained source archive and fully resolved checkpoint configurations
are recorded in [verification.json](../../results/amplified_fallback/verification.json).

## Question and comparisons

Two score banks receive the same selected source inventory from a retained checkpoint:

- `reference`: its existing effective historical column, including the frozen
  pair/incidence mixing coefficient.
- `amplified`: replace that column with frequency-weighted historical-bundle cosine
  overlap raised to power 1, 2 or 3, optionally retaining the top 100/500/all bundles.

Other sources remain identical, including any PPMI/SVD, descriptions, personalization,
propagated historical score or multi-seed concordance. These are fused-bank tests,
not isolated raw-score tests. A fitted historical coefficient can be zero.

Each bank gets three matched fits:

1. `weighted`: nonnegative coordinate-selected weighted sum.
2. `gate`: candidate-frequency fallback, with kappa 0/.05/.5/1/5/20 and unit or
   fitting-set matched fallback scale.
3. `constant`: candidate-independent fallback mixture with fraction
   0/.05/.25/.5/.75/1 and those same two scale options.

The constant mixture tests whether frequency is needed, rather than merely changing
the relative scale of fallback scores. All three share each bank's fitted base
weights and fallback source definitions. Both banks use the same grouped 70% fitting /
30% tuning validation partition. Amplification and coordinate weights use fitting
queries; gate/constant choices use full-catalogue tuning NDCG@20. There is no refit.
Every controlled recipe is frozen before test scores are generated.

A seventh row, `original_reference`, evaluates the retained production checkpoint
directly. It was selected on all earlier validation queries, so differences from the
newly refitted systems are descriptive rather than a matched selection comparison.

## Inputs and evaluation scope

- **Spotify_sparse:** retain the research-profile graph/features, full recipe and
  complete validation cache from `checkpoints/publication_mechanisms/spotify_sparse`.
  That earlier queue was interrupted after the full recipe/cache had been saved;
  this follow-up does not resume or overwrite that study. Use all 1,248 supplied
  validation and 2,498 supplied test queries.
- **Electronic, food and clothing:** retain the verified extended checkpoints from
  `checkpoints/reorganized_162dd8c/<dataset>/seed_0/model.pt` and reconstruct exactly
  their original seed-0 split. These small datasets use the full validation/test sets.
- **Dunnhumby:** retain its corresponding extended checkpoint, seed-0 random split
  and original 800-validation / 2,000-test query ceilings. This is not the separate
  chronological-retail robustness protocol.

These profiles preserve the earlier dataset-specific model inputs rather than
silently switching the retail/commerce experiments to a different feature pipeline.
The profile differences are recorded per dataset. Training/query hashes, input-file
hashes and input checkpoint checksums are verified/recorded. Personalization uses
the original aligned query users. Source arrays remain training/catalogue-derived.
Source inventory was previously selected on the inspected validation data, so the
new partition does not make source selection independent: this is a conditional,
exploratory study, not untouched confirmation.

Full-catalogue Recall@20 and NDCG@20 exclude all observed items and use the shared
deterministic tie policy. Save per-query metrics and top-20 predictions. Cluster by
complete bundle for 2,000 bootstrap draws and 10,000 paired sign flips. Reserve 60
Holm tests across six planned contrasts, two metrics and five new datasets.
Three sampled production-score comparisons verify the reference decomposition;
retail/commerce original aggregate metrics must also reproduce within 2e-5.

## Retention and logs

Inputs are read-only. New outputs are isolated under:

- `results/amplified_fallback/<dataset>/`: source/runtime manifest, protocol,
  fitting/tuning indices, amplified grid, base weights, candidate tuning results,
  frozen recipes, parity checks, query results and final report/statistics.
- `checkpoints/amplified_fallback/<dataset>/`: persistent query-major score caches
  with progress markers. Recipes reference the input checkpoint; they are sufficient
  to reconstruct the new scorer without refitting.
- `results/amplified_fallback/queue.log`: sequential queue status.

Existing checkpoints, caches and incomplete files are never automatically deleted.
Resume uses identical source/protocol/input hashes and a per-dataset lock. Completed
datasets are skipped by the queue only after their summary reports completion.
Code fingerprints reject resume after an implementation change; keep the launched
sources unchanged. Smoke outputs are separate and never benchmark evidence.

```sh
.venv/bin/python -m unittest discover -s tests -p test_amplified_followup.py -v
.venv/bin/python -m unittest experiments.novelty.test_operators -v
.venv/bin/python -u -m experiments.amplified_fallback.run \
  --dataset electronic --smoke --output results/amplified_fallback_smoke
screen -dmS amplified_fallback zsh scripts/run_amplified_fallback.sh
tail -n 100 -F results/amplified_fallback/queue.log
tail -n 100 -F results/amplified_fallback/spotify_sparse/run.log
```

Resume one unfinished dataset:

```sh
.venv/bin/python -u -m experiments.amplified_fallback.run --dataset dunnhumby
```

The first three datasets' earlier results remain under
`results/novelty_attribution/{pog_dense,spotify,pog}`; they are not rerun or overwritten.
