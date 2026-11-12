# Independent neural pilot

**Stopped at the user's request during Clothing.** Electronic and Food completed. All model
checkpoints, including partial Clothing runs, remain on disk. See [stop record](stopped.json).
The separate [revised pilot](../neural_revision/README.md) is the current experiment.

Launched on 2026-11-07 from branch `independent-neural-small`.
The detached session is `independent_neural_small`.

This is an independent model trained on masked training bundles and catalogue attributes.
It has no BundleFuse teacher, source scores, interaction matrices, or residual connection.
BundleFuse appears only as a comparator. The neural item-ID table is learned from scratch.

The queue runs Electronic, Food, then Clothing. Each dataset has three architectures
(`dot`, `relations`, `contextual`) and three training seeds (0, 1, 2): 27 neural fits total.
Training uses at most 80 epochs, full-catalogue validation every two epochs, and early stopping.
All nine checkpoint selections for a dataset are frozen before neural test evaluation.

## Follow progress

```sh
tail -F results/neural_small/queue.log
tail -F results/neural_small/electronic/run.log
tail -F results/neural_small/food/run.log
tail -F results/neural_small/clothing/run.log
```

Food/Clothing logs appear when their dataset starts. Logs append on resume.

- [Electronic report](electronic/REPORT.md)
- Food report will appear at `food/REPORT.md`.
- Clothing report will appear at `clothing/REPORT.md`.
- [Protocol and architecture](../../experiments/neural/README.md)
- [Verification](verification.json)

Read each dataset's `summary.json` status to distinguish running work from complete results.
Epoch `history.json` files contain training loss and validation metrics, not test estimates.
Do not interpret the small smoke-run metrics as benchmark evidence.

## Verified Electronic comparisons

| Method | Recall@20 | NDCG@20 |
|---|---:|---:|
| Original BundleFuse, including auxiliary/user data | 0.486802 | 0.343218 |
| BundleFuse with neural-input-matched data | 0.460453 | 0.344215 |
| Popularity | 0.059875 | 0.025772 |
| Co-occurrence | 0.314644 | 0.225382 |
| ItemKNN | 0.323459 | 0.237884 |
| ContentKNN | 0.195328 | 0.092470 |
| Local p3alpha implementation | 0.327408 | 0.215574 |
| Local rp3beta implementation | 0.317293 | 0.223772 |

These use all 346 identical Electronic test queries and the full 3,499-item catalogue.
The original checkpoint reproduced its saved report and remained unchanged by checksum.
The input-matched control was refitted on validation without auxiliary/user histories.
Classical baselines use repository defaults; they were not exhaustively tuned. Published
CLHE/RaMen results on different datasets are not included. These are previously inspected
splits, so the pilot is exploratory rather than a fresh confirmation experiment.

## Checkpoint retention

Every evaluated epoch, optimizer state, and RNG state remains under
`checkpoints/neural_small/<dataset>/<variant>_seedN/epoch_NNNN.pt`.
`selected.json` identifies the validation-selected checkpoint; no checkpoints are deleted.
Comparator models, immutable run protocols and source archives are retained alongside them.
Re-running `zsh scripts/run_neural_small.sh` resumes the same source/protocol; changes fail closed.
