# Independent neural retry: completed evidence

The old queue was stopped during Clothing. All checkpoints were retained. Two diagnosis-driven revisions were tested on Electronic; the final architecture was selected using Electronic validation, then transferred unchanged to Food and Clothing.

| Dataset | Neural Recall@20 | Neural NDCG@20 | Matched BF Recall@20 | Matched BF NDCG@20 | Trainable parameters |
|---|---:|---:|---:|---:|---:|
| electronic | 0.457852 | 0.329754 | 0.460453 | 0.344215 | 38,525 |
| food | 0.563724 | 0.440778 | 0.575071 | 0.449279 | 38,461 |
| clothing | 0.503489 | 0.379798 | 0.474663 | 0.388848 | 37,403 |

Neural values are means across three seeds. All methods use the same test queries, catalogue, observed-item masks and metric implementation. The matched BF control also has category inputs, which the final neural architecture does not consume; the neural model uses a subset of those inputs. Original BF additionally uses auxiliary/user information. No RaMen comparison with unmatched published splits is made here.

## Why the model changed

- V1 compressed historical relationships into item vectors, exposed training frequency as a predictive input, and did not simulate genuinely unseen inputs when dropping identities.
- Frozen validation probes found weak linear recovery of category, text, propagated-text and joint-history signals. Sparse learned link weights and neural attribute propagation improved source recovery, but did not by themselves match BF.
- Validation interventions isolated harmful unrestricted category penalties. The final model removes that table and constrains direct relationship weights to be nonnegative during optimization.
- The final model learns a small text-residual encoder, one graph propagation layer, and one scalar per observed directed training pair. It has no item-ID embedding table, no frequency input, no BF score or teacher, and no fixed frequency fallback.
- Training uses every eligible catalogue item and simulates unseen items by masking all history paths incident on selected IDs. The final architecture has no separate joint-score term; that controlled candidate did not win Electronic validation.

## How signal recovery was verified

Synthetic assays verify exact supported pair/triple coefficient representation, gradient learning, set invariance, cold-history masking, checkpoint replay and deterministic resume. These are capacity/correctness checks, not accuracy guarantees. Separate Ridge probes fit BF source targets only to frozen representations on half of validation queries and evaluate the other half. They never train the completion network.

Probe dimension and representation type differ between revisions; source standardization and shared sparse support affect recoverability. Higher probe R2 does not prove causal mechanism recovery. The final scorer does not reproduce every BF raw score, nor does it need to copy each one to rank well. Spectral, Rao and category-saturation equivalence has not been established.

## Evidence and logs

- [Initial source probes](../neural_revision/diagnostics/electronic/SIGNAL_DIAGNOSTIC.md)
- [Revision findings and validation interventions](../neural_revision/FINDINGS.md)
- [Electronic final report](../neural_constrained/electronic/REPORT.md)
- [Food transfer report](../neural_transfer/food/REPORT.md)
- [Clothing transfer report](../neural_transfer/clothing/REPORT.md)
- [Implementation and protocol](../../experiments/neural_revision/README.md)
- [Machine-readable three-seed summaries](summary.json)

Checkpoints remain under checkpoints/neural_small, checkpoints/neural_revision, checkpoints/neural_constrained and checkpoints/neural_transfer. Every evaluated epoch retains optimizer/RNG state; source/protocol changes use separate roots and preserve the earlier source archive.

These are exploratory results on previously inspected splits. Numerical closeness, small seed variation or an interval crossing zero is not a non-inferiority certificate or evidence of universal superiority.
