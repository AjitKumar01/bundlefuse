# Historical manuscript

This draft and its compiled PDF predate the corrected evaluation refresh. Current paired results
are in [../results/evaluation_refresh/REPORT.md](../results/evaluation_refresh/REPORT.md) and the
root README. The manuscript tables, hardcoded exposure/regime figures, and runtime claims must be
reconciled before using this draft as a report of the current implementation.

The [supplied-split comparison](../results/provided_splits/REPORT.md) evaluates only BundleFuse
on the released POG/Spotify split files. The draft incorrectly describes POG_dense/Spotify_sparse
as newly constructed local variants: both are present in the original CLHE paper. Its published
variant scores and stated split ratio also differ from the historical local reproductions;
the new report distinguishes these sources explicitly.

The [completed component study](../results/component_study/ANALYSIS.md) now tests retuned
ablations and the raw-information hybrid controls on all four supplied splits. Gains over both
controls are supported after the planned multiple-comparison adjustment. Component benefits
are conditional: higher-order effects are clearest on music; bridging can reduce POG NDCG;
most of the R@20 gap over HardSwitch survives removal of all higher-order terms. The manuscript's
component explanations must reflect these results before it is presented as the current paper.

The corrected runner does not reproduce the DDBC F1 or triple2vec+adaLoyal tables: their evaluation
paths are absent from this checkout. RaMen's published ablation is evidence about its own model;
it does not precisely identify the cause of the cross-model performance gap.

The existing figure generator still contains historical inputs for some figures. Regenerating the
PDFs does not turn those inputs into newly measured experimental evidence.
