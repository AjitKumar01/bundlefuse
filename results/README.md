# `results/` — what each file is

Everything here is regenerated from the pipeline; `docs/EXPERIMENTS.md` gives the exact command
behind each one. Most `*.json`/`*.csv` are git-ignored and regenerated on demand; the few that are
version-controlled on purpose (the canonical matrix, the weight sweeps, the matched baselines) are
whitelisted in `.gitignore`.

## Canonical headline numbers
| File | What | Produced by |
|---|---|---|
| `matrix_zeroparam.csv`, `matrix_zeroparam.json` | The 8-dataset zero-parameter matrix (R/N@10,20 + cross-seed std) — the paper's `tab:main`. | `scripts/regen_zeroparam.py` (+ `scripts/add_retail_stds.py` for the retail error bars) |

## Weight / selection studies
| File | What | Produced by |
|---|---|---|
| `wsweep_pog.json`, `wsweep_spotify.json` (+ `.log`) | Per-signal fusion-weight sensitivity; back `paper/figs/fig_wsweep`. | `WSWEEP=1 python -m bundlefuse.eval.fullrank` |
| `selection_sensitivity.md`, `selsens.log` | Selection-objective sensitivity study (NDCG@K plateau; cutoff-free losses). | `SEL_OBJ=… SELDUMP=1` runs; see EXPERIMENTS.md |

## Matched-protocol baselines (Phase 2)
| File | What | Produced by |
|---|---|---|
| `baselines_<ds>.json` (8) | ItemKNN / P3α / RP3β / SLIM / iALS / BundleKNN / LinearCombiner on identical splits. | `BASELINES=1 python -m bundlefuse.eval.<evaluator>` |
| `baselines_all8.log`, `baselines_fullrank.log` | Full run transcripts. | same |

## Diagnostics & honest-negative provenance (one-off transcripts)
`ablation_zeroparam.log` (stage ablation → `tab:ablation`), `ease_baseline.log`, `encoder_head2head.log`
(encoder on/off), `image_ab_electronic.log` (multimodal-image negative result), `valgap_fullrank.log` /
`valgap_retail.log` (selection generalization gap), `data_efficiency_pog.log` (train-fraction curve),
`kfold_ab.log` (the retracted k-fold probe), `regen_zeroparam.log`, `retail_stds.log`. These are kept as
evidence for claims in `RESULTS.md`; regenerate via the `ABLATE` / `VALGAP` / `ENCODER` modes and the
`scripts/` helpers documented in `docs/EXPERIMENTS.md`.

## Case study (methodology)
`case_study_pog.json` — one real `pog_dense` query dumped end-to-end (every signal + the four fusion
stages) for `notes/methodology.tex`. Produced by `CASE_STUDY=1 BUNDLE_DS=pog_dense python -m bundlefuse.eval.fullrank`.
