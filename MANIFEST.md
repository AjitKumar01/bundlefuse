# Salvage manifest

This repo was forked from the prior `simba/` codebase at its last **verified-honest committed state**
(git `HEAD = ededd65`), deliberately *before* a series of uncommitted edits that were found to be
fabricated, tautological, or non-reproducible. Below is exactly what was kept, cleaned, or dropped.

## Kept (independently verified correct)

| Component | File | Why kept |
|---|---|---|
| Collaborative operators (cooc `B^T B`, PPMI, PPMI-SVD, row-stochastic 2-hop) | `bundlefuse/operators.py` | Correct math; reproduce; "popularity is signal" holds |
| Dual-spectrum taxonomic ontology (PMI + slot exclusion + soft clusters) | `bundlefuse/operators.py` | Real, contributes on content-driven sets |
| Style-disentangled & graph-bridged content | `bundlefuse/operators.py` | Real cold-start prior signal |
| Honest conjugate shrinkage `(1-u)Y + u·mu0`, `u = k/(f+k)` | `bundlefuse/operators.py` | The working engine; classical, correct |
| Honest OLS variance diagnostic (reports negative slope, clips, defers to grid) | `bundlefuse/operators.py` | Truthful diagnostic; no tautology |
| Full-ranking / e-commerce / retail evaluators | `bundlefuse/{fullrank,bundlerec,dunnhumby}_eval.py` | "Ours" path verified clean; numbers reproduce qualitatively |
| Genuine sweep runner (real inference) | `bundlefuse/real_sensitivity.py` | Measured (electronic/dunnhumby β reproduce bit-exactly) |
| 7 genuine tests incl. batch-invariance (no-leakage) | `tests/test_core.py` | All pass; real guarantees |

## Dropped (and why)

| Removed / excluded | Reason |
|---|---|
| Fabricated `run_sensitivity_sweeps()` with hardcoded constants + synthetic ramps | Fabricated experimental data (e.g. `warm_r20=1.0` for a dataset with 0 warm items). Replaced by `real_sensitivity.py`. |
| Tautological variance "fix" (`v_rate = v / f`, regress on `1/f`) | Mechanically yields a positive slope even when the true `sigma_eps^2 = 0`; proven by synthetic test. |
| Per-exposure sample-mean "fix" (`score_pairwise_sample_mean`, `build_decomposed_observation`, `inv_f_exposure`) | Forces a positive slope by dividing the collaborative score by `f_c`, which *removes popularity* (the key signal) and craters the base (~49%→37%); made Clothing *worse*. |
| Circular "Test 8" (`test_exposure_sample_mean_positive_variance_slope`) | Asserts a result it constructs; was defined but never invoked by the runner. |
| "Simplicial-Spectral Empirical-Bayes" headline framing | Oversells two inert estimators (+0.00–0.34pp) as core mechanisms. |
| "Data-driven kappa / 8-of-8 positive slopes" claims | Not backed by the committed results (6/8 slopes are negative). `kappa` is grid-selected. |
| Baseline-weakening ablations (moving diffusion/bridge out of Stage 0) | Redistributes credit to inert operators; not a demonstration they help in the full model. |
| All pre-computed numbers in `results/` | Stale / non-reproducible under current code; everything will be regenerated once, deterministically. |

## Open tasks (to make this paper-ready)

1. **Regenerate all 8-dataset tables once, on CPU, with fixed seeds**, plus multi-seed error bars.
2. **Encoder on/off head-to-head** (closed-form zero-param vs. +0.3M encoder) on identical seeds — decide
   whether the encoder stays dropped (preferred) or returns as an optional module.
3. **Baseline-fairness check:** confirm reproduced CLHE matches published CLHE where both exist.
4. **RaMen image ablation:** run RaMen text-only to substantiate "the e-commerce gap is the private images."
5. **Unify / reconcile protocols** so the same method/dataset reports one consistent number across tables.
6. Decide the honest framing: an efficiency + reproducibility + regime-analysis paper built around the
   DDBC protocol audit and the cost/regime story — not a novel-architecture claim.
