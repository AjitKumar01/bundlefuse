# BundleFuse

A cheap, interpretable, **regime-adaptive** baseline for **bundle completion** (given a partial
bundle `S`, rank catalogue items to recover the held-out members). It runs on a single CPU core in
tens of seconds and is competitive with heavy GPU encoders, a 7B LLM, and a diffusion model on the
cold/sparse regimes where those methods are supposed to dominate.

This repository is a **cleaned, honest fork** of the prior `simba/` codebase. It keeps only what was
independently verified to be correct and drops everything that was fabricated, inert, or
non-reproducible. See **[MANIFEST.md](MANIFEST.md)** for the exact salvage/drop list and the reasons.

## What this method actually is

```
score(c | S) = (1 - u_c) * Y_fused(c | S)  +  u_c * mu_0(c | S) ,    u_c = kappa / (f_c + kappa)
```

- **`Y_fused`** — a validation-weighted fusion of closed-form evidence: raw collaborative
  co-occurrence (`B^T B`), PPMI / PPMI-SVD, dual-spectrum taxonomic PMI, style-disentangled and
  graph-bridged content. (Raw co-occurrence is kept *un-debiased* on purpose — **popularity is
  signal**.)
- **`mu_0`** — a metadata/ontology+content prior; cold items (`f_c = 0`) are ranked entirely by it.
- **`u_c`** — frequency-confidence **cold-start shrinkage**: low-exposure candidates are pulled
  toward the prior. This is classical Bell–Koren / RLFM shrinkage; `kappa` is selected by
  **validation grid search** (the variance-component model does *not* fit this data — see below).

An optional ~0.3M set-attention contrastive encoder exists as an **off-by-default** fused signal.
It was never the engine; the headline results do not depend on it.

## Honest findings (the contribution)

1. **Cold-start shrinkage is the working mechanism** — it lifts cold-bucket Recall@20 by +2–5pp across
   the cold/sparse datasets. This is the real driver of the wins.
2. **A tiny CPU model is regime-competitive** with 9.5–16M-param GPU encoders (CLHE, RaMen), a 7B LLM
   (Bundle-MLLM), and a diffusion model (DDBC), on POG / Spotify / sparse / retail.
3. **Protocol audit (DDBC):** on DDBC's own 100-candidate protocol, a *parameter-free co-occurrence
   baseline* beats the diffusion SOTA — evidence the reported gains there are a protocol artifact.
4. **Where heavy models win is data, not modelling:** RaMen's e-commerce edge comes from private
   Amazon product images absent from the public release, not from a better bundle model.

### Negative results we report honestly (not hidden)

- **Higher-order estimators are inert.** Simplicial hyperedge incidence (`score_simplicial_incidence_only`)
  and Rao quadratic concordance (`score_rao_concordance`) add only **+0.00 to +0.34pp** once a strong
  pairwise base is in place. They are retained only to *document* this negative result; they are not
  part of the headline method.
- **The empirical-Bayes variance model does not fit.** The homoscedastic model implies the OLS slope
  `sigma_eps^2 > 0`, but it is **negative on 7/8 datasets** because the collaborative score is
  dominated by popularity. We therefore do **not** claim a data-driven `kappa`; `kappa` is grid-selected
  on validation and we say so. (Prior attempts to force a positive slope by dividing residuals or scores
  by `f_c` were tautological or removed the popularity signal; they are excluded here.)

## Reproducibility

Default policy is **reproducibility first**: CPU execution with fixed seeds (`bundlefuse/_device.py`).
Earlier accelerator (MPS) runs drifted several percent run-to-run and were not reproducible; CPU runs
here are **bit-identical across repeats**. Strict deterministic kernels (`torch.use_deterministic_algorithms`)
are available via `DETERMINISTIC=1` but are off by default because they force ~10x-slower fallback kernels
and are not needed for reproducibility here.

```bash
# genuine unit / theoretical / no-leakage tests (7)
python3 tests/test_core.py

# evaluators (CPU-deterministic by default; set FORCE_DEVICE=mps|cuda for speed, NOT for published numbers)
BUNDLE_DS=pog        python3 -m bundlefuse.fullrank_eval
BRDS=electronic      python3 -m bundlefuse.bundlerec_eval
SPLIT_MODE=random    python3 -m bundlefuse.dunnhumby_eval

# full 8-dataset matrix
python3 -m bundlefuse.run_matrix
```

## Status

Every table in `results/` was regenerated **once, deterministically** from this pipeline — nothing is
carried over as a pre-computed number from the old `simba/` repo. See **[RESULTS.md](RESULTS.md)** for
the full zero-parameter matrix (with error bars), the encoder on/off head-to-head (encoder dropped),
the validation-selection overfitting probe (negative — selection generalizes), and the multimodal-image
experiment (honest negative result). Remaining open tasks are listed there and in MANIFEST.md.

The write-up lives in **[`paper/`](paper/)** (`main.tex`; `pdflatex main.tex` → `main.pdf`). It is the honest
BundleFuse paper — zero-parameter framing, regenerated tables with error bars, the documented negatives, the
CLHE baseline-fairness accounting, and the RaMen data-availability analysis — with every SIMBA-era fabrication removed.

Data lives one level up (`../BundleRec`, `../bundlemllm`, `../dunnhumby`) and is not vendored here.
