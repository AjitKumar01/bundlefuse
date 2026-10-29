# BundleFuse

A cheap, interpretable, **regime-adaptive** baseline for **bundle completion**: given a partial
bundle `S`, rank catalogue items to recover the held-out members. It runs on a single CPU core in
tens of seconds and is competitive with heavy GPU encoders, a 7B LLM, and a diffusion model on the
cold/sparse regimes where those methods are supposed to dominate.

The scorer has **zero trained parameters** — its handful of fusion weights are *selected* on a
validation split, not learned by gradient descent.

```
score(c | S) = (1 - u_c) · Y_fused(c | S)  +  u_c · μ₀(c | S) ,    u_c = κ_f / (f_c + κ_f)
```

- **`Y_fused`** — a validation-weighted fusion of closed-form evidence: raw collaborative
  co-occurrence (`BᵀB`), PPMI / PPMI-SVD, dual-spectrum taxonomic PMI, style-disentangled and
  graph-bridged content. (Raw co-occurrence is kept *un-debiased* on purpose — **popularity is signal**.)
- **`μ₀`** — a metadata/ontology+content prior; cold items (`f_c = 0`) are ranked entirely by it.
- **`u_c`** — frequency-confidence **cold-start shrinkage** (classical Bell–Koren / RLFM); `κ_f` is
  selected by validation grid search.

An optional ~0.3M set-attention contrastive encoder exists as an **off-by-default** fused signal
(`ENCODER=1`); the headline results do not depend on it.

## Results at a glance (Recall@20)

| Dataset | **Ours** (0 params) | Best matched baseline† | Heavy models (CLHE / RaMen)‡ |
|---|---|---|---|
| pog            | **0.0653** | 0.0259 (BundleKNN) | 0.0284 / 0.0375 |
| pog_dense      | 0.3737     | 0.2842 (RP3β)      | **0.3815** / — |
| spotify        | **0.1562** | 0.1314 (Combiner)  | 0.1082 / 0.1091 |
| spotify_sparse | **0.1653** | 0.1180 (Combiner)  | 0.0969 / — |
| electronic     | 0.5192     | 0.3341 (iALS)      | 0.4721 / **0.8371** |
| food           | 0.5870     | 0.3362 (iALS)      | 0.5077 / **0.8459** |
| clothing       | **0.5904** | 0.2651 (ItemKNN)   | — |
| dunnhumby      | **0.1501** | 0.1370 (Combiner)  | — |

**Ours beats the best of six tuned matched baselines on all eight datasets**, and beats the heavy
GPU encoder CLHE on six of six where both run (it edges us only on dense POG, within noise). RaMen
wins e-commerce (electronic/food) — but via its heavy Collaborative Strategy Encoder, not a better
bundle model (see MANIFEST/paper). Numbers are the committed `results/matrix_zeroparam.csv`,
`results/baselines_*.json`, and reproduced/published heavy baselines.

<sub>†ItemKNN / P3α / RP3β / SLIM / iALS / BundleKNN / LinearCombiner, same splits/candidates/metrics.
‡CLHE = 9.5–16M-param GPU encoder; RaMen = attention-GNN encoder.</sub>

## Repository layout

```
bundlefuse/                 the package (library core)
  operators.py              the scorer: collaborative / taxonomic / content operators + shrinkage
  selection.py              validation selection objectives (ndcg@K, recall@K, mrr, ndcgfull, nll)
  protocol.py               the bundle-completion masking protocol
  baselines/                matched baselines, one file per method + OOP base class
    base.py itemknn.py randomwalk.py slim.py bundleknn.py ials.py combiner.py suite.py
  paths.py                  single source of truth for data / results locations
  eval/                     one runnable evaluator per dataset family (+ run_matrix)
    fullrank.py             POG / POG_dense / Spotify / Spotify_sparse
    bundlerec.py            electronic / food / clothing
    dunnhumby.py            transaction baskets
    run_matrix.py           orchestrates all eight into one table
  diagnostics/              sensitivity sweeps + the legacy OFAT ablation driver
docs/        EXPERIMENTS.md (the reproducibility index) · MANIFEST.md (salvage log)
notes/       methodology.tex / .pdf (how a partial bundle is filled, worked end-to-end)
paper/       the write-up (main.tex → main.pdf)
results/     regenerated logs + json (indexed in results/README.md)
tests/       test_core.py (7 theoretical / numerical / no-leakage tests)
```

The same five-step algorithm runs on every dataset — only the loader and the available content
features change. It is written out once, with the exact command behind every reported number, in
**[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**. The step-by-step derivation — one real POG query
(a bag + sandals, recovering earrings and a dress) carried through *every* equation and figure with
the live scorer's actual numbers — is in **[notes/methodology.tex](notes/methodology.tex)** (20 pp;
`pdflatex methodology.tex`).

## Quickstart

```bash
python3 tests/test_core.py                                  # 7 unit tests

BUNDLE_DS=pog_dense MATRIX=1 python -m bundlefuse.eval.fullrank    # POG / Spotify family
BRDS=electronic    MATRIX=1 python -m bundlefuse.eval.bundlerec    # e-commerce family
MATRIX=1                     python -m bundlefuse.eval.dunnhumby    # retail baskets

python -m bundlefuse.eval.run_matrix                        # full 8-dataset matrix
```

CPU-deterministic by default (set `FORCE_DEVICE=mps|cuda` for speed, **not** for published numbers).
Data lives one level up (`../bundlemllm`, `../BundleRec`, `../dunnhumby`) and is not vendored.

## Honest findings (the contribution)

1. **Cold-start shrinkage is the working mechanism** — it lifts cold-bucket Recall@20 by +2–5pp across
   the cold/sparse datasets. This is the real driver of the wins.
2. **A tiny CPU model is regime-competitive** with 9.5–16M-param GPU encoders (CLHE, RaMen), a 7B LLM
   (Bundle-MLLM), and a diffusion model (DDBC), on POG / Spotify / sparse / retail.
3. **Protocol audit (DDBC):** on DDBC's own 100-candidate protocol a *parameter-free co-occurrence
   baseline* beats the diffusion SOTA — evidence the reported gains there are a protocol artifact.
4. **Where heavy models win is data, not modelling:** RaMen's e-commerce edge comes from its heavy
   Collaborative Strategy Encoder (a multi-head attention GNN), not from a better bundle model.

### Negative results we report honestly (not hidden)

- **Higher-order estimators are nearly inert.** Simplicial incidence and Rao concordance add at most
  +0.79pp (Spotify), ≤+0.26pp on seven of eight datasets, once a strong pairwise base is in place.
  They are retained only to *document* this.
- **The empirical-Bayes variance model does not fit.** The homoscedastic model implies a positive OLS
  slope, but it is negative on most datasets because the collaborative score is dominated by
  popularity. We therefore do **not** claim a data-driven `κ_f`; it is grid-selected on validation,
  and we say so.

## Provenance

This repository is a **cleaned, honest fork** of the prior `simba/` codebase — it keeps only what was
independently verified correct and drops what was fabricated, inert, or non-reproducible. See
**[docs/MANIFEST.md](docs/MANIFEST.md)** for the salvage/drop list, **[RESULTS.md](RESULTS.md)** for the
full result tables with error bars and the documented negatives, and **[paper/](paper/)** for the
write-up (`pdflatex main.tex`).
