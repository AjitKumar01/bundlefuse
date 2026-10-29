# Experiments — the algorithm, and one command per result

This is the map from *every reported number* to the exact command that produces it. If you only read
one file to understand the repo, read this one.

Data is **not** vendored. It lives one directory above the repo and is resolved automatically by
`bundlefuse/paths.py`:

```
<parent of repo>/bundlemllm/datasets/<ds>/   POG, POG_dense, Spotify, Spotify_sparse
<parent of repo>/BundleRec/dataset/<ds>/     electronic, food, clothing
<parent of repo>/dunnhumby/                  transaction baskets
```

Point the whole thing elsewhere with `BUNDLEFUSE_DATA_ROOT=/path/to/data` (then it expects
`$BUNDLEFUSE_DATA_ROOT/bundlemllm/...`, `.../BundleRec/...`, `.../dunnhumby/`), or override a single
dataset with `DATA_DIR=` / `BR_DIR=` / `DUNN_DIR=`. All runs are CPU-deterministic by default
(`bundlefuse/_device.py`); numbers are bit-identical across repeats, no GPU required.

**Where to get the data.** POG / POG_dense / Spotify / Spotify_sparse are the **BundleConstruction /
CLHE** release (Ma et al., WSDM'24; HF dataset `xhLiu/BundleConstruction`) — these already ship the
pre-extracted BLIP/CLAP content features the scorer loads. **BundleRec** (electronic/food/clothing) is
the BundleRec benchmark. **Dunnhumby** is "The Complete Journey" transaction log. The scorer needs only
the files each evaluator reads (`bi_*.txt` / `count.json` / `item_info.json` for POG/Spotify;
`bundle_item.csv` etc. for BundleRec; `transaction_data.csv` + `product.csv` for Dunnhumby).

---

## The algorithm (identical for all 8 datasets)

Given a partial bundle `S` (the seeds), score every catalogue item `c` and return the top-K. The
scorer has **zero trained parameters**; its few fusion weights are *selected*, not learned, on the
validation split. Same five steps everywhere — only the data loader and the available features differ.

```
1. BUILD (train fold only)      B (bundle×item incidence) → co-occurrence C=BᵀB (diag 0),
   bundlefuse/operators.py      PPMI + its 128-d SVD, degree/volume discounts, 2-hop propagation,
                                category PMI (DualSpectrumOntology), style-disentangled content.
2. SCORE each signal z_k        collaborative (cooc, ppmi, csvd, simp, diff2, pop, Δinc, rao),
                                taxonomic (cat, homo/softcat), content (style/image/audio, bridge).
3. FIT on validation            greedy-select fusion weights wₖ, the higher-order mix (α_inc, γ_rao),
   (maximise SEL_OBJ)           and the shrinkage κ_f — all to maximise a validation objective
   bundlefuse/selection.py      decoupled from the reported cutoff (default NDCG@20).
4. FUSE (4 stages)              Y_base=Σ wₖzₖ → Y=(1−α)Y_pair+αY_inc+γ z_rao → prior μ₀ → 
   operators.EmpiricalBayes     θ̂=(1−u_c)Y+u_c μ₀,  u_c=κ_f/(f_c+κ_f)   [cold-start shrinkage].
5. RANK                         mask the seeds, return argsort(θ̂)[:K].
```

Per-dataset differences, and nothing else:

| Family | Evaluator | Content features | Side labels |
|---|---|---|---|
| POG / Spotify | `bundlefuse.eval.fullrank` | external pretrained: POG image/desc = **BLIP**, title = **BGE**; Spotify audio = **CLAP** | category (POG) / artist+album (Spotify) |
| electronic / food / clothing | `bundlefuse.eval.bundlerec` | in-pipeline **TF-IDF + truncated SVD** over titles+category paths | category path (2 levels) |
| dunnhumby | `bundlefuse.eval.dunnhumby` | in-pipeline **TF-IDF + truncated SVD** over product text | sub-commodity |

---

## Reproduce the headline matrix (`tab:main`)

```bash
# all 8 datasets → results/matrix.csv (+ merges published CLHE/RaMen rows)
python -m bundlefuse.eval.run_matrix
# a subset
python -m bundlefuse.eval.run_matrix pog pog_dense

# the version-controlled canonical matrix with cross-seed error bars:
python scripts/regen_zeroparam.py && python scripts/add_retail_stds.py   # → results/matrix_zeroparam.{csv,json}
```

Per dataset (prints `popularity / co-occurrence / content-kNN / item2vec / … / Ours`):

| Dataset | Command | Ours R@20 |
|---|---|---|
| pog            | `BUNDLE_DS=pog MATRIX=1 python -m bundlefuse.eval.fullrank` | 0.0653 |
| pog_dense      | `BUNDLE_DS=pog_dense MATRIX=1 python -m bundlefuse.eval.fullrank` | 0.3737 |
| spotify        | `BUNDLE_DS=spotify MATRIX=1 python -m bundlefuse.eval.fullrank` | 0.1562 |
| spotify_sparse | `BUNDLE_DS=spotify_sparse MATRIX=1 python -m bundlefuse.eval.fullrank` | 0.1653 |
| electronic     | `BRDS=electronic MATRIX=1 python -m bundlefuse.eval.bundlerec` | 0.5192 |
| food           | `BRDS=food MATRIX=1 python -m bundlefuse.eval.bundlerec` | 0.5870 |
| clothing       | `BRDS=clothing MATRIX=1 python -m bundlefuse.eval.bundlerec` | 0.5904 |
| dunnhumby      | `MATRIX=1 python -m bundlefuse.eval.dunnhumby` | 0.1501 |

(The `Ours R@20` column is the committed `results/matrix_zeroparam.csv`; a fresh run must match it.)

---

## Other experiments (env-mode → table/figure)

Every mode is an environment flag on the same evaluator; each prints/writes and exits. Example:
`BASELINES=1 BRDS=food python -m bundlefuse.eval.bundlerec`.

| Mode (env) | What it measures | Output | Paper |
|---|---|---|---|
| *(default)* | the full-ranking matrix | stdout; `results/matrix*.csv` | `tab:main` |
| `BASELINES=1` | matched ItemKNN/P3α/RP3β/SLIM/iALS/BundleKNN/LinearCombiner | `results/baselines_<ds>.json` | `tab:neigh` |
| `EASE=1` | EASE closed-form baseline (bundlerec/dunnhumby, small n) | stdout row | `tab:suite` |
| `ABLATE=1` | stage ablation (base→+inc→+rao→+both→+shrink) + exposure buckets + variance components | stdout | `tab:ablation`, `tab:exposure_buckets` |
| `VALGAP=1` | validation→test generalization gap of weight selection | stdout | selection-overfitting probe |
| `WSWEEP=1` | per-signal fusion-weight sensitivity (fullrank only) | `results/wsweep_<ds>.json` | `fig_wsweep` |
| `SEL_OBJ=<obj> SELDUMP=1` | selection objective ∈ {ndcg\<K\>, recall\<K\>, mrr, ndcgfull, nll} | `results/selcmp_<ds>_<obj>.json` | §selK plateau |
| `ENCODER=1` | add the optional ~0.3M set-attention encoder as a fused signal | stdout | encoder on/off |
| `CASE_STUDY=1` | dump one real query end-to-end (fullrank only) | `results/case_study_<ds>.json` | `notes/methodology.tex` |
| `TIMING=1` | learned params + ms/bundle inference (fullrank only) | stdout | efficiency claim |

Diagnostics not wired into an evaluator:

```bash
python -m bundlefuse.diagnostics.sensitivity   # measured operator sweeps (β_clique, γ/τ, κ_f) → robustness §
python -m bundlefuse.diagnostics.ablation       # legacy OFAT driver (exposure/variance diagnostics)
```

One-off scripts (CLIP image harvest, EASE grid, k-fold probe, train-fraction curve) are indexed in
`scripts/README.md`.

---

## Tests

```bash
python3 tests/test_core.py   # 7 theoretical / numerical / batch-invariance / no-leakage tests
```

## Leak-freedom (why the numbers are honest)

All collaborative structures (`C`, PPMI, `csvd`, `Pcat`, `freq`) are built from the **train fold only**;
test queries are scored from seeds alone with held-out members masked to `-1e9`; the fusion weights,
higher-order mix, and shrinkage are frozen on validation before any test item is seen, so test scoring
is independent of test-batch composition (asserted by `test_batch_invariance_with_frozen_prior_scale`).
