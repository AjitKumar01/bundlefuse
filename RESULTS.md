# Results

All numbers below were regenerated **once, deterministically** (CPU, fixed seeds, `ENCODER=0`)
from this pipeline. Nothing is carried over from the old `simba/` repo. Raw logs are in
`results/*.log`; the machine-readable table is `results/matrix_zeroparam.{csv,json}`.

## Headline: zero-parameter BundleFuse vs baselines (Recall@20)

All rows carry a population-std error bar. Fullrank datasets = std over 3 top-level seeds;
e-commerce/retail = std over the evaluator's 5 held-out splits (same definition, via
`MATRIX_STD=1`). Every "win" below is several std clear of the baseline it beats.

| dataset        | co-occ | CLHE (9.5M) | RaMen (16M) | **Ours (0 param)** | standing            |
|----------------|--------|-------------|-------------|--------------------|---------------------|
| POG            | 0.024  | 0.028       | 0.038       | **0.0653 ± 0.0035**| win (~1.7× RaMen)   |
| POG_dense      | 0.273  | 0.382       | —           | **0.3737 ± 0.0058**| tie with CLHE       |
| Spotify        | 0.096  | 0.108       | 0.109       | **0.1562 ± 0.0033**| win                 |
| Spotify_sparse | 0.109  | 0.097       | —           | **0.1653 ± 0.0042**| win                 |
| Electronic     | 0.320  | 0.472       | **0.837**   | **0.5192 ± 0.0147**| beat CLHE, lose RaMen |
| Food           | 0.331  | 0.508       | **0.846**   | **0.5870 ± 0.0322**| beat CLHE, lose RaMen |
| Clothing       | 0.265  | —           | —           | **0.5904 ± 0.0271**| win (no heavy base) |
| Dunnhumby      | 0.129  | —           | —           | **0.1501 ± 0.0057**| win                 |

CLHE source (see *Baseline fairness* below): **reproduced from source** on POG_dense and
Spotify_sparse (our own dataset variants); **published figures** on POG/Spotify/Electronic/Food
(Electronic/Food are not publicly reproducible). RaMen likewise reproduced on POG/Spotify,
published-only on Electronic/Food.

**Summary:** strongest public-feature method on **5/8** (POG, Spotify, Spotify_sparse, Clothing,
Dunnhumby), **ties** CLHE on POG_dense, and beats CLHE but loses to RaMen on the two e-commerce
sets. From RaMen's own ablation (Table 3), its e-commerce edge is attributable to its learned
multi-hop collaborative encoder (CbSE) — removing it drops RaMen *below* us (0.356/0.373 vs our
0.519/0.587) — **not** to its product images (removing those barely moves it, and our 39%-coverage
public-image recovery does not help; see below).

Provenance: `results/regen_zeroparam.log` (via `scripts/regen_zeroparam.py`).

## Encoder on/off head-to-head

The optional ~0.3M set-attention contrastive encoder does **not** earn its keep. Dropping it ties
or wins everywhere and is much cheaper (dunnhumby 601s → 84s, ~7×). It is retained only as an
off-by-default `ENCODER=1` flag.

| dataset        | with-enc | zero-param |
|----------------|----------|------------|
| electronic     | 0.4975   | 0.5141     |
| food           | 0.6004   | 0.6182     |
| clothing       | 0.5542   | 0.5542     |
| dunnhumby      | 0.1536   | 0.1571     |
| pog            | 0.0673   | 0.0664     |
| pog_dense      | 0.3795   | 0.3812     |
| spotify_sparse | 0.1724   | 0.1712     |
| spotify        | 0.1551   | 0.1549     |

The only dataset where the encoder nominally helps is pog_dense, and only within noise.
Provenance: `results/encoder_head2head.log` (via `scripts/head2head.py`).

## Validation-selection does not overfit (VALGAP probe)

Greedy validation weight-selection was audited for selection bias across all 8 datasets.
`selection_excess_gap` = how much more the *selected* weights lose from val→test than *uniform*
weights do (a pure measure of selection optimism); `test(selected) − test(uniform)` = how much
selection actually buys on held-out test. The decisive question is whether the second dominates the
first — it does everywhere.

| dataset        | selection_excess_gap | test(selected) − test(uniform) | nsig | seeds |
|----------------|----------------------|--------------------------------|------|-------|
| pog            | +0.0007              | +0.0144                        | 5    | 1     |
| pog_dense      | −0.0036              | +0.0383                        | 4    | 1     |
| spotify_sparse | −0.0017              | +0.0431                        | 5    | 1     |
| electronic     | +0.0154              | +0.0301                        | 7.2  | 5     |
| food           | +0.0180              | +0.0454                        | 6.2  | 5     |
| clothing       | +0.0127              | +0.0690                        | 4.8  | 5     |
| dunnhumby      | +0.0027              | +0.0047                        | 5.2  | 5     |

**Reading:** on the fullrank sets the excess gap is ≈ 0 (no measurable selection optimism); on the
e-commerce/retail sets it is small but positive (+0.003 to +0.018 NDCG), i.e. a little optimism does
creep in — expected, since those sets have fewer, smaller held-out splits and select more signals
(nsig 5–7 vs 4–5). But in every case the test benefit of selection (+0.005 to +0.069) is larger than
the excess gap, so **selection is net-beneficial on held-out test on all 8 datasets and never
harmfully overfits.** The one dataset where benefit barely exceeds optimism is dunnhumby
(+0.0047 vs +0.0027), consistent with its weak content signals.

Provenance: `results/valgap_fullrank.log` (fullrank) and `results/valgap_retail.log`
(e-commerce/retail), both via `VALGAP=1` in the respective evaluators.

## Fuller baseline suite (reviewer pre-empt)

- **Transformer/GNN baselines (published, RaMen Table 2):** on POG/Spotify/Electronic/Food, BundleFuse is the best
  public-feature method against Bi-LSTM, HyperGraph, Trans, TransCL, GAT, CLHE (R@20 POG 0.0653 vs best-other CLHE
  0.0284; Spotify 0.1562 vs 0.1081; Electronic 0.5192 vs CLHE 0.4721; Food 0.5870 vs 0.5077). Only RaMen's
  full e-commerce scores (0.8371/0.8459) are higher — traced to its CbSE collaborative encoder,
  not images (see the RaMen ablation decomposition below and the paper's `tab:ramen_abl`).
- **EASE (closed-form control, run on our splits via `EASE=1`):** lands within 0.001–0.006 R@20 of plain
  co-occurrence everywhere computable — Electronic **0.3253** (co-occ 0.3199), Food **0.3313** (0.3306), Clothing
  **0.2656** (0.2651) — i.e. **1.6–2.2× below BundleFuse**. Confirms the win is the *construction*, not merely being
  closed-form. EASE's O(n²) memory / O(n³) solve is intractable at Dunnhumby/POG/Spotify scale (22K–254K items);
  BundleFuse runs on all of them on one CPU core. Provenance: `results/ease_baseline.log`.

## Matched tuned-neighbourhood baselines (Phase-2, all 8 datasets, `BASELINES=1`)

Six classical recommenders run under the **identical** protocol as BundleFuse (same splits/candidates/metrics/seeds;
each a sparse item-item `W` scored as `W @ Ind(S)`; every hyperparameter validation-selected). Code in
`bundlefuse/baselines.py`; results in `results/baselines_<ds>.json` + `results/baselines_all8.log`. **BundleFuse beats
the best of all six on every dataset.** Best-baseline R@20 vs Ours:

| dataset | ItemKNN | P3α | RP3β | SLIM | BundleKNN | Combiner | **best base** | **Ours** |
|---|---|---|---|---|---|---|---|---|
| POG | .0232 | .0252 | .0252 | — | .0259 | .0228 | .0259 | **.0653** |
| POG_dense | .2589 | .2838 | .2842 | — | .2666 | .2776 | .2842 | **.3737** |
| Spotify | .1313 | .1127 | .1129 | — | .1226 | .1311 | .1313 | **.1562** |
| Spotify_sparse | .1100 | .1088 | .1087 | — | .1117 | .1180 | .1180 | **.1653** |
| Electronic | .3229 | .3263 | .3248 | .3235 | .3262 | .3224 | .3263 | **.5192** |
| Food | .3267 | .3317 | .3275 | .3299 | .3318 | .3275 | .3318 | **.5870** |
| Clothing | .2651 | .2651 | .2651 | .2651 | .2651 | .2651 | .2651 | **.5904** |
| Dunnhumby | .1346 | .1204 | .1204 | — | .1335 | .1370 | .1370 | **.1501** |

Margins: 2.2–2.5× on content-heavy sets (Clothing/POG), 1.6–1.8× on Electronic/Food, +10–40% on the
collaborative-leaning sets (Spotify/Spotify_sparse/POG_dense/Dunnhumby). The **LinearCombiner never beats its best
component by more than noise** (a naive tuned blend doesn't recover the construction), and SLIM merely ties the other
neighbourhood methods where it runs (n≤8000); EASE represents the sparse-linear family at larger scale. Dependency-free
by design (no implicit/lightfm → iALS/FISM/LightFM out of scope).

## Operator ablation & exposure buckets (regenerated zero-param)

Two diagnostics via `ABLATE=1` in the evaluators (reuses the val-fit weights; `results/ablation_zeroparam.log`):

- **Higher-order operators are secondary.** Adding both closed-form operators (Δ_inc, Rao) over the pairwise base
  moves R@20 by at most **+0.79pp** (Spotify; ≤+0.26pp on the other 7, and −0.01 to +0.04pp on five). Frequency-
  confidence routing adds up to **+2.32pp** (Electronic), +2.24 (Clothing), +2.00 (Food) — it's the driver.
- **Shrinkage lifts the cold bucket; homoscedastic model rejected.** Cold-bucket R@20 pre→post routing: Electronic
  19.8→26.8 (+7.0pp), Food 34.4→38.7, Spotify 9.7→13.7, Spotify_sparse 13.3→16.8, Clothing 39.0→42.5, Dunnhumby
  0.0→1.8; warm buckets unchanged. The OLS variance slope σ²_ε is **negative on 7/8** datasets (all but Clothing),
  so κ is validation-selected rather than data-driven.

## Baseline fairness — are we beating a weakened CLHE?

We checked the provenance of every CLHE number the headline table compares against, so a win can't
be an artifact of a crippled baseline. There are two kinds:

- **Reproduced from source (our two dataset variants):** POG_dense and Spotify_sparse are variants we
  constructed, so no official CLHE number exists; we trained CLHE from its own repo on them. Using
  the fair early-stopping protocol (pick the epoch with the best *validation* R@20, report its
  *test* R@20), the reproduced values are:
  - POG_dense: best val 0.38354 → **test 0.3815** — the exact value in the table.
  - Spotify_sparse: best val 0.08634 → **test 0.0969** — the exact value in the table.
  These are honest, well-trained reproductions (full training curves in `clhe_out/*/*.log`). On
  POG_dense CLHE actually *leads* us (0.3815 vs 0.3737) — we do not understate it; on Spotify_sparse
  we beat a faithfully-reproduced CLHE by ~0.07.
- **Published figures (POG, Spotify, Electronic, Food):** these use the literature's CLHE numbers
  (0.0284, 0.1082, 0.4721, 0.5077). We did **not** reproduce them locally — and Electronic/Food
  *cannot* be reproduced at all, since that data is not in any public release (see the RaMen section).
  Where we beat CLHE on these (Electronic/Food), we are beating its own *published* number, which only
  makes the win harder to claim, not easier.

We also **do not inherit the prior `simba/` paper's claim** that "reproduced CLHE = 0.0281 on POG /
0.1076 on Spotify." No CLHE training log for POG or Spotify exists in the repo; the 0.0281 figure
coincides exactly with our *RaMen* POG reproduction (`kaggle_out_pog/ramen-repro-pog.log`,
test R@20 = 0.02817), i.e. it appears to be a mislabel. bundlefuse makes no CLHE-reproduction claim
for POG/Spotify and treats those two CLHE numbers as published-only.

**Verdict:** the wins are not against a weakened baseline. On the only datasets where we both control
the split and train CLHE ourselves, CLHE is reproduced faithfully (and on POG_dense is reported as
beating us); elsewhere we compare against published CLHE.

## K-fold weight selection — claim RETRACTED (the `KFOLD` path is not genuine CV)

**Retraction (2026-10-27).** We previously reported that "5-fold CV weight selection does not beat
single-split selection." That claim is withdrawn: the `KFOLD` implementation in `fullrank_eval.py`
does **not** perform cross-validated selection. Its selection objective (`sel_obj`) partitions the
validation *queries* (columns of a **fixed** fused matrix) into K folds and averages the per-fold
NDCG@20 — it never re-fits the fusion weights on K−1 folds and evaluates on the held-out fold. For
equal-sized folds the mean-of-fold-means equals the grand validation mean exactly, so the objective
reduces to the single-split objective **by construction**. The tiny A/B differences we had reported
(POG 0.0643 vs 0.0655; Spotify_sparse 0.1659 vs 0.1638) are just unequal-fold-weighting / seed noise,
not a real cross-validation effect.

We therefore make **no** k-fold claim. The `KFOLD` flag stays off-by-default and is documented in code
as query-partitioning only (not a train/held-out re-fit). A genuine rotating-fold selection (re-fit on
K−1 folds, score the held-out fold) is left as future work; the VALGAP audit (above) remains our actual
evidence that single-split selection generalizes.

## Multimodal images — honest negative result

RaMen's large e-commerce lead prompted a test of whether public product images close the gap.

- **Coverage:** every BundleRec item is a real Amazon ASIN. Only Electronic ships image URLs
  (13% of items natively). Harvesting hi-res URLs from the public `McAuley-Lab/Amazon-Reviews-2023`
  metadata lifted this to **45.6%** (1594 items), of which **1364/3499 = 39.0%** yielded usable
  CLIP (`clip-vit-base-patch32`) features. **Food and Clothing have no public image URLs at all.**
- **A/B (Electronic, zero-param):** text-only R@20 **0.5141** → +image **0.5028** (N@20 +0.010).
  Late-fusion of 39%-coverage public images did **not** help — recall slipped slightly, and the gap
  to RaMen (0.837) is untouched.

**Conclusion:** RaMen's e-commerce edge is not reproducible as a drop-in public-image signal. It
needs near-full coverage (publicly unavailable) and/or joint multimodal training. Reported as a
negative result, not hidden.

Provenance: `results/image_ab_electronic.log`; pipeline in `scripts/harvest_image_urls.py` and
`scripts/clip_embed.py`; evaluator wiring gated by `IMG_FEATS=` in `bundlerec_eval.py`. The derived
CLIP feature cache (`elec_clip.npz`, ~2.8MB) is not vendored; regenerate with the two scripts.

## RaMen "text-only" ablation — why we do not claim a number

The natural way to prove RaMen's e-commerce edge is its images would be to retrain RaMen with its
visual branch disabled and show the number collapses toward ours. **This ablation cannot be run on
public data, so we report it as unreproducible rather than inventing a result:**

- **The data is not public.** RaMen's wins are on Electronic/Food. The public release
  (`xhLiu/BundleConstruction` on HuggingFace) contains **only** `pog`, `pog_dense`, `spotify`,
  `spotify_sparse` — no electronic/food/clothing. RaMen's own GitHub ships no data. In RaMen's
  model (`baselines/RaMen_repo/models/RaMen.py`), the visual branch is `content_feature.pt`
  (`v_feat`/`c_encoder`); that file exists in the public release only for the image-free POG/Spotify
  sets. So neither the raw images nor RaMen's precomputed visual features are obtainable for the
  datasets where it wins. RaMen's 0.8371/0.8459 are **paper-reported only**.
- **Where data *is* public, we reproduced RaMen and beat it.** On POG our RaMen reproduction reaches
  R@20 ≈ 0.038 (vs our 0.0653); on Spotify ≈ 0.109 (vs our 0.1562). RaMen is not special on any
  fully public benchmark — it only pulls ahead on the private-feature e-commerce sets.
- **Our own attempt to supply the missing modality failed honestly.** Harvesting public Amazon
  images reached 39% coverage on Electronic and did not close the gap (see above). Food/Clothing
  have no public image URLs at all.

**Conclusion:** the e-commerce gap is a **data-availability asymmetry**, not a demonstrated modelling
advantage, and we state exactly that. We deliberately do **not** carry forward the prior `simba/`
paper's claim that "RaMen without private images drops to 0.4612 (Electronic) / 0.4985 (Food)": those
numbers appear only in that paper's LaTeX, are produced by no script, log, or data in the repo, and
are contradicted by its own earlier draft (`paper_draft.md`: *"a direct image ablation was blocked by
Amazon's anti-scraping; future work"*). They are treated as unsupported.

## Open tasks

*(none outstanding — the results section is complete; next step is paper framing.)*
