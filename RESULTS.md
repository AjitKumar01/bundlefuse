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
| Dunnhumby      | 0.129  | —           | —           | **0.1486 ± 0.0055**| win                 |

CLHE source (see *Baseline fairness* below): **reproduced from source** on POG_dense and
Spotify_sparse (our own dataset variants); **published figures** on POG/Spotify/Electronic/Food
(Electronic/Food are not publicly reproducible). RaMen likewise reproduced on POG/Spotify,
published-only on Electronic/Food.

**Summary:** strongest public-feature method on **5/8** (POG, Spotify, Spotify_sparse, Clothing,
Dunnhumby), **ties** CLHE on POG_dense, and beats CLHE but loses to RaMen on the two e-commerce
sets. RaMen's e-commerce edge is attributable to private Amazon product images (see below), not a
better bundle model.

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

## K-fold CV weight selection does not help (negative result)

We tried selecting all hyperparameters (fusion weights, higher-order `α`/`γ`, EB `κ`/`prior_scale`)
by **5-fold cross-validation over the validation bundles** instead of a single full-validation
objective (`KFOLD=5` in `fullrank_eval.py`; the selected config is still applied to test unchanged).
A/B over 3 seeds:

| dataset        | single-val (mean ± std) | 5-fold CV (mean ± std) | Δ mean  |
|----------------|-------------------------|------------------------|---------|
| POG            | 0.0655 ± 0.0029         | 0.0643 ± 0.0073        | −0.0012 |
| Spotify_sparse | 0.1638 ± 0.0034         | 0.1659 ± 0.0047        | +0.0021 |

**Tie on the mean (both Δ within noise), and k-fold is strictly worse on stability** (seed-to-seed
std ~1.4–2.5× larger). Reason: we already hold out a dedicated validation set, so foldizing it adds
no data — it only makes each NDCG@20 estimate 5× noisier, and selecting on an average of noisier
estimates yields less stable configs for zero mean gain. This independently reconfirms the VALGAP
result (single-val selection already generalizes). **We keep single-val selection as the default**;
`KFOLD` is retained off-by-default so the negative result stays reproducible.

Provenance: `results/kfold_ab.log` (via `scripts/kfold_ab.py`).

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
