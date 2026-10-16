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

Greedy validation weight-selection was audited for selection bias. `selection_excess_gap`
(how much more the *selected* weights lose from val→test than *uniform* weights do) is ≈ 0, while
the selected weights beat uniform on the test set by +1.4 to +4.3pp. So selection generalizes.

| dataset        | selection_excess_gap | test(selected) − test(uniform) | nsig |
|----------------|----------------------|--------------------------------|------|
| pog            | +0.0007              | +0.0144                        | 5    |
| pog_dense      | −0.0036              | +0.0383                        | 4    |
| spotify_sparse | −0.0017              | +0.0431                        | 5    |

Provenance: `results/valgap_fullrank.log` (`VALGAP=1` in `fullrank_eval.py`).

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

- Baseline-fairness check (reproduced vs published CLHE).
- Extend the VALGAP probe to the bundlerec/dunnhumby evaluators.
