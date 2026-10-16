# Results

All numbers below were regenerated **once, deterministically** (CPU, fixed seeds, `ENCODER=0`)
from this pipeline. Nothing is carried over from the old `simba/` repo. Raw logs are in
`results/*.log`; the machine-readable table is `results/matrix_zeroparam.{csv,json}`.

## Headline: zero-parameter BundleFuse vs baselines (Recall@20)

Fullrank datasets = mean ± std over 3 seeds. E-commerce/retail = mean over the evaluator's
internal 5 seeds; cross-run std was not measured (`std=0.0000` means "not measured", **not**
"zero variance").

| dataset        | co-occ | CLHE (9.5M) | RaMen (16M) | **Ours (0 param)** | standing            |
|----------------|--------|-------------|-------------|--------------------|---------------------|
| POG            | 0.024  | 0.028       | 0.038       | **0.0653 ± 0.0035**| win (~1.7× RaMen)   |
| POG_dense      | 0.273  | 0.382       | —           | **0.3737 ± 0.0058**| tie with CLHE       |
| Spotify        | 0.096  | 0.108       | 0.109       | **0.1562 ± 0.0033**| win                 |
| Spotify_sparse | 0.109  | 0.097       | —           | **0.1653 ± 0.0042**| win                 |
| Electronic     | 0.320  | 0.472       | **0.837**   | 0.5192             | beat CLHE, lose RaMen |
| Food           | 0.331  | 0.508       | **0.846**   | 0.5870             | beat CLHE, lose RaMen |
| Clothing       | 0.265  | —           | —           | **0.5904**         | win (no heavy base) |
| Dunnhumby      | 0.129  | —           | —           | **0.1486**         | win                 |

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

## Open tasks

- Multi-seed std for the e-commerce/retail rows (currently single-aggregate).
- RaMen text-only ablation to further substantiate the "edge is images" claim.
- k-fold CV weight selection for the final tables.
- Baseline-fairness check (reproduced vs published CLHE).
- Extend the VALGAP probe to the bundlerec/dunnhumby evaluators.
