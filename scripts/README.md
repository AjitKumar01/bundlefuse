# `scripts/` — one-off drivers and table builders

These wrap the evaluators in `bundlefuse.eval` for specific tables/figures or data prep. They are not
part of the core library; run them from the repo root (`python scripts/<name>.py`).

| Script | Purpose | Feeds |
|---|---|---|
| `regen_zeroparam.py` | Re-run all 8 evaluators and write the canonical zero-parameter matrix. | `results/matrix_zeroparam.{csv,json}` → `tab:main` |
| `add_retail_stds.py` | Add cross-seed error bars to the e-commerce/retail rows of that matrix. | `results/matrix_zeroparam.*` std column |
| `head2head.py` | Encoder on/off head-to-head on identical seeds. | `results/encoder_head2head.log` |
| `kfold_ab.py` | K-fold vs single-split weight selection probe (claim later **retracted** — see RESULTS.md). | `results/kfold_ab.log` |
| `trainfrac_study.py` | Train-fraction data-efficiency curve on POG. | `results/data_efficiency_pog.log` |
| `harvest_image_urls.py` | Harvest product image URLs for the multimodal-image experiment. | CLIP pipeline input |
| `clip_embed.py` | Embed harvested images with CLIP into an `.npz` cache. | `IMG_FEATS=` input to `eval.bundlerec` |

See `docs/EXPERIMENTS.md` for how these relate to the env-gated modes built into the evaluators.
