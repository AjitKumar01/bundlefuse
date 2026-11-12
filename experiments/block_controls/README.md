# Controls for the between-category block

Tests the claim that bundle history replaces the between-category part of description similarity
(`cat`), while similarity is kept within a category (`style`). Run on POG and POG_DENSE. Spotify is
not tested, because its selected recipes do not use `cat`.

Status: exploratory. It uses the filtered population, fitting/tuning partition and retained models
of `results/sufficiency`, whose test split had already been inspected. Decision rules and arms were
written to `protocol.json` before the run, and every arm was frozen before its test columns were
built. Both recorded references reproduce on tuning and test (tolerance 1e-6).

Null: Curveball trades on training bundles keep item frequencies and bundle sizes. Only `cat` is
rebuilt from them; history stays real. There are 5 seeds.

## Primary results, NDCG@20 (Holm over 8 tests)

| Contrast | POG | POG_DENSE |
|---|---|---|
| `cat_style` | .0400 | .2549 |
| raw `content`, refit | .0173 | .2470 |
| shuffled `cat`, mean of 5 | .0325 | .2435 |
| vs raw `content`, 95% CI | +.0226 (.0196, .0258) | +.0079 (.0062, .0095) |
| vs shuffled mean, 95% CI | +.0075 (.0061, .0089) | +.0114 (.0096, .0131) |
| Holm p, all 8 tests | .0008 | .0008 |
| Smallest gain over one seed | +.0067 | +.0079 |

Both pre-set rules pass on both datasets.

## Limits

- POG_DENSE never selects raw or residual description. Its `content`, `title`, `samecat_style` and
  `style_only` arms all reduce to history + popularity (.2470), and `cat_style` selects `cat` + `pop`
  + `hist`. The contrast with raw `content` is therefore `cat` against no description at all. The
  within-category half of the claim has support on POG only.
- The POG_DENSE shuffled gap includes harm from selecting a noisy column: the shuffled mean (.2435)
  is below history + popularity (.2470). The gain over that base is +.0079.
- The slot penalty stays active under the null (POG 67 categories real vs 66 null; POG_DENSE 64 vs
  62). It works as a near-constant "do not repeat a category" rule, not as exclusivity learned from
  history. Slot alone adds about +.0023 to +.0026, and lift alone adds +.0084 (POG) and +.0075
  (POG_DENSE).
- `samecat` has zero weight on both datasets, but the weight grid is non-negative, so this arm cannot
  express a negative same-category effect. `cat_slot` covers that direction.
- Cold-target Recall@20 on POG_DENSE is 0 in every arm, including the reference. On POG, `cat_style`
  reaches .0523, raw `content` .0041, and `style_only` .0405.
- This supports the mechanism's value, not its priority. Category-level association is long
  established.

## Post hoc, added after reading the first report (POG, NDCG@20, unadjusted)

- `style_only` vs raw title with the same input: +.0101 (.0081, .0123).
- `cat_style` vs `content` + `cat`: +.0121 (.0097, .0145).

## Run

```sh
python -m experiments.block_controls.run --dataset pog
python -m experiments.block_controls.run --dataset pog_dense
python -m experiments.block_controls.run --report
```

Requires `checkpoints/publication_mechanisms/<dataset>/artifacts.pt`. Results are in
`results/block_controls/`. Score columns (about 38 GB) were deleted after the run and are rebuilt
by rerunning; delete `checkpoints/block_controls/<dataset>` first, because its progress marker
would otherwise skip them. Per-query files stay local.
