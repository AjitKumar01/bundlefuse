# Frozen neural representation diagnostic

R2 is held-out linear decodability from frozen hidden features, not proof that a source is absent or unlearnable. Correlation is not a mechanism test.

| Source | Relations probe R2 | Contextual probe R2 |
|---|---:|---:|
| pair | 0.2812 | 0.2088 |
| simp | 0.2586 | 0.1912 |
| cooc | 0.3017 | 0.2122 |
| ppmi | 0.3191 | 0.2169 |
| cat | 0.0290 | 0.0398 |
| homo | 0.1665 | 0.1252 |
| content | 0.1271 | 0.1225 |
| bridge | 0.0649 | 0.0532 |
| csvd | 0.2053 | 0.1264 |
| rao | 0.0536 | 0.0321 |
| pop | 0.9995 | 0.9995 |
| joint_surplus | 0.0787 | 0.0146 |

Only separate Ridge probes fit source targets; neural weights remain frozen. No probe enters inference.

All source operators are constructed from training bundles. Probe rows contain a fixed mix of source top candidates and uniform candidates. Split is by query, not candidate row.
