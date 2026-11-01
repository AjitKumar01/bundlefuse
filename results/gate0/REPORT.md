# Gate 0 - candidate-coverage probe (v3: category-PMI slate + content leg, evaluator tie policy)

Recall ceiling = fraction of held-out items kept within a size-M shortlist (rank < M under
score desc / item-id asc). `union_all` = in top-M of any leg (budget M each; shortlist <= 4M).
`catpmi` is cross-category complementarity (BundleFuse's `cat`); `content` is a content-embedding
ANN. The discrete hard ceiling counts held items with positive collaborative OR category-PMI
evidence (content excluded, since cosine reaches everything at large M). BundleFuse R@20 is the bar.

## pog_dense - original

- catalogue 31217 items, 5938 test queries, 9185 held-out items
- held reachable by collaboration (positive C/W_pair): **53.29%**
- held reachable by category-PMI complementarity: **99.9%**
- **discrete hard ceiling** (collab OR category-PMI): **99.92%** (content ANN can retrieve beyond this at finite M)
- **BundleFuse actual R@20 = 0.395223** (bar to clear)

| retriever | M=20 | M=50 | M=100 | M=200 | M=500 | M=1000 | M=2000 | M=5000 |
|---|---|---|---|---|---|---|---|---|
| cooc       | 0.285 | 0.325 | 0.358 | 0.396 | 0.444 | 0.481 | 0.519 | 0.577 |
| wpair      | 0.310 | 0.349 | 0.377 | 0.409 | 0.449 | 0.483 | 0.519 | 0.577 |
| catpmi     | 0.003 | 0.007 | 0.013 | 0.023 | 0.061 | 0.131 | 0.228 | 0.437 |
| content    | 0.000 | 0.001 | 0.002 | 0.004 | 0.010 | 0.018 | 0.036 | 0.102 |
| union_all  | 0.316 | 0.356 | 0.390 | 0.429 | 0.492 | 0.558 | 0.641 | 0.781 |

## pog_dense - single_seed

- catalogue 31217 items, 5938 test queries, 9185 held-out items
- held reachable by collaboration (positive C/W_pair): **29.9%**
- held reachable by category-PMI complementarity: **98.24%**
- **discrete hard ceiling** (collab OR category-PMI): **98.4%** (content ANN can retrieve beyond this at finite M)
- **BundleFuse actual R@20 = 0.395223** (bar to clear)

| retriever | M=20 | M=50 | M=100 | M=200 | M=500 | M=1000 | M=2000 | M=5000 |
|---|---|---|---|---|---|---|---|---|
| cooc       | 0.212 | 0.233 | 0.246 | 0.265 | 0.283 | 0.302 | 0.328 | 0.394 |
| wpair      | 0.212 | 0.234 | 0.247 | 0.266 | 0.283 | 0.302 | 0.327 | 0.394 |
| catpmi     | 0.002 | 0.004 | 0.008 | 0.015 | 0.035 | 0.081 | 0.161 | 0.314 |
| content    | 0.000 | 0.000 | 0.001 | 0.003 | 0.008 | 0.016 | 0.034 | 0.107 |
| union_all  | 0.217 | 0.239 | 0.256 | 0.280 | 0.314 | 0.365 | 0.443 | 0.601 |

## spotify - original

- catalogue 254155 items, 4000 test queries, 126753 held-out items
- held reachable by collaboration (positive C/W_pair): **80.4%**
- held reachable by category-PMI complementarity: **95.05%**
- **discrete hard ceiling** (collab OR category-PMI): **95.26%** (content ANN can retrieve beyond this at finite M)
- **BundleFuse actual R@20 = 0.155646** (bar to clear)

| retriever | M=20 | M=50 | M=100 | M=200 | M=500 | M=1000 | M=2000 | M=5000 |
|---|---|---|---|---|---|---|---|---|
| cooc       | 0.079 | 0.140 | 0.205 | 0.284 | 0.407 | 0.503 | 0.593 | 0.695 |
| wpair      | 0.090 | 0.158 | 0.228 | 0.312 | 0.439 | 0.535 | 0.622 | 0.715 |
| catpmi     | 0.007 | 0.015 | 0.026 | 0.047 | 0.100 | 0.169 | 0.277 | 0.477 |
| content    | 0.005 | 0.011 | 0.019 | 0.033 | 0.063 | 0.102 | 0.163 | 0.282 |
| union_all  | 0.103 | 0.180 | 0.261 | 0.358 | 0.508 | 0.622 | 0.728 | 0.841 |

## spotify - single_seed

- catalogue 254155 items, 4000 test queries, 126753 held-out items
- held reachable by collaboration (positive C/W_pair): **38.02%**
- held reachable by category-PMI complementarity: **73.73%**
- **discrete hard ceiling** (collab OR category-PMI): **76.52%** (content ANN can retrieve beyond this at finite M)
- **BundleFuse actual R@20 = 0.155646** (bar to clear)

| retriever | M=20 | M=50 | M=100 | M=200 | M=500 | M=1000 | M=2000 | M=5000 |
|---|---|---|---|---|---|---|---|---|
| cooc       | 0.043 | 0.079 | 0.118 | 0.168 | 0.240 | 0.293 | 0.338 | 0.378 |
| wpair      | 0.043 | 0.079 | 0.118 | 0.169 | 0.241 | 0.293 | 0.339 | 0.378 |
| catpmi     | 0.001 | 0.003 | 0.006 | 0.013 | 0.033 | 0.065 | 0.122 | 0.247 |
| content    | 0.003 | 0.006 | 0.009 | 0.016 | 0.033 | 0.055 | 0.090 | 0.166 |
| union_all  | 0.051 | 0.092 | 0.138 | 0.197 | 0.290 | 0.367 | 0.447 | 0.553 |
