"""Per-dataset evaluators (run as scripts / ``python -m``).

One module per dataset family, each sharing the same closed-form algorithm from
:mod:`bundlefuse.operators` and the same selection objective from :mod:`bundlefuse.selection`:

- :mod:`bundlefuse.eval.fullrank`   -- POG / POG_dense / Spotify / Spotify_sparse (torch full-ranking)
- :mod:`bundlefuse.eval.bundlerec`  -- electronic / food / clothing (TF-IDF content)
- :mod:`bundlefuse.eval.dunnhumby`  -- Dunnhumby transaction baskets
- :mod:`bundlefuse.eval.run_matrix` -- orchestrates all eight into one table

See ``docs/EXPERIMENTS.md`` for the exact command + env flags behind every reported number.
"""
