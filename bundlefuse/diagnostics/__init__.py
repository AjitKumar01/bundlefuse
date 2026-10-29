"""Diagnostic / sensitivity drivers (not part of the headline pipeline).

- :mod:`bundlefuse.diagnostics.sensitivity` -- measured operator hyperparameter sweeps
  (beta_clique, Rao's gamma/tau, shrinkage kappa_f). Cited by the paper's robustness section.
- :mod:`bundlefuse.diagnostics.ablation`    -- the older one-factor-at-a-time ablation driver
  inherited from the SIMBA fork. Superseded for the headline stage-ablation by the ``ABLATE`` mode
  built into the evaluators; retained for its exposure-bucket and variance-component diagnostics.
"""
