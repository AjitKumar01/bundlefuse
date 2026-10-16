"""BundleFuse: a cheap, interpretable, regime-adaptive baseline for bundle completion.

A closed-form evidence-fusion scorer (collaborative co-occurrence + taxonomic ontology
+ style-disentangled/graph-bridged content) with cold-start shrinkage toward a metadata
prior. Validation-selected fusion weights; optional (off-by-default) set-attention encoder.

See MANIFEST.md for exactly what was salvaged from the prior "SIMBA" codebase and what was
dropped, and why.
"""
from ._device import DEVICE, set_determinism
from .operators import (
    SimplicialGraphOperator,
    DualSpectrumOntology,
    BilateralContentBridge,
    EmpiricalBayesRouter,
    zcols,
)
