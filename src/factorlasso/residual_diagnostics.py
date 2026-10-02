"""Historical location of the residual diagnostics.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.diagnostics`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.diagnostics._residuals import (
    ArrayLike, diagnose_residuals, effective_sparsity, marchenko_pastur_edge,
    missing_factor_components, null_threshold, _PARTITION_SHARE_COLUMNS, _partition_share_row,
    partition_variance_share, raw_offdiagonal_mass, residual_correlation, ResidualDiagnostics,
    Sparsity, suggest_tolerance,
)

__all__ = [
    "ResidualDiagnostics",
    "diagnose_residuals",
    "marchenko_pastur_edge",
    "missing_factor_components",
    "null_threshold",
    "raw_offdiagonal_mass",
    "residual_correlation",
    "effective_sparsity",
    "Sparsity",
    "suggest_tolerance",
    "partition_variance_share",
]

guard_legacy_module(__name__)
