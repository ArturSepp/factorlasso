"""Diagnostics of fitted models and their covariance history.

Strict-factor-structure tests on residual panels, effective sparsity, the partition variance
share, and offline persistent labelling (lineage) of the risk clusters recorded in a rolling
covariance history. The names are also exported from :mod:`factorlasso`.
"""

from factorlasso.diagnostics._lineage import (
    RiskClusterReport, TaxonomyConfig, analyze_cluster_lineage, run_cluster_lineage_report,
)
from factorlasso.diagnostics._residuals import (
    ResidualDiagnostics, Sparsity, diagnose_residuals, effective_sparsity,
    marchenko_pastur_edge, missing_factor_components, null_threshold,
    partition_variance_share, raw_offdiagonal_mass, residual_correlation, suggest_tolerance,
)

__all__ = [
    "ResidualDiagnostics",
    "RiskClusterReport",
    "Sparsity",
    "TaxonomyConfig",
    "analyze_cluster_lineage",
    "diagnose_residuals",
    "effective_sparsity",
    "marchenko_pastur_edge",
    "missing_factor_components",
    "null_threshold",
    "partition_variance_share",
    "raw_offdiagonal_mass",
    "residual_correlation",
    "run_cluster_lineage_report",
    "suggest_tolerance",
]
