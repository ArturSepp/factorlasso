"""
factorlasso — Sparse factor model estimation with constrained LASSO
===================================================================

Estimate sparse multi-output regression coefficients with sign
constraints, prior-centered regularisation, and hierarchical group
structure, then assemble consistent factor covariance matrices. The
group penalty is offered in two modes: a row-grouped penalty
(``HIERARCHICAL_CLUSTER_GROUP_LASSO``, HCGL) that groups each asset's factor
loadings, and a cluster-by-factor block penalty
(``FACTOR_CLUSTER_GROUP_LASSO``, FCGL) that groups the loadings of a
cluster's assets on each factor.

Quick start
-----------
>>> from factorlasso import LassoModel, LassoModelType
>>> model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-4)
>>> model.fit(x=X, y=Y)

Cross-validated regularisation
------------------------------
>>> from factorlasso import LassoModelCV
>>> cv = LassoModelCV(n_splits=5).fit(x=X, y=Y)
>>> cv.best_lambda_
1e-4

Residual validation
-------------------
A sparse factor model asserts that the residual covariance is diagonal. Nothing
in the estimation enforces the assertion, so test it.

>>> from factorlasso import LassoModelDiagonalityCV, diagnose_residuals
>>> sel = LassoModelDiagonalityCV(n_splits=5).fit(x=X, y=Y)
>>> sel.passed_           # do held-out residuals look diagonal?
False
>>> sel.missing_factors_  # the components the factor set does not carry

Full pipeline
-------------
>>> from factorlasso import LassoModel, CurrentFactorCovarData, VarianceColumns

Citation
--------
If you use factorlasso in academic work, please cite the software paper
and the methodology paper; see ``CITATION.cff`` or the README for the
BibTeX entries.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from factorlasso.cluster_lineage import (
    RiskClusterReport,
    TaxonomyConfig,
    analyze_cluster_lineage,
    run_cluster_lineage_report,
)
from factorlasso.cluster_smoothing import (
    ClusterSmootherType,
    RollingClusterData,
    apply_partition_distance_bonus,
    compute_rolling_smoothed_clusters,
    smooth_similarity_ewma,
)
from factorlasso.cluster_utils import (
    ClusterCorrelationTransform,
    ClusterCorrelationTransformResult,
    DistanceTransform,
    apply_cluster_correlation_transform,
    compute_clusters_from_corr_matrix,
    get_clusters_by_freq,
    get_cutoffs_by_freq,
    get_linkage_array,
    get_linkages_by_freq,
    remove_first_principal_component,
)
from factorlasso.cv import LassoModelCV
from factorlasso.dependence_utils import (
    DependenceMeasure,
    compute_dependence_matrix,
    compute_gerber_matrix,
)
from factorlasso.diagonality import LassoModelDiagonalityCV
from factorlasso.ewm_utils import (
    compute_ewm,
    compute_ewm_covar,
    compute_expanding_power,
    set_group_loadings,
)
from factorlasso.factor_covar import (
    CurrentFactorCovarData,
    RollingFactorCovarData,
    VarianceColumns,
)
from factorlasso.lasso_estimator import (
    LassoEstimationResult,
    LassoModel,
    LassoModelType,
    get_x_y_np,
    solve_cooperative_group_lasso_cvx_problem,
    solve_group_lasso_cvx_problem,
    solve_group_lasso_path,
    solve_lasso_cvx_problem,
    solve_unilasso_cvx_problem,
)
from factorlasso.residual_diagnostics import (
    ResidualDiagnostics,
    Sparsity,
    diagnose_residuals,
    effective_sparsity,
    marchenko_pastur_edge,
    missing_factor_components,
    null_threshold,
    raw_offdiagonal_mass,
    residual_correlation,
    suggest_tolerance,
)
from factorlasso.sign_constraints import (
    derive_sign_constraints,
    validate_cluster_signs,
)

try:
    __version__ = _pkg_version("factorlasso")
except PackageNotFoundError:  # pragma: no cover - editable install before metadata exists
    __version__ = "0.0.0+unknown"

__all__ = [
    # Core estimator
    "LassoModel",
    "LassoModelCV",
    "LassoModelType",
    "LassoEstimationResult",
    "ClusterSmootherType",
    "RollingClusterData",
    "solve_lasso_cvx_problem",
    "solve_group_lasso_cvx_problem",
    "solve_group_lasso_path",
    "solve_cooperative_group_lasso_cvx_problem",
    "solve_unilasso_cvx_problem",
    "get_x_y_np",
    # Factor covariance assembly
    "CurrentFactorCovarData",
    "RollingFactorCovarData",
    "VarianceColumns",
    # Offline cluster lineage
    "RiskClusterReport",
    "TaxonomyConfig",
    "analyze_cluster_lineage",
    "run_cluster_lineage_report",
    # Dependence measures
    "DependenceMeasure",
    "compute_dependence_matrix",
    "compute_gerber_matrix",
    # Clustering utilities
    "ClusterCorrelationTransform",
    "ClusterCorrelationTransformResult",
    "DistanceTransform",
    "apply_cluster_correlation_transform",
    "compute_clusters_from_corr_matrix",
    "get_clusters_by_freq",
    "get_cutoffs_by_freq",
    "get_linkage_array",
    "get_linkages_by_freq",
    "remove_first_principal_component",
    "apply_partition_distance_bonus",
    "compute_rolling_smoothed_clusters",
    "smooth_similarity_ewma",
    # EWMA / group-loading utilities
    "compute_ewm",
    "compute_ewm_covar",
    "compute_expanding_power",
    "set_group_loadings",
    # Sign-constraint derivation
    "derive_sign_constraints",
    "validate_cluster_signs",
    # Residual validation
    "LassoModelDiagonalityCV",
    "ResidualDiagnostics",
    "Sparsity",
    "diagnose_residuals",
    "effective_sparsity",
    "marchenko_pastur_edge",
    "missing_factor_components",
    "null_threshold",
    "raw_offdiagonal_mass",
    "residual_correlation",
    "suggest_tolerance",
]
