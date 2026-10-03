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

from factorlasso.cluster import (
    ClusterCorrelationTransform,
    ClusterCorrelationTransformResult,
    ClusterSmootherType,
    ClusterStabilityStatistics,
    DependenceMeasure,
    DistanceTransform,
    RollingClusterData,
    StabilityPoolingType,
    apply_cluster_correlation_transform,
    apply_partition_distance_bonus,
    compute_cluster_stability_statistics,
    compute_clusters_from_corr_matrix,
    compute_co_association_panel,
    compute_dependence_matrix,
    compute_gerber_matrix,
    compute_rolling_smoothed_clusters,
    get_clusters_by_freq,
    get_cutoffs_by_freq,
    get_linkage_array,
    get_linkages_by_freq,
    remove_first_principal_component,
    score_with_stability_pooled_clusters,
    smooth_similarity_ewma,
)
from factorlasso.covariance import (
    CurrentFactorCovarData,
    ResidualCorrelationData,
    ResidualType,
    RollingFactorCovarData,
    VarianceColumns,
    estimate_residual_correlation,
)
from factorlasso.diagnostics import (
    ResidualDiagnostics,
    RiskClusterReport,
    Sparsity,
    TaxonomyConfig,
    analyze_cluster_lineage,
    diagnose_residuals,
    effective_sparsity,
    marchenko_pastur_edge,
    missing_factor_components,
    null_threshold,
    partition_variance_share,
    raw_offdiagonal_mass,
    residual_correlation,
    run_cluster_lineage_report,
    suggest_tolerance,
)
from factorlasso.linear_model import (
    LassoEstimationResult,
    LassoModel,
    LassoModelType,
    LassoNowcastResult,
    solve_cooperative_group_lasso_cvx_problem,
    solve_group_lasso_cvx_problem,
    solve_group_lasso_path,
    solve_lasso_cvx_problem,
    solve_unilasso_cvx_problem,
)
from factorlasso.model_selection import LassoModelCV, LassoModelDiagonalityCV
from factorlasso.priors import (
    Ar1PriorInterval,
    ExpertPriorResolution,
    GaussianDominanceInformation,
    PriorHacGeometry,
    compute_ar1_prior_interval,
    compute_expert_prior_statistics,
    compute_prior_hac_geometry,
    derive_sign_constraints,
    gaussian_dominance_information,
    gaussian_prior_critical_value,
    map_expert_factor_priors,
    two_factor_limit_risk,
    two_factor_minimax_radius,
    validate_cluster_signs,
)
from factorlasso.utils import (
    compute_ewm,
    compute_ewm_covar,
    compute_expanding_power,
    get_x_y_np,
    set_group_loadings,
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
    "LassoNowcastResult",
    "ClusterSmootherType",
    "ClusterStabilityStatistics",
    "StabilityPoolingType",
    "RollingClusterData",
    "solve_lasso_cvx_problem",
    "solve_group_lasso_cvx_problem",
    "solve_group_lasso_path",
    "solve_cooperative_group_lasso_cvx_problem",
    "solve_unilasso_cvx_problem",
    "get_x_y_np",
    "ExpertPriorResolution",
    "map_expert_factor_priors",
    "compute_expert_prior_statistics",
    "PriorHacGeometry",
    "Ar1PriorInterval",
    "compute_prior_hac_geometry",
    "gaussian_prior_critical_value",
    "compute_ar1_prior_interval",
    "GaussianDominanceInformation",
    "gaussian_dominance_information",
    "two_factor_limit_risk",
    "two_factor_minimax_radius",
    # Factor covariance assembly
    "CurrentFactorCovarData",
    "ResidualType",
    "ResidualCorrelationData",
    "estimate_residual_correlation",
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
    "compute_co_association_panel",
    "compute_cluster_stability_statistics",
    "compute_rolling_smoothed_clusters",
    "score_with_stability_pooled_clusters",
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
    "partition_variance_share",
]
