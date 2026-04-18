"""
factorlasso — Sparse factor model estimation with constrained LASSO
===================================================================

Estimate sparse multi-output regression coefficients with sign
constraints, prior-centered regularisation, and hierarchical group
structure (HCGL), then assemble consistent factor covariance matrices.

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

Full pipeline
-------------
>>> from factorlasso import LassoModel, CurrentFactorCovarData, VarianceColumns
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("factorlasso")
except PackageNotFoundError:  # pragma: no cover - editable install before metadata exists
    __version__ = "0.0.0+unknown"

# --- Core estimator ---
from factorlasso.cv import LassoModelCV

# --- EWMA / group-loading utilities ---
from factorlasso.ewm_utils import (
    compute_ewm,
    compute_ewm_covar,
    compute_expanding_power,
    set_group_loadings,
)

# --- Clustering utilities ---
from factorlasso.cluster_utils import (
    compute_clusters_from_corr_matrix,
    get_clusters_by_freq,
    get_cutoffs_by_freq,
    get_linkage_array,
    get_linkages_by_freq,
)

# --- Factor covariance assembly ---
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
    solve_group_lasso_cvx_problem,
    solve_lasso_cvx_problem,
)

__all__ = [
    # Estimator
    "LassoModelType",
    "LassoModel",
    "LassoModelCV",
    "LassoEstimationResult",
    "solve_lasso_cvx_problem",
    "solve_group_lasso_cvx_problem",
    "get_x_y_np",
    # Factor covariance
    "VarianceColumns",
    "CurrentFactorCovarData",
    "RollingFactorCovarData",
    # Clustering utilities
    "compute_clusters_from_corr_matrix",
    "get_linkage_array",
    "get_clusters_by_freq",
    "get_linkages_by_freq",
    "get_cutoffs_by_freq",
    # EWMA / group-loading utilities
    "compute_ewm",
    "compute_ewm_covar",
    "compute_expanding_power",
    "set_group_loadings",
]
