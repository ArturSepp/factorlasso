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

Full pipeline
-------------
>>> from factorlasso import LassoModel, CurrentFactorCovarData, VarianceColumns
"""

__version__ = "0.1.0"

# --- Core estimator ---
# --- Utilities ---
from factorlasso.ewm_utils import (
    compute_ewm,
    compute_ewm_covar,
    compute_expanding_power,
    set_group_loadings,
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
    compute_clusters_from_corr_matrix,
    get_x_y_np,
    solve_group_lasso_cvx_problem,
    solve_lasso_cvx_problem,
)

__all__ = [
    # Estimator
    "LassoModelType",
    "LassoModel",
    "LassoEstimationResult",
    "solve_lasso_cvx_problem",
    "solve_group_lasso_cvx_problem",
    "get_x_y_np",
    "compute_clusters_from_corr_matrix",
    # Factor covariance
    "VarianceColumns",
    "CurrentFactorCovarData",
    "RollingFactorCovarData",
    # Utilities
    "compute_ewm",
    "compute_ewm_covar",
    "compute_expanding_power",
    "set_group_loadings",
]
