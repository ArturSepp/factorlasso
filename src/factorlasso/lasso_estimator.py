"""Historical location of the estimator, its result types and its solvers.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.linear_model`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.cluster._dependence import (
    compute_dependence_matrix, DEFAULT_DEPENDENCE_MEASURE, DEFAULT_GERBER_THRESHOLD,
    DependenceMeasure,
)
from factorlasso.cluster._hierarchical import (
    apply_cluster_correlation_transform, ClusterCorrelationTransform,
    compute_clusters_from_corr_matrix, DEFAULT_CLUSTER_CORRELATION_TRANSFORM,
    DEFAULT_CUTOFF_FRACTION, DEFAULT_DISTANCE_TRANSFORM, DEFAULT_LINKAGE_METHOD, DistanceTransform,
    VALID_LINKAGE_METHODS,
)
from factorlasso.cluster._smoothing import ClusterSmootherType
from factorlasso.linear_model._estimator import LassoModel
from factorlasso.linear_model._preparation import _PreparedFit
from factorlasso.linear_model._settings import _selected_prior_factors
from factorlasso.linear_model._solvers.common import (
    _build_loading_bound_constraints, _build_sign_constraints, _clean_beta_prior,
    _compute_solver_diagnostics, _compute_solver_weights, _derive_valid_mask_from_y, _nan_result,
    _solve_with_fallback, _validate_loss_normalization, _weighted_squared_loss,
)
from factorlasso.linear_model._solvers.cooperative import solve_cooperative_group_lasso_cvx_problem
from factorlasso.linear_model._solvers.group_lasso import (
    _build_group_lasso_problem, solve_group_lasso_cvx_problem, solve_group_lasso_path,
)
from factorlasso.linear_model._solvers.lasso import solve_lasso_cvx_problem
from factorlasso.linear_model._solvers.unilasso import solve_unilasso_cvx_problem
from factorlasso.linear_model._types import (
    LassoEstimationResult, LassoModelType, LassoNowcastResult, _MODES_WITHOUT_SIGN_CONSTRAINTS,
)
from factorlasso.priors._bounds import _compute_expert_prior_bounds, _validate_expert_bound_settings
from factorlasso.priors._ols import (
    _compute_joint_ols_prior, _compute_ols_prior, _validate_prior_selection_type,
    _zero_incompatible_priors,
)
from factorlasso.utils._ewm import (
    compute_ewm, compute_expanding_power, set_group_loadings, _validate_span,
)
from factorlasso.utils._panel import get_x_y_np

guard_legacy_module(__name__)
