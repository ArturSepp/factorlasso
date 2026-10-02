"""Sparse factor-model estimation.

:class:`LassoModel` estimates the loadings of every mode (LASSO, UniLasso, group LASSO, HCGL,
FCGL and the cooperative estimators) with CVXPY; the solver functions are also callable
directly on prepared arrays. The names are also exported from :mod:`factorlasso`.
"""

from factorlasso.linear_model._types import (
    LassoEstimationResult, LassoModelType, LassoNowcastResult,
)
from factorlasso.linear_model._estimator import LassoModel
from factorlasso.linear_model._solvers.cooperative import (
    solve_cooperative_group_lasso_cvx_problem,
)
from factorlasso.linear_model._solvers.group_lasso import (
    solve_group_lasso_cvx_problem, solve_group_lasso_path,
)
from factorlasso.linear_model._solvers.lasso import solve_lasso_cvx_problem
from factorlasso.linear_model._solvers.unilasso import solve_unilasso_cvx_problem

__all__ = [
    "LassoEstimationResult",
    "LassoModel",
    "LassoModelType",
    "LassoNowcastResult",
    "solve_cooperative_group_lasso_cvx_problem",
    "solve_group_lasso_cvx_problem",
    "solve_group_lasso_path",
    "solve_lasso_cvx_problem",
    "solve_unilasso_cvx_problem",
]
