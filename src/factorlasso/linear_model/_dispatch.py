"""Route a prepared fit to the CVXPY solver of its mode.

This is the only module of the estimator layer that calls the solvers: tests that intercept a
solve patch the solver names here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from factorlasso.linear_model._preparation import _PreparedFit
from factorlasso.linear_model._solvers.cooperative import (
    solve_cooperative_group_lasso_cvx_problem,
)
from factorlasso.linear_model._solvers.group_lasso import (
    solve_group_lasso_cvx_problem, solve_group_lasso_path,
)
from factorlasso.linear_model._solvers.lasso import solve_lasso_cvx_problem
from factorlasso.linear_model._solvers.unilasso import solve_unilasso_cvx_problem
from factorlasso.linear_model._types import LassoEstimationResult, LassoModelType, _mode_spec
from factorlasso.utils._ewm import set_group_loadings

def group_penalty_geometry(model_type: LassoModelType, prep: _PreparedFit) -> Dict:
    """Solver keywords of the group penalty.

    GROUP_LASSO and HCGL penalise each response's loadings (rows); FCGL penalises the
    loadings of a cluster's responses on each factor (cluster-by-factor blocks), so its
    problem is not block-separable across responses.
    """
    if _mode_spec(model_type).block_mode == "cluster_factor":
        return dict(block_mode="cluster_factor", col_weights=prep.col_weights_np)
    return dict(row_weights=prep.row_weights_np)


def solve_prepared(model, prep: _PreparedFit, x_np: np.ndarray, y_np: np.ndarray,
                   valid_mask: np.ndarray, eff_span: Optional[float],
                   verbose: bool) -> LassoEstimationResult:
    """Solve one prepared fit at ``model.reg_lambda``."""
    solver = _mode_spec(model.model_type).solver
    if prep.is_lasso_mode:
        return solve_lasso_cvx_problem(
            x=x_np, y=y_np, valid_mask=valid_mask,
            reg_lambda=model.reg_lambda, span=eff_span,
            verbose=verbose, solver=model.solver,
            solver_fallbacks=model.solver_fallbacks,
            loss_normalization=model.loss_normalization,
            nonneg=model.nonneg,
            factors_beta_loading_signs=prep.signs_np,
            factors_beta_prior=prep.prior_np,
            beta_lower_bounds=prep.lower_bounds_np,
            beta_upper_bounds=prep.upper_bounds_np,
            penalty_weights=prep.penalty_weights_np,
        )
    if solver == "group":
        gl = set_group_loadings(group_data=prep.asset_clusters)
        return solve_group_lasso_cvx_problem(
            x=x_np, y=y_np, group_loadings=gl.to_numpy(),
            valid_mask=valid_mask,
            reg_lambda=model.reg_lambda, span=eff_span,
            verbose=verbose, solver=model.solver,
            solver_fallbacks=model.solver_fallbacks,
            loss_normalization=model.loss_normalization,
            nonneg=model.nonneg,
            factors_beta_loading_signs=prep.signs_np,
            factors_beta_prior=prep.prior_np,
            beta_lower_bounds=prep.lower_bounds_np,
            beta_upper_bounds=prep.upper_bounds_np,
            group_penalty=model.group_penalty,
            l1_weight=model.l1_weight,
            penalty_weights=prep.penalty_weights_np,
            **group_penalty_geometry(model.model_type, prep),
        )
    if solver == "cooperative":
        gl = set_group_loadings(group_data=prep.asset_clusters)
        return solve_cooperative_group_lasso_cvx_problem(
            x=x_np, y=y_np, group_loadings=gl.to_numpy(),
            valid_mask=valid_mask,
            reg_lambda=model.reg_lambda, span=eff_span,
            verbose=verbose, solver=model.solver,
            solver_fallbacks=model.solver_fallbacks,
            loss_normalization=model.loss_normalization,
            factors_beta_prior=prep.prior_np,
            group_penalty=model.group_penalty,
            l1_weight=model.l1_weight,
        )
    if solver == "unilasso":
        return solve_unilasso_cvx_problem(
            x=x_np, y=y_np, valid_mask=valid_mask,
            reg_lambda=model.reg_lambda, span=eff_span,
            verbose=verbose, solver=model.solver,
            solver_fallbacks=model.solver_fallbacks,
            loo=model.unilasso_loo,
            non_negative=model.unilasso_non_negative,
        )
    raise NotImplementedError(f"Unsupported model_type: {model.model_type}")


def solve_prepared_path(model, prep: _PreparedFit, x_np: np.ndarray, y_np: np.ndarray,
                        valid_mask: np.ndarray, eff_span: Optional[float],
                        reg_lambdas: Sequence[float],
                        verbose: bool) -> List[LassoEstimationResult]:
    """Solve a prepared group-penalty fit at every ``reg_lambda`` with one canonical form."""
    gl = set_group_loadings(group_data=prep.asset_clusters).to_numpy()
    return solve_group_lasso_path(
        x=x_np, y=y_np, group_loadings=gl, reg_lambdas=reg_lambdas,
        valid_mask=valid_mask, span=eff_span, verbose=verbose,
        solver=model.solver, solver_fallbacks=model.solver_fallbacks,
        nonneg=model.nonneg,
        factors_beta_loading_signs=prep.signs_np,
        factors_beta_prior=prep.prior_np,
        beta_lower_bounds=prep.lower_bounds_np,
        beta_upper_bounds=prep.upper_bounds_np,
        group_penalty=model.group_penalty, l1_weight=model.l1_weight,
        penalty_weights=prep.penalty_weights_np,
        loss_normalization=model.loss_normalization,
        **group_penalty_geometry(model.model_type, prep),
    )
