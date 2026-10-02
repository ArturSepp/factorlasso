"""Element-wise L1 LASSO solver."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import cvxpy as cvx
import numpy as np

from factorlasso.utils._ewm import _validate_span
from factorlasso.linear_model._types import LassoEstimationResult
from factorlasso.linear_model._solvers.common import (
    _compute_solver_diagnostics, _compute_solver_weights, _weighted_squared_loss, _clean_beta_prior,
    _derive_valid_mask_from_y, _build_sign_constraints, _nan_result, _solve_with_fallback,
    _build_loading_bound_constraints,
)


def solve_lasso_cvx_problem(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    reg_lambda: float = 1e-8,
    span: Optional[float] = None,
    verbose: bool = False,
    solver: str = 'CLARABEL',
    solver_fallbacks: Optional[Sequence[str]] = None,
    nonneg: bool = False,
    factors_beta_loading_signs: Optional[np.ndarray] = None,
    factors_beta_prior: Optional[np.ndarray] = None,
    penalty_weights: Optional[np.ndarray] = None,
    loss_normalization: str = "sample",
    beta_lower_bounds: Optional[np.ndarray] = None,
    beta_upper_bounds: Optional[np.ndarray] = None,
) -> LassoEstimationResult:
    r"""
    L1-regularised (LASSO) multi-output regression via CVXPY.

    Minimises

    .. math::

        \frac{1}{T}\|W \odot (X\beta^\top - Y)\|_F^2
        + \lambda\|\beta - \beta_0\|_1

    where β is ``(N × M)``, X is ``(T × M)``, Y is ``(T × N)``.

    Parameters
    ----------
    x : np.ndarray, shape (T, M)
        Regressor matrix.
    y : np.ndarray, shape (T, N)
        Response matrix.
    valid_mask : np.ndarray, shape (T, N), optional
        Binary validity mask.  Derived from ``y`` if ``None``.
    loss_normalization : {"sample", "weight_sum"}, default "sample"
        The displayed loss uses ``sample``. ``weight_sum`` instead divides each
        response's weighted squared error by its valid squared-weight mass.
        Recalibrate lambda when switching; zero-mass responses contribute no loss.
    reg_lambda : float, default 1e-8
        L1 regularisation strength.
    span : float, optional
        EWMA span for observation weighting.  Must be ≥ 1 when provided.
        Float accepted.
    verbose : bool, default False
        Print CVXPY solver output.
    solver : str, default 'CLARABEL'
        CVXPY solver name.
    nonneg : bool, default False
        Constrain all β ≥ 0.
    factors_beta_loading_signs : np.ndarray, shape (N, M), optional
        Element-wise sign constraints.
    factors_beta_prior : np.ndarray, shape (N, M), optional
        Prior β₀.  NaN entries → zero prior.

    beta_lower_bounds, beta_upper_bounds : ndarray, shape (N, M), optional
        Individual coefficient bounds; NaN cells add no constraint. These arrays
        are already resolved by the caller, including any sign precedence.

    Returns
    -------
    LassoEstimationResult
    """
    _validate_span(span)
    assert y.ndim == 2 and x.ndim == 2 and x.shape[0] == y.shape[0]
    t, n_x = x.shape
    n_y = y.shape[1]

    if valid_mask is None:
        y, valid_mask = _derive_valid_mask_from_y(y)
    if t < 5:
        warnings.warn(f"insufficient observations for lasso: t={t}")
        return _nan_result(n_y, n_x)

    # Variable and constraints
    if factors_beta_loading_signs is not None:
        beta = cvx.Variable((n_y, n_x))
        constraints = _build_sign_constraints(beta, factors_beta_loading_signs)
    else:
        beta = cvx.Variable((n_y, n_x), nonneg=nonneg)
        constraints = []

    constraints.extend(_build_loading_bound_constraints(
        beta, beta_lower_bounds, beta_upper_bounds))
    weights = _compute_solver_weights(t, n_y, span, valid_mask)
    prior = _clean_beta_prior(factors_beta_prior, n_y, n_x)

    # L1 penalty term: weighted elementwise if penalty_weights supplied
    # (Zou 2006 adaptive Lasso); plain L1 otherwise.
    if penalty_weights is not None:
        if penalty_weights.shape != (n_y, n_x):
            raise ValueError(
                f"penalty_weights shape {penalty_weights.shape} "
                f"!= expected ({n_y}, {n_x})"
            )
        l1_term = reg_lambda * cvx.sum(
            cvx.multiply(penalty_weights, cvx.abs(beta - prior))
        )
    else:
        l1_term = reg_lambda * cvx.norm1(beta - prior)

    objective = cvx.Minimize(
        _weighted_squared_loss(x @ beta.T - y, weights, t, loss_normalization)
        + l1_term
    )
    problem = cvx.Problem(objective, constraints) if constraints else cvx.Problem(objective)
    _solve_with_fallback(problem, solver, solver_fallbacks, verbose=verbose)

    if beta.value is None:
        warnings.warn("lasso problem not solved")
        return _nan_result(n_y, n_x)

    alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
        x, y, beta.value, weights
    )
    return LassoEstimationResult(
        estimated_beta=beta.value, alpha=alpha,
        ss_total=ss_total, ss_res=ss_res, r2=r2,
    )
