"""Zero-centred squared-L2 regression with the shared weighted factor-model loss.

References
----------
Hoerl, A. E., and Kennard, R. W. (1970), "Ridge Regression: Biased Estimation
for Nonorthogonal Problems", Technometrics 12(1), 55-67.
https://doi.org/10.1080/00401706.1970.10488634

EW observation weights, response-specific missing masks and the two loss
normalizations are package adaptations. No inferential or optimal-tuning
guarantee is inherited. The caller owns centring; the solver does not estimate
or penalize an intercept, standardize factors, or impose signs or prior targets.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import cvxpy as cvx
import numpy as np

from factorlasso.linear_model._solvers.common import (
    _compute_solver_diagnostics, _compute_solver_weights, _derive_valid_mask_from_y,
    _nan_result, _solve_with_fallback, _validate_loss_normalization, _weighted_squared_loss,
)
from factorlasso.linear_model._types import LassoEstimationResult
from factorlasso.utils._ewm import _validate_span


def solve_ridge_cvx_problem(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    reg_lambda: float = 1e-5,
    span: Optional[float] = None,
    verbose: bool = False,
    solver: str = 'CLARABEL',
    solver_fallbacks: Optional[Sequence[str]] = None,
    loss_normalization: str = 'sample',
) -> LassoEstimationResult:
    """Minimise the shared weighted squared loss plus lambda times squared beta.

    Parameters
    ----------
    x, y : np.ndarray
        Prepared, finite factor and response arrays, shaped (T, M) and (T, N).
        Without an explicit mask, NaN responses are masked and zero-filled.
    valid_mask : np.ndarray, optional
        Response validity mask of shape (T, N).
    reg_lambda : float, default 1e-5
        Nonnegative squared-L2 penalty. Zero returns the minimum-norm weighted
        least-squares solution; positive infinity returns exactly zero loadings.
    span : float, optional
        EW observation-weight span. Centring is performed by the estimator.
    verbose : bool, default False
        Whether to print solver output.
    solver : str, default 'CLARABEL'
        CVXPY solver used at finite positive penalties.
    solver_fallbacks : sequence of str, optional
        Shared opt-in solver retry chain.
    loss_normalization : {'sample', 'weight_sum'}, default 'sample'
        Divide SSE by T or each response's valid squared-weight mass. There is
        no factor of one half in the loss or penalty. Empty responses have zero
        loss and zero loadings.

    Returns
    -------
    LassoEstimationResult
        Coefficients of shape (N, M) and shared residual diagnostics.
    """
    _validate_span(span)
    _validate_loss_normalization(loss_normalization)
    if (isinstance(reg_lambda, (bool, np.bool_))
            or not isinstance(reg_lambda, (int, float, np.integer, np.floating))
            or np.isnan(reg_lambda) or reg_lambda < 0):
        raise ValueError('RIDGE reg_lambda must be nonnegative')
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError('Ridge x and y must be two-dimensional with matching rows')
    t, n_x = x.shape
    n_y = y.shape[1]
    if valid_mask is None:
        y, valid_mask = _derive_valid_mask_from_y(y)
    if valid_mask.shape != y.shape:
        raise ValueError('Ridge valid_mask must match the response shape')
    if t == 0:
        warnings.warn('insufficient observations for ridge: t=0')
        return _nan_result(n_y, n_x)
    weights = _compute_solver_weights(t, n_y, span, valid_mask)
    mass = np.sum(np.square(weights), axis=0)

    if np.isinf(reg_lambda):
        estimated_beta = np.zeros((n_y, n_x))
    elif reg_lambda == 0:
        # A scalar loss denominator does not affect each response's minimizer.
        # lstsq fixes the minimum-norm convention even for rank-deficient designs.
        estimated_beta = np.vstack([
            np.linalg.lstsq(x * weights[:, j, None], y[:, j] * weights[:, j],
                            rcond=None)[0]
            for j in range(n_y)
        ])
    else:
        beta = cvx.Variable((n_y, n_x))
        # Scale both terms equally to avoid an unbounded objective coefficient
        # for very large finite penalties. The minimizer is unchanged.
        scale = max(1.0, float(reg_lambda))
        objective = cvx.Minimize(
            _weighted_squared_loss(x @ beta.T - y, weights, t, loss_normalization) / scale
            + (reg_lambda / scale) * cvx.sum_squares(beta)
        )
        problem = cvx.Problem(objective)
        _solve_with_fallback(problem, solver, solver_fallbacks, verbose=verbose)
        if beta.value is None:
            warnings.warn('ridge problem not solved')
            return _nan_result(n_y, n_x)
        estimated_beta = beta.value
    estimated_beta[mass == 0.0, :] = 0.0
    alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
        x, y, estimated_beta, weights,
    )
    return LassoEstimationResult(
        estimated_beta=estimated_beta, alpha=alpha,
        ss_total=ss_total, ss_res=ss_res, r2=r2,
    )
