"""UniLasso solver: univariate-guided two-stage LASSO."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import cvxpy as cvx
import numpy as np

from factorlasso.utils._ewm import _validate_span
from factorlasso.linear_model._types import LassoEstimationResult
from factorlasso.linear_model._solvers.common import (
    _compute_solver_diagnostics, _compute_solver_weights, _derive_valid_mask_from_y, _nan_result,
    _solve_with_fallback,
)


def solve_unilasso_cvx_problem(
    x: np.ndarray,
    y: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    reg_lambda: float = 1e-8,
    span: Optional[float] = None,
    verbose: bool = False,
    solver: str = 'CLARABEL',
    solver_fallbacks: Optional[Sequence[str]] = None,
    loo: bool = True,
    non_negative: bool = True,
) -> LassoEstimationResult:
    r"""UniLasso univariate-guided sparse regression, per response (no groups).

    Two-stage estimator of Chatterjee, Hastie & Tibshirani (2025). For each
    response, stage 1 fits the M univariate slopes
    :math:`\hat\beta^{uni}_j = (x_j^\top y) / (x_j^\top x_j)` and forms the
    prevalidated (leave-one-out) univariate fits :math:`\hat\eta_{\cdot,j}`.
    Stage 2 regresses y on those fits,

    .. math::

        \min_{\theta \ge 0}\ \frac1T\|y - \hat\eta\,\theta\|_2^2
          + \lambda \|\theta\|_1,

    and the final coefficient :math:`\hat\beta_j = \hat\theta_j\,
    \hat\beta^{uni}_j` inherits the univariate sign through
    :math:`\theta_j \ge 0`. There is no grouping, no clustering, and no
    significance gate. Sign preservation is indirect via the stage-2
    non-negativity, in contrast to the hard sign constraint of the cluster
    modes.

    Parameters
    ----------
    x : np.ndarray, shape (T, M)
    y : np.ndarray, shape (T, N)
    valid_mask, reg_lambda, span, verbose, solver
        See :func:`solve_group_lasso_cvx_problem`. ``reg_lambda`` weights the
        stage-2 L1 penalty on theta.
    loo : bool, default True
        Use leave-one-out (prevalidated) univariate fits in stage 2 (the
        published UniLasso). False uses in-sample univariate fits.
    non_negative : bool, default True
        Constrain theta >= 0 so the coefficient inherits the univariate sign.

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
        warnings.warn(f"insufficient observations for unilasso: t={t}")
        return _nan_result(n_y, n_x)

    weights = _compute_solver_weights(t, n_y, span, valid_mask)
    beta = np.zeros((n_y, n_x), dtype=float)

    for k in range(n_y):
        w = valid_mask[:, k] > 0
        xk = x[w]
        yk = y[w, k]
        tk = xk.shape[0]
        if tk < 5:
            continue
        s_xx = np.einsum('ij,ij->j', xk, xk)        # (M,) sum x^2
        s_xy = xk.T @ yk                            # (M,) sum x y
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = np.where(s_xx > 0.0, s_xy / s_xx, 0.0)   # full-sample (M,)
        if loo:
            num = s_xy[None, :] - xk * yk[:, None]   # (tk, M) leave-one-out
            den = s_xx[None, :] - xk * xk
            with np.errstate(divide='ignore', invalid='ignore'):
                slope_loo = np.where(den > 0.0, num / den, 0.0)
            eta = xk * slope_loo                     # (tk, M)
        else:
            eta = xk * slope[None, :]
        theta = cvx.Variable(n_x, nonneg=non_negative)
        fit = (1.0 / tk) * cvx.sum_squares(eta @ theta - yk)
        pen = reg_lambda * cvx.norm1(theta)
        prob = cvx.Problem(cvx.Minimize(fit + pen))
        _solve_with_fallback(prob, solver, solver_fallbacks, verbose=verbose)
        th = theta.value if theta.value is not None else np.zeros(n_x)
        beta[k, :] = th * slope

    alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
        x, y, beta, weights
    )
    return LassoEstimationResult(
        estimated_beta=beta, alpha=alpha,
        ss_total=ss_total, ss_res=ss_res, r2=r2,
    )
