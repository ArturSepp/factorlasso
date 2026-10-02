"""Shared solver building blocks.

Observation weights, the weighted squared loss, sign and loading-bound constraints,
in-sample diagnostics and the solver fallback chain used by every CVXPY solver.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cvxpy as cvx
import numpy as np

from factorlasso.utils._ewm import compute_expanding_power
from factorlasso.linear_model._types import LassoEstimationResult


def _compute_solver_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    estimated_beta: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """In-sample fit diagnostics from solver weights.

    The returned ``alpha`` is the EWMA-weighted mean of residuals on the
    demeaned data the solver received, computed in the **nominal-span EWMA
    norm** (per-observation weight ``lambda^k``) — the same norm as the
    solver's error term. It is **not** the regression
    intercept in original units; see the docstring of
    :class:`LassoEstimationResult` for the distinction. The economic
    intercept of ``y = α + Xβ + ε`` is computed in
    :meth:`LassoModel.fit` from the sample means of the original (pre-
    demean) ``y`` and ``X``, and is exposed as ``model.alpha_const_``.

    Solver ``weights`` carry ``sqrt(lambda)`` decay by construction (they
    are squared inside the quadratic loss), so any *linear* EWMA statistic
    here must use ``weights**2``. Reusing the sqrt-decay weights linearly
    silently doubles the effective span of the statistic (a linear EWMA
    with decay ``lambda^(1/2)`` has effective span ``≈ 2·span``); this was
    the behaviour of versions before 0.5.0.
    """
    w_sq = np.square(weights)
    col_sums = np.sum(w_sq, axis=0)
    norm_w = np.divide(w_sq, col_sums, out=np.zeros_like(w_sq),
                       where=col_sums != 0)

    residuals = y - x @ estimated_beta.T
    alpha = np.sum(norm_w * residuals, axis=0)
    ss_res = np.sum(norm_w * np.square(residuals), axis=0)
    y_wmean = np.sum(norm_w * y, axis=0)
    ss_total = np.sum(norm_w * np.square(y - y_wmean), axis=0)
    r2 = np.zeros_like(ss_res)
    np.divide(ss_res, ss_total, out=r2, where=ss_total > 0.0)
    r2 = 1.0 - r2
    return alpha, ss_total, ss_res, r2


def _compute_solver_weights(
    t: int, n_y: int, span: Optional[float], valid_mask: np.ndarray
) -> np.ndarray:
    """Observation weights: EWMA decay × validity mask.

    The returned weights carry ``sqrt(lambda)`` decay: the quadratic solver
    loss squares them, so the loss norm is the nominal-span EWMA. Design
    rule: these are sqrt-decay row scalings and may only enter quadratic
    forms; any linear-EWMA statistic must use ``weights**2`` (see
    :func:`_compute_solver_diagnostics`).
    """
    if span is not None:
        w = compute_expanding_power(
            n=t,
            power_lambda=np.sqrt(1.0 - 2.0 / (span + 1.0)),
            reverse_columns=True,
        )
    else:
        w = np.ones(t)

    if valid_mask.ndim == 2:
        w = np.tile(w, (n_y, 1)).T

    return w * valid_mask


def _validate_loss_normalization(loss_normalization: str) -> None:
    """Validate the explicit objective convention without changing defaults."""
    if loss_normalization not in ('sample', 'weight_sum'):
        raise ValueError("loss_normalization must be 'sample' or 'weight_sum'")


def _weighted_squared_loss(residual, weights: np.ndarray, t: int,
                           loss_normalization: str):
    """Build a loss with legacy row scaling or per-response valid weight mass.

    ``weights`` are square-root observation weights. Diagnostics continue to
    use the original weights; only the optimization loss is normalized. A
    response with zero mass contributes zero loss, with no division by zero.
    """
    _validate_loss_normalization(loss_normalization)
    if loss_normalization == 'sample':
        return (1.0 / t) * cvx.sum_squares(cvx.multiply(weights, residual))
    mass = np.sum(np.square(weights), axis=0)
    normalized = np.divide(weights, np.sqrt(mass), out=np.zeros_like(weights),
                           where=mass > 0.0)
    return cvx.sum_squares(cvx.multiply(normalized, residual))


def _clean_beta_prior(
    factors_beta_prior: Optional[np.ndarray], n_y: int, n_x: int
) -> np.ndarray:
    """Return clean prior (N × M): NaN → 0, None → zeros."""
    if factors_beta_prior is not None:
        return np.where(np.isnan(factors_beta_prior), 0.0, factors_beta_prior)
    return np.zeros((n_y, n_x))


def _derive_valid_mask_from_y(
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive validity mask from NaN positions and zero-fill NaNs."""
    nan_mask = np.isnan(y)
    return np.where(nan_mask, 0.0, y), (~nan_mask).astype(float)


def _build_sign_constraints(
    beta: cvx.Variable,
    signs: np.ndarray,
) -> list:
    """Build CVXPY constraints from sign matrix."""
    constraints = []
    zero_mask = np.isclose(signs, 0.0).astype(float)
    nonneg_mask = np.greater(signs, 0.0).astype(float)
    nonpos_mask = np.less(signs, 0.0).astype(float)

    if np.any(zero_mask > 0):
        constraints.append(cvx.multiply(zero_mask, beta) == 0)
    if np.any(nonneg_mask > 0):
        constraints.append(cvx.multiply(nonneg_mask, beta) >= 0)
    if np.any(nonpos_mask > 0):
        constraints.append(cvx.multiply(nonpos_mask, beta) <= 0)
    return constraints


def _nan_result(n_y: int, n_x: int) -> LassoEstimationResult:
    """Return NaN-filled result for failed solves."""
    return LassoEstimationResult(
        estimated_beta=np.full((n_y, n_x), np.nan),
        alpha=np.full(n_y, np.nan),
        ss_total=np.full(n_y, np.nan),
        ss_res=np.full(n_y, np.nan),
        r2=np.full(n_y, np.nan),
    )


def _solve_with_fallback(problem: cvx.Problem,
                         solver: str,
                         solver_fallbacks: Optional[Sequence[str]] = None,
                         verbose: bool = False,
                         warm_start: bool = False) -> None:
    """solve ``problem`` with ``solver``, optionally retrying with each solver
    in ``solver_fallbacks`` when the primary solver raises or returns a
    non-optimal status.

    With ``solver_fallbacks=None`` (the default) the primary solver is invoked
    exactly once and any exception propagates, so behaviour is identical to a
    direct ``problem.solve`` call and the caller's downstream
    ``beta.value is None`` handling is unchanged. A non-empty
    ``solver_fallbacks`` activates the retry chain, letting production callers
    prefer graceful degradation over a hard failure on a single solver.

    Parameters
    ----------
    problem : cvxpy.Problem
        the already-constructed convex program.
    solver : str
        primary solver name, e.g. ``'CLARABEL'``.
    solver_fallbacks : sequence of str, optional
        ordered fallback solver names, tried only if the primary fails.
    verbose : bool, default False
    warm_start : bool, default False

    Raises
    ------
    cvxpy.error.SolverError
        only when ``solver_fallbacks`` is non-empty and the primary solver and
        every fallback fail to reach an optimal status.
    """
    if not solver_fallbacks:
        if warm_start:
            problem.solve(verbose=verbose, solver=solver, warm_start=True)
        else:
            problem.solve(verbose=verbose, solver=solver)
        return
    chain = [solver, *solver_fallbacks]
    last_error: Optional[Exception] = None
    for name in chain:
        try:
            problem.solve(verbose=verbose, solver=name, warm_start=warm_start)
        except Exception as error:  # retry on any solver failure
            last_error = error
            continue
        if problem.status in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
            return
    raise cvx.error.SolverError(
        f"factorlasso: all solvers failed (tried {chain}); last status "
        f"{problem.status!r}, last error {last_error!r}"
    )


def _build_loading_bound_constraints(beta, lower, upper):
    """Create finite-cell individual bounds; NaN means no constraint."""
    constraints = []
    arrays = []
    for values, is_lower in ((lower, True), (upper, False)):
        if values is None:
            arrays.append(None)
            continue
        values = np.asarray(values, dtype=float)
        if values.shape != beta.shape or np.isinf(values).any():
            raise ValueError('Loading bounds must match beta shape and be finite or NaN')
        mask = np.isfinite(values)
        if mask.any():
            constraints.append(beta[mask] >= values[mask] if is_lower
                               else beta[mask] <= values[mask])
        arrays.append(values)
    if all(value is not None for value in arrays) and np.any(arrays[0] > arrays[1]):
        raise ValueError('Loading lower bounds exceed upper bounds')
    return constraints
