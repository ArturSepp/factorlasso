"""Joint-loading shrinkage toward supplied response groups' WLS averages.

References
----------
Hoerl, A. E., and Kennard, R. W. (1970), "Ridge Regression: Biased Estimation
for Nonorthogonal Problems", Technometrics 12(1), 55-67.
https://doi.org/10.1080/00401706.1970.10488634

The estimated group target, explicit observation weights and unpenalized
asset-specific intercept are package adaptations. This module does not inherit
a dominance, sampling-distribution or optimal-tuning guarantee from ridge theory.
It reuses the existing zero-centred solver after subtracting the target fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import cvxpy as cvx
import numpy as np
import pandas as pd

from factorlasso.linear_model._solvers.ridge import solve_ridge_cvx_problem
from factorlasso.utils._pooled_inputs import _validate_pooled_inputs, _weighted_center


@dataclass(frozen=True)
class PooledLoadingShrinkageResult:
    """Labeled pooled-target fit and independently recomputed diagnostics.

    Attributes
    ----------
    coefficients, pooled_target, individual_wls : pandas.DataFrame
        Response-by-factor slopes in the supplied analysis coordinates.
    intercept, target_intercept, y_mean : pandas.Series
        Asset-specific fitted/target intercepts and weighted response means.
    x_mean : pandas.Series
        Weighted factor means in the supplied coordinates.
    groups, date_weights : pandas.Series
        Response-group IDs and strictly positive mean-one observation weights.
    reg_lambda : float
        Squared-distance penalty, including the exact zero/infinite endpoints.
    design_rank : int
        Full rank of the weighted-centered slope design.
    weighted_sse : float
        Sum of weighted squared original-coordinate residuals, without dividing by T.
    penalty : float
        Squared Frobenius distance of coefficients from the pooled target.
    objective : float or None
        Weighted SSE/T plus reg_lambda times penalty; None at infinity.
    stationarity_residual : float or None
        Normalized finite-penalty half-gradient residual; None at infinity.
    target_violation : float
        Maximum absolute loading distance from the pooled target.

    Notes
    -----
    Returned pandas objects own their data and labels. A frozen dataclass
    prevents field reassignment, but does not make pandas contents immutable.
    A small stationarity residual does not bound coefficient error for an
    ill-conditioned design. These diagnostics are not a CVXPY status report.
    """

    coefficients: pd.DataFrame
    pooled_target: pd.DataFrame
    individual_wls: pd.DataFrame
    intercept: pd.Series
    target_intercept: pd.Series
    y_mean: pd.Series
    x_mean: pd.Series
    groups: pd.Series
    date_weights: pd.Series
    reg_lambda: float
    design_rank: int
    weighted_sse: float
    penalty: float
    objective: float | None
    stationarity_residual: float | None
    target_violation: float


def _maximum_absolute(values: np.ndarray) -> float:
    """Return the infinity norm of a nonempty finite array."""
    return float(np.max(np.abs(values)))


def _stationarity(beta, target, hessian, cross, penalty):
    """Certify the scaled half-gradient without a penalty-times-distance overflow."""
    scale = max(1.0, penalty)
    fitted = beta @ (hessian / scale)
    rhs = cross / scale
    pull = (penalty / scale) * (beta - target)
    residual = fitted - rhs + pull
    values = (fitted, rhs, pull, residual)
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError("Pooled shrinkage stationarity is not finite")
    normalized = _maximum_absolute(residual) / max(
        1.0, *(_maximum_absolute(value) for value in values[:3]))
    if normalized > 1e-7:
        raise RuntimeError(
            f"Pooled shrinkage failed stationarity certification: {normalized:.6g} > 1e-7")
    return normalized


def _solver_coefficients(x, y, penalty, solver, fallbacks):
    """Reuse ridge and reject failed or nonfinite coefficients without a substitute fit."""
    try:
        result = solve_ridge_cvx_problem(
            x=x, y=y, reg_lambda=penalty, span=None, solver=solver,
            solver_fallbacks=fallbacks, loss_normalization="sample")
    except (cvx.error.SolverError, np.linalg.LinAlgError) as error:
        raise RuntimeError("Pooled shrinkage ridge solver failed") from error
    beta = np.asarray(result.estimated_beta, dtype=float)
    if beta.shape != (y.shape[1], x.shape[1]) or not np.isfinite(beta).all():
        raise RuntimeError("Pooled shrinkage ridge solver returned invalid coefficients")
    return beta.copy()


def fit_pooled_loading_shrinkage(
    x: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    reg_lambda: float,
    *,
    weights: pd.Series | None = None,
    solver: str = "CLARABEL",
    solver_fallbacks: Sequence[str] | None = None,
) -> PooledLoadingShrinkageResult:
    """Shrink joint WLS loadings toward response-group averages.

    Parameters
    ----------
    x, y : pandas.DataFrame
        Complete real finite factors (T by M) and responses (T by N), with
        identical unique ordered row indexes and unique nonmissing columns.
        A single response is a one-column frame. No alignment, row dropping,
        imputation or standardization is performed. T must be at least M+1.
    groups : pandas.Series
        Response-group IDs indexed exactly by y.columns, in the same order.
        IDs are strings or finite real numbers, excluding booleans. The
        arithmetic mean of each group's joint WLS slopes is its target.
    reg_lambda : float
        Nonnegative squared-distance penalty in weighted SSE/T units. Zero
        returns individual joint WLS; positive infinity returns the exact
        pooled target with asset-specific unpenalized intercepts.
    weights : pandas.Series, optional
        Strictly positive finite date weights indexed exactly by x.index.
        None means equal weights; supplied weights are normalized to mean one.
        The caller owns original-date EW weights, including when subsetting.
    solver : str, default 'CLARABEL'
        Existing CVXPY ridge solver used at finite positive penalties.
    solver_fallbacks : sequence of str, optional
        Existing opt-in solver retry chain. No other estimator is substituted.

    Returns
    -------
    PooledLoadingShrinkageResult
        Owned labeled coefficients, intercepts, target and diagnostics in the
        supplied coordinates. Finite fits pass normalized stationarity <=1e-7.

    Raises
    ------
    TypeError
        A panel, group or supplied weight container has the wrong type.
    ValueError
        Values, labels, scalar settings or weighted-centered rank are invalid.
        Full joint WLS rank is required even at positive/infinite penalties.
    RuntimeError
        Arithmetic is unrepresentable, the solver fails, or independently
        computed stationarity is unacceptable. No partial fit is returned.

    Notes
    -----
    With weighted-centered Xc and Yc, mean-one w, target G and N by M slopes B,
    the objective is sum(w * (Yc - Xc @ B.T)**2)/T + lambda*||B-G||_F**2.
    There is no factor one-half or sign restriction. The intercept is
    y_mean - B @ x_mean and is not penalized. A target passed to an existing
    L1/group prior implements a different penalty from this squared distance.

    Group discovery, training-only scales and tuning belong to the caller.
    Rebuild the target within each training prefix. Changing factor units at
    fixed scalar lambda generally changes isotropic shrinkage. This fit does
    not estimate premiums or decide which alpha enters a return forecast.
    """
    if isinstance(reg_lambda, (bool, np.bool_)) or not isinstance(reg_lambda, Real):
        raise ValueError("reg_lambda must be nonnegative or positive infinity")
    try:
        lam = float(reg_lambda)
    except (OverflowError, ValueError) as error:
        raise ValueError("reg_lambda must be representable as a float") from error
    if np.isposinf(lam) and reg_lambda != np.inf:
        raise ValueError("finite reg_lambda must be representable as a float")
    if np.isnan(lam) or lam < 0:
        raise ValueError("reg_lambda must be nonnegative or positive infinity")
    if not isinstance(solver, str) or not solver:
        raise ValueError("solver must be a nonempty solver name")
    if solver_fallbacks is not None:
        if (isinstance(solver_fallbacks, (str, bytes))
                or not isinstance(solver_fallbacks, Sequence)
                or any(not isinstance(name, str) or not name for name in solver_fallbacks)):
            raise ValueError("solver_fallbacks must be a sequence of nonempty solver names")
    panel = _validate_pooled_inputs(x, y, groups, weights)
    t, m = panel.x.shape
    if t <= m:
        raise ValueError("Pooled shrinkage requires T >= M+1 for its joint WLS target")
    w = panel.date_weights.to_numpy(copy=True)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            xc, xm = _weighted_center(panel.x, w)
            yc, ym = _weighted_center(panel.y, w)
            root = np.sqrt(w[:, None])
            xw, yw = root * xc, root * yc
            rank = int(np.linalg.lstsq(xw, np.zeros(t), rcond=None)[2])
            if rank != m:
                raise ValueError("Pooled shrinkage target requires full weighted design rank")
            hessian, cross = xw.T @ xw / t, yw.T @ xw / t
            if not np.isfinite(hessian).all() or not np.isfinite(cross).all():
                raise RuntimeError("Pooled shrinkage quadratic geometry is not finite")
            individual = _solver_coefficients(xw, yw, 0.0, solver, solver_fallbacks)
            _stationarity(individual, individual, hessian, cross, 0.0)
            target = np.empty_like(individual)
            for members in panel.members:
                # Dividing first avoids overflow in the sum of finite group slopes.
                target[members] = np.sum(individual[members] / len(members), axis=0)
            if lam == 0:
                beta = individual.copy()
            elif np.isposinf(lam):
                beta = target.copy()
            else:
                residual = root * (yc - xc @ target.T)
                beta = target + _solver_coefficients(
                    xw, residual, lam, solver, solver_fallbacks)
            intercept, target_intercept = ym - beta @ xm, ym - target @ xm
            # Centered residuals avoid cancellation of large unpenalized levels.
            residual = yc - xc @ beta.T
            weighted_sse = float(np.sum(w[:, None] * np.square(residual)))
            distance = beta - target
            penalty = float(np.sum(np.square(distance)))
            objective = None if np.isposinf(lam) else weighted_sse / t + lam * penalty
            stationarity = (None if np.isposinf(lam) else
                            _stationarity(beta, target, hessian, cross, lam))
            target_violation = _maximum_absolute(distance)
            arrays = (beta, target, individual, intercept, target_intercept, xm, ym)
            scalars = (weighted_sse, penalty, target_violation)
            if (not all(np.isfinite(value).all() for value in arrays)
                    or not all(np.isfinite(value) and value >= 0 for value in scalars)
                    or (objective is not None and not np.isfinite(objective))):
                raise RuntimeError("Pooled shrinkage produced nonfinite fit diagnostics")
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        raise RuntimeError("Pooled shrinkage arithmetic is not representable") from error

    def frame(values):
        """Construct an owned response-by-factor frame."""
        return pd.DataFrame(values.copy(), index=panel.response_columns.copy(deep=True),
                            columns=panel.factor_columns.copy(deep=True))

    def response_series(values):
        """Construct an owned response-indexed series."""
        return pd.Series(values.copy(), index=panel.response_columns.copy(deep=True))

    return PooledLoadingShrinkageResult(
        coefficients=frame(beta), pooled_target=frame(target), individual_wls=frame(individual),
        intercept=response_series(intercept), target_intercept=response_series(target_intercept),
        y_mean=response_series(ym),
        x_mean=pd.Series(xm.copy(), index=panel.factor_columns.copy(deep=True)),
        groups=panel.groups.copy(deep=True), date_weights=panel.date_weights.copy(deep=True),
        reg_lambda=lam, design_rank=rank, weighted_sse=weighted_sse, penalty=penalty,
        objective=objective, stationarity_residual=stationarity, target_violation=target_violation)
