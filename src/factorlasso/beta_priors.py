"""Internal weighted-OLS prior construction for sparse factor models.

Selection is a package heuristic: the factor with highest weighted
univariate R-squared receives its full slope; all other automatic priors
are zero. These are empirical penalty centres, not posterior means or
fixed loadings.

Weighted regressions include an intercept on the original observations. They
use the LASSO loss span, without its rolling-mean preprocessing or cluster
pooling. The moments are evaluated together with pairwise finite masks.
"""
from typing import Optional, Tuple

import numpy as np

from factorlasso.ewm_utils import _validate_span, compute_expanding_power


def _validate_prior_selection_type(prior_selection_type: str) -> None:
    """Reject unknown OLS-prior selectors without coercing constructor parameters."""
    allowed = ('highest_r2',)
    if not isinstance(prior_selection_type, str) or prior_selection_type not in allowed:
        raise ValueError(f'prior_selection_type must be one of {allowed!r}')


def _centred_finite_values(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shift by a finite column value to stabilise weighted central moments."""
    finite = np.isfinite(values)
    last = len(values) - 1 - np.argmax(finite[::-1], axis=0)
    origin = values[last, np.arange(values.shape[1])]
    origin = np.where(finite.any(axis=0), origin, 0.0)
    return np.where(finite, values - origin, 0.0), finite.astype(float)


def _compute_ols_prior(
    x: np.ndarray,
    y: np.ndarray,
    span: Optional[float],
    min_periods: int = 3,
    prior_selection_type: str = 'highest_r2',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return response-by-factor slopes, weighted R-squared and raw priors.

    Parameters
    ----------
    x, y : ndarray
        Aligned original observations, shaped T by M and T by N. Missing rows
        retain their time-grid age; each pair uses its own finite observations.
    span : float or None
        Effective LASSO squared-loss span. None gives equal weights. Span one
        has only one positive-weight row and hence no estimable slope.
    min_periods : int, default 3
        Minimum pairwise finite observations, normally at least model warmup.
    prior_selection_type : str, default 'highest_r2'
        The only supported selector. Assign the full slope to the factor with
        highest weighted R-squared. R-squared equals one minus weighted residual
        sum of squares divided by the weighted sum of squares about the response
        mean.

    Returns
    -------
    beta, r_squared, prior : ndarray
        N by M matrices. Undefined statistics are NaN; undefined priors are
        zero. Ties use input factor order. The highest-R-squared winner
        receives its full slope. Sign constraints and explicit overrides
        are applied by the caller later.
    """
    _validate_span(span)
    _validate_prior_selection_type(prior_selection_type)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]
            or not x.shape[0] or not x.shape[1] or not y.shape[1]):
        raise ValueError('OLS priors require nonempty aligned two-dimensional panels')
    x0, mx = _centred_finite_values(x)
    y0, my = _centred_finite_values(y)
    if span is None:
        weights = np.ones(len(x))
    elif span == 1.0:
        weights = np.zeros(len(x))
        weights[-1] = 1.0
    else:
        weights = compute_expanding_power(
            n=len(x), power_lambda=1.0 - 2.0 / (span + 1.0), reverse_columns=True)
    w = weights[:, None]
    sw = my.T @ (w * mx)
    sx = my.T @ (w * x0)
    sy = y0.T @ (w * mx)
    sxx = my.T @ (w * x0 * x0)
    syy = (y0 * y0).T @ (w * mx)
    sxy = y0.T @ (w * x0)
    nobs = my.T @ mx
    denominator = np.where(sw > 0.0, sw, 1.0)
    vx = sxx - sx * sx / denominator
    vy = syy - sy * sy / denominator
    cov = sxy - sx * sy / denominator
    tolerance = 64.0 * np.finfo(float).eps
    valid = ((nobs >= max(3, min_periods)) & (sw > 0.0)
             & (vx > tolerance * sxx) & (vy > tolerance * syy))
    beta = np.divide(cov, vx, out=np.full_like(cov, np.nan), where=valid)
    r_squared = np.divide(cov * cov, vx * vy, out=np.full_like(cov, np.nan), where=valid)
    r_squared = np.clip(r_squared, 0.0, 1.0)
    prior = np.zeros_like(beta)
    rows = np.flatnonzero(valid.any(axis=1))
    winners = np.argmax(np.where(valid, r_squared, -np.inf), axis=1)[rows]
    prior[rows, winners] = beta[rows, winners]
    return beta, r_squared, prior


def _compute_joint_ols_prior(
    x: np.ndarray, y: np.ndarray, span: Optional[float], min_periods: int = 3,
) -> np.ndarray:
    """Estimate joint slopes with intercept, loss-span weights and complete finite rows.

    Age weights retain the original observation grid when rows are missing.
    Undefined or rank-deficient joint regressions yield neutral zero centres.
    Single-factor requests retain the existing pairwise-moment implementation.
    """
    _validate_span(span)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not x.shape[1]:
        raise ValueError('joint OLS priors require aligned X and one response')
    neutral = np.zeros(x.shape[1])
    if span is None:
        weights = np.ones(len(x))
    elif span == 1.0:
        return neutral
    else:
        weights = compute_expanding_power(
            n=len(x), power_lambda=1.0 - 2.0 / (span + 1.0), reverse_columns=True)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y) & (weights > 0.0)
    if valid.sum() < max(min_periods, x.shape[1] + 1, 3):
        return neutral
    xx, yy, ww = x[valid], y[valid], weights[valid]
    # Shift before centring for stability on large-offset observations.
    xx, yy = xx - xx[-1], yy - yy[-1]
    xx -= np.average(xx, axis=0, weights=ww)
    yy -= np.average(yy, weights=ww)
    root = np.sqrt(ww / ww.sum())
    design = xx * root[:, None]
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0.0):
        return neutral
    slopes, _, rank, _ = np.linalg.lstsq(design / scales, yy * root, rcond=None)
    slopes = slopes / scales
    if rank != x.shape[1] or not np.isfinite(slopes).all():
        return neutral
    return slopes


def _zero_incompatible_priors(
    prior: np.ndarray, signs: Optional[np.ndarray], nonneg: bool = False,
) -> np.ndarray:
    """Zero forbidden prior cells after selection, without selecting replacements."""
    if signs is None:
        return np.maximum(prior, 0.0) if nonneg else prior.copy()
    blocked = ((signs == 0.0) | ((signs > 0.0) & (prior < 0.0))
               | ((signs < 0.0) & (prior > 0.0)))
    return np.where(blocked, 0.0, prior)
