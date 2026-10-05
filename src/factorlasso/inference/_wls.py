"""Weighted least-squares statistics on a possibly incomplete observation grid.

Newey and West (1987), Econometrica 55, 703-708, doi:10.2307/1913610.
Observation weighting, zero scores at gaps and n/(n-p) are implementation choices.
HAC standard errors do not by themselves provide finite-sample Gaussian coverage.
"""
from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import numpy as np

from factorlasso.inference._sandwich import lag_covariance


@dataclass(frozen=True)
class WlsHacStatistics:
    """Coefficients, HAC uncertainty and support for one weighted regression.

    Coefficients and standard_error have length P, including a leading intercept
    when requested. Covariance is either a P by P matrix or None. Unidentified
    fits return NaN arrays with an explicit status. Effective_n describes weight
    concentration, not serial dependence or residual degrees of freedom.
    """

    coefficients: np.ndarray
    standard_error: np.ndarray
    covariance: Optional[np.ndarray]
    observations: int
    effective_n: float
    calendar_gaps: int
    hac_lags: int
    status: str


def compute_wls_hac_statistics(x, y, weights=None, hac_lags=0, min_periods=3, *,
                               fit_intercept=True, return_covariance=False):
    """Estimate weighted regression coefficients and original-grid Bartlett HAC.

    Parameters
    ----------
    x : array-like, shape (T, M)
        Regressors without an intercept when fit_intercept is True. Zero columns
        are allowed for an intercept-only mean. Nonfinite rows are missing.
    y : array-like, shape (T,)
        Response in original row order; nonfinite observations are missing.
    weights : array-like, shape (T,), optional
        Finite nonnegative loss weights. None gives equal weights. Missing rows
        receive zero weight; a common positive weight rescaling cancels.
    hac_lags : int, default 0
        Nonnegative Bartlett bandwidth in original observation-grid periods.
    min_periods : int, default 3
        Minimum usable count; at least three observations and n > P are required.
    fit_intercept : bool, default True
        Add a leading intercept and center regressors for numerical stability.
    return_covariance : bool, default False
        Also return the full coefficient covariance. Neither mode builds a dense
        T by T matrix; finite-sample correction is n/(n-P).

    Returns
    -------
    WlsHacStatistics
        Coefficients, standard errors, optional covariance and support status.
        No interval calibration or adjustment for factor selection is implied.
    """
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xx.ndim != 2 or not len(xx) or yy.shape != (len(xx),):
        raise ValueError('x and y must have shapes (T, M) and (T,) with T > 0')
    for value, name in ((hac_lags, 'hac_lags'), (min_periods, 'min_periods')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    for value, name in ((fit_intercept, 'fit_intercept'), (return_covariance, 'return_covariance')):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be boolean')
    p = xx.shape[1] + int(fit_intercept)
    if not p:
        raise ValueError('at least one coefficient is required')
    weights = np.ones(len(xx)) if weights is None else np.asarray(weights, dtype=float)
    if (weights.shape != (len(xx),) or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError('weights must be finite, nonnegative and match rows')
    valid = np.isfinite(xx).all(axis=1) & np.isfinite(yy) & (weights > 0)
    weights = np.where(valid, weights, 0.0)
    n = int(valid.sum())
    positions = np.flatnonzero(valid)
    gaps = int(positions[-1] - positions[0] + 1 - n) if n else 0
    if n:
        weights /= weights.max()
    effective_n = float(weights.sum()**2 / (weights @ weights)) if n else 0.0

    def unidentified(status):
        """Return support diagnostics without inventing unidentified coefficients."""
        covariance = np.full((p, p), np.nan) if return_covariance else None
        return WlsHacStatistics(np.full(p, np.nan), np.full(p, np.nan), covariance,
                                n, effective_n, gaps, hac_lags, status)

    if n < max(3, min_periods) or n <= p:
        return unidentified('insufficient_observations')
    w = weights[valid]
    observed_x, observed_y = xx[valid].copy(), yy[valid].copy()
    x_mean, y_mean = np.zeros(xx.shape[1]), 0.
    if fit_intercept:
        observed_x -= xx[valid][-1]
        observed_y -= yy[valid][-1]
        x_offset = np.average(observed_x, axis=0, weights=w)
        y_offset = np.average(observed_y, weights=w)
        x_mean = xx[valid][-1] + x_offset
        y_mean = yy[valid][-1] + y_offset
        observed_x -= x_offset
        observed_y -= y_offset
    scales = np.sqrt(np.average(observed_x**2, axis=0, weights=w))
    if np.any(scales == 0):
        return unidentified('rank_deficient')
    design, response = np.zeros((len(xx), p)), np.zeros(len(xx))
    start = int(fit_intercept)
    if fit_intercept:
        design[valid, 0] = 1.
    design[valid, start:] = observed_x/scales
    response[valid] = observed_y
    root = np.sqrt(weights)
    weighted_design = root[:, None]*design
    u, singular, vt = np.linalg.svd(weighted_design, full_matrices=False)
    if singular[-1] <= singular[0]*max(weighted_design.shape)*np.finfo(float).eps:
        return unidentified('rank_deficient')
    coefficients = vt.T @ ((u.T @ (root*response))/singular)
    bread = (vt.T/singular**2) @ vt
    residuals = response - design @ coefficients
    scores = weights[:, None]*design*residuals[:, None]
    influence = scores @ bread
    variance = lag_covariance(influence, hac_lags, diagonal=True)*n/(n-p)
    estimates = coefficients.copy()
    estimates[start:] /= scales
    standard_error = np.sqrt(np.maximum(variance, 0.))
    standard_error[start:] /= scales
    if fit_intercept:
        estimates[0] += y_mean - x_mean @ estimates[1:]
        intercept_influence = influence[:, 0] - influence[:, 1:] @ (x_mean/scales)
        intercept_variance = lag_covariance(intercept_influence[:, None], hac_lags,
                                            diagonal=True)[0]*n/(n-p)
        standard_error[0] = np.sqrt(max(intercept_variance, 0.))
    covariance = None
    if return_covariance:
        original_influence = influence.copy()
        original_influence[:, start:] /= scales
        if fit_intercept:
            original_influence[:, 0] = intercept_influence
        covariance = lag_covariance(original_influence, hac_lags)*n/(n-p)
    return WlsHacStatistics(estimates, standard_error, covariance, n, effective_n,
                            gaps, hac_lags, 'ok')
