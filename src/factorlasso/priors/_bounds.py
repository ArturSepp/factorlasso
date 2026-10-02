"""Individual selected-OLS uncertainty and sign-oriented loading bounds.

No cluster pooling enters this calculation. Bartlett HAC products retain the
original observation grid, with zero scores at missing rows. EWMA weights are
sampling weights, so each score contains w (and its outer product w squared).
"""
from numbers import Integral, Real
from typing import Optional

import numpy as np
import pandas as pd

from factorlasso.utils._ewm import _validate_span


def _validate_hac_lags(hac_lags):
    """Require a nonnegative integer bandwidth without coercing constructor input."""
    if isinstance(hac_lags, (bool, np.bool_)) or not isinstance(hac_lags, Integral) or hac_lags < 0:
        raise ValueError('expert_prior_hac_lags must be a nonnegative integer')


def _validate_expert_bound_settings(n_std, hac_lags, frequency_map):
    """Validate optional floor size and explicit per-cadence HAC bandwidths."""
    if n_std is not None and (
        isinstance(n_std, (bool, np.bool_)) or not isinstance(n_std, Real)
        or not np.isfinite(n_std) or n_std < 0
    ):
        raise ValueError('expert_prior_bound_n_std must be finite, nonnegative or None')
    _validate_hac_lags(hac_lags)
    if frequency_map is not None:
        if not isinstance(frequency_map, dict) or not frequency_map:
            raise ValueError('expert_prior_hac_lags_freq_dict must be a nonempty dict or None')
        for frequency, value in frequency_map.items():
            if not isinstance(frequency, str) or not frequency:
                raise ValueError('expert prior HAC frequency keys must be nonempty strings')
            _validate_hac_lags(value)


def compute_expert_prior_statistics(
    x: pd.DataFrame, y: pd.Series, span: Optional[float] = None,
    hac_lags: int = 0, min_periods: int = 3,
) -> pd.DataFrame:
    """Estimate selected-factor EWMA OLS slopes and Bartlett HAC standard errors.

    Parameters
    ----------
    x : pandas.DataFrame
        Selected factors only, on the original ordered observation grid. One
        column gives univariate OLS; multiple columns give joint partial slopes.
    y : pandas.Series
        One response with exactly the same unique index as x. Nonfinite rows
        are missing, not zero returns; pre-inception rows have zero score.
    span : float or None
        Squared-loss EWMA span; None gives uniform observation weights.
    hac_lags : int, default 0
        Bartlett bandwidth in observation-grid periods. Missing internal rows
        retain their positions. Zero gives heteroskedasticity-robust inference.
    min_periods : int, default 3
        Minimum usable observations, normally the estimator warmup. Positive
        residual degrees of freedom and a full-rank design are also required.

    Returns
    -------
    pandas.DataFrame
        Factor-indexed reference_beta, se_hac, observations, effective_n,
        calendar_gaps, hac_lags and status. Unidentified regressions report NaN
        coefficients/errors and an explicit status, never an invented floor.
        The covariance uses n/(n-p), where p includes the intercept. Effective
        sample size is diagnostic and does not replace n in that correction.

    Notes
    -----
    This is the weighted-score Newey-West (1987) sandwich. It describes the
    selected small regression, not confidence coverage for the full penalized
    multifactor coefficient or uncertainty in factor selection/de-smoothing.
    """
    _validate_span(span)
    _validate_hac_lags(hac_lags)
    if (isinstance(min_periods, (bool, np.bool_)) or not isinstance(min_periods, Integral)
            or min_periods < 0):
        raise ValueError('min_periods must be a nonnegative integer')
    if (not isinstance(x, pd.DataFrame) or not isinstance(y, pd.Series)
            or not x.index.equals(y.index) or not x.index.is_unique
            or not x.columns.is_unique or x.empty):
        raise ValueError('expert OLS requires nonempty aligned unique DataFrame/Series axes')
    if isinstance(x.index, pd.DatetimeIndex) and not x.index.is_monotonic_increasing:
        raise ValueError('expert OLS observation dates must be increasing')
    xx, yy = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    weights = (np.ones(len(x)) if span is None else
               np.power(1.0 - 2.0 / (span + 1.0), np.arange(len(x)-1, -1, -1)))
    valid = np.isfinite(xx).all(axis=1) & np.isfinite(yy) & (weights > 0)
    weights = np.where(valid, weights, 0.0)
    n, p = int(valid.sum()), len(x.columns) + 1
    positions = np.flatnonzero(valid)
    gaps = int(positions[-1] - positions[0] + 1 - n) if n else 0
    if n:
        weights /= weights.max()  # common rescaling leaves OLS and its sandwich unchanged
    effective_n = weights.sum()**2 / (weights @ weights) if n else 0.0
    result = pd.DataFrame(dict(reference_beta=np.nan, se_hac=np.nan,
                               observations=n, effective_n=effective_n,
                               calendar_gaps=gaps, hac_lags=hac_lags,
                               status='insufficient_observations'), index=x.columns)
    if n < max(3, min_periods) or n <= p:
        return result
    # Centre and scale before solving to protect large offsets and factor units.
    observed_x = xx[valid] - xx[valid][-1]
    observed_y = yy[valid] - yy[valid][-1]
    w = weights[valid]
    observed_x -= np.average(observed_x, axis=0, weights=w)
    observed_y -= np.average(observed_y, weights=w)
    scales = np.sqrt(np.average(observed_x**2, axis=0, weights=w))
    if np.any(scales == 0):
        result['status'] = 'rank_deficient'
        return result
    design = np.zeros((len(x), p))
    response = np.zeros(len(x))
    design[valid, 0] = 1.0
    design[valid, 1:] = observed_x / scales
    response[valid] = observed_y
    root = np.sqrt(weights)
    weighted_design = root[:, None] * design
    u, singular, vt = np.linalg.svd(weighted_design, full_matrices=False)
    if singular[-1] <= singular[0] * max(weighted_design.shape) * np.finfo(float).eps:
        result['status'] = 'rank_deficient'
        return result
    coefficients = vt.T @ ((u.T @ (root * response)) / singular)
    bread = (vt.T / singular**2) @ vt
    residuals = response - design @ coefficients
    scores = weights[:, None] * design * residuals[:, None]
    # Direct influence form of bread @ HAC(scores) @ bread; no compression of gaps.
    influence = scores @ bread
    variance = np.sum(influence**2, axis=0)
    for lag in range(1, min(hac_lags, len(x)-1) + 1):
        variance += 2 * (1 - lag / (hac_lags + 1)) * np.sum(
            influence[lag:] * influence[:-lag], axis=0)
    variance *= n / (n-p)
    result['reference_beta'] = coefficients[1:] / scales
    result['se_hac'] = np.sqrt(np.maximum(variance[1:], 0.0)) / scales
    result['status'] = 'ok'
    return result


def _compute_expert_prior_bounds(x, y, selections, raw_prior, effective_prior,
                                 manual_prior, span, n_std, hac_lags, min_periods):
    """Resolve each selected OLS cell's floor after manual and hard-sign precedence."""
    lower = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    upper = lower.copy()
    records = []
    for asset, factors in selections.items():
        if asset not in y.columns or not factors:
            continue
        statistics = None
        for factor in factors:
            target = raw_prior.at[asset, factor]
            row = dict(asset=asset, factor=factor, selection=tuple(factors),
                       reference_beta=target, se_hac=np.nan, floor=np.nan,
                       direction=float(np.sign(target)), n_std=n_std,
                       hac_lags=hac_lags, imposed=False)
            if manual_prior is not None and pd.notna(manual_prior.at[asset, factor]):
                row['status'] = 'manual_prior_remains_soft'
            elif not np.isfinite(target) or target == 0:
                row['status'] = 'undefined_or_zero_OLS_prior'
            elif effective_prior.at[asset, factor] == 0:
                row['status'] = 'hard_constraint_has_precedence'
            else:
                if statistics is None:
                    statistics = compute_expert_prior_statistics(
                        x.loc[:, list(factors)], y[asset], span=span,
                        hac_lags=hac_lags, min_periods=min_periods)
                row.update(statistics.loc[factor].to_dict())
                if row['status'] == 'ok':
                    if not np.isclose(row['reference_beta'], target, rtol=1e-8, atol=1e-10):
                        raise ValueError('Expert OLS prior and bound regression disagree')
                    floor = max(0.0, abs(target) - n_std * row['se_hac'])
                    row.update(floor=floor, imposed=floor > 0,
                               status='imposed' if floor > 0 else 'zero_floor')
                    if floor > 0:
                        if target > 0:
                            lower.at[asset, factor] = floor
                        else:
                            upper.at[asset, factor] = -floor
            records.append(row)
    columns = ['asset', 'factor', 'selection', 'reference_beta', 'se_hac', 'floor',
               'direction', 'n_std', 'hac_lags', 'imposed', 'status',
               'observations', 'effective_n', 'calendar_gaps']
    return lower, upper, pd.DataFrame(records, columns=columns)
