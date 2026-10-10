"""Individual selected-OLS uncertainty and sign-oriented loading bounds.

A selected target c (the marginal or joint weighted OLS slope of a response on
its selected factors) receives the magnitude floor max(0, |c| - kappa). The
slack kappa is either k times the Bartlett HAC standard error of c, or the
exposure-budget slack

    kappa = sum_j (sign(c) * Gamma_j * s_j)_+ * d_j,

where d_j >= 0 is a resolved budget on the loading of a non-selected factor j,
s_j its hard sign (a free sign counts as |Gamma_j|) and Gamma_j the coefficient
of the selected factor in the weighted OLS regression of factor j on the selected
factors. By the omitted-variable identity, E[c] equals the selected loading plus
sum_j Gamma_j * beta_j on the same rows and weights, so kappa is the largest
displacement of c that the budgets admit in the direction of c.

The caller resolves declared or cluster-derived budgets before this calculation.
Bartlett HAC products retain the
original observation grid, with zero scores at missing rows. EWMA weights are
sampling weights, so each score contains w (and its outer product w squared).

References
----------
Newey and West (1987), Econometrica 55, 703-708, doi:10.2307/1913610, for the
Bartlett sandwich. Sepp (2026), "Exposure-budget floors for factor loadings
under correlated factors" (working paper), for the exposure-budget slack and
the floor-centred penalty. The sign-aware many-factor slack is stated there
without a many-factor optimality proof; this module implements the slack, not
an optimality guarantee. Data-derived budgets are experimental limits, not coverage claims.
"""
from numbers import Integral, Real
from typing import Optional

import numpy as np
import pandas as pd

from factorlasso.inference._wls import compute_wls_hac_statistics
from factorlasso.utils._ewm import _validate_span


def _validate_hac_lags(hac_lags):
    """Require a nonnegative integer bandwidth without coercing constructor input."""
    if isinstance(hac_lags, (bool, np.bool_)) or not isinstance(hac_lags, Integral) or hac_lags < 0:
        raise ValueError('expert_prior_hac_lags must be a nonnegative integer')


#: Penalty centres of a floored loading and the selectors that receive floors.
BOUND_CENTRES = ('prior', 'floor')
BOUND_SCOPES = ('all', 'named')


def _validate_exposure_budget(budget):
    """Require a response-by-factor frame of finite nonnegative budgets or NaN cells."""
    if budget is None:
        return
    if not isinstance(budget, pd.DataFrame):
        raise TypeError('expert_prior_exposure_budget must be a pandas DataFrame or None')
    if not budget.index.is_unique or not budget.columns.is_unique:
        raise ValueError('expert_prior_exposure_budget labels must be unique')
    try:
        values = budget.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('expert_prior_exposure_budget must be numeric') from error
    finite = np.isfinite(values)
    if np.isinf(values).any() or np.any(values[finite] < 0):
        raise ValueError('expert_prior_exposure_budget must be finite and nonnegative or NaN')


def _validate_expert_bound_settings(n_std, hac_lags, frequency_map, budget=None,
                                    centre='prior', scope='all'):
    """Validate the floor size or budget, its policies and per-cadence HAC bandwidths."""
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
    _validate_exposure_budget(budget)
    if not isinstance(centre, str) or centre not in BOUND_CENTRES:
        raise ValueError(f'expert_prior_bound_centre must be one of {BOUND_CENTRES!r}')
    if not isinstance(scope, str) or scope not in BOUND_SCOPES:
        raise ValueError(f'expert_prior_bound_scope must be one of {BOUND_SCOPES!r}')


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
    statistics = compute_wls_hac_statistics(xx, yy, weights, hac_lags, min_periods)
    return pd.DataFrame(dict(reference_beta=statistics.coefficients[1:],
                             se_hac=statistics.standard_error[1:],
                             observations=statistics.observations,
                             effective_n=statistics.effective_n,
                             calendar_gaps=statistics.calendar_gaps, hac_lags=hac_lags,
                             status=statistics.status), index=x.columns)



def _observation_weights(n, span):
    """EWMA sampling weights on the original grid, newest observation last."""
    if span is None:
        return np.ones(n)
    return np.power(1.0 - 2.0 / (span + 1.0), np.arange(n - 1, -1, -1))


def _weighted_correlation(a, b, w):
    """Weighted Pearson correlation of two finite vectors; NaN without variation."""
    w = w / w.sum()
    a, b = a - w @ a, b - w @ b
    va, vb = w @ (a * a), w @ (b * b)
    if va <= 0 or vb <= 0:
        return np.nan
    return float(w @ (a * b) / np.sqrt(va * vb))


def _competitor_geometry(x, y, factors, span, min_periods):
    """Slopes and correlations of the other factors on the selected ones, on the prior's rows.

    ``slopes.loc[k, j]`` is the coefficient of selected factor k in the weighted OLS regression
    of factor j on the selected factors with an intercept, using the EWMA weights of the prior
    regression and only the rows on which the response is observed. It is the coefficient of
    the omitted-variable identity for the selected slope. ``correlations.loc[k, j]`` is the
    weighted correlation of factors k and j on the same rows. Undefined entries are NaN.
    """
    factors = list(factors)
    others = [column for column in x.columns if column not in factors]
    slopes = pd.DataFrame(np.nan, index=factors, columns=others)
    correlations = slopes.copy()
    weights = _observation_weights(len(x), span) * np.isfinite(y.to_numpy(dtype=float))
    selected = x.loc[:, factors].to_numpy(dtype=float)
    for other in others:
        values = x[other].to_numpy(dtype=float)
        statistics = compute_wls_hac_statistics(selected, values, weights, 0, min_periods)
        if statistics.status == 'ok':
            slopes.loc[:, other] = statistics.coefficients[1:]
        valid = np.isfinite(selected).all(axis=1) & np.isfinite(values) & (weights > 0)
        if valid.sum() >= max(3, min_periods):
            for position, factor in enumerate(factors):
                correlations.loc[factor, other] = _weighted_correlation(
                    selected[valid, position], values[valid], weights[valid])
    return slopes, correlations


def _declared_budgets(budget, asset, factors):
    """The response's declared budgets on its non-selected factors, or None if it has none."""
    if budget is None or asset not in budget.index:
        return None
    row = budget.loc[asset].astype(float)
    if not row.notna().any():
        return None
    return row.drop(labels=[f for f in factors if f in row.index]).fillna(0.0)


def _budget_slack(slopes, budgets, signs, direction):
    """Sign-aware slack sum_j (direction * slope_j * s_j)_+ * d_j; NaN if a slope is missing."""
    total = 0.0
    for factor, budget in budgets.items():
        if budget == 0.0:
            continue
        sign = signs.get(factor, np.nan)
        if sign == 0.0:
            continue
        slope = slopes.get(factor, np.nan)
        if not np.isfinite(slope):
            return np.nan
        weight = abs(slope) if not np.isfinite(sign) else max(direction * slope * sign, 0.0)
        total += weight * budget
    return float(total)


def _competitor_diagnostics(slopes, correlations, signs, slack):
    """The most correlated permitted competitor, its slope and the slack read as its budget."""
    allowed = [f for f in correlations.index if signs.get(f, np.nan) != 0.0
               and np.isfinite(correlations[f])]
    if not allowed:
        return dict(competitor=None, competitor_slope=np.nan,
                    competitor_correlation=np.nan, implied_budget=np.nan)
    competitor = max(allowed, key=lambda f: abs(correlations[f]))
    slope = slopes.get(competitor, np.nan)
    implied = slack / abs(slope) if np.isfinite(slope) and slope != 0 else np.nan
    return dict(competitor=competitor, competitor_slope=float(slope),
                competitor_correlation=float(correlations[competitor]),
                implied_budget=float(implied))


def _compute_expert_prior_bounds(x, y, selections, raw_prior, effective_prior,
                                 manual_prior, span, n_std, hac_lags, min_periods, *,
                                 budget=None, signs=None, scope='all', named=(),
                                 geometry_cache=None):
    """Resolve each selected OLS cell's floor after scope, manual and hard-sign precedence.

    A response with at least one finite declared budget uses the exposure-budget slack; other
    responses use ``n_std`` HAC standard errors, or receive no floor when ``n_std`` is None.
    With ``scope='named'`` only responses in ``named`` receive floors.
    """
    lower = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    upper = lower.copy()
    if signs is None:
        signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    records = []
    for asset, factors in selections.items():
        if asset not in y.columns or not factors:
            continue
        declared = _declared_budgets(budget, asset, factors)
        statistics = geometry = None
        for factor in factors:
            target = raw_prior.at[asset, factor]
            row = dict(asset=asset, factor=factor, selection=tuple(factors),
                       reference_beta=target, se_hac=np.nan, floor=np.nan,
                       direction=float(np.sign(target)), n_std=n_std,
                       hac_lags=hac_lags, imposed=False, slack=np.nan, slack_source=None,
                       competitor=None, competitor_slope=np.nan,
                       competitor_correlation=np.nan, implied_budget=np.nan)
            if scope == 'named' and asset not in named:
                row['status'] = 'automatic_selection_not_floored'
            elif manual_prior is not None and pd.notna(manual_prior.at[asset, factor]):
                row['status'] = 'manual_prior_remains_soft'
            elif not np.isfinite(target) or target == 0:
                row['status'] = 'undefined_or_zero_OLS_prior'
            elif effective_prior.at[asset, factor] == 0:
                row['status'] = 'hard_constraint_has_precedence'
            elif declared is None and n_std is None:
                row['status'] = 'no_exposure_budget'
            else:
                if statistics is None:
                    statistics = compute_expert_prior_statistics(
                        x.loc[:, list(factors)], y[asset], span=span,
                        hac_lags=hac_lags, min_periods=min_periods)
                row.update(statistics.loc[factor].to_dict())
                if row['status'] == 'ok':
                    if not np.isclose(row['reference_beta'], target, rtol=1e-8, atol=1e-10):
                        raise ValueError('Expert OLS prior and bound regression disagree')
                    if geometry is None:
                        geometry = _competitor_geometry(x, y[asset], factors, span, min_periods)
                        if geometry_cache is not None:
                            geometry_cache[asset] = geometry
                    slopes = geometry[0].loc[factor]
                    asset_signs = signs.loc[asset]
                    if declared is not None:
                        slack = _budget_slack(slopes, declared, asset_signs, np.sign(target))
                        row['slack_source'] = 'budget'
                    else:
                        slack = n_std * row['se_hac']
                        row['slack_source'] = 'n_std'
                    row['slack'] = slack
                    row.update(_competitor_diagnostics(
                        slopes, geometry[1].loc[factor], asset_signs, slack))
                    if not np.isfinite(slack):
                        row['status'] = 'budget_slope_undefined'
                    else:
                        floor = max(0.0, abs(target) - slack)
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
               'observations', 'effective_n', 'calendar_gaps', 'slack', 'slack_source',
               'competitor', 'competitor_slope', 'competitor_correlation', 'implied_budget']
    return lower, upper, pd.DataFrame(records, columns=columns)
