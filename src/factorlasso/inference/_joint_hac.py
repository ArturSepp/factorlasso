"""Joint coefficient HAC and first-order uncertainty for prespecified targets.

Newey and West (1987), Econometrica 55, 703-708, doi:10.2307/1913610,
provide the Bartlett covariance construction. The delta method propagates a
target Jacobian through that covariance. Calendar aggregation and finite-sample
score corrections are explicit implementation choices, not coverage guarantees.
No factor selection, shrinkage adjustment or future-return prediction is implied.
"""
from numbers import Integral

import numpy as np
import pandas as pd

from factorlasso.inference._intervals import linear_confidence_intervals
from factorlasso.inference._sandwich import lag_covariance
from factorlasso.inference._validation import _confidence


def aggregate_influence_calendar(influence, *, frequency='Q', start=None, end=None):
    """Sum dated coefficient influences on a complete coarser calendar.

    Parameters
    ----------
    influence : pandas.DataFrame
        Finite influence columns on a sorted unique, timezone-naive DatetimeIndex
        or PeriodIndex. Include zero rows at missing observations of identified
        equations; do not substitute zeros for unidentified coefficients.
    frequency : str, default 'Q'
        Pandas period frequency for aggregation. A monthly/quarterly system can
        sum monthly scores within quarters and place quarterly scores at endpoints.
    start, end : period-like, optional
        Inclusive outer bounds of the resulting grid. Defaults to observed bounds.
        Bounds may extend the calendar with zeros, but cannot discard input rows.

    Returns
    -------
    pandas.DataFrame
        Influence sums on a regular PeriodIndex, with zero-filled empty periods.
        Columns, units and existing small-sample corrections are preserved.

    Notes
    -----
    Inputs are coefficient influences, not returns. Coarsening changes the HAC
    convention: one quarterly lag is not equivalent to three monthly lags.
    The caller chooses a common grid before joining equations with ragged histories.
    """
    if (not isinstance(influence, pd.DataFrame) or influence.empty
            or not isinstance(influence.index, (pd.DatetimeIndex, pd.PeriodIndex))
            or not influence.index.is_unique or not influence.index.is_monotonic_increasing
            or influence.index.hasnans or not influence.columns.is_unique
            or not np.isfinite(influence.to_numpy(dtype=float)).all()):
        raise ValueError('influence requires finite columns on a nonempty ordered unique calendar')
    index = influence.index
    if isinstance(index, pd.DatetimeIndex):
        if index.tz is not None:
            raise ValueError('calendar must be timezone-naive')
        groups = index.to_period(frequency)
    else:
        groups = index.asfreq(frequency, how='end')
    first = groups.min() if start is None else pd.Period(start, freq=frequency)
    last = groups.max() if end is None else pd.Period(end, freq=frequency)
    if first > groups.min() or last < groups.max():
        raise ValueError('calendar bounds cannot discard input observations')
    return influence.groupby(groups).sum().reindex(
        pd.period_range(first, last, freq=frequency), fill_value=0.)


def compute_hac_covariance(influence, hac_lags=0):
    """Compute joint Bartlett covariance from aligned coefficient influences.

    Parameters
    ----------
    influence : array-like, shape (T, P)
        Finite coefficient influences on a regular common grid. P may be zero.
        The WLS statistics API's influence output already includes n/(n-P)
        through its square-root score correction; no extra correction is applied.
    hac_lags : int, default 0
        Nonnegative Bartlett bandwidth in common-grid periods.

    Returns
    -------
    numpy.ndarray, shape (P, P)
        Joint estimation-error covariance with all cross-equation terms retained.
        Scores are not demeaned and no division by the grid length is applied.
    """
    values = np.asarray(influence, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('influence must be a finite (T, P) array with T > 0')
    if (isinstance(hac_lags, (bool, np.bool_)) or not isinstance(hac_lags, Integral)
            or hac_lags < 0):
        raise ValueError('hac_lags must be a nonnegative integer')
    return lag_covariance(values, hac_lags)


def delta_method_hac_intervals(estimate, influence, jacobian, *, hac_lags=0,
                               confidence_levels=(.95,)):
    """Propagate joint coefficient uncertainty into prespecified target intervals.

    Parameters
    ----------
    estimate : array-like, shape (K,)
        Finite plug-in target values, e.g. fixed portfolio sums or products of
        fitted slopes. These are evaluated by the caller in the intended units.
    influence : array-like, shape (T, P)
        Aligned influences for every required coefficient, including every stage
        of a conversion regression. Unknown coefficients must not be zero-filled.
    jacobian : array-like, shape (K, P)
        Derivatives of the targets with respect to those coefficients, in the
        same order. For a product a*b the two entries are b and a; omitting either
        loses that stage's uncertainty and its cross-covariance with the other.
    hac_lags : int, default 0
        Nonnegative Bartlett lag count on the supplied common grid.
    confidence_levels : iterable of float, default (.95,)
        Distinct pointwise normal confidence levels strictly between zero and one.
        Request ``(.8, .95)`` explicitly to include both 80% and 95% intervals.

    Returns
    -------
    dict
        ``estimate``, transformed ``influence``, target ``covariance`` and
        ``intervals`` keyed by confidence level. Interval entries have the same
        schema as linear_confidence_intervals. Results are first-order normal
        approximations for fixed models, not intervals around penalised estimates.

    Notes
    -----
    A vanishing derivative can produce zero first-order variance without making
    a nonlinear target known. Small effective samples, weak identification and
    selected models require separate assessment; this API does not validate them.
    """
    values = np.asarray(estimate, dtype=float)
    scores = np.asarray(influence, dtype=float)
    gradient = np.asarray(jacobian, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('estimate must be a nonempty finite vector')
    if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('influence must be a finite (T, P) array with T > 0')
    if gradient.shape != (len(values), scores.shape[1]) or not np.isfinite(gradient).all():
        raise ValueError('jacobian must be finite with shape (K, P)')
    levels = tuple(_confidence(level) for level in confidence_levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError('confidence_levels must be nonempty and distinct')
    target_influence = scores @ gradient.T
    covariance = compute_hac_covariance(target_influence, hac_lags)
    intervals = {level: linear_confidence_intervals(values, covariance, confidence=level)
                 for level in levels}
    return dict(estimate=values.copy(), influence=target_influence, covariance=covariance,
                intervals=intervals)
