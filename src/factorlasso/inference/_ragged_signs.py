"""Conditional observed-cell pooling on asynchronous response histories.

White (1980), Econometrica 48, 817--838, doi:10.2307/1912934.
Common-intercept observed-cell loss, date-count weighting and the date-based
HC1 correction are implementation choices. A changing observed group mix may
change the estimated target; no missing-at-random or selection calibration is
claimed. This is not an asset-fixed-effects or serial-HAC estimator.
"""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from factorlasso.inference._pooled_signs import _validate_gate_policy
from factorlasso.inference._wls import compute_wls_hac_statistics
from factorlasso.utils._pooled_inputs import _validate_pooled_inputs


@dataclass(frozen=True)
class RaggedPooledSignGateResult:
    """Conditional sign evidence with per-group support on the original grid.

    Attributes
    ----------
    slopes, standard_errors, t_stats, signs, restrictions : pandas.DataFrame
        Response-by-factor evidence. Identified signs are -1, 0 or +1.
        Restrictions follow the selected action. Explicitly released
        unidentified groups have NaN in every evidence/restriction cell.
    groups, date_weights : pandas.Series
        Copied groups and positive mean-one original-grid date weights.
    observation_mask : pandas.DataFrame
        True only for an observed response with all factors observed that date.
        Missing cells are never fitted as zero returns.
    diagnostics : pandas.DataFrame
        First-appearance group order. Records status, usable dates, residual
        degrees of freedom, observed response cells, complete shared dates,
        effective loss-weight size, HC1 correction and original-grid gaps.
        Loss weights include the observed-member count; HC1 uses dates, not
        response cells or effective size.
    threshold, action, on_unidentified : float, str, str
        Requested threshold, weak-evidence action and explicit failure policy.
    evidence, variance_estimator : str
        Fixed values ``conditional`` and ``HC1_date``.
    fit_intercept : bool
        True: a shared unpenalized intercept within each pooled group.
    hac_lags : int
        Zero: contemporaneous date dependence, not serial-dependence correction.

    Notes
    -----
    Pandas objects are owned copies, but are not deeply immutable.
    """

    slopes: pd.DataFrame
    standard_errors: pd.DataFrame
    t_stats: pd.DataFrame
    signs: pd.DataFrame
    restrictions: pd.DataFrame
    groups: pd.Series
    date_weights: pd.Series
    observation_mask: pd.DataFrame
    diagnostics: pd.DataFrame
    threshold: float
    action: str
    on_unidentified: str
    evidence: str
    variance_estimator: str
    fit_intercept: bool
    hac_lags: int


def compute_ragged_pooled_sign_gate(
    x: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    *,
    weights: pd.Series | None = None,
    threshold: float = 2.0,
    action: str = 'exclude',
    on_unidentified: str = 'raise',
) -> RaggedPooledSignGateResult:
    """Fit joint group slopes to observed cells, with original-date HC1 scores.

    Parameters
    ----------
    x, y : pandas.DataFrame
        Real numeric factor and response panels with exactly matching unique,
        nonmissing ordered indexes and columns. NaN denotes a missing cell;
        infinity is invalid. A date with any missing factor contributes no
        cells to that group's joint regression. No alignment or imputation.
    groups : pandas.Series
        Nonmissing group IDs in exactly y.columns order, as for the complete
        ``compute_pooled_sign_gate`` API. Groups are supplied, not learned here.
    weights : pandas.Series, optional
        Strictly positive finite original-grid date weights. None gives equal
        weights. Subsetting never restarts the EWMA calendar. Internally each
        date's loss weight is multiplied by its observed member count.
    threshold : float, default 2.0
        Finite nonnegative nonboolean absolute t-statistic threshold.
    action : {'exclude', 'release', 'zero_only'}, default 'exclude'
        Same identified-cell restriction mapping as the complete-panel gate.
    on_unidentified : {'raise', 'release'}, default 'raise'
        Raise if a group lacks T > M+1 usable dates or design rank. Explicit
        release returns NaN evidence/restrictions, records the status and warns.
        It never manufactures weak zeros or falls back to marginal evidence.
        Arithmetic failures still raise under either policy.

    Returns
    -------
    RaggedPooledSignGateResult
        Labeled evidence, original observed-cell mask and per-group diagnostics.

    Notes
    -----
    Let n_t count observed members, and ybar_t their arithmetic mean. Regressing
    ybar_t on all factors with loss weights w_t*n_t gives exactly the shared
    intercept/slopes of observed-cell stacked WLS. Its score at each date equals
    the sum of the member scores, retaining contemporaneous covariance. The
    existing WLS engine supplies HC1 with T_g/(T_g-M-1), where T_g counts usable
    dates, not cells. Balanced panels agree with the complete-panel API.

    A changing observed-member mix changes this observed-cell estimand; the
    function does not correct informative missingness or individual intercepts.
    Caller-owned point-in-time clustering, publication availability and units
    remain essential. Explicit restrictions can be passed to LassoModel with
    automatic signs disabled; this additive API does not change its defaults.
    """
    panel = _validate_pooled_inputs(x, y, groups, weights, allow_missing=True)
    threshold = _validate_gate_policy(threshold, action)
    if not isinstance(on_unidentified, str) or on_unidentified not in ('raise', 'release'):
        raise ValueError('on_unidentified must be raise or release')
    t, m = panel.x.shape
    shape = (panel.y.shape[1], m)
    slopes, errors, stats, signs = [np.full(shape, np.nan) for _ in range(4)]
    finite_x = np.isfinite(panel.x).all(axis=1)
    observed = np.isfinite(panel.y) & finite_x[:, None]
    w = panel.date_weights.to_numpy(copy=True)
    records, released = [], []
    for members in panel.members:
        group = panel.groups.iloc[members[0]]
        mask = observed[:, members]
        counts = mask.sum(axis=1)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                # Zero is only an accumulator for absent cells; count-zero rows
                # retain a missing response and zero loss weight in the engine.
                scaled = np.divide(panel.y[:, members], counts[:, None],
                                   out=np.zeros((t, len(members))), where=mask)
                mean_response = scaled.sum(axis=1)
                mean_response[counts == 0] = np.nan
                fitted = compute_wls_hac_statistics(
                    panel.x, mean_response, weights=w*counts,
                    hac_lags=0, fit_intercept=True,
                )
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise RuntimeError('conditional observed-cell WLS arithmetic failed') from exc
        n = fitted.observations
        records.append(dict(
            group=group, status=fitted.status, observations=n, residual_df=n-m-1,
            response_observations=int(mask.sum()), complete_dates=int(mask.all(axis=1).sum()),
            effective_n=fitted.effective_n,
            hc1_correction=n/(n-m-1) if n > m+1 else np.nan,
            calendar_gaps=fitted.calendar_gaps,
        ))
        if fitted.status != 'ok':
            if on_unidentified == 'raise':
                raise ValueError(
                    f'conditional gate unidentified for group {group!r}: {fitted.status}')
            released.append((group, fitted.status))
            continue
        beta, se = fitted.coefficients[1:], fitted.standard_error[1:]
        if (not np.isfinite(beta).all() or not np.isfinite(se).all() or np.any(se < 0)):
            raise RuntimeError('conditional observed-cell WLS returned invalid evidence')
        statistic = np.zeros(m)
        with np.errstate(over='ignore', divide='ignore'):
            np.divide(beta, se, out=statistic, where=se > 0)
        if not np.isfinite(statistic[se > 0]).all():
            raise RuntimeError('conditional observed-cell t-statistic is not representable')
        exact = (se == 0) & (beta != 0)
        statistic[exact] = np.copysign(np.inf, beta[exact])
        detected = np.where((beta != 0) & (np.abs(statistic) >= threshold), np.sign(beta), 0.)
        slopes[members], errors[members] = beta, se
        stats[members], signs[members] = statistic, detected
    restrictions = signs.copy()
    if action == 'release':
        restrictions[signs == 0] = np.nan
    elif action == 'zero_only':
        restrictions[np.abs(signs) == 1] = np.nan
    if released:
        warnings.warn(f'Conditional sign gate unidentified groups {released!r}; automatic '
                      'restrictions released; no marginal fallback', UserWarning, stacklevel=2)

    def frame(values):
        """Return independently owned response-by-factor evidence."""
        return pd.DataFrame(values.copy(), index=panel.response_columns.copy(deep=True),
                            columns=panel.factor_columns.copy(deep=True))

    return RaggedPooledSignGateResult(
        slopes=frame(slopes), standard_errors=frame(errors), t_stats=frame(stats),
        signs=frame(signs), restrictions=frame(restrictions), groups=panel.groups,
        date_weights=panel.date_weights,
        observation_mask=pd.DataFrame(observed.copy(), index=panel.row_index.copy(deep=True),
                                      columns=panel.response_columns.copy(deep=True)),
        diagnostics=pd.DataFrame(records).set_index('group'), threshold=threshold,
        action=action, on_unidentified=on_unidentified, evidence='conditional',
        variance_estimator='HC1_date', fit_intercept=True, hac_lags=0,
    )
