"""Multi-factor pooled sign screening from complete labeled panels.

The gate of the sign-guided Lasso: pooled multi-factor (WLS, date-score HC1)
slopes within response groups, thresholded into explicit restrictions.

References
----------
White, H. (1980), "A Heteroskedasticity-Consistent Covariance Matrix Estimator
and a Direct Test for Heteroskedasticity", Econometrica 48, 817--838,
doi:10.2307/1912934.

Arithmetic response pooling, positive date weights, the T/(T-M-1) HC1
correction and threshold/action mapping are implementation choices. The
screen is not calibrated for learned groups, serial dependence or selection;
a group-mean loading need not have every member's sign or support.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from factorlasso.inference._wls import compute_wls_hac_statistics
from factorlasso.utils._pooled_inputs import _validate_pooled_inputs


@dataclass(frozen=True)
class PooledSignGateResult:
    """Inspect a pooled multi-factor gate in the supplied analysis coordinates.

    Attributes
    ----------
    slopes, standard_errors, t_stats, signs, restrictions : pandas.DataFrame
        Responses by factors in input column order. Detected signs are -1, 0,
        +1. Restrictions use -1/+1 for half-lines, zero for a hard zero, and
        NaN for unrestricted cells in an explicit-only sign composition.
    groups, date_weights : pandas.Series
        Copied response groups and strictly positive mean-one row weights.
    threshold : float
        Nonzero slopes pass at absolute t-statistic greater than or equal to
        this threshold.
    action : str
        One of ``exclude``, ``release`` or ``zero_only``.
    evidence, variance_estimator : str
        Fixed values ``conditional`` and ``HC1_date``.
    hac_lags : int
        Zero: only same-date score products enter the covariance.
    fit_intercept : bool
        True: each pooled regression includes an unpenalized intercept.
    observations, residual_df : int
        T and T-M-1 for the complete joint design.
    effective_n, hc1_correction : float
        Weight concentration (sum w)^2 / sum(w^2), and T/(T-M-1).
        Effective_n is metadata, not the HC1 degrees-of-freedom correction.

    Notes
    -----
    The factory returns owned copies. The dataclass is frozen, but its pandas
    contents are mutable; this is not a deeply immutable result object.
    """

    slopes: pd.DataFrame
    standard_errors: pd.DataFrame
    t_stats: pd.DataFrame
    signs: pd.DataFrame
    restrictions: pd.DataFrame
    groups: pd.Series
    date_weights: pd.Series
    threshold: float
    action: str
    evidence: str
    variance_estimator: str
    hac_lags: int
    fit_intercept: bool
    observations: int
    residual_df: int
    effective_n: float
    hc1_correction: float


def _validate_gate_policy(threshold, action):
    """Validate the shared threshold and weak-cell action without changing data."""
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise ValueError('threshold must be a finite nonnegative nonboolean real')
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('threshold must be a representable finite nonnegative real') from exc
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be a finite nonnegative nonboolean real')
    if not isinstance(action, str) or action not in ('exclude', 'release', 'zero_only'):
        raise ValueError('action must be exclude, release or zero_only')
    return threshold


def compute_pooled_sign_gate(
    x: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    *,
    weights: pd.Series | None = None,
    threshold: float = 2.0,
    action: str = 'exclude',
) -> PooledSignGateResult:
    """Screen group-mean loadings using joint WLS and date-score HC1.

    This is the gate step of the sign-guided Lasso: pooled multi-factor
    signs with explicit exclusion, release or zero-only restrictions.
    ``compute_sign_guided_gate`` is the same function under that name.

    Parameters
    ----------
    x, y : pandas.DataFrame
        Complete real finite panels of shape (T, M) and (T, N), with identical
        unique nonmissing ordered row indexes and unique nonmissing columns.
        No alignment, deletion, imputation or standardization is performed.
    groups : pandas.Series
        Response groups with index exactly equal to y.columns. IDs are
        nonmissing strings or finite nonboolean reals. First appearance and
        identity are preserved; numeric 1 and string "1" are different IDs.
    weights : pandas.Series, optional
        Strictly positive finite date weights indexed exactly as x.index.
        Copied and normalized to mean one; None means equal weights. Supply
        original-date subset weights explicitly when splitting a sample.
    threshold : float, default 2.0
        Finite nonnegative nonboolean threshold. Exact zero slopes remain weak
        even at zero. A nonzero slope passes when abs(t) >= threshold.
    action : {'exclude', 'release', 'zero_only'}, default 'exclude'
        Exclude imposes passing directions and weak zeros; release imposes
        passing directions and leaves weak cells unrestricted; zero_only
        imposes weak zeros and leaves passing cells unrestricted.

    Returns
    -------
    PooledSignGateResult
        Labeled response-by-factor evidence and explicit restrictions, copied
        groups/weights and fixed HC1 metadata in the supplied coordinates.

    Raises
    ------
    TypeError
        An input is not the required pandas container.
    ValueError
        Data, labels, weights or settings are invalid; T <= M+1; or the joint
        weighted design fails the existing WLS engine's scaled-SVD rank rule.
    RuntimeError
        Valid finite inputs lead to unrepresentable arithmetic or nonfinite
        coefficients/standard errors. No single-factor or neutral fallback is used.

    Notes
    -----
    Pool arithmetic mean responses before squaring date scores to retain
    contemporaneous member covariance. The intercept is unpenalized and the
    HC1 correction is T/(T-M-1), not a Kish effective-size correction. For
    exactly zero standard error, nonzero slope gives signed infinite t and
    zero slope gives t=0; no standard-error floor is introduced.

    To pass ``restrictions`` into LassoModel, set auto_sign_constraints=False:
    explicit NaNs otherwise fall back to its configured automatic sign
    layer. Neither learning groups nor using EW weights supplies significance
    calibration or a guarantee for each member's loading. The caller
    owns training-prefix selection, information availability and units.
    """
    panel = _validate_pooled_inputs(x, y, groups, weights)
    threshold = _validate_gate_policy(threshold, action)
    t, m = panel.x.shape
    if t <= m + 1:
        raise ValueError('conditional gate requires T > M+1 residual degrees of freedom')
    w = panel.date_weights.to_numpy(copy=True)
    shape = (panel.y.shape[1], m)
    slopes, errors, stats, signs = [np.zeros(shape, dtype=float) for _ in range(4)]
    for members in panel.members:
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                mean_response = panel.y[:, members].mean(axis=1)
                fitted = compute_wls_hac_statistics(
                    panel.x, mean_response, weights=w.copy(), hac_lags=0, fit_intercept=True,
                )
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise RuntimeError('conditional pooled WLS arithmetic failed') from exc
        if fitted.status != 'ok':
            raise ValueError(f'conditional pooled WLS failed: {fitted.status}')
        beta, se = fitted.coefficients[1:], fitted.standard_error[1:]
        if (not np.isfinite(beta).all() or not np.isfinite(se).all() or np.any(se < 0)):
            raise RuntimeError('conditional pooled WLS returned invalid slopes or standard errors')
        statistic = np.zeros(m, dtype=float)
        with np.errstate(over='ignore', divide='ignore'):
            np.divide(beta, se, out=statistic, where=se > 0)
        if not np.isfinite(statistic[se > 0]).all():
            raise RuntimeError('conditional pooled t-statistic is not representable')
        statistic[(se == 0) & (beta != 0)] = np.copysign(np.inf, beta[(se == 0) & (beta != 0)])
        detected = np.where((beta != 0) & (np.abs(statistic) >= threshold), np.sign(beta), 0.)
        slopes[members], errors[members] = beta, se
        stats[members], signs[members] = statistic, detected
    restrictions = signs.copy()
    if action == 'release':
        restrictions[signs == 0] = np.nan
    elif action == 'zero_only':
        restrictions[signs != 0] = np.nan

    def frame(values):
        """Build an independently owned labeled response-by-factor result."""
        return pd.DataFrame(values.copy(), index=panel.response_columns.copy(deep=True),
                            columns=panel.factor_columns.copy(deep=True))

    return PooledSignGateResult(
        slopes=frame(slopes), standard_errors=frame(errors), t_stats=frame(stats),
        signs=frame(signs), restrictions=frame(restrictions), groups=panel.groups,
        date_weights=panel.date_weights, threshold=float(threshold), action=action,
        evidence='conditional', variance_estimator='HC1_date', hac_lags=0,
        fit_intercept=True, observations=t, residual_df=t-m-1,
        effective_n=float(w.sum()**2 / (w @ w)), hc1_correction=t/(t-m-1),
    )


#: The gate under the method's name (sign-guided Lasso); the same object as
#: ``compute_pooled_sign_gate``, so signatures, results and ``evidence`` labels agree.
compute_sign_guided_gate = compute_pooled_sign_gate
