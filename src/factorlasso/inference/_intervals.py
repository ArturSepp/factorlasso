"""Fixed linear-target intervals and explicitly unvalidated bootstrap calibration.

Normal/Bartlett intervals are asymptotic approximations. The studentized option
uses residual-bootstrap errors and re-estimated standard errors, supplied by the
caller; its finite-sample adequacy must be established for the application.
Bonferroni normal and maximum studentized-error options specify family scope.
"""
import numpy as np
from scipy.stats import norm

from factorlasso.inference._validation import _confidence, _psd_root


def linear_confidence_intervals(mean, covariance, *, confidence=.95, method='normal',
                                errors=None, replicate_standard_errors=None,
                                simultaneous=False):
    """Compute intervals for fixed linear estimates using an explicit calibration.

    Parameters
    ----------
    mean : array-like, shape (N,)
        Fixed target estimates. For contrasts, transform both mean and covariance.
    covariance : array-like, shape (N, N)
        Joint estimation-error covariance, not residual return risk.
    confidence : float, default .95
        Requested pointwise or family confidence level.
    method : {'normal', 'bootstrap_t', 'unavailable'}, default 'normal'
        Normal uses Gaussian multipliers. Bootstrap uses absolute studentized
        replicate errors. Unavailable retains estimates but supplies no endpoints.
    errors, replicate_standard_errors : array-like, shape (draws, N), optional
        Centred estimation errors and re-estimated SEs from the same residual
        bootstrap. Both are required for bootstrap_t, with at least 100 draws.
    simultaneous : bool, default False
        Apply normal Bonferroni or the bootstrap maximum over all supplied targets.

    Returns
    -------
    dict
        Estimate, SE, bounds, critical values, confidence, method, scope and status.
        Bootstrap results explicitly remain unvalidated until separately assessed.
        Degenerate bootstrap studentization gives unavailable bounds, not certainty.
    """
    confidence = _confidence(confidence)
    mean = np.asarray(mean, dtype=float)
    if mean.ndim != 1 or len(mean) == 0 or not np.isfinite(mean).all():
        raise ValueError('mean must be a nonempty finite vector')
    if not isinstance(simultaneous, (bool, np.bool_)):
        raise ValueError('simultaneous must be boolean')
    _psd_root(covariance, len(mean))
    se = np.sqrt(np.maximum(np.diag(covariance), 0.))
    if method == 'normal':
        family = len(mean) if simultaneous else 1
        critical = np.full(len(mean), norm.ppf(1-(1-confidence)/(2*family)))
        status = 'approximate_conditional'
    elif method == 'bootstrap_t':
        errors = np.asarray(errors, dtype=float)
        replicate_se = np.asarray(replicate_standard_errors, dtype=float)
        if (errors.ndim != 2 or errors.shape[1] != len(mean) or len(errors) < 100
                or replicate_se.shape != errors.shape or not np.isfinite(errors).all()
                or not np.isfinite(replicate_se).all() or np.any(replicate_se < 0)):
            raise ValueError('bootstrap requires at least 100 aligned finite error/SE draws')
        ratios = np.divide(np.abs(errors), replicate_se, out=np.full_like(errors, np.inf),
                           where=replicate_se > 0)
        # A zero target with zero errors and variance may be a structural contrast;
        # keep it zero without masking failed nonzero-error studentization.
        ratios[(replicate_se == 0) & (errors == 0)] = 0.
        statistic = np.max(ratios, axis=1) if simultaneous else ratios
        critical = np.broadcast_to(np.quantile(statistic, confidence, axis=0), mean.shape).copy()
        status = ('bootstrap_approximation_unvalidated' if np.isfinite(critical).all()
                  else 'unsupported_degenerate_studentization')
    elif method == 'unavailable':
        critical = np.full(len(mean), np.nan)
        status = 'unavailable'
    else:
        raise ValueError('unknown linear interval method')
    radius = critical*se
    lower, upper = mean-radius, mean+radius
    invalid = ~np.isfinite(critical)
    lower[invalid], upper[invalid] = np.nan, np.nan
    return dict(estimate=mean.copy(), standard_error=se, lower=lower, upper=upper,
                critical_value=critical, confidence=confidence, interval_method=method,
                scope='simultaneous' if simultaneous else 'pointwise', status=status)
