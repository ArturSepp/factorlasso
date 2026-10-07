"""Scalar Gaussian confidence bounds for quadratic and positive-part functionals.

The construction here inverts Chernoff bounds from the Gaussian quadratic MGF.
For Y ~ N(theta, S), Q=||Y||^2 and k=||theta||^2, its log MGF is bounded by
  -.5 sum(log(1-2*t*lambda_j)) + t*k/(1-2*t*lambda_max).
This holds for positive and negative t, uniformly over the direction of theta.
The positive-part construction uses ||(theta+e)_+|| <= ||theta_+ + e|| for the
upper tail, and the supporting hyperplane of ||theta_+|| for the lower tail.
Both tails receive (1-confidence)/2. There is no fitted active-set conditioning.

References
----------
Hsu, Kakade and Zhang (2012), A tail inequality for quadratic forms of subgaussian
random vectors, Electronic Communications in Probability 17, 52,
doi:10.1214/ECP.v17-2079, provides background on quadratic MGF tail bounds.
The scalar inversions and positive-part combination below are derived here for
Gaussian errors, not a claim of the source's more general subgaussian coverage.
Chen and Fang (2019), Journal of Econometrics 210, 459-481,
doi:10.1016/j.jeconom.2019.01.011, discusses first-order degeneracy: ordinary
Wald or percentile-bootstrap intervals need not work at a zero quadratic signal.

Known-covariance coverage does not transfer automatically to plug-in HAC V.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from factorlasso.inference._validation import _confidence, _psd_root


def _means(mean):
    """Keep a fixed shared coordinate axis for one or several estimated means."""
    values = np.asarray(mean, dtype=float)
    scalar = values.ndim == 1
    if scalar:
        values = values[None, :]
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError('mean must be finite with shape (N,) or (draws, N)')
    return values, scalar


def _spectrum(covariance):
    """Remove only negative eigensolver roundoff, retaining every positive mode."""
    value = (covariance+covariance.T)/2
    _, eig, _ = _psd_root(value, len(value))
    return np.maximum(eig, 0.)


def _endpoint(observed, eigenvalues, log_tail, upper):
    """Invert the MGF envelope with a bounded one-dimensional parameter.

    Any parameter in the optimization domain gives a conservative endpoint.
    Missing the global optimum can only widen the interval, not invalidate it.
    Grid candidates make the numerical precision check independent of convergence.
    """
    largest = eigenvalues.max()
    ratios, q = eigenvalues/largest, observed/largest

    def value(u):
        if upper:
            logs = np.log1p(-u*(1-ratios))-np.log1p(-u)
            return q/(1-u)-(2/u)*(log_tail+.5*logs.sum())
        logs = np.log1p(-u*ratios)
        return (1-u)*(q+(2/u)*(log_tail+.5*logs.sum()))

    sign = 1. if upper else -1.
    result = minimize_scalar(lambda u: sign*value(u), bounds=(1e-10, 1-1e-10),
                             method='bounded', options={'xatol': 2e-12, 'maxiter': 200})
    if not result.success or not np.isfinite(result.fun):
        raise ArithmeticError('scalar Gaussian tail inversion failed')
    candidates = [value(result.x), *[value(u) for u in [.001, .01, .1, .5, .9, .99, .999]]]
    endpoint = (min(candidates) if upper else max(candidates))*largest
    # Outward rounding at floating-point optimization precision, not a rigorous
    # interval-arithmetic enclosure or a change to statistical error allocation.
    padding = 1e-10*(observed+eigenvalues.sum()+largest)
    return max(0., endpoint+padding if upper else endpoint-padding)


def _bounds(observed, eigenvalues, confidence, positive_part):
    """Return scalar bounds without treating a selected mean direction as known."""
    largest = eigenvalues.max(initial=0.)
    if largest == 0:
        return observed.copy(), observed.copy()
    log_tail = np.log((1-confidence)/2)
    lower = np.array([_endpoint(q, eigenvalues, log_tail, False) for q in observed])
    if positive_part:
        # g(theta)=||theta_+|| is a convex support function. Its true supporting
        # direction has Gaussian error variance at most lambda_max; this is not
        # the observed, data-selected positive set.
        upper = (np.sqrt(observed)+norm.isf((1-confidence)/2)*np.sqrt(largest))**2
    else:
        upper = np.array([_endpoint(q, eigenvalues, log_tail, True) for q in observed])
    if np.any(lower > upper):
        raise ArithmeticError('scalar Gaussian interval endpoints are inconsistent')
    return lower, upper


def _summary(observed, eigenvalues, confidence, scalar, positive_part):
    """Describe the fixed-covariance contract separately from plug-in interpretation."""
    lower, upper = _bounds(observed, eigenvalues, confidence, positive_part)
    noise = float(eigenvalues.sum())
    rank_tolerance = 100*np.finfo(float).eps*len(eigenvalues)*eigenvalues.max(initial=0.)
    label = 'positive_part_support' if positive_part else 'quadratic_chernoff'
    result = dict(observed=observed, lower=lower, upper=upper,
                  noise=np.nan if positive_part else noise,
                  noise_adjusted=(np.full_like(observed, np.nan)
                                  if positive_part else observed-noise),
                  confidence=confidence,
                  uncertainty_rank=int(np.count_nonzero(eigenvalues > rank_tolerance)),
                  error_covariance_trace=noise,
                  error_covariance_squared_trace=float(eigenvalues @ eigenvalues),
                  largest_error_eigenvalue=float(eigenvalues.max(initial=0.)),
                  interval_method='scalar_gaussian_'+label,
                  scope='pointwise_fixed_metric',
                  status='known_covariance_formula_plugin_unvalidated',
                  direction_treatment=('uniform Gaussian MGF envelope; '
                                       'no estimated-direction plug-in'))
    if scalar:
        for key in ['observed', 'lower', 'upper', 'noise_adjusted']:
            result[key] = float(result[key][0])
    return result


def quadratic_scalar_confidence_summary(mean, covariance, metric, *, confidence=.95):
    """Invert scalar Gaussian tail bounds for a fixed PSD quadratic of the true mean.

    Parameters
    ----------
    mean : array-like, shape (N,) or (draws, N)
        Estimated mean, or a batch sharing one covariance and metric.
    covariance : array-like, shape (N, N)
        Gaussian estimation-error covariance, positive semidefinite. Estimated
        HAC covariance is accepted but does not confer known-covariance coverage.
    metric : array-like, shape (N, N)
        Fixed positive-semidefinite quadratic metric, including the zero matrix.
    confidence : float, default .95
        Pointwise confidence, with equal upper/lower tail error allocation.

    Returns
    -------
    dict
        Observed quadratic, trace noise, signed noise correction, lower/upper
        bounds and inference metadata. Batch-dependent values have shape (draws,).
        Bounds target the true quadratic, not its noise-contaminated expectation.
        They are conservative for known Gaussian covariance and all signal
        directions, including zero signal. They do not estimate covariance or
        account for fitted metric uncertainty. Existing norm-bound APIs are unchanged.
    """
    values, scalar = _means(mean)
    confidence = _confidence(confidence)
    _psd_root(covariance, values.shape[1])
    root, _, _ = _psd_root(metric, values.shape[1])
    transformed = values @ root
    spectrum = _spectrum(root.T @ np.asarray(covariance, float) @ root)
    observed = np.sum(transformed**2, axis=1)
    return _summary(observed, spectrum, confidence, scalar, False)


def positive_part_confidence_summary(mean, covariance, weights, *, confidence=.95):
    """Bound the weighted positive-part squared norm of a Gaussian mean vector.

    Parameters
    ----------
    mean : array-like, shape (N,) or (draws, N)
        Estimated mean or a batch sharing one covariance and weighting scheme.
    covariance : array-like, shape (N, N)
        Gaussian estimation-error covariance. Cross-coordinate errors are retained.
        Plug-in HAC covariance requires separate empirical coverage assessment.
    weights : array-like, shape (N,)
        Fixed finite nonnegative weights in sum(weights * max(true_mean, 0)**2).
        These are metric weights, not portfolio holdings or estimator weights.
    confidence : float, default .95
        Pointwise confidence; half the error probability goes to each tail.

    Returns
    -------
    dict
        Observed positive-part quadratic, conservative scalar bounds and metadata.
        Noise correction is unavailable: subtracting a fixed trace is invalid
        across changing positive sets. The upper bound uses the true supporting
        direction's variance envelope, not a selected-positive-set Wald interval.
        Known Gaussian covariance is required for the finite-sample interpretation;
        fixed weights, zero signals and singular covariance are supported.
    """
    values, scalar = _means(mean)
    confidence = _confidence(confidence)
    _psd_root(covariance, values.shape[1])
    w = np.asarray(weights, float)
    if w.shape != (values.shape[1],) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('weights must be a finite nonnegative vector matching mean columns')
    root = np.sqrt(w)
    spectrum = _spectrum(np.asarray(covariance, float)*np.outer(root, root))
    observed = np.sum(np.maximum(values, 0.)**2*w, axis=1)
    return _summary(observed, spectrum, confidence, scalar, True)
