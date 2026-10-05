"""Central Gaussian PSD quadratic quantiles and conservative norm bounds.

Imhof (1961), Biometrika 48, 419-426, doi:10.1093/biomet/48.3-4.419.
We invert the characteristic function with sine/cosine Fourier quadrature.
The exponential subtraction removes the sine-integrand singularity at zero.
Reported quadrature errors are numerical diagnostics, not rigorous enclosures.
Known-covariance coverage does not extend to an estimated covariance by substitution.
"""
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.stats import chi2

from factorlasso.inference._validation import _confidence, _probability, _psd_root


def _quadratic_cdf(value, eigenvalues):
    """Invert a normalized positive quadratic using convergent Fourier amplitudes."""
    if value <= 0:
        return 0.

    def amplitudes(t):
        """Remove the removable zero singularities without cancellation."""
        if t == 0:
            return 1., float(eigenvalues.sum())
        z = 2*eigenvalues*t
        log_modulus = -.25*np.log1p(z*z).sum()
        phase = .5*np.arctan(z).sum()
        cosine_minus_one = -2*np.sin(phase/2)**2
        real_difference = (np.expm1(log_modulus)*np.cos(phase)
                           + cosine_minus_one-np.expm1(-t))
        return real_difference/t, np.exp(log_modulus)*np.sin(phase)/t

    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        try:
            sine, sine_error = quad(lambda t: amplitudes(t)[0], 0., np.inf,
                                    weight='sin', wvar=value, epsabs=2e-9,
                                    limit=250, limlst=250)
            cosine, cosine_error = quad(lambda t: amplitudes(t)[1], 0., np.inf,
                                        weight='cos', wvar=value, epsabs=2e-9,
                                        limit=250, limlst=250)
        except IntegrationWarning as error:
            raise ArithmeticError('quadratic CDF quadrature did not converge') from error
    if sine_error+cosine_error > 2e-7:
        raise ArithmeticError('quadratic CDF error exceeds numerical tolerance')
    result = .5+(np.arctan(value)+sine-cosine)/np.pi
    if not np.isfinite(result) or result < -1e-7 or result > 1+1e-7:
        raise ArithmeticError('quadratic CDF is outside its numerical probability range')
    return float(np.clip(result, 0., 1.))


def gaussian_quadratic_quantile(eigenvalues, confidence=.95):
    """Return the quantile of a positive weighted sum of independent chi-square ones.

    Parameters
    ----------
    eigenvalues : array-like, shape (R,)
        Finite nonnegative coefficients; exact zeros are allowed. These are
        eigenvalues of the covariance transformed by the quadratic metric.
    confidence : float, default .95
        Interior CDF probability. No Monte Carlo approximation is used.

    Returns
    -------
    float
        Central quadratic quantile. Zero/equal positive coefficients use the
        exact degenerate/scaled chi-square reference. Other cases use Fourier
        quadrature and root finding, with explicit failure on nonconvergence.
    """
    confidence = _probability(confidence, 'confidence')
    values = np.asarray(eigenvalues, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('eigenvalues must be a finite nonnegative vector')
    values = values[values > 0]
    if not len(values):
        return 0.
    if np.all(values == values[0]):
        return float(values[0]*chi2.ppf(confidence, len(values)))
    scale = values.sum()
    if not np.isfinite(scale):
        raise ValueError('eigenvalue sum must be finite')
    normalized = values/scale
    upper = float(normalized.max()*chi2.ppf(confidence, len(values)))
    root = brentq(lambda x: _quadratic_cdf(x, normalized)-confidence,
                  0., upper*(1+1e-8), xtol=2e-9, rtol=2e-10)
    return float(scale*root)


def quadratic_confidence_summary(mean, covariance, metric, *, confidence=.95,
                                 method='weighted_chi2', errors=None):
    """Describe a fixed PSD quadratic and conservative error-norm interval.

    Parameters
    ----------
    mean : array-like, shape (N,)
        Estimated means, unbiased for the target when interpreting noise correction.
    covariance : array-like, shape (N, N)
        Estimation-error covariance. Plug-in use does not confer known-V coverage.
    metric : array-like, shape (N, N)
        Fixed positive-semidefinite metric; not estimated in this conditional layer.
    confidence : float, default .95
        Pointwise confidence under the specified error model.
    method : {'spectral', 'weighted_chi2', 'bootstrap_error_norm'}
        Spectral retains the legacy maximum-eigenvalue bound. Weighted chi-square
        uses the actual Gaussian error-energy quantile. Bootstrap uses centred
        estimation errors supplied by the caller and remains unvalidated.
    errors : array-like, shape (draws, N), optional
        At least 100 centred residual-bootstrap error draws, required for bootstrap.
        These must not be Gaussian draws centred on the estimated mean.

    Returns
    -------
    dict
        Observed quadratic, estimated noise, signed adjustment, bounds and audit.
        Bounds use a triangle inequality and allow zero under a zero-signal null.
        None of the methods accounts for fitted factor selection automatically.
    """
    confidence = _confidence(confidence)
    mean = np.asarray(mean, dtype=float)
    if mean.ndim != 1 or len(mean) == 0 or not np.isfinite(mean).all():
        raise ValueError('mean must be a nonempty finite vector')
    root, _, _ = _psd_root(covariance, len(mean))
    _psd_root(metric, len(mean))
    metric, covariance = np.asarray(metric, float), np.asarray(covariance, float)
    projected = root.T @ metric @ root
    eig = np.linalg.eigvalsh((projected+projected.T)/2)
    tolerance = 100*np.finfo(float).eps*len(mean)*np.max(np.abs(eig), initial=0.)
    rank = int(np.count_nonzero(eig > tolerance))
    if method == 'spectral':
        radius = np.sqrt(max(0., eig.max(initial=0.))*chi2.ppf(confidence, rank)) if rank else 0.
        label = 'conservative_Gaussian_region_plugin_covariance'
    elif method == 'weighted_chi2':
        radius = np.sqrt(gaussian_quadratic_quantile(eig[eig > tolerance], confidence))
        label = 'weighted_chi2_error_norm_plugin_covariance'
    elif method == 'bootstrap_error_norm':
        errors = np.asarray(errors, float)
        if (errors.ndim != 2 or errors.shape[1] != len(mean) or len(errors) < 100
                or not np.isfinite(errors).all()):
            raise ValueError('bootstrap requires at least 100 aligned finite centred errors')
        norms_squared = np.einsum('ij,ij->i', errors @ metric, errors)
        radius = np.sqrt(max(0., float(np.quantile(norms_squared, confidence))))
        label = 'bootstrap_error_norm_unvalidated'
    else:
        raise ValueError('unknown quadratic interval method')
    observed = float(mean @ metric @ mean)
    noise = float(np.trace(metric @ covariance))
    magnitude = np.sqrt(max(0., observed))
    return dict(observed=observed, noise=noise, noise_adjusted=observed-noise,
                lower=float(max(0., magnitude-radius)**2) if radius else observed,
                upper=float((magnitude+radius)**2) if radius else observed,
                confidence=confidence, uncertainty_rank=rank, interval_method=label,
                error_norm_radius=float(radius), scope='pointwise_fixed_metric',
                status=('bootstrap_approximation_unvalidated' if method == 'bootstrap_error_norm'
                        else 'known_covariance_formula_plugin_unvalidated'))
