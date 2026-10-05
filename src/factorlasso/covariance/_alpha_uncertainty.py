"""Conditional uncertainty of recursive EWMA residual means.

The Bartlett weighted-score sandwich follows Newey and West (1987),
Econometrica 55, 703-708, doi:10.2307/1913610. Exact recursive influence weights,
the common calendar kernel and the n/(n-1) mean correction are implementation
choices. HAC intervals are approximate, conditional on the fitted factor model
and a stationary constant residual-mean model. They do not account for LASSO
selection, fitted betas, covariance estimation or a drifting endpoint.

For leading missing observations the default zero initialization can give
weight mass m below one. The target is m times the residual mean, matching the
saved estimator, rather than silently renormalizing its mean or covariance.
"""
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import norm

from factorlasso.inference._ar1 import compute_ar1_interval
from factorlasso.inference._gaussian import gaussian_critical_value
from factorlasso.inference._mean import compute_weighted_mean_hac_geometry
from factorlasso.inference._quadratic import quadratic_confidence_summary
from factorlasso.inference._validation import _confidence, _psd_root
from factorlasso.utils._ewm import compute_ewm, _validate_span
from factorlasso.utils._hac import bartlett_kernel, score_covariance


def _axis_values(value, columns, name):
    """Accept a scalar or an exactly aligned labelled vector."""
    if isinstance(value, pd.Series):
        if not value.index.equals(columns):
            raise ValueError(f'{name} must match the exact asset order')
        values = value.to_numpy(dtype=float)
    elif np.isscalar(value) and not isinstance(value, (bool, np.bool_)):
        values = np.full(len(columns), float(value))
    else:
        raise ValueError(f'{name} must be a scalar or aligned Series')
    if not np.isfinite(values).all():
        raise ValueError(f'{name} must be finite')
    return values


@dataclass(frozen=True)
class AlphaUncertainty:
    """Owned labelled conditional estimates, covariance, support and influence weights.

    Fields use the input residual units multiplied by ``scale``. Covariance is
    estimation-error covariance, not the portfolio residual return covariance.
    ``diagnostics`` includes pointwise approximate normal intervals, weight mass
    and effective_n. ``overlap`` records simultaneous nonmissing observations.
    ``method`` records the fixed-model interpretation and calendar bandwidth.
    """

    estimates: pd.Series
    covariance: pd.DataFrame
    diagnostics: pd.DataFrame
    weights: pd.DataFrame
    overlap: pd.DataFrame
    method: dict


def estimate_alpha_uncertainty(residuals, spans, *, calendar, bandwidth=6., scale=1.,
                               confidence=.95, min_observations=3, min_overlap=3):
    """Estimate recursive EWMA residual alpha and its joint calendar HAC covariance.

    Parameters
    ----------
    residuals : pandas.DataFrame
        Residual histories on an ordered unique common date axis. NaNs are gaps.
        Producer-declared units are retained; already annualized panels use scale=1.
    spans : float or pandas.Series
        Exact native-observation EWMA spans, aligned with the residual columns.
    calendar : array-like
        Strictly increasing numeric coordinates for every original row, including
        gaps. Month ordinals are suitable for monthly/quarterly reporting panels.
    bandwidth : float, default 6
        Calendar distance at which the Bartlett kernel reaches zero. A six-month
        distance gives five monthly lags and one quarterly lag. One common kernel
        governs all pairs; independently chosen per-asset kernels are not spliced.
    scale : float or pandas.Series, default 1
        Explicit multiplier from stored residual units to requested alpha units.
    confidence : float, default .95
        Level for approximate pointwise normal intervals conditional on the fit.
    min_observations, min_overlap : int, default 3
        Minimum individual history and pairwise simultaneous support. Inadequate
        support raises rather than silently implying independent alpha errors.

    Returns
    -------
    AlphaUncertainty
        Exact production-recursion estimates and a PSD score covariance. Effective
        sample size is diagnostic; n/(n-1) charges only the conditional mean fit.
        Weight mass below one changes the target to the initialized weighted mean.
    """
    confidence = _confidence(confidence)
    if (not isinstance(residuals, pd.DataFrame) or residuals.empty
            or not residuals.index.is_unique or not residuals.index.is_monotonic_increasing
            or not residuals.columns.is_unique):
        raise ValueError('residuals must have ordered unique rows and unique columns')
    for name, value in [('min_observations', min_observations), ('min_overlap', min_overlap)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
            raise ValueError(f'{name} must be an integer at least two')
    x = residuals.to_numpy(dtype=float)
    if np.isinf(x).any():
        raise ValueError('infinite residuals are not missing observations')
    valid = np.isfinite(x)
    counts = valid.sum(axis=0)
    if np.any(counts < min_observations):
        raise ValueError('insufficient residual observations')
    overlaps = valid.astype(int).T @ valid.astype(int)
    if np.any(overlaps < min_overlap):
        raise ValueError('insufficient overlapping history for joint alpha covariance')
    span_values = _axis_values(spans, residuals.columns, 'spans')
    scales = _axis_values(scale, residuals.columns, 'scale')
    for value in span_values:
        _validate_span(value)
    if np.any(scales <= 0):
        raise ValueError('scale must be positive')
    kernel = bartlett_kernel(calendar, bandwidth)
    if len(kernel) != len(x):
        raise ValueError('calendar must match residual rows')
    weights = np.zeros_like(x)
    estimates = np.zeros(x.shape[1])
    rows, cache = [], {}
    for j, name in enumerate(residuals):
        mask, span = valid[:, j], span_values[j]
        key = (mask.tobytes(), span)
        if key not in cache:
            # Reuse the actual recursion: impulses expose every initialization
            # and missing-row convention without a second EWMA implementation.
            impulses = np.full((len(x), counts[j]), np.nan)
            impulses[mask] = np.eye(counts[j])
            cache[key] = compute_ewm(impulses, span=span)[-1]
        q = cache[key]
        weights[mask, j] = q
        estimates[j] = q @ x[mask, j]
        mass = q.sum()
        if mass <= 0 or np.count_nonzero(q) < 2:
            raise ValueError('EWMA has insufficient positive-weight observations')
        where = np.flatnonzero(mask)
        rows.append(dict(asset=name, observations=int(counts[j]), span=float(span),
                         weight_mass=float(mass), effective_n=float(mass**2/(q @ q)),
                         first_observation=residuals.index[where[0]],
                         last_observation=residuals.index[where[-1]],
                         missing_rows_inside_history=int(where[-1]-where[0]+1-counts[j]),
                         last_calendar_distance=float(np.asarray(calendar)[-1]
                                                      - np.asarray(calendar)[where[-1]]),
                         initialization='observation' if mask[0] else 'zero_before_inception'))
    masses = weights.sum(axis=0)
    center = estimates/masses
    scores = weights*np.where(valid, x-center, 0.)
    scores *= scales*np.sqrt(counts/(counts-1))
    covariance = score_covariance(scores, kernel)
    _psd_root(covariance, len(estimates))
    estimates *= scales
    se = np.sqrt(np.maximum(np.diag(covariance), 0.))
    critical = norm.ppf((1+confidence)/2)
    diagnostics = pd.DataFrame(rows).set_index('asset')
    diagnostics['alpha'] = estimates
    diagnostics['standard_error'] = se
    diagnostics['lower'] = estimates-critical*se
    diagnostics['upper'] = estimates+critical*se
    diagnostics['status'] = np.where(se > 0, 'approximate_conditional', 'zero_sample_variance')
    index = residuals.columns.copy()
    return AlphaUncertainty(
        pd.Series(estimates, index=index, name='alpha'),
        pd.DataFrame(covariance, index=index, columns=index), diagnostics,
        pd.DataFrame(weights, index=residuals.index.copy(), columns=index),
        pd.DataFrame(overlaps, index=index, columns=index),
        dict(scope='fixed_model_initialized_EWMA_mean', kernel='Bartlett',
             bandwidth=float(bandwidth), confidence=confidence,
             interval='approximate_pointwise_normal', correction='sqrt(n/(n-1)) per score',
             target='weight_mass times constant residual mean',
             calendar=np.asarray(calendar, dtype=float).tolist(), scales=scales.tolist(),
             residual_sha256=sha256(x.tobytes()).hexdigest(),
             beta_selection_uncertainty=False))


def calibrate_alpha_uncertainty(result, residuals, *, method='normal', confidence=.95,
                                covariance_shapes=None, phi_max=.7, cells=401,
                                adaptive=False, delta=.01):
    """Calibrate marginal alpha intervals while retaining the exact saved estimator.

    Parameters
    ----------
    result : AlphaUncertainty
        Output of estimate_alpha_uncertainty with calendar and scale metadata.
    residuals : pandas.DataFrame
        The original residual panel, including its exact row/asset axes and gaps.
    method : {'normal', 'known_shape', 'ar1'}, default 'normal'
        Normal is approximate. Known-shape Gaussian requires a caller-supplied
        true covariance shape for each observed series. AR1 requires complete
        regular native support and the declared Gaussian AR1 family.
    confidence : float, default .95
        Pointwise coverage level under the selected method's assumptions.
    covariance_shapes : dict, optional
        Asset-keyed positive-definite observation covariance matrices, in each
        asset's observed row order. Fitted shapes do not confer exact coverage.
    phi_max, cells, adaptive, delta
        Passed unchanged to compute_ar1_interval; no data-driven mode selection.

    Returns
    -------
    pandas.DataFrame
        Estimates, standard errors, endpoints, critical values, method, scope,
        support status and per-asset calibration audits. Irregular AR1 support
        has unavailable bounds. These are not projected or simultaneous CIs.
    """
    confidence = _confidence(confidence)
    if method not in ('normal', 'known_shape', 'ar1'):
        raise ValueError('unknown alpha interval method')
    if (not isinstance(residuals, pd.DataFrame)
            or not residuals.index.equals(result.weights.index)
            or not residuals.columns.equals(result.estimates.index)):
        raise ValueError('residual panel must match the saved row and asset axes')
    if method == 'known_shape' and (not isinstance(covariance_shapes, dict)
            or set(covariance_shapes) != set(residuals.columns)):
        raise ValueError('known_shape requires one covariance shape per asset')
    if not all(key in result.method for key in ('calendar', 'scales', 'residual_sha256')):
        raise ValueError('recompute the uncertainty result to retain calibration provenance')
    times = np.asarray(result.method['calendar'], dtype=float)
    scales = np.asarray(result.method['scales'], dtype=float)
    x = residuals.to_numpy(dtype=float)
    if sha256(x.tobytes()).hexdigest() != result.method['residual_sha256']:
        raise ValueError('residual values differ from the saved estimator')
    valid = np.isfinite(x)
    if np.isinf(x).any() or not np.array_equal(valid.sum(axis=0),
                                             result.diagnostics.observations.to_numpy()):
        raise ValueError('residual support differs from the saved estimator')
    q = result.weights.to_numpy()
    if np.any(q[~valid] != 0):
        raise ValueError('residual gaps differ from the saved estimator')
    replay = np.sum(q*np.where(valid, x, 0.), axis=0)*scales
    if not np.allclose(replay, result.estimates, rtol=1e-12, atol=1e-14):
        raise ValueError('residual values do not replay the saved alpha estimates')
    rows, cache = [], {}
    for j, asset in enumerate(residuals):
        mask = valid[:, j]
        calendar = times[mask]
        row = dict(asset=asset, estimate=float(result.estimates.iloc[j]),
                   standard_error=float(result.diagnostics.standard_error.iloc[j]),
                   lower=np.nan, upper=np.nan, critical_value=np.nan,
                   confidence=confidence, interval_method=method, scope='pointwise_fixed_model',
                   status='approximate_conditional', calibration_audit={})
        if method == 'ar1' and not np.allclose(np.diff(calendar), np.diff(calendar)[0]):
            row['status'] = 'unsupported_irregular_calendar'
            rows.append(row)
            continue
        if method == 'normal':
            critical = norm.ppf((1+confidence)/2)
        else:
            geometry = compute_weighted_mean_hac_geometry(
                q[mask, j], calendar=calendar, bandwidth=result.method['bandwidth'],
                scale=scales[j])
            estimate, se = geometry.statistics(x[mask, j])
            if (not np.allclose(estimate[0], row['estimate'], rtol=1e-11, atol=1e-14)
                    or not np.allclose(se[0], row['standard_error'], rtol=1e-10, atol=1e-14)):
                raise ValueError('alpha geometry does not replay saved statistics')
            if method == 'known_shape':
                critical = gaussian_critical_value(geometry, covariance_shapes[asset],
                                                    alpha=1-confidence)
                row['status'] = 'known_shape_gaussian_model'
            else:
                # Deterministic full-domain calibration is shared across identical
                # shapes. Adaptive retained cells depend on responses and are not cached.
                key = (geometry.linear.tobytes(), geometry.quadratic.tobytes())
                if adaptive or key not in cache:
                    interval = compute_ar1_interval(geometry, x[mask, j], phi_max=phi_max,
                                                    cells=cells, alpha=1-confidence,
                                                    adaptive=adaptive, delta=delta)
                    if not adaptive:
                        cache[key] = interval
                else:
                    interval = cache[key]
                critical = float(interval.critical_value[0])
                row['status'] = 'bounded_ar1_gaussian_model'
                row['calibration_audit'] = {
                    name: value.tolist() if isinstance(value, np.ndarray) else value
                    for name, value in vars(interval).items()
                    if name not in ('estimate', 'standard_error', 'lower', 'upper')}
        row.update(critical_value=float(critical),
                   lower=row['estimate']-critical*row['standard_error'],
                   upper=row['estimate']+critical*row['standard_error'])
        if row['standard_error'] == 0:
            row.update(status='zero_sample_variance', lower=np.nan, upper=np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index('asset')


def sample_gaussian_estimates(mean, covariance, *, draws=1000, seed=0):
    """Generate joint plug-in Gaussian sensitivity scenarios, not posterior draws.

    Parameters
    ----------
    mean : array-like, shape (N,)
        Finite estimated means in the same units as covariance.
    covariance : array-like, shape (N, N)
        Positive semidefinite estimation-error covariance, possibly singular.
    draws : int, default 1000
        Number of scenarios.
    seed : int, default 0
        Seed for an isolated NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Shape (draws, N). Rank/sign frequencies are sensitivity diagnostics.
    """
    mean = np.asarray(mean, dtype=float)
    if mean.ndim != 1 or len(mean) == 0 or not np.isfinite(mean).all():
        raise ValueError('mean must be a nonempty finite vector')
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, Integral) or draws < 1:
        raise ValueError('draws must be a positive integer')
    root, _, _ = _psd_root(covariance, len(mean))
    return mean + np.random.default_rng(seed).normal(size=(draws, len(mean))) @ root.T


def gaussian_quadratic_summary(mean, covariance, metric, *, confidence=.95, method='spectral'):
    """Describe a PSD quadratic and conservative Gaussian confidence-region bounds.

    Parameters
    ----------
    mean : array-like, shape (N,)
        Estimated mean vector; the bias correction assumes unbiasedness for the target.
    covariance : array-like, shape (N, N)
        Estimation-error covariance. Coverage is conditional on treating it as known.
    metric : array-like, shape (N, N)
        Fixed positive semidefinite quadratic metric.
    confidence : float, default .95
        Confidence level for a conservative transformed Gaussian region.
    method : {'spectral', 'weighted_chi2'}, default 'spectral'
        Preserve the original spectral bound or use the actual central Gaussian
        error-energy quantile. Both use a triangle inequality and plug-in covariance.

    Returns
    -------
    dict
        Observed quadratic, trace noise, signed noise-adjusted value, and bounds.
        The displacement radius uses lambda_max * chi2(rank), then the triangle
        inequality. Bounds allow zero under the null. Estimated HAC covariance
        makes their empirical interpretation approximate, not exact coverage.
    """
    if method not in ('spectral', 'weighted_chi2'):
        raise ValueError('unknown quadratic interval method')
    result = quadratic_confidence_summary(mean, covariance, metric,
                                           confidence=confidence, method=method)
    # Retain the legacy dictionary contract for callers relying on its exact keys.
    return {key: value for key, value in result.items()
            if key not in ('error_norm_radius', 'scope', 'status')}
