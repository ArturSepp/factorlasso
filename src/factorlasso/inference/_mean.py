"""Fixed weighted-mean geometry without renormalizing the estimator.

The score covariance uses Newey and West (1987), doi:10.2307/1913610.
The observed-count correction is a convention, not a coverage result.
"""
import numpy as np

from factorlasso.inference._geometry import LinearHacGeometry
from factorlasso.utils._hac import bartlett_kernel, score_covariance


def compute_weighted_mean_hac_geometry(weights, *, calendar, bandwidth=6., scale=1.):
    """Represent an unnormalized weighted mean and its calendar HAC variance.

    Parameters
    ----------
    weights : array-like, shape (T,)
        Fixed nonnegative weights on observed support, with at least two positive
        entries. Their mass is retained: the constant-mean target is sum(weights)
        times scale times the mean. Missing dates must retain calendar distances.
    calendar : array-like, shape (T,)
        Strictly increasing observation coordinates in the bandwidth's units.
    bandwidth : float, default 6
        Calendar distance at which the Bartlett weight is zero.
    scale : float, default 1
        Positive multiplier of the estimate; variance scales by its square.

    Returns
    -------
    LinearHacGeometry
        Exact fixed linear estimate and centred variance with T/(T-1) correction.
        Zero weights do not remove observed rows from that count. This builder
        supplies geometry, not a Gaussian, AR1 or post-selection guarantee.
    """
    q = np.asarray(weights, dtype=float)
    if (q.ndim != 1 or len(q) < 2 or not np.isfinite(q).all()
            or np.any(q < 0) or np.count_nonzero(q) < 2):
        raise ValueError('weights require at least two positive finite observed entries')
    if (isinstance(scale, (bool, np.bool_)) or not np.isscalar(scale)
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('scale must be finite and positive')
    kernel = bartlett_kernel(calendar, bandwidth)
    if len(kernel) != len(q):
        raise ValueError('calendar must match observed weights')
    mass = q.sum()
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('weight mass must be finite and positive')
    h = scale*q
    residual = np.eye(len(q))-np.outer(np.ones(len(q)), q/mass)
    influence = h[:, None]*residual
    quadratic = len(q)/(len(q)-1)*score_covariance(influence, kernel)
    return LinearHacGeometry(np.ones((len(q), 1)), h, quadratic, residual,
                             np.array([scale*mass]))


def weighted_mean_hac_expectation(weights, *, observed, calendar, covariance_shape,
                                   bandwidth=6., scale=1.):
    """Compute exact joint weighted-mean and expected HAC covariance factors.

    Parameters
    ----------
    weights : array-like, shape (T, N)
        Fixed nonnegative estimator weights, zero off observed support. Masses
        are retained, including initialization effects.
    observed : boolean array, shape (T, N)
        Observation masks. Observed zero weights still count in n/(n-1).
    calendar : array-like, shape (T,)
        Strictly increasing coordinates, including unobserved dates.
    covariance_shape : array-like, shape (T, T)
        Declared common positive-semidefinite temporal covariance R. Under the
        separable model Cov(y_ti, y_sj) = C_ij R_ts, multiply each returned matrix
        elementwise by spatial covariance C to obtain the corresponding moments.
    bandwidth : float, default 6
        Bartlett bandwidth in calendar units.
    scale : float or array-like, shape (N,), default 1
        Positive estimate multipliers, applied to both covariance axes.

    Returns
    -------
    dict
        true_covariance and expected_hac_covariance factors under the declared R.
        Constant means are removed using the same weighted score centering as
        the alpha estimator. These are model diagnostics, not estimated covariance
        matrices or confidence intervals. No elementwise correction is applied:
        such a correction to sample HAC covariance need not remain PSD. Quarterly
        samples of a monthly process and quarterly averages have different R;
        this common-shape model does not silently equate those observations.

    Notes
    -----
    The expectation follows directly by expanding the two centered scores in
    the Bartlett sandwich (Newey and West, 1987). Gaussianity is unnecessary
    for this second-moment identity; fixed masks and correct R are required.
    """
    from factorlasso.inference._validation import _psd_root
    q, mask = np.asarray(weights, float), np.asarray(observed)
    if (q.ndim != 2 or not q.size or mask.shape != q.shape or mask.dtype != bool
            or not np.isfinite(q).all() or np.any(q < 0) or np.any(q[~mask] != 0)
            or np.any(np.count_nonzero(q, axis=0) < 2)):
        raise ValueError('weights must be finite nonnegative with exact boolean observed support')
    kernel = bartlett_kernel(calendar, bandwidth)
    if len(kernel) != len(q):
        raise ValueError('calendar must match weight rows')
    sigma = np.asarray(covariance_shape, float)
    _psd_root(sigma, len(q))
    scales = np.asarray(scale, float)
    if scales.ndim == 0:
        scales = np.full(q.shape[1], scales)
    if scales.shape != (q.shape[1],) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('scale must be positive finite and scalar or match the asset count')
    masses = q.sum(axis=0)
    if not np.isfinite(masses).all():
        raise ValueError('weight masses must be finite')
    p = q/masses
    rp, kq = sigma @ p, kernel @ q
    correction = scales*np.sqrt(mask.sum(axis=0)/(mask.sum(axis=0)-1))
    cross = q.T @ (rp*kq)
    expected = q.T @ (kernel*sigma) @ q-cross-cross.T+(p.T @ rp)*(q.T @ kq)
    expected *= np.outer(correction, correction)
    truth = (q*scales).T @ sigma @ (q*scales)
    return dict(true_covariance=(truth+truth.T)/2,
                expected_hac_covariance=(expected+expected.T)/2)
