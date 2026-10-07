"""Deterministic covariance moment matching by a PSD-preserving congruence.

This is a linear-algebra identity, not a confidence-interval calibration theorem.
For fixed M = E[S] and target V, T = V^(1/2) M^(-1/2) satisfies
E[T S T'] = V. Estimated or misspecified moments do not inherit that identity.
"""
import numpy as np

from factorlasso.inference._validation import _psd_root


def calibrate_covariance_moments(covariance, *, expected_covariance, target_covariance):
    """Match declared covariance moments while preserving positive semidefiniteness.

    Parameters
    ----------
    covariance : array-like, shape (N, N) or (R, N, N)
        One PSD covariance estimate or a batch of estimates in identical units.
    expected_covariance : array-like, shape (N, N)
        Declared expectation M of the uncorrected estimator. Must be numerically
        positive definite; no ridge, pseudoinverse or eigenvalue repair is applied.
    target_covariance : array-like, shape (N, N)
        Declared target covariance V. May be singular PSD.

    Returns
    -------
    dict
        ``covariance`` has the input shape and equals T S T'; ``transform`` is T.
        Diagnostics report the relative moment residual and condition number of M.
        ``status`` identifies conditional moment matching, not calibrated coverage.

    Notes
    -----
    Principal symmetric roots define T = V^(1/2) M^(-1/2). If M = E[S] and M,V
    are fixed, E[T S T'] = V exactly. This operation does not reduce covariance
    estimation noise or integrate it into subsequent inference. A fitted M or V
    makes the transformation random and requires separate validation. It changes
    the covariance estimate, never the point estimate or its economic units.
    """
    values = np.asarray(covariance, float)
    if (values.ndim not in (2, 3) or values.shape[-1] == 0
            or values.shape[-2] != values.shape[-1]
            or (values.ndim == 3 and len(values) == 0)):
        raise ValueError('covariance must be a nonempty square matrix or batch')
    size = values.shape[-1]
    batch = values[None] if values.ndim == 2 else values
    for value in batch:
        _psd_root(value, size)
    m = np.asarray(expected_covariance, float)
    root_m, eig_m, tolerance = _psd_root(m, size)
    if eig_m.min() <= tolerance:
        raise ValueError('expected_covariance must be numerically positive definite')
    root_v, eig_v, _ = _psd_root(target_covariance, size)
    # _psd_root returns eigenvectors times sqrt(eigenvalues), not a symmetric root.
    vectors_m = root_m/np.sqrt(eig_m)
    inverse_root_m = (vectors_m/np.sqrt(eig_m)) @ vectors_m.T
    sqrt_v = np.sqrt(np.maximum(eig_v, 0.))
    vectors_v = np.divide(root_v, sqrt_v, out=np.zeros_like(root_v), where=sqrt_v > 0)
    symmetric_root_v = root_v @ vectors_v.T
    transform = symmetric_root_v @ inverse_root_m
    corrected = transform @ values @ transform.T
    corrected = (corrected+corrected.swapaxes(-1, -2))/2
    target = np.asarray(target_covariance, float)
    residual = np.linalg.norm(transform @ m @ transform.T-target)
    denominator = np.linalg.norm(target)
    return dict(covariance=corrected, transform=transform,
                relative_moment_error=float(residual/denominator if denominator else residual),
                expected_condition_number=float(eig_m[-1]/eig_m[0]),
                status='conditional_moment_matching_not_coverage_calibration')
