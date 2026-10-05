"""Joint residual dependent-wild bootstrap of fixed weighted means.

Shao (2010), The Dependent Wild Bootstrap, JASA 105, 218-235,
doi:10.1198/jasa.2009.tm08744. Shared Gaussian calendar multipliers preserve
observed score dependence and missing masks. Fixed-span initialized means and
ragged multivariate use here are adaptations, not inherited coverage guarantees.
Every replicate re-centres residuals and re-estimates the Bartlett HAC SE.
"""
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix

from factorlasso.utils._hac import bartlett_kernel, score_covariance


def bootstrap_weighted_means(residuals, weights, *, calendar, bandwidth=6., scale=1.,
                             contrasts=None, draws=1000, seed=0, batch_size=32,
                             return_residuals=False):
    """Resample a residual panel using common calendar multipliers and fixed weights.

    Parameters
    ----------
    residuals, weights : array-like, shape (T, N)
        Residuals may have NaNs. Fixed nonnegative estimator weights must be zero
        at missing cells. Their column masses are retained rather than normalized.
    calendar : array-like, shape (T,)
        Original strictly increasing observation coordinates, including gaps.
    bandwidth : float, default 6
        Common Bartlett zero-weight distance for multiplier dependence and HAC.
    scale : float or array-like, shape (N,), default 1
        Positive multipliers of the estimates, applied once.
    contrasts : array-like, shape (J, N), optional
        Fixed linear targets. None means the original N means.
    draws : int, default 1000
        At least 100 residual replicates for studentized calibration.
    seed : int, default 0
        Isolated NumPy random generator seed.
    batch_size : int, default 32
        Maximum replicates held in the temporary score array.
    return_residuals : bool, default False
        Also return a (draws, T, N) residual panel in the original input units,
        preserving every missing cell. This opt-in allocation supports caller-owned
        factor refits with the same joint multipliers; scale and contrasts affect
        the mean outputs only. Refitting does not establish calibrated coverage.

    Returns
    -------
    dict
        Original target estimates/covariance, centred bootstrap errors, re-estimated
        target SEs and explicit unvalidated method metadata. The bootstrap does
        not fit factors, repair incomplete overlap or establish post-selection CIs.
    """
    if not isinstance(return_residuals, (bool, np.bool_)):
        raise ValueError('return_residuals must be boolean')
    x, q = np.asarray(residuals, float), np.asarray(weights, float)
    if (x.ndim != 2 or min(x.shape) == 0 or q.shape != x.shape or np.isinf(x).any()
            or not np.isfinite(q).all() or np.any(q < 0)):
        raise ValueError('residuals and finite nonnegative weights must share a panel shape')
    valid = np.isfinite(x)
    counts, masses = valid.sum(axis=0), q.sum(axis=0)
    if (np.any(counts < 3) or np.any(masses <= 0) or not np.isfinite(masses).all()
            or np.any(np.count_nonzero(q, axis=0) < 2) or np.any(q[~valid] != 0)):
        raise ValueError('weights must respect missing cells and sufficient observed support')
    overlap = valid.astype(int).T @ valid.astype(int)
    if np.any(overlap < 3):
        raise ValueError('insufficient overlapping observations for joint resampling')
    scales = np.broadcast_to(np.asarray(scale, float), (x.shape[1],))
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('scale must be positive and finite for every asset')
    c = np.eye(x.shape[1]) if contrasts is None else np.asarray(contrasts, float)
    if c.ndim != 2 or c.shape[1] != x.shape[1] or not len(c) or not np.isfinite(c).all():
        raise ValueError('contrasts must be a finite matrix with one column per asset')
    for name, value, minimum in [('draws', draws, 100), ('batch_size', batch_size, 1)]:
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                or value < minimum):
            raise ValueError(f'{name} must be an integer at least {minimum}')
    kernel = bartlett_kernel(calendar, bandwidth)
    if len(kernel) != len(x):
        raise ValueError('calendar must match residual rows')
    # Distinct dates with finite Bartlett distance give a positive-definite kernel;
    # fail if numerically unresolved rather than silently modifying its covariance.
    root = np.linalg.cholesky(kernel)
    sparse_kernel = csr_matrix(kernel)
    centre = np.sum(q*np.where(valid, x, 0.), axis=0)/masses
    innovations = np.where(valid, x-centre, 0.)
    h = q*scales
    correction = np.sqrt(counts/(counts-1))
    scores = h*innovations*correction
    covariance = score_covariance(scores, kernel)
    target_covariance = c @ covariance @ c.T
    target_covariance = (target_covariance+target_covariance.T)/2
    errors = np.empty((draws, len(c)))
    standard_errors = np.empty_like(errors)
    panels = np.empty((draws, *x.shape)) if return_residuals else None
    rng = np.random.default_rng(seed)
    for start in range(0, draws, batch_size):
        stop = min(start+batch_size, draws)
        multipliers = rng.normal(size=(stop-start, len(x))) @ root.T
        innovation_draws = innovations[None, :, :]*multipliers[:, :, None]
        if panels is not None:
            panels[start:stop] = np.where(valid, centre+innovation_draws, np.nan)
        mean_shift = np.sum(q*innovation_draws, axis=1)/masses
        errors[start:stop] = (mean_shift*masses*scales) @ c.T
        scores_draws = h*(innovation_draws-mean_shift[:, None, :])*correction
        target_scores = scores_draws @ c.T
        flat = target_scores.transpose(1, 0, 2).reshape(len(x), -1)
        variance = np.sum(flat*sparse_kernel.dot(flat), axis=0).reshape(stop-start, len(c))
        standard_errors[start:stop] = np.sqrt(np.maximum(variance, 0.))
    result = dict(estimate=c @ (centre*masses*scales), covariance=target_covariance,
                errors=errors, standard_errors=standard_errors,
                method='residual_dependent_wild_bootstrap', bandwidth=float(bandwidth),
                draws=draws, seed=seed, scope='fixed_weight_fixed_model',
                status='bootstrap_approximation_unvalidated')
    if panels is not None:
        result['residual_draws'] = panels
    return result
