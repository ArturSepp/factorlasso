"""Fixed-group Gaussian reference geometry and clipped directional-gate risk.

The WLS covariance follows linear transformation of a matrix-normal error law.
The risk calculation integrates elementary one-sided normal moments, following
the truncated-normal moment principle in Tallis (1961), JRSS B 23, 223-229,
doi:10.1111/j.2517-6161.1961.tb00408.x. The two-stage gate and its risk decomposition
are the explicit reference construction below, not a theorem for learned groups,
estimated HC1 thresholds, or the correlated constrained HCGL/FCGL estimator.
No estimator configuration or production restriction is changed by these helpers.
"""
from dataclasses import dataclass, fields
from numbers import Real

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr

from factorlasso.inference._validation import _psd_root


@dataclass(frozen=True)
class ConditionalGaussianGeometry:
    """Known-law WLS slopes and pooled gates for a fixed complete panel.

    Arrays are owned and read-only. ``coefficient_map`` maps observations into
    individual factor slopes. ``factor_covariance`` is its covariance under the
    declared date covariance, before multiplying by asset residual covariance.
    Gate/fit arrays use (group, factor)/(asset, factor); joint arrays use
    (asset, group, factor). A zero-variance pair has reported correlation zero.
    This convention does not assert statistical independence of random pairs.
    """

    coefficient_map: np.ndarray
    factor_covariance: np.ndarray
    coefficients: np.ndarray
    pooling_weights: np.ndarray
    residual_covariance: np.ndarray
    date_covariance: np.ndarray
    pooled_means: np.ndarray
    fit_sd: np.ndarray
    gate_sd: np.ndarray
    fit_gate_covariance: np.ndarray
    fit_gate_correlation: np.ndarray

    def __post_init__(self):
        """Own result arrays without sharing mutable caller storage."""
        for field in fields(self):
            value = np.asarray(getattr(self, field.name), dtype=float).copy()
            value.setflags(write=False)
            object.__setattr__(self, field.name, value)


def compute_conditional_gaussian_geometry(
    x, coefficients, residual_covariance, pooling_weights, *, weights=None,
    fit_intercept=True, date_covariance=None,
):
    """Compute exact known-covariance geometry for fixed-group conditional gates.

    Parameters
    ----------
    x : array-like, shape (T, K)
        Fixed finite factor design, without an intercept and without implicit
        standardization or demeaning. Positive-weight rows must identify all
        declared slopes and the optional intercept.
    coefficients : array-like, shape (N, K)
        True individual loading vectors in the declared linear mean model.
    residual_covariance : array-like, shape (N, N)
        Known positive-semidefinite asset covariance Omega. The error law is
        Gaussian with Cov(error[t,i], error[s,j]) = Gamma[t,s] * Omega[i,j].
    pooling_weights : array-like, shape (G, N)
        Fixed nonnegative averaging rows summing to one. Groups may overlap.
        No clustering is learned or sign coherence assumed by this function.
    weights : array-like, shape (T,), optional
        Common nonnegative WLS loss weights. None uses equal weights. Uniform
        rescaling does not change the estimator. These are not noise precisions
        unless the supplied error covariance explicitly makes them so.
    fit_intercept : bool, default True
        Include an unrestricted nuisance intercept before extracting slopes.
    date_covariance : array-like, shape (T, T), optional
        Known positive-semidefinite Gamma. None uses independent unit-variance
        dates. A deterministic response/design transformation requires its
        correspondingly transformed mean model and error covariance; recursive
        demeaning does not preserve independent dates automatically.

    Returns
    -------
    ConditionalGaussianGeometry
        Slopes have mean B; pooled gates have mean A B. With coefficient map L,
        V = L Gamma L', Var(fit[i,j]) = Omega[i,i] V[j,j], and
        Cov(fit[i,j], gate[g,j]) = (Omega A')[i,g] V[j,j]. The full across-factor
        covariance is retained in V. No sample-estimated standard errors occur.

    Notes
    -----
    These statements require fixed groups/design/weights and a correct exogenous
    Gaussian linear mean model. A known average loading need not represent a
    minority-sign member. Singular residual/date covariance is supported, but a
    rank-deficient mean design raises; no ridge or missing-data fallback is used.
    """
    xx = np.asarray(x, dtype=float)
    beta = np.asarray(coefficients, dtype=float)
    a = np.asarray(pooling_weights, dtype=float)
    if xx.ndim != 2 or min(xx.shape, default=0) == 0 or not np.isfinite(xx).all():
        raise ValueError('x must be a nonempty finite two-dimensional design')
    n, k = xx.shape
    if (beta.ndim != 2 or len(beta) == 0 or beta.shape[1] != k
            or not np.isfinite(beta).all()):
        raise ValueError('coefficients must be finite with shape (N, K)')
    if (a.ndim != 2 or len(a) == 0 or a.shape[1] != len(beta)
            or not np.isfinite(a).all() or np.any(a < 0)
            or not np.allclose(a.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError('pooling_weights must have nonnegative rows summing to one')
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError('fit_intercept must be boolean')
    omega = np.asarray(residual_covariance, dtype=float)
    gamma = np.eye(n) if date_covariance is None else np.asarray(date_covariance, dtype=float)
    _psd_root(omega, len(beta))
    _psd_root(gamma, n)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError('weights must be finite, nonnegative and have positive support')
    w = w/w.max()
    design = np.column_stack([np.ones(n), xx]) if fit_intercept else xx
    if np.count_nonzero(w) < design.shape[1]:
        raise ValueError('positive-weight design must have full column rank')
    weighted = np.sqrt(w[:, None])*design
    scale = np.linalg.norm(weighted, axis=0)
    if np.any(scale == 0) or not np.isfinite(scale).all():
        raise ValueError('positive-weight design must have finite nonzero columns')
    u, singular, vt = np.linalg.svd(weighted/scale, full_matrices=False)
    tolerance = singular[0]*max(weighted.shape)*np.finfo(float).eps
    if singular[-1] <= tolerance:
        raise ValueError('positive-weight design must have full column rank')
    full_map = ((vt.T/singular) @ u.T)*np.sqrt(w)[None, :]/scale[:, None]
    if not np.allclose(full_map @ design, np.eye(design.shape[1]), atol=2e-8, rtol=2e-8):
        raise ValueError('design is too ill-conditioned for an unbiased numerical WLS map')
    linear = full_map[1:] if fit_intercept else full_map
    factor_covariance = linear @ gamma @ linear.T
    factor_covariance = (factor_covariance+factor_covariance.T)/2
    diagonal = np.maximum(np.diag(factor_covariance), 0.)
    pooled_covariance = a @ omega @ a.T
    fit_sd = np.sqrt(np.maximum(np.diag(omega), 0.)[:, None]*diagonal)
    gate_sd = np.sqrt(np.maximum(np.diag(pooled_covariance), 0.)[:, None]*diagonal)
    covariance = (omega @ a.T)[:, :, None]*diagonal
    denominator = fit_sd[:, None, :]*gate_sd[None, :, :]
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance),
                            where=denominator > 0)
    if np.max(abs(correlation), initial=0.) > 1+1e-10:
        raise ArithmeticError('inconsistent Gaussian covariance correlation')
    correlation = np.clip(correlation, -1., 1.)
    return ConditionalGaussianGeometry(
        linear, factor_covariance, beta, a, omega, gamma, a @ beta,
        fit_sd, gate_sd, covariance, correlation,
    )


@dataclass(frozen=True)
class GaussianSignGateRisk:
    """Exact-law moments evaluated by deterministic one-dimensional quadrature.

    ``positive_mse``, ``negative_mse`` and ``weak_mse`` are unconditional
    contributions whose sum is ``mse``. Wrong-sign probability concerns the
    gate relative to nonzero individual beta, not the realized fitted sign.
    False exclusion is weak-gate probability for nonzero beta under exclusion;
    false admission is active-gate probability for a zero beta. Neither is a
    final-support recovery probability. Clipping probability is the probability
    of a strict fitted/gate sign conflict on a passing gate.
    """

    estimate_mean: float
    estimate_second_moment: float
    bias: float
    variance: float
    mse: float
    baseline_mse: float
    positive_mse: float
    negative_mse: float
    weak_mse: float
    probability_positive: float
    probability_negative: float
    probability_weak: float
    probability_wrong_sign: float
    probability_false_exclusion: float
    probability_false_admission: float
    probability_clipped: float
    integration_error: float


def _one_sided_moments(mean, sd, beta, direction):
    """Return mean, second moment, bias, MSE and clipping probability."""
    if sd == 0:
        estimate = max(mean, 0.) if direction > 0 else min(mean, 0.)
        return np.array([estimate, estimate**2, estimate-beta, (estimate-beta)**2,
                         float(direction*mean < 0)])
    q = direction*mean/sd
    keep, reject = ndtr(q), ndtr(-q)
    density = np.exp(-.5*q*q)/np.sqrt(2*np.pi)
    delta = mean-beta
    first = mean*keep+direction*sd*density
    second = (mean*mean+sd*sd)*keep+direction*mean*sd*density
    bias = delta*keep+direction*sd*density-beta*reject
    mse = ((delta*delta+sd*sd)*keep
           + direction*sd*(mean-2*beta)*density+beta*beta*reject)
    # Only roundoff may make nonnegative truncated moments slightly negative.
    return np.array([first, max(second, 0.), bias, max(mse, 0.), reject])


def _normal_interval_probability(lower, upper):
    """Stable probability for a standard normal interval, including tails."""
    if lower >= 0:
        return float(ndtr(-lower)-ndtr(-upper))
    return float(ndtr(upper)-ndtr(lower))


def _integrate_normal(function, lower, upper, split=None):
    """Integrate normal-weighted moments without missing central mass on wide limits."""
    if lower >= upper:
        return np.zeros(5), 0.
    points = [-8., -4., -1., 0., 1., 4., 8.]
    if split is not None and np.isfinite(split):
        points.append(float(split))
    boundaries = [lower]+sorted(set(p for p in points if lower < p < upper))+[upper]
    result, error = np.zeros(5), 0.
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        value, err = quad_vec(
            lambda u: function(u)*np.exp(-.5*u*u)/np.sqrt(2*np.pi),
            start, stop, epsabs=1e-11, epsrel=1e-11,
        )
        result += value
        error += err
    return result, float(error)


def gaussian_sign_gate_risk(
    beta, gate_mean, fit_sd, gate_sd, correlation, *, threshold=1.0, action='exclude',
):
    """Evaluate Gaussian risk of a pooled gate followed by scalar sign clipping.

    Parameters
    ----------
    beta, gate_mean : float
        Means of individual unrestrained estimate Z and pooled gate G. They may
        differ: a cluster's mean direction need not represent every member.
    fit_sd, gate_sd : float
        Known nonnegative standard deviations of Z and G, not estimated SEs.
    correlation : float
        Correlation of Z and G, in [-1, 1]. It encodes same-sample dependence.
        Ignored if either standard deviation is zero, but still range checked.
    threshold : float, default 1.0
        Nonnegative threshold tau. Positive G >= tau*gate_sd imposes Z >= 0;
        negative G <= -tau*gate_sd imposes Z <= 0. A deterministic zero gate is
        weak even when tau=0. Positive infinity explicitly makes all gates weak.
    action : {'exclude', 'release'}, default 'exclude'
        A weak gate returns zero for exclusion and the unrestricted Z for release.
        Passing gates return max(Z, 0) or min(Z, 0), respectively.

    Returns
    -------
    GaussianSignGateRisk
        Moments, gate probabilities and squared-error contributions in coefficient
        units. Multiplying ``mse`` by a fixed factor premium squared gives that
        component's attribution MSE. Total expected-return risk also needs
        across-factor error cross-moments, not only these marginal scalar risks.

    Raises
    ------
    ValueError
        If input parameters are invalid or the raw second moment
        ``beta**2 + fit_sd**2`` cannot be represented as a finite float.

    Notes
    -----
    Conditional on U=(G-gate_mean)/gate_sd, Z is normal with mean
    beta + correlation*fit_sd*U and variance fit_sd**2*(1-correlation**2).
    The implementation integrates one-sided normal moments in each gate region,
    including correlation endpoints and deterministic fits/gates. This is an
    exact distributional reference evaluated numerically, not Monte Carlo.
    Coordinate clipping equals unpenalized constrained WLS only with diagonal
    slope-loss geometry. It is not the risk of general correlated, penalized
    HCGL/FCGL fitting, learned clusters or sample-estimated HC1 thresholds.
    """
    values = [beta, gate_mean, fit_sd, gate_sd, correlation]
    if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
           or not np.isfinite(v) for v in values):
        raise ValueError('means, standard deviations and correlation must be finite scalars')
    beta, gate_mean, fit_sd, gate_sd, correlation = map(float, values)
    if fit_sd < 0 or gate_sd < 0 or abs(correlation) > 1:
        raise ValueError('standard deviations must be nonnegative and correlation in [-1, 1]')
    # Inputs are Python floats here: overflowing products are inf without
    # NumPy warnings, and testing the actual sum also catches joint overflow.
    if not np.isfinite(beta * beta + fit_sd * fit_sd):
        raise ValueError('raw second moment beta**2 + fit_sd**2 must be finite')
    if (not isinstance(threshold, Real) or isinstance(threshold, (bool, np.bool_))
            or np.isnan(threshold) or threshold < 0):
        raise ValueError('threshold must be nonnegative, finite or positive infinity')
    threshold = float(threshold)
    if action not in ('exclude', 'release'):
        raise ValueError("action must be 'exclude' or 'release'")
    positive, negative, weak = np.zeros(5), np.zeros(5), np.zeros(5)
    error = 0.
    if np.isinf(threshold) or gate_sd == 0:
        direction = 0 if np.isinf(threshold) else np.sign(gate_mean)
        pp, pn, pw = float(direction > 0), float(direction < 0), float(direction == 0)
        if direction:
            value = _one_sided_moments(beta, fit_sd, beta, direction)
            if direction > 0:
                positive = value
            else:
                negative = value
        elif action == 'exclude':
            weak = np.array([0., 0., -beta, beta*beta, 0.])
        else:
            weak = np.array([beta, beta*beta+fit_sd*fit_sd, 0., fit_sd*fit_sd, 0.])
    else:
        standardized = gate_mean/gate_sd
        lower, upper = -threshold-standardized, threshold-standardized
        pp, pn = float(ndtr(-upper)), float(ndtr(lower))
        pw = _normal_interval_probability(lower, upper)
        conditional_sd = fit_sd*np.sqrt(max((1-correlation)*(1+correlation), 0.))
        slope = correlation*fit_sd
        split = -beta/slope if slope != 0 else None
        for direction, left, right in ((1, upper, np.inf), (-1, -np.inf, lower)):
            value, err = _integrate_normal(
                lambda u: _one_sided_moments(beta+slope*u, conditional_sd, beta, direction),
                left, right, split,
            )
            error += err
            if direction > 0:
                positive = value
            else:
                negative = value
        if action == 'exclude':
            weak = np.array([0., 0., -beta*pw, beta*beta*pw, 0.])
        else:
            def unrestrained(u):
                mean, delta = beta+slope*u, slope*u
                return np.array([mean, mean*mean+conditional_sd**2, delta,
                                 delta*delta+conditional_sd**2, 0.])
            weak, err = _integrate_normal(unrestrained, lower, upper, split)
            error += err
    moments = positive+negative+weak
    first, second, bias, mse, clipped = map(float, moments)
    variance = max(mse-bias*bias, 0.)
    if not np.isfinite([first, second, bias, mse, variance, error]).all():
        raise ArithmeticError('Gaussian moments exceed finite numerical precision')
    return GaussianSignGateRisk(
        first, second, bias, variance, mse, fit_sd*fit_sd,
        float(positive[3]), float(negative[3]), float(weak[3]), pp, pn, pw,
        pn if beta > 0 else (pp if beta < 0 else 0.),
        pw if beta != 0 and action == 'exclude' else 0.,
        pp+pn if beta == 0 else 0., float(np.clip(clipped, 0., 1.)), error,
    )
