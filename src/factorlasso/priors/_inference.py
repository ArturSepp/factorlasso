# Portions adapted from OptimalPortfolios research references (MIT).
# This FactorLasso implementation is distributed under GPL-3.0-or-later;
# the original permission notice for those portions is retained below.
# MIT License
#
# Copyright (c) 2024 Artur Sepp
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Optional finite-Gaussian calibration of weighted OLS/HAC prior uncertainty.

These routines do not alter LassoModel or its default prior floors. The mean
model and factor identity must be fixed independently of the response. Coverage
is for that model's coefficient, not a selected factor or a drifting endpoint.
All rows must be observed on a regular calendar for the AR(1) construction.

References
----------
Imhof (1961), Biometrika 48, 419-426, doi:10.1093/biomet/48.3-4.419:
Gaussian quadratic forms. Here a one-positive-eigenvalue pivot uses a
nonoscillatory integral, rather than Imhof's general inversion algorithm.
Tyler (1987), Biometrika 74, 579-589, doi:10.1093/biomet/74.3.579:
angular central Gaussian densities. A fixed-mixture Markov inequality is used
here to obtain a residual-direction confidence set.
Berger and Boos (1994), JASA 89, 1012-1016,
doi:10.1080/01621459.1994.10476836: nuisance confidence sets and error allocation.
The AR cell domination bound below is an explicit implementation choice.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp


def _probability(value, name):
    """Validate a strictly interior, finite probability."""
    if not np.isscalar(value) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f'{name} must be finite and strictly between zero and one')
    return float(value)


def _symmetric(matrix, size, name, positive_definite=False):
    """Validate a finite symmetric covariance or quadratic form without silently symmetrizing."""
    value = np.asarray(matrix, dtype=float)
    if value.shape != (size, size) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite {size} by {size} matrix')
    scale = np.max(np.abs(value))
    if scale == 0 or not np.allclose(value, value.T, rtol=0, atol=1e-12*scale):
        raise ValueError(f'{name} must be nonzero and symmetric')
    value = (value+value.T)/2
    if positive_definite:
        try:
            np.linalg.cholesky(value)
        except np.linalg.LinAlgError as error:
            raise ValueError(f'{name} must be positive definite') from error
    elif np.linalg.eigvalsh(value)[0] < -1e-10*scale:
        raise ValueError(f'{name} must be positive semidefinite')
    return value


@dataclass(frozen=True)
class PriorHacGeometry:
    """Fixed linear estimate and Bartlett HAC quadratic form.

    Construct with ``compute_prior_hac_geometry``. Arrays are in observation
    order and must not be modified after construction. ``linear`` is h and
    ``quadratic`` is Q in b=h'y and HAC variance y'Qy. The design includes
    the intercept if the intended regression has one.
    """

    design: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray
    residual_map: np.ndarray

    def statistics(self, responses):
        """Return estimate and HAC SE arrays, one entry per response row.

        Parameters
        ----------
        responses : array-like, shape (T,) or (N, T)
            Complete finite responses on the design's original row grid.

        Returns
        -------
        estimates, standard_errors : ndarray
            One-dimensional arrays, including for a single input response.
        """
        y = _responses(responses, len(self.design))
        # Remove the fitted mean first to avoid cancellation in y'Qy.
        residual = y @ self.residual_map.T
        variance = np.einsum('ij,jk,ik->i', residual, self.quadratic, residual)
        return y @ self.linear, np.sqrt(np.maximum(variance, 0.))


def _responses(responses, size):
    """Validate a complete scalar or batched response, without dropping calendar gaps."""
    values = np.asarray(responses, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if (values.ndim != 2 or values.shape[1] != size or len(values) == 0
            or not np.isfinite(values).all()):
        raise ValueError('responses must be finite with shape (T,) or (N, T); gaps are unsupported')
    return values


def compute_prior_hac_geometry(design, weights=None, hac_lags=0, coefficient=1):
    """Build the exact weighted OLS/Bartlett-HAC linear and quadratic forms.

    Parameters
    ----------
    design : array-like, shape (T, P)
        Fixed finite full-rank mean design, including an intercept if required.
        No response-dependent selection, implicit centering or row deletion.
    weights : array-like, shape (T,), optional
        Nonnegative observation weights; None means equal weights. EWMA weights
        are ``(1 - 2/(span+1)) ** np.arange(T-1, -1, -1)``. Scaling cancels.
    hac_lags : int, default 0
        Bartlett bandwidth on the original row grid.
    coefficient : int, default 1
        Zero-based design column to infer. Conventionally column zero is the
        intercept and column one is the named factor.

    Returns
    -------
    PriorHacGeometry
        Uses the existing prior statistic's n_obs/(n_obs-P) correction. No
        effective-sample-size substitution or additional degrees-of-freedom fit.
    """
    d = np.asarray(design, dtype=float)
    if d.ndim != 2 or min(d.shape) == 0 or not np.isfinite(d).all():
        raise ValueError('design must be a nonempty finite matrix')
    n, p = d.shape
    if (isinstance(hac_lags, (bool, np.bool_)) or not isinstance(hac_lags, Integral)
            or hac_lags < 0):
        raise ValueError('hac_lags must be a nonnegative integer')
    if (isinstance(coefficient, (bool, np.bool_)) or not isinstance(coefficient, Integral)
            or not 0 <= coefficient < p):
        raise ValueError('coefficient must index a design column')
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('weights must be finite, nonnegative and match design rows')
    nobs = np.count_nonzero(w)
    if nobs <= p:
        raise ValueError('positive residual degrees of freedom required')
    w = w / np.max(w)
    root = np.sqrt(w)
    scales = np.linalg.norm(root[:, None]*d, axis=0)
    if np.any(scales == 0):
        raise ValueError('design must have full weighted column rank')
    u, singular, vt = np.linalg.svd(root[:, None]*(d/scales), full_matrices=False)
    if singular[-1] <= singular[0]*max(d.shape)*np.finfo(float).eps:
        raise ValueError('design must have full weighted column rank')
    hmap = ((vt.T/singular) @ u.T)*root[None, :]/scales[:, None]
    h = hmap[coefficient]
    residual = np.eye(n)-d @ hmap
    distance = np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])
    kernel = np.maximum(0., 1-distance/(hac_lags+1))
    influence = h[:, None]*residual
    q = nobs/(nobs-p)*(influence.T @ kernel @ influence)
    return PriorHacGeometry(d.copy(), h.copy(), (q+q.T)/2, residual)


def _pivot_tail(critical, a, b):
    """Integrate the central Gaussian ratio tail via its single positive eigenvalue."""
    if critical == 0:
        return 1.
    matrix = np.outer(a, a)-critical**2*b
    eigen = np.linalg.eigvalsh((matrix+matrix.T)/2)
    tolerance = 1e-12*max(np.max(np.abs(eigen)), np.finfo(float).tiny)
    positive = eigen[eigen > tolerance]
    if len(positive) == 0:
        return 0.
    if len(positive) != 1:
        raise ArithmeticError('Pivot numerator has more than one positive eigenvalue')
    ratios = -eigen[eigen < -tolerance]/positive[0]

    def integrand(angle):
        """Evaluate the stable nonoscillatory Gaussian tail integrand."""
        return np.exp(-.5*np.log1p(ratios/np.sin(angle)**2).sum())

    value, error = quad(integrand, 0., np.pi/2, epsabs=2e-10, epsrel=2e-10)
    if error > 1e-7:
        raise ArithmeticError('Gaussian pivot quadrature did not converge')
    return float(2/np.pi*value)


def _gaussian_critical(h, q, covariance, alpha):
    """Calibrate an already validated nondegenerate Gaussian pivot."""
    c = np.linalg.cholesky(covariance)
    a = c.T @ h
    b = c.T @ q @ c
    scale = np.linalg.norm(a)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError('nonzero finite coefficient variance required')
    a, b = a/scale, b/scale**2
    if np.trace(b) <= 0:
        raise ValueError('nonzero HAC variance form required')
    upper = 2.
    while _pivot_tail(upper, a, b) > alpha:
        upper *= 2
        if upper > 1e6:
            raise ArithmeticError('Could not bracket Gaussian pivot critical value')
    return float(brentq(lambda k: _pivot_tail(k, a, b)-alpha, 0., upper, xtol=1e-9))


def gaussian_prior_critical_value(geometry, covariance_shape, alpha=.05):
    """Calibrate a two-sided prior interval under a known Gaussian covariance shape.

    Parameters
    ----------
    geometry : PriorHacGeometry
        Fixed design geometry returned by ``compute_prior_hac_geometry``.
    covariance_shape : array-like, shape (T, T)
        Known symmetric positive-definite shape V. The noise covariance is
        sigma squared times V; the unknown scalar cancels from the pivot.
    alpha : float, default 0.05
        Two-sided error probability, strictly between zero and one.

    Returns
    -------
    float
        k such that estimate +/- k * HAC_SE has model-based coverage 1-alpha.
        Quadrature and eigensolver tolerances are numerical, not an interval-
        arithmetic proof. A fitted covariance shape does not inherit validity.
    """
    alpha = _probability(alpha, 'alpha')
    h = np.asarray(geometry.linear, float)
    if h.ndim != 1 or not np.isfinite(h).all():
        raise ValueError('geometry.linear must be a finite vector')
    q = _symmetric(geometry.quadratic, len(h), 'quadratic')
    covariance = _symmetric(covariance_shape, len(h), 'covariance_shape', True)
    return _gaussian_critical(h, q, covariance, alpha)


def _ar_covariance(size, phi):
    """Return stationary AR(1) unit-marginal covariance on a regular grid."""
    return phi**np.abs(np.arange(size)[:, None]-np.arange(size)[None, :])


def _angular_log_density(design, responses, phis):
    """Evaluate residual-direction densities using tridiagonal AR precision."""
    # An orthonormal basis preserves the mean space and protects rescaled designs.
    d = np.linalg.qr(design, mode='reduced')[0]
    n, p = d.shape
    y = responses-(responses @ d) @ d.T
    yy = np.sum(y*y, axis=1)
    if np.any(yy <= np.finfo(float).eps**2*np.sum(responses*responses, axis=1)):
        raise ValueError('AR shape inference requires a nonzero residual direction')
    if np.any(yy <= np.finfo(float).tiny):
        raise ValueError('AR shape inference requires a nonzero residual direction')
    adjacent = np.sum(y[:, 1:]*y[:, :-1], axis=1)
    interior = np.sum(y[:, 1:-1]**2, axis=1)
    dt0 = y @ d
    dt1 = y[:, 1:] @ d[:-1]+y[:, :-1] @ d[1:]
    dt2 = y[:, 1:-1] @ d[1:-1]
    m0 = d.T @ d
    m1 = d[1:].T @ d[:-1]+d[:-1].T @ d[1:]
    m2 = d[1:-1].T @ d[1:-1]
    logdet0 = np.linalg.slogdet(m0)[1]
    result = []
    for phi in phis:
        matrix = m0-phi*m1+phi*phi*m2
        vector = dt0-phi*dt1+phi*phi*dt2
        rss = yy-2*phi*adjacent+phi*phi*interior
        rss -= np.sum(vector*np.linalg.solve(matrix, vector.T).T, axis=1)
        if np.any(rss <= 0):
            raise ArithmeticError('Nonpositive profiled residual quadratic form')
        result.append(.5*np.log1p(-phi*phi)-.5*np.linalg.slogdet(matrix)[1]
                      +.5*logdet0-.5*(n-p)*np.log(rss/yy))
    return np.column_stack(result)


@dataclass(frozen=True)
class Ar1PriorInterval:
    """Batch of model-based intervals and their continuous-cell calibration audit.

    Estimate, standard_error, lower, upper and critical_value have shape (N,).
    ``retained_cells`` has shape (N, cells); it contains all cells in full-domain
    mode. ``cell_critical_values`` use the reported ``calibration_alpha``.
    This is a pointwise fixed-factor interval, not simultaneous or post-selection.
    """

    estimate: np.ndarray
    standard_error: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    critical_value: np.ndarray
    phi_centres: np.ndarray
    phi_edges: np.ndarray
    retained_cells: np.ndarray
    cell_critical_values: np.ndarray
    alpha: float
    delta: float
    calibration_alpha: float
    adaptive: bool


def compute_ar1_prior_interval(geometry, responses, *, phi_max=.7, cells=401,
                               alpha=.05, adaptive=False, delta=.01):
    """Calibrate intervals over a continuous, bounded stationary AR(1) family.

    Parameters
    ----------
    geometry : PriorHacGeometry
        Fixed complete regular-grid design and weighted HAC geometry.
    responses : array-like, shape (T,) or (N, T)
        Responses with mean in the design's column space and Gaussian AR(1)
        errors of unknown scale. Batched responses reuse deterministic calibration.
    phi_max : float, default 0.7
        Declared bound 0 < phi_max < 1 on the absolute AR coefficient.
    cells : int, default 401
        Number of equal cells in atanh(phi), at least three.
    alpha : float, default 0.05
        Two-sided interval error probability.
    adaptive : bool, default False
        False maximizes over the whole AR domain. True restricts to a certified
        residual-direction confidence cover. Choosing the narrower realized
        interval from the two modes is not covered by either guarantee.
    delta : float, default 0.01
        Confidence-cover error when adaptive=True; must be below alpha. Ignored
        in full-domain mode, which spends all alpha on slope calibration.

    Returns
    -------
    Ar1PriorInterval
        Arrays with one interval per response row and a reproducible cell audit.
        Coverage is at least 1-alpha under the declared model, subject to numerical
        calibration tolerances. It does not cover omitted means, irregularly spaced
        rows, estimated factor selection, arbitrary drift or non-Gaussian errors.
    """
    alpha = _probability(alpha, 'alpha')
    if not isinstance(adaptive, (bool, np.bool_)):
        raise ValueError('adaptive must be boolean')
    delta = _probability(delta, 'delta') if adaptive else 0.
    if delta >= alpha:
        raise ValueError('adaptive delta must be smaller than alpha')
    if not np.isscalar(phi_max) or not np.isfinite(phi_max) or not 0 < phi_max < 1:
        raise ValueError('phi_max must be finite and strictly between zero and one')
    if isinstance(cells, (bool, np.bool_)) or not isinstance(cells, Integral) or cells < 3:
        raise ValueError('cells must be an integer at least three')
    n = len(geometry.design)
    y = _responses(responses, n)
    endpoint = np.arctanh(phi_max)
    psi_edges = np.linspace(-endpoint, endpoint, cells+1)
    phis = np.tanh((psi_edges[:-1]+psi_edges[1:])/2)
    half_width = endpoint/cells
    corrected_alpha = (alpha-delta)*np.exp(-2*n*half_width)
    if corrected_alpha <= 1e-9:
        raise ValueError('cells too coarse for stable calibration; increase cells')
    q = _symmetric(geometry.quadratic, n, 'quadratic')
    critical = np.array([
        _gaussian_critical(geometry.linear, q, _ar_covariance(n, phi), corrected_alpha)
        for phi in phis])
    if adaptive:
        density = _angular_log_density(geometry.design, y, phis)
        mixture = logsumexp(density, axis=1)-np.log(cells)
        allowance = 2*(n-geometry.design.shape[1])*half_width
        retained = density+allowance >= np.log(delta)+mixture[:, None]
        if not retained.any(axis=1).all():
            raise ArithmeticError('Empty AR confidence cover')
    else:
        retained = np.ones((len(y), cells), dtype=bool)
    selected = np.max(np.where(retained, critical[None, :], -np.inf), axis=1)
    estimate, standard_error = geometry.statistics(y)
    width = selected*standard_error
    return Ar1PriorInterval(estimate, standard_error, estimate-width, estimate+width,
                            selected, phis, np.tanh(psi_edges), retained, critical,
                            alpha, delta, float(corrected_alpha), bool(adaptive))
