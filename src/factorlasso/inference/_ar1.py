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

"""Continuous bounded stationary AR(1) Gaussian calibration.

Tyler (1987), Biometrika 74, 579-589, doi:10.1093/biomet/74.3.579:
angular central Gaussian densities. Berger and Boos (1994), JASA 89,
1012-1016, doi:10.1080/01621459.1994.10476836: nuisance-set error allocation.
The fixed-mixture cover and continuous-cell domination are implementation choices.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.special import logsumexp

from factorlasso.inference._gaussian import _gaussian_critical
from factorlasso.inference._validation import (
    _probability, _responses, _symmetric, _validate_geometry,
)


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
class Ar1Interval:
    """Batch of model-based intervals and their continuous-cell calibration audit.

    Estimate, standard_error, lower, upper and critical_value have shape (N,).
    ``retained_cells`` has shape (N, cells); it contains all cells in full-domain
    mode. ``cell_critical_values`` use the reported ``calibration_alpha``.
    This is a pointwise fixed-target interval, not simultaneous or post-selection.
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



def _ar1_interval(geometry, responses, *, phi_max=.7, cells=401,
                               alpha=.05, adaptive=False, delta=.01):
    """Calibrate intervals over a continuous, bounded stationary AR(1) family.

    Parameters
    ----------
    geometry : geometry-like
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
    Ar1Interval
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
    return Ar1Interval(estimate, standard_error, estimate-width, estimate+width,
                            selected, phis, np.tanh(psi_edges), retained, critical,
                            alpha, delta, float(corrected_alpha), bool(adaptive))



def compute_ar1_interval(geometry, responses, *, phi_max=.7, cells=401,
                         alpha=.05, adaptive=False, delta=.01):
    """Return pointwise intervals over a continuous bounded Gaussian AR(1) family.

    Parameters
    ----------
    geometry : LinearHacGeometry
        Validated fixed target and complete regular-grid mean design.
    responses : array-like, shape (T,) or (N, T)
        Finite responses with mean in the design and stationary Gaussian AR errors.
    phi_max : float, default 0.7
        Declared upper bound on absolute AR coefficient, strictly below one.
    cells : int, default 401
        At least three equal cells in atanh(phi); includes off-grid calibration.
    alpha : float, default 0.05
        Two-sided interval error probability.
    adaptive : bool, default False
        Restrict calibration to a residual-direction confidence cover when True.
        Otherwise maximize over the complete domain and spend all alpha there.
    delta : float, default 0.01
        Cover error when adaptive, strictly below alpha; otherwise ignored.

    Returns
    -------
    Ar1Interval
        Estimates, SEs, interval bounds and continuous-cell audit. Coverage is at
        least 1-alpha under the declared model, subject to numerical tolerances.
        Choosing the narrower realized mode, response selection, irregular grids
        and arbitrary drift are outside this guarantee.
    """
    _validate_geometry(geometry)
    return _ar1_interval(geometry, responses, phi_max=phi_max, cells=cells,
                         alpha=alpha, adaptive=adaptive, delta=delta)
