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

"""Known-shape Gaussian calibration of a centred linear/quadratic pivot.

Imhof (1961), Biometrika 48, 419-426, doi:10.1093/biomet/48.3-4.419.
This uses a specialized one-positive-eigenvalue integral, not general Imhof
inversion. Coverage requires the fixed correct Gaussian model up to scale.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from factorlasso.inference._validation import _probability, _symmetric, _validate_geometry


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



def _known_shape_critical(geometry, covariance_shape, alpha=.05):
    """Calibrate a two-sided interval under a known Gaussian covariance shape.

    Parameters
    ----------
    geometry : geometry-like
        Fixed linear target and quadratic variance form supplied by an adapter.
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



def gaussian_critical_value(geometry, covariance_shape, alpha=.05):
    """Calibrate a two-sided multiplier under known Gaussian covariance shape.

    Parameters
    ----------
    geometry : LinearHacGeometry
        Fixed mean and target contract. Validated before calibration.
    covariance_shape : array-like, shape (T, T)
        Known positive-definite V in noise covariance sigma squared times V.
        Plugging in an estimated V does not inherit the coverage guarantee.
    alpha : float, default 0.05
        Two-sided error probability strictly between zero and one.

    Returns
    -------
    float
        k for estimate +/- k * HAC_SE. Calibration is finite-model Gaussian,
        subject to floating-point eigenvalue, quadrature and root tolerances.
    """
    _validate_geometry(geometry)
    return _known_shape_critical(geometry, covariance_shape, alpha)
