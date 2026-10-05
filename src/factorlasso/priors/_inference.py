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

"""Prior-facing compatibility types and fixed-model inference entry points.

Numerical implementations live in factorlasso.inference. Existing class module
paths, fields, signatures and validation behaviour remain available here.
"""
from dataclasses import dataclass

import numpy as np

from factorlasso.inference._ar1 import _ar1_interval
from factorlasso.inference._gaussian import _known_shape_critical
from factorlasso.inference._geometry import _geometry_statistics, _wls_geometry_arrays


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
        return _geometry_statistics(self, responses)


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
    return PriorHacGeometry(*_wls_geometry_arrays(design, weights, hac_lags, coefficient))



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
    return _known_shape_critical(geometry, covariance_shape, alpha)



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
    result = _ar1_interval(geometry, responses, phi_max=phi_max, cells=cells,
                           alpha=alpha, adaptive=adaptive, delta=delta)
    return Ar1PriorInterval(**vars(result))
