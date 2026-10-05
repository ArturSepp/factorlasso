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

"""Linear estimates and Bartlett HAC forms for fixed weighted regression.

Newey and West (1987), Econometrica 55, 703-708, doi:10.2307/1913610.
The n/(n-p) correction is retained from prior statistics, not a coverage theorem.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from factorlasso.inference._validation import _responses, _validate_geometry
from factorlasso.utils._hac import bartlett_kernel, score_covariance


def _geometry_statistics(geometry, responses):
    """Evaluate the estimate and variance after removing the fitted mean."""
    y = _responses(responses, len(geometry.design))
    residual = y @ geometry.residual_map.T
    variance = np.einsum('ij,jk,ik->i', residual, geometry.quadratic, residual)
    return y @ geometry.linear, np.sqrt(np.maximum(variance, 0.))


@dataclass(frozen=True)
class LinearHacGeometry:
    """Fixed target h'y and variance y'Qy with an explicit mean model D theta.

    The target is target_contrast' theta. Construction checks h'D=c', QD=0,
    and residual-map consistency. Arrays are owned and read-only. Statistical
    independence of the design and response noise remains a caller assumption.
    Build with compute_wls_hac_geometry for WLS or provide validated forms for
    another unbiased linear estimator under the declared mean model.
    """

    design: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray
    residual_map: np.ndarray
    target_contrast: np.ndarray

    def __post_init__(self):
        """Validate and own the arrays that define the inference target."""
        values = _validate_geometry(self)
        names = ('design', 'linear', 'quadratic', 'residual_map', 'target_contrast')
        for name, value in zip(names, values):
            owned = value.copy()
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)

    def statistics(self, responses):
        """Return estimate and HAC SE arrays for responses of shape (T,) or (N, T)."""
        return _geometry_statistics(self, responses)


def _wls_geometry_arrays(design, weights=None, hac_lags=0, coefficient=1, contrast=None):
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
    contrast : ndarray, shape (P,), optional
        Validated coefficient contrast; when supplied, it replaces coefficient.

    Returns
    -------
    tuple of ndarray
        Design, linear target, quadratic variance and residual map. Uses the
        existing prior statistic's n_obs/(n_obs-P) correction. No
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
    h = hmap[coefficient] if contrast is None else contrast @ hmap
    residual = np.eye(n)-d @ hmap
    kernel = bartlett_kernel(np.arange(n), hac_lags+1)
    influence = h[:, None]*residual
    q = nobs/(nobs-p)*score_covariance(influence, kernel)
    return d.copy(), h.copy(), (q+q.T)/2, residual



def compute_wls_hac_geometry(design, weights=None, hac_lags=0, *, coefficient=None,
                             contrast=None):
    """Build fixed WLS geometry for one coefficient or a linear contrast.

    Parameters
    ----------
    design : array-like, shape (T, P)
        Complete fixed design, including an intercept column if intended.
    weights : array-like, shape (T,), optional
        Nonnegative loss weights; None means equal weights. Scaling cancels.
    hac_lags : int, default 0
        Bartlett bandwidth on the original row grid; zero gives robust variance.
    coefficient : int, optional
        Zero-based column to infer. Supply exactly one of coefficient or contrast.
    contrast : array-like, shape (P,), optional
        Nonzero fixed coefficient contrast. No response-driven selection is covered.

    Returns
    -------
    LinearHacGeometry
        Dense linear/quadratic forms with n/(n-P) correction. Missing rows are
        unsupported. Use compute_wls_hac_statistics when dense forms are unnecessary.
    """
    if (coefficient is None) == (contrast is None):
        raise ValueError('supply exactly one of coefficient or contrast')
    d = np.asarray(design, dtype=float)
    if d.ndim != 2 or min(d.shape) == 0:
        raise ValueError('design must be a nonempty matrix')
    if contrast is None:
        arrays = _wls_geometry_arrays(d, weights, hac_lags, coefficient)
        target = np.eye(d.shape[1])[coefficient]
    else:
        target = np.asarray(contrast, dtype=float)
        if (target.shape != (d.shape[1],) or not np.isfinite(target).all()
                or not np.any(target)):
            raise ValueError('contrast must be a nonzero finite coefficient vector')
        arrays = _wls_geometry_arrays(d, weights, hac_lags, 0, target)
    return LinearHacGeometry(*arrays, target)
