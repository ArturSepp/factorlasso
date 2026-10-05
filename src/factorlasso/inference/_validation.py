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

"""Validation for fixed linear estimates and Gaussian covariance shapes."""
import numpy as np


def _confidence(value):
    """Validate the confidence convention shared with the alpha application."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not 0 < value < 1:
        raise ValueError('confidence must be strictly between zero and one')
    return float(value)


def _psd_root(matrix, size):
    """Factor PSD covariance, removing only negative eigensolver roundoff."""
    value = np.asarray(matrix, dtype=float)
    if (value.shape != (size, size) or not np.isfinite(value).all()
            or not np.allclose(value, value.T, rtol=1e-12, atol=0)):
        raise ValueError('matrix must be finite, symmetric and match the vector')
    eig, vectors = np.linalg.eigh(value)
    tolerance = 100*np.finfo(float).eps*max(1, size)*np.max(np.abs(eig), initial=0.)
    if eig.min(initial=0.) < -tolerance:
        raise ValueError('matrix must be positive semidefinite; no repair is performed')
    return vectors*np.sqrt(np.maximum(eig, 0.)), eig, tolerance


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



def _responses(responses, size):
    """Validate a complete scalar or batched response, without dropping calendar gaps."""
    values = np.asarray(responses, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if (values.ndim != 2 or values.shape[1] != size or len(values) == 0
            or not np.isfinite(values).all()):
        raise ValueError('responses must be finite with shape (T,) or (N, T); gaps are unsupported')
    return values



def _validate_geometry(geometry):
    """Check the explicit target and mean-removal contract of generic geometry."""
    d = np.asarray(geometry.design, dtype=float)
    if d.ndim != 2 or min(d.shape) == 0 or not np.isfinite(d).all():
        raise ValueError('design must be a nonempty finite matrix')
    n, p = d.shape
    scales = np.linalg.norm(d, axis=0)
    if n <= p or np.any(scales == 0):
        raise ValueError('design must have full rank and positive residual dimension')
    normalized = d/scales
    singular = np.linalg.svd(normalized, compute_uv=False)
    if singular[-1] <= singular[0]*max(d.shape)*np.finfo(float).eps:
        raise ValueError('design must have full column rank')
    h = np.asarray(geometry.linear, dtype=float)
    target = np.asarray(geometry.target_contrast, dtype=float)
    residual = np.asarray(geometry.residual_map, dtype=float)
    if h.shape != (n,) or not np.isfinite(h).all() or not np.any(h):
        raise ValueError('linear must be a nonzero finite observation vector')
    if target.shape != (p,) or not np.isfinite(target).all() or not np.any(target):
        raise ValueError('target_contrast must be a nonzero finite coefficient vector')
    if residual.shape != (n, n) or not np.isfinite(residual).all():
        raise ValueError('residual_map must be a finite square observation matrix')
    q = _symmetric(geometry.quadratic, n, 'quadratic')
    # Normalize columns and operators so changes of factor units do not change checks.
    tolerance = 1e-9
    if not np.allclose(h @ normalized, target/scales, rtol=0,
                       atol=tolerance*np.linalg.norm(h)):
        raise ValueError('linear does not estimate the declared target_contrast')
    rnorm = max(np.linalg.norm(residual), np.finfo(float).tiny)
    qnorm = max(np.linalg.norm(q), np.finfo(float).tiny)
    if np.linalg.norm(residual @ normalized) > tolerance*rnorm:
        raise ValueError('residual_map must annihilate the declared mean design')
    if np.linalg.norm(q @ normalized) > tolerance*qnorm:
        raise ValueError('quadratic must annihilate the declared mean design')
    if np.linalg.norm(q @ residual-q) > tolerance*qnorm*max(1., rnorm):
        raise ValueError('residual_map must preserve the quadratic variance form')
    return d, h, q, residual, target
