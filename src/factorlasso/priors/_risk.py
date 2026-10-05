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

"""Optional two-factor floor-risk and Gaussian information diagnostics.

The radius is an analytic result for zero-centred, nonnegative two-factor
Lasso with half-quadratic loss. It is not a default or tuning theorem for
prior-centred FCGL. First let correlation tend to one, then let signal/noise
increase; a joint limit additionally needs theta*sqrt(1-rho)/s_B -> 0.
Bounded prior offsets are fixed, not selected from the observed noise.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from factorlasso.inference._validation import _symmetric


def _nonnegative(value, name):
    """Validate a finite nonnegative scalar budget or penalty."""
    if not np.isscalar(value) or not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    return float(value)


def two_factor_minimax_radius(secondary_bound, bias_bound, penalty):
    """Return the minimax radius in the iid strong-signal, weak-contrast limit.

    Parameters
    ----------
    secondary_bound : float
        d >= 0 bounds the fixed secondary exposure eta in [0, d].
    bias_bound : float
        tau >= 0 bounds a fixed additive error e in the marginal OLS prior.
    penalty : float
        lambda >= 0 in 0.5*b'G*b - z'b + lambda*(b1+b2), b >= 0.
        This is not automatically the literal LassoModel regularization scale.

    Returns
    -------
    float
        Radius r >= 0 for the floor (z1 - rho*d + e - r)_+.
        The result assumes independent marginal and orthogonal Gaussian score
        components, correct primary-factor identity, fixed d/tau/penalty and
        the limiting experiment stated in this module. It is not an SE multiplier.
    """
    d = _nonnegative(secondary_bound, 'secondary_bound')
    tau = _nonnegative(bias_bound, 'bias_bound')
    lam = _nonnegative(penalty, 'penalty')
    # Normalize to protect large, but finite, budgets against quadratic overflow.
    scale = max(d, tau, lam)
    if scale == 0:
        return 0.
    d, tau, lam = d/scale, tau/scale, lam/scale
    numerator = (lam-2*d)*tau-lam*(lam-d)
    if lam == 0 or d >= lam/2 or numerator <= 0:
        return 0.
    return float(scale*numerator/(4*tau+2*d-lam))


def two_factor_limit_risk(radius, secondary_bound, bias_bound, penalty,
                          score_variance=1., cross_moment=0.):
    """Return worst limiting coefficient risk over fixed exposure and bias budgets.

    Parameters
    ----------
    radius, secondary_bound, bias_bound, penalty : float
        Nonnegative r, d, tau and lambda in ``two_factor_minimax_radius``.
    score_variance : float, default 1
        Variance of marginal score noise A, strictly positive.
    cross_moment : float, default 0
        E[A*1{B>0}] = Cov(A,B)/(sd(B)*sqrt(2*pi)), where B is the orthogonal
        Gaussian score component. Zero gives the iid formula. A nonzero value
        need not have the optimum returned by ``two_factor_minimax_radius``.

    Returns
    -------
    float
        Maximum limiting coefficient MSE over eta in [0,d] and e in [-tau,tau].
        This is neither finite-sample minimax risk nor prediction error.
    """
    r = _nonnegative(radius, 'radius')
    d = _nonnegative(secondary_bound, 'secondary_bound')
    tau = _nonnegative(bias_bound, 'bias_bound')
    lam = _nonnegative(penalty, 'penalty')
    variance = _nonnegative(score_variance, 'score_variance')
    if variance == 0:
        raise ValueError('score_variance must be positive')
    if (not np.isscalar(cross_moment) or not np.isfinite(cross_moment)
            or abs(cross_moment) > np.sqrt(variance/(2*np.pi))):
        raise ValueError('cross_moment violates the Gaussian covariance bound')
    left, right = d+r-tau, d+r+tau
    candidates = [left, right]
    if left <= lam <= right:
        candidates.append(lam)
    values = []
    for a in candidates:
        for eta in (0., d):
            value = variance+(eta-a)**2+eta**2
            if a > lam:
                value += lam*(lam-a)-2*cross_moment*(a-lam)
            values.append(value)
    return float(max(values))


@dataclass(frozen=True)
class GaussianDominanceInformation:
    """Known-covariance Gaussian information after profiling an unknown intercept.

    ``sole_variance`` is 1/I11, ``coupling`` is I12/I11, and
    ``secondary_variance`` is 1/(I22-I12**2/I11). All coefficients must use
    the same factor units as the weighted fit to which this is compared.
    """

    information: np.ndarray
    sole_variance: float
    coupling: float
    secondary_variance: float

    def risk_rate(self, secondary_bound):
        """Return R with R/32 <= minimax coefficient risk <= 2R.

        The class has theta >= 0, 0 <= eta <= secondary_bound, an unrestricted
        intercept and known Gaussian covariance. This is an oracle information
        benchmark, not a guarantee for a particular EWMA estimator.
        """
        d = _nonnegative(secondary_bound, 'secondary_bound')
        return float(self.sole_variance+(1+self.coupling**2)
                     *min(d**2, self.secondary_variance))


def gaussian_dominance_information(factors, covariance):
    """Compute the two-factor known-Gaussian minimax information benchmark.

    Parameters
    ----------
    factors : array-like, shape (T, 2)
        Complete finite factor panel. An unrestricted intercept is profiled
        internally. Coefficient units are those of these original columns.
    covariance : array-like, shape (T, T)
        Known full Gaussian noise covariance, including its scale.

    Returns
    -------
    GaussianDominanceInformation
        Information and variance components computed with whitening and stable
        residual contrasts. Requires an identifiable intercept and two factors.
    """
    x = np.asarray(factors, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('factors must be a finite T by 2 matrix with T >= 3')
    omega = _symmetric(covariance, len(x), 'covariance', True)
    chol = np.linalg.cholesky(omega)
    one = solve_triangular(chol, np.ones(len(x)), lower=True)
    whitened = solve_triangular(chol, x, lower=True)
    residual = whitened-np.outer(one, one @ whitened)/(one @ one)
    scales = np.linalg.norm(residual, axis=0)
    if np.any(scales == 0) or np.linalg.matrix_rank(residual/scales) < 2:
        raise ValueError('intercept and both factor coefficients must be identifiable')
    information = residual.T @ residual
    coupling = information[0, 1]/information[0, 0]
    contrast = residual[:, 1]-coupling*residual[:, 0]
    return GaussianDominanceInformation(information, float(1/information[0, 0]),
                                         float(coupling), float(1/(contrast @ contrast)))
