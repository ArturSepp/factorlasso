"""
Tests for the valid-observation slope/t-statistic computation under
heterogeneous inception dates (leading-NaN panels).

Background
----------
Prior to v0.4.1 the cluster-pooled and per-response sign-derivation paths
zero-filled NaN cells and then computed the slope, SSR, and degrees of
freedom over *all* T rows with nominal sample length. For panels with
leading-NaN prefixes (assets with later inception dates) this:

  * biased the pooled slope (the cross-asset sum mixes assets with
    different valid windows, and the denominator over-counted rows), and
  * inflated the t-statistic (zero-filled rows deflate the residual
    variance because the dof used nominal T, not the valid-row count).

The fix evaluates the slope numerator/denominator, the SSR, and the dof
over genuine (row, response) observations only. These tests pin the
corrected behaviour:

1. On a leading-NaN panel, the per-response slope equals the honest
   drop-NaN univariate OLS slope to machine precision.
2. The gated sign on such a panel matches the honest drop-NaN gate.
3. On a fully observed panel, the corrected code reproduces the previous
   nominal-T formula exactly (backward compatibility).
"""
from __future__ import annotations

import numpy as np

from factorlasso.sign_constraints import (
    _compute_sign_matrix_per_response,
    _compute_sign_vector,
)


def _honest_univariate(x_arr, y_col):
    """Drop-NaN univariate no-intercept OLS slope and t for one response."""
    m = ~np.isnan(y_col)
    xv, yv = x_arr[m], y_col[m]
    xx = (xv * xv).sum(axis=0)
    slope = (xv.T @ yv) / xx
    df = max(m.sum() - 1, 1)
    ssr = (yv * yv).sum() - slope ** 2 * xx
    se = np.sqrt(np.maximum(ssr, 0.0) / df / xx)
    t = np.where(se > 0, slope / se, 0.0)
    return slope, t


def test_per_response_slope_matches_drop_nan():
    rng = np.random.default_rng(1)
    T, N, M = 80, 12, 4
    X = rng.standard_normal((T, M))
    beta = rng.standard_normal((N, M)) * 0.5
    Y = X @ beta.T + 0.4 * rng.standard_normal((T, N))
    Y[:30, :4] = np.nan  # first four assets miss the first 30 observations

    _, slopes = _compute_sign_matrix_per_response(
        X, Y, auto_sign_threshold_t=None, return_slopes=True
    )
    for k in range(N):
        honest_slope, _ = _honest_univariate(X, Y[:, k])
        np.testing.assert_allclose(slopes[k], honest_slope, atol=1e-12)


def test_per_response_gate_matches_drop_nan():
    rng = np.random.default_rng(7)
    T, N, M = 90, 8, 5
    X = rng.standard_normal((T, M))
    beta = rng.standard_normal((N, M)) * 0.4
    Y = X @ beta.T + 0.5 * rng.standard_normal((T, N))
    Y[:40, :3] = np.nan

    tau = 0.75
    signs = _compute_sign_matrix_per_response(X, Y, auto_sign_threshold_t=tau)
    for k in range(N):
        slope, t = _honest_univariate(X, Y[:, k])
        honest = np.where(np.abs(t) >= tau, np.sign(slope), 0.0)
        np.testing.assert_array_equal(signs[k], honest)


def test_full_data_reproduces_nominal_T_formula():
    """No NaN ⇒ corrected code must equal the previous nominal-T formula."""
    rng = np.random.default_rng(3)
    T, N, M = 100, 10, 6
    X = rng.standard_normal((T, M))
    beta = rng.standard_normal((N, M)) * 0.5
    Y = X @ beta.T + 0.3 * rng.standard_normal((T, N))

    tau = 0.75
    signs = _compute_sign_matrix_per_response(X, Y, auto_sign_threshold_t=tau)

    # Reference: old nominal-T behaviour (df = T-1, denom = x'x).
    xx = (X * X).sum(axis=0)
    yy = (Y * Y).sum(axis=0)
    xy = X.T @ Y
    sl = (xy / xx[:, None]).T
    ssr = yy[:, None] - sl * sl * xx[None, :]
    sigma2 = np.maximum(ssr, 0.0) / (T - 1)
    se = np.sqrt(sigma2 / xx[None, :])
    t = sl / se
    ref = np.sign(sl)
    ref[np.abs(t) < tau] = 0.0
    np.testing.assert_array_equal(signs, ref)


def test_cluster_pooled_slope_matches_drop_nan_union():
    """
    Cluster-pooled slope on a leading-NaN panel equals the pooled slope
    computed on the per-response valid union (no zero-fill leakage).
    """
    rng = np.random.default_rng(11)
    T, N, M = 120, 6, 3
    X = rng.standard_normal((T, M))
    beta = rng.standard_normal((N, M)) * 0.5
    Y = X @ beta.T + 0.4 * rng.standard_normal((T, N))
    Y[:50, :2] = np.nan  # two assets late inception

    # Single cluster containing all factors → pooled across all responses.
    clusters = np.zeros(M, dtype=int)
    sign_vec, slopes = _compute_sign_vector(
        x_arr=X, y_arr=Y, clusters=clusters, auto_sign_threshold_t=None
    )

    # Honest pooled slope for factor j: Σ_k Σ_{t valid} x_tj y_tk
    #                                   / Σ_k Σ_{t valid} x_tj²  (x_agg = mean of all factors)
    x_agg = X.mean(axis=1)
    valid_y = ~np.isnan(Y)
    y_filled = np.nan_to_num(Y, nan=0.0)
    num = float(x_agg @ y_filled.sum(axis=1))
    den = float((x_agg ** 2) @ valid_y.sum(axis=1))
    honest = num / den
    # All factors in one cluster share the aggregated slope.
    np.testing.assert_allclose(slopes, np.full(M, honest), atol=1e-10)
