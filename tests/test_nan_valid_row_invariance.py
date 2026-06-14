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


# ─────────────────────────────────────────────────────────────────────────
# NaN in X (factor columns): the closed-form SSR must mask the response
# sum-of-squares by valid_x, exactly as the denominator and dof are masked.
#
# Regression for the bug where the SSR used a per-response (or global) Σ_t y²
# not restricted to rows where the factor is observed. Under a NaN-bearing
# factor this over-counts SSR_j, inflates σ², shrinks |t|, and over-gates that
# factor's sign constraint. The slope path was never affected; only the gate.
# Each test builds a scenario in which the over-count flips a gate decision and
# asserts (a) the fixed code matches the honest drop-NaN gate, and (b) that
# honest gate differs from the pre-fix global-Σy² gate, so the test cannot pass
# on buggy code.
# ─────────────────────────────────────────────────────────────────────────


def _honest_per_response_xy(x_arr, y_arr, tau):
    """Drop-NaN per-(k, j) univariate gate, masking by BOTH x_j and y_k."""
    N = y_arr.shape[1]
    M = x_arr.shape[1]
    slopes = np.zeros((N, M))
    signs = np.zeros((N, M))
    for k in range(N):
        for j in range(M):
            m = (~np.isnan(x_arr[:, j])) & (~np.isnan(y_arr[:, k]))
            if m.sum() == 0:
                continue
            xj, yk = x_arr[m, j], y_arr[m, k]
            xx = float((xj * xj).sum())
            if xx <= 0:
                continue
            b = float(xj @ yk) / xx
            slopes[k, j] = b
            df = max(int(m.sum()) - 1, 1)
            ssr = float((yk * yk).sum()) - b * b * xx
            se = np.sqrt(max(ssr, 0.0) / df / xx)
            t = b / se if se > 0 else 0.0
            signs[k, j] = 0.0 if abs(t) < tau else np.sign(b)
    return slopes, signs


def _buggy_per_response_signs(x_arr, y_arr, tau):
    """Pre-fix gate: per-response Σ_t y² over y-valid rows, NOT x_j-restricted."""
    N = y_arr.shape[1]
    M = x_arr.shape[1]
    signs = np.zeros((N, M))
    for k in range(N):
        yk_all = y_arr[~np.isnan(y_arr[:, k]), k]
        yy = float((yk_all * yk_all).sum())
        for j in range(M):
            m = (~np.isnan(x_arr[:, j])) & (~np.isnan(y_arr[:, k]))
            if m.sum() == 0:
                continue
            xj, yk = x_arr[m, j], y_arr[m, k]
            xx = float((xj * xj).sum())
            if xx <= 0:
                continue
            b = float(xj @ yk) / xx
            df = max(int(m.sum()) - 1, 1)
            ssr = yy - b * b * xx          # BUG: global per-response Σ y²
            se = np.sqrt(max(ssr, 0.0) / df / xx)
            t = b / se if se > 0 else 0.0
            signs[k, j] = 0.0 if abs(t) < tau else np.sign(b)
    return signs


def _pooled_gate(x_arr, y_arr, tau, buggy):
    """Drop-NaN cluster-pooled gate; buggy=True uses the global Σy² over-count."""
    M = x_arr.shape[1]
    N = y_arr.shape[1]
    q_eff = int((np.sum(~np.isnan(y_arr), axis=0) > 0).sum())
    yss_global = 0.0
    for k in range(N):
        yk = y_arr[~np.isnan(y_arr[:, k]), k]
        yss_global += float((yk * yk).sum())
    slopes = np.zeros(M)
    signs = np.zeros(M)
    for j in range(M):
        vx = ~np.isnan(x_arr[:, j])
        num = den = yss = 0.0
        nval = 0
        for k in range(N):
            m = vx & (~np.isnan(y_arr[:, k]))
            xj, yk = x_arr[m, j], y_arr[m, k]
            num += float(xj @ yk)
            den += float((xj * xj).sum())
            yss += float((yk * yk).sum())
            nval += int(m.sum())
        if den <= 0:
            continue
        b = num / den
        slopes[j] = b
        df = max(nval - q_eff, 1)
        ssr = (yss_global if buggy else yss) - b * b * den
        se = np.sqrt(max(ssr, 0.0) / df / den)
        t = b / se if se > 0 else 0.0
        signs[j] = 0.0 if abs(t) < tau else np.sign(b)
    return slopes, signs


def test_per_response_gate_matches_drop_nan_with_nan_in_x():
    # Factor 0 is NaN on the first 30 rows, which also carry very-large-variance
    # y; the pre-fix global Σ y² inflates SSR_0 and (wrongly) gates factor 0.
    rng = np.random.default_rng(7)
    T, N, M = 60, 5, 3
    X = rng.standard_normal((T, M))
    beta = np.tile(np.array([0.9, 0.0, 0.6]), (N, 1))
    Y = X @ beta.T + 0.25 * rng.standard_normal((T, N))
    Y[:30, :] += 15.0 * rng.standard_normal((30, N))    # large early variance
    X[:30, 0] = np.nan                                 # factor 0 late inception

    tau = 0.75
    signs, slopes = _compute_sign_matrix_per_response(
        X, Y, auto_sign_threshold_t=tau, return_slopes=True
    )
    h_slopes, h_signs = _honest_per_response_xy(X, Y, tau)
    buggy = _buggy_per_response_signs(X, Y, tau)

    np.testing.assert_allclose(slopes, h_slopes, atol=1e-10)
    np.testing.assert_array_equal(signs, h_signs)
    assert not np.array_equal(h_signs, buggy)          # scenario triggers the bug
    assert (signs[:, 0] != 0).any()                    # corrected gate keeps factor 0


def test_cluster_pooled_gate_matches_drop_nan_with_nan_in_x():
    rng = np.random.default_rng(11)
    T, q, M = 60, 4, 3
    X = rng.standard_normal((T, M))
    beta = np.tile(np.array([0.9, 0.0, 0.6]), (q, 1))
    Y = X @ beta.T + 0.25 * rng.standard_normal((T, q))
    Y[:30, :] += 15.0 * rng.standard_normal((30, q))
    X[:30, 0] = np.nan

    tau = 0.75
    sign_vec, slopes = _compute_sign_vector(
        x_arr=X, y_arr=Y, clusters=None, master_constraints=None,
        auto_sign_threshold_t=tau,
    )
    h_slopes, h_signs = _pooled_gate(X, Y, tau, buggy=False)
    _, buggy_signs = _pooled_gate(X, Y, tau, buggy=True)

    np.testing.assert_allclose(slopes, h_slopes, atol=1e-10)
    np.testing.assert_array_equal(sign_vec, h_signs)
    assert not np.array_equal(h_signs, buggy_signs)    # scenario triggers the bug
    assert sign_vec[0] != 0                            # corrected gate keeps factor 0


def test_full_data_ssr_unchanged_by_valid_x_mask():
    # With every factor fully observed the per-factor mask is a no-op: the
    # complete-X fast path must reproduce the honest global-Σy² gate exactly.
    rng = np.random.default_rng(3)
    T, N, M = 90, 8, 4
    X = rng.standard_normal((T, M))
    beta = rng.standard_normal((N, M)) * 0.5
    Y = X @ beta.T + 0.4 * rng.standard_normal((T, N))
    tau = 0.75
    signs = _compute_sign_matrix_per_response(X, Y, auto_sign_threshold_t=tau)
    _, h_signs = _honest_per_response_xy(X, Y, tau)
    np.testing.assert_array_equal(signs, h_signs)
