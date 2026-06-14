"""
Regression tests for three v0.5.1 fixes.

1. Cluster-mode sign derivation under NaN-bearing x: the slope numerator
   must be masked by ``x_agg_valid`` exactly as the denominator is. The
   pre-fix code included rows with partially-missing cluster members in
   the numerator (with a biased zero-filled aggregate) while excluding
   them from the denominator, inflating the slope by up to ~40% in the
   half-missing-member scenario below.

2. ``LassoModel.fit`` accepts a 1-D ndarray ``x`` as one regressor of
   length T, mirroring the 1-D ``y`` and ``pd.Series`` conventions. The
   pre-fix code turned shape ``(T,)`` into a ``(1, T)`` row and failed
   on index alignment.

3. ``LassoModel.copy`` returns a fresh, unfitted estimator and does not
   corrupt ``estimation_result_`` into a plain dict.
"""
import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType
from factorlasso.lasso_estimator import LassoEstimationResult
from factorlasso.sign_constraints import _compute_sign_vector


# ─────────────────────────────────────────────────────────────────────────
# 1. Cluster-mode slope under heterogeneous NaN in x
# ─────────────────────────────────────────────────────────────────────────

def _honest_cluster_slope(x_arr, y_arr, idx):
    """Valid-row pooled slope of the cluster-mean regressor.

    Rows enter only where every cluster member is observed; response
    cells enter only where y is observed.
    """
    agg_valid = ~np.isnan(x_arr[:, idx]).any(axis=1)
    x_agg = np.nanmean(x_arr[:, idx], axis=1)
    valid_y = ~np.isnan(y_arr)
    y_filled = np.nan_to_num(y_arr, nan=0.0)
    num = float((x_agg * agg_valid) @ y_filled.sum(axis=1))
    den = float(((x_agg ** 2) * agg_valid) @ valid_y.sum(axis=1))
    return num / den


def test_cluster_slope_masks_numerator_with_nan_in_x():
    """Slope equals the honest valid-row estimator when one cluster
    member has a leading-NaN prefix (heterogeneous inception)."""
    rng = np.random.default_rng(1)
    T, M = 300, 4
    x = rng.standard_normal((T, M))
    x[:150, 1] = np.nan                       # member 1 launches mid-sample
    beta_true = np.array([0.8, 0.8, -0.6, -0.6])
    y = (np.nan_to_num(x) @ beta_true).reshape(-1, 1) \
        + 0.3 * rng.standard_normal((T, 1))
    clusters = np.array([0, 0, 1, 1])

    _, slopes = _compute_sign_vector(
        x_arr=x.copy(), y_arr=y.copy(),
        clusters=clusters, auto_sign_threshold_t=None,
    )

    ref0 = _honest_cluster_slope(x, y, [0, 1])
    ref1 = _honest_cluster_slope(x, y, [2, 3])
    np.testing.assert_allclose(slopes[[0, 1]], ref0, atol=1e-12)
    np.testing.assert_allclose(slopes[[2, 3]], ref1, atol=1e-12)


def test_cluster_slope_unchanged_on_complete_x():
    """On fully observed x the fix is a no-op: the masked and unmasked
    numerators coincide bit-for-bit."""
    rng = np.random.default_rng(2)
    T, M = 200, 4
    x = rng.standard_normal((T, M))
    y = rng.standard_normal((T, 3))
    clusters = np.array([0, 0, 1, 1])

    _, slopes = _compute_sign_vector(
        x_arr=x.copy(), y_arr=y.copy(),
        clusters=clusters, auto_sign_threshold_t=None,
    )
    ref0 = _honest_cluster_slope(x, y, [0, 1])
    ref1 = _honest_cluster_slope(x, y, [2, 3])
    np.testing.assert_allclose(slopes[[0, 1]], ref0, atol=1e-12)
    np.testing.assert_allclose(slopes[[2, 3]], ref1, atol=1e-12)


def test_cluster_gate_consistent_with_fixed_slope_under_nan_x():
    """The t-gate decision under NaN-bearing x matches a brute-force
    valid-row computation of slope, SSR, and dof.

    The scenario plants high-variance noise on exactly the rows where the
    cluster aggregate is undefined. Pre-fix, those rows leaked into the
    numerator and inflated the slope; the resulting (slope, SSR) pair
    violated the closed-form identity and could flip the gate.
    """
    rng = np.random.default_rng(3)
    T, M, q = 240, 2, 5
    x = rng.standard_normal((T, M))
    x[:120, 1] = np.nan
    beta = 0.35
    y = beta * np.nan_to_num(x[:, [0]]) + 0.5 * rng.standard_normal((T, q))
    y[:120, :] += 5.0 * rng.standard_normal((120, q))
    clusters = np.array([0, 0])
    tau = 0.75

    sign_vec, slopes = _compute_sign_vector(
        x_arr=x.copy(), y_arr=y.copy(),
        clusters=clusters, auto_sign_threshold_t=tau,
    )

    # Brute-force valid-row reference
    agg_valid = ~np.isnan(x).any(axis=1)
    x_agg = np.nanmean(x, axis=1)
    valid_y = ~np.isnan(y)
    y_f = np.nan_to_num(y)
    num = float((x_agg * agg_valid) @ y_f.sum(axis=1))
    D = float(((x_agg ** 2) * agg_valid) @ valid_y.sum(axis=1))
    slope = num / D
    Y_ss = float(agg_valid.astype(float) @ (y_f * y_f).sum(axis=1))
    ssr = Y_ss - slope * slope * D
    n_valid = float((agg_valid[:, None] & valid_y).sum())
    q_eff = int((valid_y.sum(axis=0) > 0).sum())
    sigma2 = max(ssr, 0.0) / max(n_valid - q_eff, 1.0)
    t_ref = slope / np.sqrt(sigma2 / D)

    np.testing.assert_allclose(slopes, slope, atol=1e-12)
    expected_sign = np.sign(slope) if abs(t_ref) >= tau else 0.0
    np.testing.assert_array_equal(sign_vec, np.full(M, expected_sign))


# ─────────────────────────────────────────────────────────────────────────
# 2. 1-D ndarray x
# ─────────────────────────────────────────────────────────────────────────

def test_fit_accepts_1d_ndarray_x():
    rng = np.random.default_rng(4)
    T = 80
    x = rng.standard_normal(T)
    y = 0.7 * x + 0.1 * rng.standard_normal(T)
    model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-8)
    model.fit(x=x, y=y)
    assert model.coef_.shape == (1, 1)
    assert abs(float(model.coef_.iloc[0, 0]) - 0.7) < 0.05


def test_fit_1d_x_matches_series_x():
    rng = np.random.default_rng(5)
    T = 80
    x = rng.standard_normal(T)
    y = -0.4 * x + 0.1 * rng.standard_normal(T)
    m_arr = LassoModel(reg_lambda=1e-8).fit(x=x, y=y)
    m_ser = LassoModel(reg_lambda=1e-8).fit(
        x=pd.Series(x, name="x0"), y=pd.Series(y, name="y0"),
    )
    np.testing.assert_allclose(
        m_arr.coef_.to_numpy(), m_ser.coef_.to_numpy(), atol=1e-10,
    )


# ─────────────────────────────────────────────────────────────────────────
# 3. copy() semantics
# ─────────────────────────────────────────────────────────────────────────

def test_copy_is_unfitted_and_preserves_result_type():
    rng = np.random.default_rng(6)
    T, M, N = 100, 3, 4
    X = pd.DataFrame(rng.standard_normal((T, M)),
                     columns=[f"f{j}" for j in range(M)])
    Y = pd.DataFrame(rng.standard_normal((T, N)),
                     columns=[f"a{k}" for k in range(N)])
    m1 = LassoModel(reg_lambda=1e-4).fit(x=X, y=Y)
    assert isinstance(m1.estimation_result_, LassoEstimationResult)

    m2 = m1.copy(kwargs={"reg_lambda": 1e-3})
    assert m2.reg_lambda == 1e-3
    assert m1.reg_lambda == 1e-4
    # Fresh estimator: no stale fitted state
    assert m2.coef_ is None
    assert m2.estimation_result_ is None
    # Fitting the copy restores a proper result dataclass
    m2.fit(x=X, y=Y)
    assert isinstance(m2.estimation_result_, LassoEstimationResult)


def test_copy_of_unfitted_model_roundtrips_params():
    m1 = LassoModel(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-3, cutoff_fraction=0.4,
        auto_sign_constraints=True, auto_sign_threshold_t=1.0,
        l1_weight=0.1,
    )
    m2 = m1.copy()
    assert m2.get_params() == m1.get_params()


# ─────────────────────────────────────────────────────────────────────────
# 4. predict() carries the economic intercept (v0.5.1)
# ─────────────────────────────────────────────────────────────────────────

def test_predict_recovers_response_mean():
    """Predictions in original units carry the asset mean via
    ``alpha_const_``; score() reaches the OLS-equivalent R²."""
    rng = np.random.default_rng(7)
    T = 200
    X = pd.DataFrame(rng.standard_normal((T, 2)), columns=["f0", "f1"])
    y = pd.DataFrame(
        X.values @ np.array([[1.0, 0.5]]).T + 0.5
        + 0.05 * rng.standard_normal((T, 1)),
        columns=["a"],
    )
    m = LassoModel(reg_lambda=1e-8).fit(x=X, y=y)
    y_hat = m.predict(X)
    assert abs(float(y_hat["a"].mean()) - float(y["a"].mean())) < 1e-6
    assert m.score(X, y) > 0.99


def test_predict_through_origin_when_demean_false():
    """demean=False is a through-origin fit: predict adds no constant."""
    rng = np.random.default_rng(8)
    T = 150
    X = pd.DataFrame(rng.standard_normal((T, 1)), columns=["f0"])
    y = pd.DataFrame(0.8 * X.values, columns=["a"])
    m = LassoModel(reg_lambda=1e-8, demean=False).fit(x=X, y=y)
    y_hat = m.predict(X)
    np.testing.assert_allclose(
        y_hat.to_numpy(), X.to_numpy() @ m.coef_.to_numpy().T, atol=1e-12,
    )


# ─────────────────────────────────────────────────────────────────────────
# 5. Valid-window demeaning under leading NaN (v0.5.1)
# ─────────────────────────────────────────────────────────────────────────

def test_demean_uses_valid_window_mean():
    """The demeaned response has zero mean over its valid window. The
    pre-fix code demeaned by the zero-diluted mean f·μ, leaving a
    residual offset of (1 − f)·μ on the valid rows."""
    from factorlasso.lasso_estimator import get_x_y_np
    rng = np.random.default_rng(9)
    T = 200
    x = pd.DataFrame(rng.standard_normal((T, 2)), columns=["f0", "f1"])
    y = pd.DataFrame(
        x.values @ np.array([[1.0, 0.5]]).T - 0.10
        + 0.1 * rng.standard_normal((T, 1)),
        columns=["a"],
    )
    y.iloc[:100] = np.nan
    _, y_np, mask = get_x_y_np(x, y, span=None, demean=True)
    valid = mask[:, 0] > 0
    assert abs(float(y_np[valid, 0].mean())) < 1e-10


def test_demean_full_panel_unchanged():
    """On a fully observed panel the fix is a no-op: the NaN-preserved
    mean equals the zero-filled mean bit-for-bit."""
    from factorlasso.lasso_estimator import get_x_y_np
    rng = np.random.default_rng(10)
    T = 120
    x = pd.DataFrame(rng.standard_normal((T, 3)))
    y = pd.DataFrame(rng.standard_normal((T, 4)))
    x_np, y_np, mask = get_x_y_np(x, y, span=None, demean=True)
    np.testing.assert_allclose(
        y_np, y.to_numpy() - y.to_numpy().mean(axis=0), atol=1e-14,
    )
    assert mask.all()


# ─────────────────────────────────────────────────────────────────────────
# 6. span=None clustering uses the sample Pearson correlation (v0.5.1)
# ─────────────────────────────────────────────────────────────────────────

def test_span_none_clusters_on_pearson_corr():
    """With uniform loss weights the HCGL partition equals Ward clustering
    of the sample Pearson correlation, as documented in the paper. The
    pre-fix code routed span=None through ``compute_ewm_covar`` whose
    ewm_lambda=0.94 default produced a trailing-window correlation."""
    from factorlasso import compute_clusters_from_corr_matrix
    rng = np.random.default_rng(11)
    T, N, M = 120, 30, 5
    X = pd.DataFrame(rng.standard_normal((T, M)))
    beta = rng.standard_normal((N, M)) * 0.5
    Y = pd.DataFrame(X.values @ beta.T + 0.4 * rng.standard_normal((T, N)),
                     columns=[f"a{k}" for k in range(N)])
    m = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                   reg_lambda=1e-4).fit(x=X, y=Y)
    ref, _, _ = compute_clusters_from_corr_matrix(Y.corr())
    # Identical partition (labels may permute; compare co-membership)
    a = m.clusters_.reindex(Y.columns).to_numpy()
    b = ref.reindex(Y.columns).to_numpy()
    co_a = a[:, None] == a[None, :]
    co_b = b[:, None] == b[None, :]
    np.testing.assert_array_equal(co_a, co_b)


def test_span_set_clusters_on_ewm_corr():
    """With an EWMA span the partition equals Ward clustering of the
    EWMA(span) correlation — the pre-existing contract, unchanged."""
    from factorlasso import compute_clusters_from_corr_matrix
    from factorlasso.ewm_utils import compute_ewm_covar
    from factorlasso.lasso_estimator import get_x_y_np
    rng = np.random.default_rng(12)
    T, N, M = 150, 20, 4
    X = pd.DataFrame(rng.standard_normal((T, M)))
    beta = rng.standard_normal((N, M)) * 0.5
    Y = pd.DataFrame(X.values @ beta.T + 0.4 * rng.standard_normal((T, N)),
                     columns=[f"a{k}" for k in range(N)])
    span = 36
    m = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                   reg_lambda=1e-4, span=span).fit(x=X, y=Y)
    _, y_np, mask = get_x_y_np(X, Y, span=span, demean=True)
    y_for_corr = np.where(mask > 0, y_np, np.nan)
    corr = compute_ewm_covar(a=y_for_corr, span=span, is_corr=True)
    ref, _, _ = compute_clusters_from_corr_matrix(
        pd.DataFrame(corr, index=Y.columns, columns=Y.columns))
    a = m.clusters_.reindex(Y.columns).to_numpy()
    b = ref.reindex(Y.columns).to_numpy()
    np.testing.assert_array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])
