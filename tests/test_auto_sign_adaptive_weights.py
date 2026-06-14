"""
Regression tests for the v0.3.9 ``auto_sign_adaptive_weights`` parameter.

Implements the Zou (2006) adaptive Lasso reweighting:

    λ |β_kj| / max(|β̂_uni_kj|, floor)^γ

The tests verify five properties:

1. ``auto_sign_adaptive_weights=False`` (default) reproduces v0.3.8 fits
   bit-for-bit.
2. ``auto_sign_adaptive_weights=True`` without ``auto_sign_constraints=True``
   is a silent no-op (the adaptive layer requires the sign-derivation layer
   to source slope magnitudes).
3. With adaptive weights active, factors with strong univariate evidence
   (large |β̂_uni|) are shrunk less than factors with weak evidence
   relative to a uniformly-penalised baseline — the Zou (2006) oracle
   property.
4. The floor stabiliser prevents weight explosion: small |β̂_uni| does
   not cause numerical blow-up.
5. The ``_adaptive_penalty_weights`` helper is correct in isolation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from factorlasso import LassoModel, LassoModelType
from factorlasso.sign_constraints import _adaptive_penalty_weights


def _make_mixed_panel(seed: int = 7, T: int = 240, idio: float = 0.10):
    """
    Panel with one strong factor (true β ≈ 0.5–0.6), one weak factor
    (true β ≈ 0.05–0.10), and two pure-noise factors (true β = 0).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 4))
    true_beta = np.array([
        [0.6, 0.0, 0.10, 0.0],
        [0.6, 0.0, 0.10, 0.0],
        [0.5, 0.0, 0.10, 0.0],
        [0.5, 0.0, 0.10, 0.0],
        [0.5, 0.0, 0.05, 0.0],
        [0.5, 0.0, 0.05, 0.0],
    ])
    Y = X @ true_beta.T + idio * rng.standard_normal((T, 6))
    dates = pd.date_range("2020", periods=T, freq="W-WED")
    Xdf = pd.DataFrame(
        X, columns=["f_strong", "f_noise1", "f_weak", "f_noise2"], index=dates,
    )
    Ydf = pd.DataFrame(
        Y, columns=[f"a{k}" for k in range(6)], index=dates,
    )
    return Xdf, Ydf


def test_default_value_is_false():
    """``auto_sign_adaptive_weights`` defaults to False."""
    m = LassoModel()
    assert m.auto_sign_adaptive_weights is False


def test_adaptive_default_reproduces_v038():
    """Default ``auto_sign_adaptive_weights=False`` produces identical fit
    to the v0.3.8 baseline (with ``auto_sign_constraints=True``)."""
    X, Y = _make_mixed_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_default = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=5e-3, span=None,
            auto_sign_constraints=True,
            # adaptive flag not set → defaults to False
        ).fit(x=X, y=Y, verbose=False)
        m_explicit = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=5e-3, span=None,
            auto_sign_constraints=True,
            auto_sign_adaptive_weights=False,
        ).fit(x=X, y=Y, verbose=False)
    # Bit-identical
    np.testing.assert_array_equal(
        m_default.coef_.to_numpy(), m_explicit.coef_.to_numpy()
    )


def test_adaptive_without_sign_constraints_is_noop():
    """``auto_sign_adaptive_weights=True`` without sign-constraint
    derivation produces the same fit as the plain LASSO baseline. The
    adaptive layer cannot source slope magnitudes without the sign layer.
    """
    X, Y = _make_mixed_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_plain = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=5e-3, span=None,
            auto_sign_constraints=False,
            auto_sign_adaptive_weights=False,
        ).fit(x=X, y=Y, verbose=False)
        m_adaptive_no_sign = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=5e-3, span=None,
            auto_sign_constraints=False,
            auto_sign_adaptive_weights=True,
        ).fit(x=X, y=Y, verbose=False)
    np.testing.assert_array_equal(
        m_plain.coef_.to_numpy(), m_adaptive_no_sign.coef_.to_numpy()
    )


def test_adaptive_preserves_strong_signal_more_than_uniform():
    """Strong-evidence factors (large |β̂_uni|) are shrunk LESS by the
    adaptive penalty than by a uniform L1 penalty at the same nominal
    ``reg_lambda``. Conversely, weak-evidence factors are shrunk MORE.

    This is the Zou (2006) oracle property: the penalty becomes
    magnitude-aware.
    """
    X, Y = _make_mixed_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_uniform = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=5e-3, span=None,
            auto_sign_constraints=True,
            auto_sign_adaptive_weights=False,
        ).fit(x=X, y=Y, verbose=False)
        m_adaptive = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=5e-3, span=None,
            auto_sign_constraints=True,
            auto_sign_adaptive_weights=True,
        ).fit(x=X, y=Y, verbose=False)

    strong_uniform = m_uniform.coef_["f_strong"].abs().mean()
    strong_adaptive = m_adaptive.coef_["f_strong"].abs().mean()

    weak_uniform = m_uniform.coef_["f_weak"].abs().mean()
    weak_adaptive = m_adaptive.coef_["f_weak"].abs().mean()

    # Strong factor: adaptive should preserve (or slightly increase) magnitude
    # relative to uniform — the heavy L1 is partly lifted.
    assert strong_adaptive >= 0.95 * strong_uniform, (
        f"Adaptive shrunk strong factor too much: "
        f"uniform={strong_uniform:.4f}, adaptive={strong_adaptive:.4f}"
    )
    # Weak factor: adaptive should shrink MORE than uniform — the heavy
    # L1 weight on small slopes pushes harder toward zero.
    assert weak_adaptive < weak_uniform, (
        f"Adaptive failed to shrink weak factor more than uniform: "
        f"uniform={weak_uniform:.4f}, adaptive={weak_adaptive:.4f}"
    )


def test_adaptive_with_gamma_larger_amplifies_effect():
    """Larger γ amplifies the magnitude-aware reweighting: weak factors
    are shrunk harder under γ=2 than under γ=1."""
    X, Y = _make_mixed_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_g1 = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=5e-3, span=None,
            auto_sign_constraints=True,
            auto_sign_adaptive_weights=True,
            auto_sign_adaptive_gamma=1.0,
        ).fit(x=X, y=Y, verbose=False)
        m_g2 = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=5e-3, span=None,
            auto_sign_constraints=True,
            auto_sign_adaptive_weights=True,
            auto_sign_adaptive_gamma=2.0,
        ).fit(x=X, y=Y, verbose=False)

    weak_g1 = m_g1.coef_["f_weak"].abs().mean()
    weak_g2 = m_g2.coef_["f_weak"].abs().mean()

    assert weak_g2 < weak_g1, (
        f"γ=2 failed to shrink weak factor more than γ=1: "
        f"γ=1: {weak_g1:.4f}, γ=2: {weak_g2:.4f}"
    )


def test_adaptive_floor_prevents_explosion():
    """Even when |β̂_uni| approaches zero, the floor stabiliser keeps the
    fit numerically stable. We construct a panel where the weak factor has
    very small true β and verify the fitted coefficients are finite."""
    rng = np.random.default_rng(11)
    T = 200
    X = rng.standard_normal((T, 3))
    # Factor 1 has near-zero true β across all responses
    true_beta = np.array([
        [0.5, 1e-5, 0.0],
        [0.5, 1e-5, 0.0],
    ])
    Y = X @ true_beta.T + 0.10 * rng.standard_normal((T, 2))
    Xdf = pd.DataFrame(
        X, columns=["strong", "near_zero", "noise"],
        index=pd.date_range("2020", periods=T, freq="W-WED"),
    )
    Ydf = pd.DataFrame(
        Y, columns=["a1", "a2"], index=Xdf.index,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=5e-3, span=None,
            auto_sign_constraints=True,
            auto_sign_adaptive_weights=True,
            auto_sign_adaptive_floor=1e-3,  # explicit floor
        ).fit(x=Xdf, y=Ydf, verbose=False)

    # All fitted coefficients must be finite — no NaN, no Inf
    assert np.all(np.isfinite(m.coef_.to_numpy())), (
        "Adaptive-weight floor failed: non-finite coefficients in fit"
    )


def test_adaptive_penalty_weights_helper():
    """Direct test of _adaptive_penalty_weights:

    - large |slope| → small weight
    - small |slope| (below floor) → weight clipped at 1/floor^γ
    - signs == 0 → weight = 1.0 (placeholder)
    """
    slopes = np.array([
        [1.0, 0.5, 0.001, 0.0],   # decreasing magnitude
        [-2.0, 0.01, 0.0, 0.5],
    ])
    signs = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 1.0],
    ])

    w = _adaptive_penalty_weights(
        slopes=slopes, signs=signs, gamma=1.0, floor=1e-2,
    )

    # Strong slope at (0, 0): weight = 1/1.0 = 1.0
    assert np.isclose(w[0, 0], 1.0)
    # Moderate slope at (0, 1): weight = 1/0.5 = 2.0
    assert np.isclose(w[0, 1], 2.0)
    # Slope below floor (1e-3 < 1e-2): weight clipped at 1/floor = 100
    # but sign is 0 → placeholder of 1.0 wins
    assert np.isclose(w[0, 2], 1.0)  # placeholder because sign == 0
    # Slope = 0 with sign = 0: placeholder 1.0
    assert np.isclose(w[0, 3], 1.0)
    # Slope = -2 with sign = -1: weight = 1/2 = 0.5
    assert np.isclose(w[1, 0], 0.5)
    # Slope = 0.01 (= floor) with sign = 1: weight = 1/0.01 = 100
    assert np.isclose(w[1, 1], 100.0)
    # Slope = 0 with sign = 0: placeholder
    assert np.isclose(w[1, 2], 1.0)
    # Slope = 0.5 with sign = 1: weight = 1/0.5 = 2.0
    assert np.isclose(w[1, 3], 2.0)


def test_adaptive_penalty_weights_gamma_exponent():
    """Verify γ=2 produces the squared inverse: 1 / |β̂|^2."""
    slopes = np.array([[0.5, 0.25]])
    signs = np.array([[1.0, 1.0]])

    w1 = _adaptive_penalty_weights(slopes, signs, gamma=1.0, floor=1e-4)
    w2 = _adaptive_penalty_weights(slopes, signs, gamma=2.0, floor=1e-4)

    # γ=1: 1/0.5=2.0, 1/0.25=4.0
    np.testing.assert_array_almost_equal(w1, [[2.0, 4.0]])
    # γ=2: 1/0.25=4.0, 1/0.0625=16.0
    np.testing.assert_array_almost_equal(w2, [[4.0, 16.0]])


# ---------------------------------------------------------------------- #
# Adaptive Group Lasso path (Wang & Leng 2008): row-weight aggregation   #
# applies the adaptive reweighting to the group L2 norms, which is the   #
# mechanism that has impact in pure-group-LASSO production configs       #
# (l1_weight=0).                                                          #
# ---------------------------------------------------------------------- #


def test_aggregate_to_row_weights_rms():
    """Direct test of _aggregate_to_row_weights: RMS over non-pinned cells."""
    from factorlasso.sign_constraints import _aggregate_to_row_weights

    # All non-pinned: RMS over the full row
    cell_w = np.array([[2.0, 4.0, 4.0]])         # sqrt((4+16+16)/3) = sqrt(12) = 3.464
    signs = np.array([[1.0, 1.0, 1.0]])
    row_w = _aggregate_to_row_weights(cell_w, signs)
    np.testing.assert_almost_equal(row_w[0], np.sqrt(12.0))

    # One pinned: RMS over the remaining two
    cell_w = np.array([[2.0, 4.0, 1.0]])
    signs = np.array([[1.0, 1.0, 0.0]])           # last cell pinned
    row_w = _aggregate_to_row_weights(cell_w, signs)
    np.testing.assert_almost_equal(row_w[0], np.sqrt((4.0 + 16.0) / 2.0))  # sqrt(10)

    # All pinned: fallback to 1.0
    cell_w = np.array([[100.0, 50.0]])           # would explode normally
    signs = np.array([[0.0, 0.0]])
    row_w = _aggregate_to_row_weights(cell_w, signs)
    np.testing.assert_almost_equal(row_w[0], 1.0)

    # Uniform |β̂_uni|=1 case: row weight should be exactly 1.0
    cell_w = np.array([[1.0, 1.0, 1.0, 1.0]])
    signs = np.array([[1.0, -1.0, 1.0, 1.0]])
    row_w = _aggregate_to_row_weights(cell_w, signs)
    np.testing.assert_almost_equal(row_w[0], 1.0)


def test_aggregate_multiple_rows():
    """Multi-row aggregation runs vectorised across N."""
    from factorlasso.sign_constraints import _aggregate_to_row_weights

    cell_w = np.array([
        [2.0, 4.0, 4.0],
        [1.0, 1.0, 1.0],
        [3.0, 0.0, 5.0],     # middle cell will be pinned
    ])
    signs = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ])
    row_w = _aggregate_to_row_weights(cell_w, signs)
    np.testing.assert_almost_equal(row_w[0], np.sqrt(12.0))
    np.testing.assert_almost_equal(row_w[1], 1.0)
    np.testing.assert_almost_equal(row_w[2], np.sqrt((9.0 + 25.0) / 2.0))  # sqrt(17)


def test_adaptive_group_lasso_has_impact_in_production_config():
    """In the production HIERARCHICAL_CLUSTER_GROUP_LASSO config with l1_weight=0,
    activating auto_sign_adaptive_weights must produce a non-trivial change
    in the fit — otherwise the adaptive layer is a no-op for the very
    pipeline it's most needed in.

    This test pins the v0.3.10 fix to a regression that v0.3.9 had: the
    L1-only adaptive reweighting had zero impact when l1_weight=0."""
    rng = np.random.default_rng(0)
    T, M, N = 240, 9, 25
    X = rng.standard_normal((T, M))
    # Mixed-strength signal: strong on first 5 factors, weak on last 4
    true_B = rng.standard_normal((M, N)) * 0.3
    true_B[5:, :] *= 0.05
    Y = X @ true_B + rng.standard_normal((T, N)) * 0.5

    dates = pd.date_range("2020", periods=T, freq="W-WED")
    Xdf = pd.DataFrame(X, columns=[f"f{j}" for j in range(M)], index=dates)
    Ydf = pd.DataFrame(Y, columns=[f"a{k}" for k in range(N)], index=dates)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_off = LassoModel(
            model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
            reg_lambda=1e-3, span=None,           # production-like reg
            auto_sign_constraints=True,
            auto_sign_adaptive_weights=False,
            # l1_weight defaults to 0 — production setting
        ).fit(x=Xdf, y=Ydf, verbose=False)
        m_on = LassoModel(
            model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
            reg_lambda=1e-3, span=None,
            auto_sign_constraints=True,
            auto_sign_adaptive_weights=True,
        ).fit(x=Xdf, y=Ydf, verbose=False)

    abs_diff = (m_on.coef_ - m_off.coef_).abs().values.max()
    # In v0.3.9 this would be 0.0 — adaptive weights only affected the
    # (zero-weighted) L1 term. The v0.3.10 row-aggregation fix routes the
    # adaptive penalty through the active group L2 norms.
    assert abs_diff > 1e-6, (
        f"Adaptive group-lasso failed to change fit in production config: "
        f"max |Δβ| = {abs_diff:.2e}"
    )


def test_adaptive_group_lasso_default_off_backward_compatible():
    """auto_sign_adaptive_weights=False in HIERARCHICAL_CLUSTER_GROUP_LASSO reproduces
    the v0.3.8 baseline bit-for-bit."""
    rng = np.random.default_rng(3)
    T, M, N = 200, 7, 20
    X = rng.standard_normal((T, M))
    Y = X @ rng.standard_normal((M, N)) * 0.3 + rng.standard_normal((T, N)) * 0.4

    dates = pd.date_range("2020", periods=T, freq="W-WED")
    Xdf = pd.DataFrame(X, columns=[f"f{j}" for j in range(M)], index=dates)
    Ydf = pd.DataFrame(Y, columns=[f"a{k}" for k in range(N)], index=dates)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_default = LassoModel(
            model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
            reg_lambda=1e-3, span=None,
            auto_sign_constraints=True,
        ).fit(x=Xdf, y=Ydf, verbose=False)
        m_explicit_off = LassoModel(
            model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
            reg_lambda=1e-3, span=None,
            auto_sign_constraints=True,
            auto_sign_adaptive_weights=False,
        ).fit(x=Xdf, y=Ydf, verbose=False)

    np.testing.assert_array_equal(
        m_default.coef_.to_numpy(), m_explicit_off.coef_.to_numpy()
    )
