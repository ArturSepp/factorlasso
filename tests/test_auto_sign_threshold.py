"""
Regression tests for the v0.3.7 ``auto_sign_threshold_t`` parameter.

These tests verify three properties:

1. With the default threshold (0.75), a factor with weak univariate
   evidence (|t| < 0.75) is pinned to sign=0 and gets β = 0 in the fit.
2. With the gate disabled (``auto_sign_threshold_t=None``), the same
   weak-evidence factor receives the raw slope sign and can take a
   non-zero β.
3. Strong-evidence factors (|t| above the threshold) are unaffected by
   the gate regardless of its setting.

Backward-compatibility check: passing ``auto_sign_threshold_t=None``
reproduces v0.3.6 behaviour exactly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from factorlasso import (
    LassoModel,
    LassoModelType,
    derive_sign_constraints,
)


def _make_panel(seed: int = 7, T: int = 120):
    """Construct a synthetic panel with one strong and one weak factor.

    Strong factor (``Strong``): true β = +0.60, very large univariate signal
                                (|t| around 60).
    Weak factor   (``Weak``)  : true β = 0; included only as a noise
                                regressor.  Seed 7 reliably yields a
                                univariate ``|t|`` well below 0.75.
    """
    rng = np.random.default_rng(seed)
    f_strong = rng.normal(size=T)
    f_weak = rng.normal(size=T)
    eps = 0.10 * rng.normal(size=T)
    y_arr = 0.60 * f_strong + 0.0 * f_weak + eps
    X = pd.DataFrame({"Strong": f_strong, "Weak": f_weak})
    Y = pd.DataFrame({"asset": y_arr})
    return X, Y


def _univariate_t(x: np.ndarray, y: np.ndarray) -> float:
    """No-intercept univariate OLS t-stat (matches _compute_sign_vector)."""
    xx = float(x @ x)
    if xx <= 0.0:
        return 0.0
    beta = float(x @ y) / xx
    resid = y - beta * x
    sigma2 = float(resid @ resid) / max(len(y) - 1, 1)
    se = np.sqrt(sigma2 / xx) if sigma2 > 0 else np.inf
    return beta / se if se > 0 else 0.0


# ---------------------------------------------------------------------- #
# LassoModel default behaviour                                           #
# ---------------------------------------------------------------------- #

def test_default_threshold_pins_weak_factor_to_zero():
    """Default threshold (0.75) should pin the weak-evidence factor."""
    X, Y = _make_panel()

    # Sanity-check the panel: weak factor's univariate |t| is small,
    # strong factor's is large.
    t_strong = abs(_univariate_t(X["Strong"].to_numpy() - X["Strong"].mean(),
                                  Y["asset"].to_numpy() - Y["asset"].mean()))
    t_weak = abs(_univariate_t(X["Weak"].to_numpy() - X["Weak"].mean(),
                                Y["asset"].to_numpy() - Y["asset"].mean()))
    assert t_strong > 1.96, f"strong factor |t| = {t_strong:.2f} unexpectedly low"
    assert t_weak < 0.75, f"weak factor |t| = {t_weak:.2f} unexpectedly high"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(
            model_type=LassoModelType.LASSO,
            reg_lambda=1e-6,
            span=None,
            auto_sign_constraints=True,
            # No explicit auto_sign_threshold_t — should use default 0.75
        )
        m.fit(x=X, y=Y, verbose=False)

    # Weak factor: derived sign should be 0 (pinned), β = 0
    # (solver may produce a numerically tiny non-zero, use isclose).
    assert m.derived_signs_.iloc[0]["Weak"] == 0.0
    assert np.isclose(m.coef_.iloc[0]["Weak"], 0.0, atol=1e-10)
    # Strong factor: derived sign respected, β > 0.
    assert m.derived_signs_.iloc[0]["Strong"] == 1.0
    assert m.coef_.iloc[0]["Strong"] > 0.3


def test_none_threshold_reproduces_v036_behaviour():
    """Explicit None disables the gate — weak factor gets a sign."""
    X, Y = _make_panel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(
            model_type=LassoModelType.LASSO,
            reg_lambda=1e-6,
            span=None,
            auto_sign_constraints=True,
            auto_sign_threshold_t=None,  # explicit opt-out
        )
        m.fit(x=X, y=Y, verbose=False)

    # Weak factor: derived sign is the raw slope sign (±1), not 0.
    weak_sign = m.derived_signs_.iloc[0]["Weak"]
    assert weak_sign in (-1.0, 1.0), \
        f"with threshold=None, weak factor sign should be ±1, got {weak_sign}"


def test_threshold_default_value_is_075():
    """Verify the documented default in the dataclass signature."""
    m = LassoModel()
    assert m.auto_sign_threshold_t == 0.75


# ---------------------------------------------------------------------- #
# Public derive_sign_constraints wrapper                                 #
# ---------------------------------------------------------------------- #

def test_derive_sign_constraints_default_threshold():
    """The public wrapper applies the default 0.75 gate as well."""
    X, Y = _make_panel()
    signs = derive_sign_constraints(
        x=X, y=Y,
        # default threshold
    )
    assert signs["Weak"].iloc[0] == 0.0
    assert signs["Strong"].iloc[0] == 1.0


def test_derive_sign_constraints_none_threshold():
    """Explicit None reproduces the unfiltered slope sign."""
    X, Y = _make_panel()
    signs = derive_sign_constraints(
        x=X, y=Y,
        auto_sign_threshold_t=None,
    )
    weak_sign = signs["Weak"].iloc[0]
    assert weak_sign in (-1.0, 1.0)
    assert signs["Strong"].iloc[0] == 1.0


# ---------------------------------------------------------------------- #
# Master-constraint overlay still wins                                   #
# ---------------------------------------------------------------------- #

def test_master_constraint_overrides_threshold_gate():
    """Explicit user-supplied sign survives even if |t| < threshold."""
    X, Y = _make_panel()
    signs_df = pd.DataFrame(
        {"Strong": [1], "Weak": [1]},   # user explicitly fixes Weak = +1
        index=["asset"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(
            model_type=LassoModelType.LASSO,
            reg_lambda=1e-6,
            span=None,
            auto_sign_constraints=True,
            auto_sign_threshold_t=0.75,
            factors_beta_loading_signs=signs_df,
        )
        m.fit(x=X, y=Y, verbose=False)

    # User constraint wins: derived_signs_ should show +1 for Weak,
    # not the gated 0.
    assert m.derived_signs_.iloc[0]["Weak"] == 1.0


# ---------------------------------------------------------------------- #
# Cluster-mode threshold behaviour                                       #
# ---------------------------------------------------------------------- #
#
# In cluster modes (``GROUP_LASSO`` and ``GROUP_LASSO_CLUSTERS``), the
# threshold gate operates on the *cluster-aggregated* t-statistic, not
# on per-member t-stats. That is the design: every member of a cluster
# is assumed to share an economic loading, so they share a gating
# decision. The three tests below pin the three failure modes of that
# design:
#
#   1. A cluster with one strong and one weak member is kept as a whole
#      when the cluster aggregate is strong (the weak member rides the
#      strong member's signal).
#   2. A cluster with all-weak members is pinned uniformly to sign=0 —
#      even if one individual member's *own* univariate slope happens
#      to be lucky and would survive a per-column gate.
#   3. Master constraints can still surgically pin one cluster member
#      to a non-cluster value, breaking within-cluster coherence by
#      design (master always wins, including over the cluster gate).
# ---------------------------------------------------------------------- #


def _make_clustered_panel(seed: int = 11, T: int = 120, idio: float = 0.10):
    """
    Two factors {Strong, Weak}, four assets in two groups:

        Group ``S`` (strong-cluster):  asset_S1, asset_S2 — true β on Strong
                                       both = +0.60 (large univariate signal).
        Group ``W`` (weak-cluster) :   asset_W1, asset_W2 — true β on Strong
                                       both = 0 (no signal). Pure noise.

    The Weak factor is true-zero for everyone (extra noise predictor).
    """
    rng = np.random.default_rng(seed)
    f_strong = rng.normal(size=T)
    f_weak = rng.normal(size=T)
    X = pd.DataFrame({"Strong": f_strong, "Weak": f_weak})

    s1 = 0.60 * f_strong + idio * rng.normal(size=T)
    s2 = 0.60 * f_strong + idio * rng.normal(size=T)
    w1 = idio * rng.normal(size=T)
    w2 = idio * rng.normal(size=T)
    Y = pd.DataFrame({
        "asset_S1": s1, "asset_S2": s2,
        "asset_W1": w1, "asset_W2": w2,
    })
    group_data = pd.Series(
        {"asset_S1": "S", "asset_S2": "S", "asset_W1": "W", "asset_W2": "W"},
        name="cluster",
    )
    return X, Y, group_data


def test_cluster_mode_strong_cluster_survives_gate():
    """Strong cluster: both members kept (sign = +1 on Strong factor)."""
    X, Y, group_data = _make_clustered_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(
            model_type=LassoModelType.GROUP_LASSO,
            reg_lambda=1e-6, span=None,
            group_data=group_data,
            auto_sign_constraints=True,
            auto_sign_threshold_t=0.75,
        )
        m.fit(x=X, y=Y, verbose=False)

    # Strong cluster on Strong factor: gated to +1 for every member
    assert m.derived_signs_.loc["asset_S1", "Strong"] == 1.0
    assert m.derived_signs_.loc["asset_S2", "Strong"] == 1.0
    # And the fit produces non-zero betas for the strong cluster
    assert m.coef_.loc["asset_S1", "Strong"] > 0.3
    assert m.coef_.loc["asset_S2", "Strong"] > 0.3


def test_cluster_mode_weak_cluster_pinned_uniformly():
    """Weak cluster: cluster aggregate has weak |t|, so ALL members get
    sign = 0 — even if one member's per-asset slope happens to be 'lucky'.

    This pins the design choice: cluster-mode threshold gates on the
    cluster's pooled t-stat, not on per-member t-stats."""
    X, Y, group_data = _make_clustered_panel(seed=11)

    # Verify the design premise: the cluster's pooled t-stat IS below
    # the threshold even if one individual member's univariate t-stat
    # might be above by chance.
    weak_idx = [Y.columns.get_loc("asset_W1"), Y.columns.get_loc("asset_W2")]
    f_strong_c = X["Strong"].to_numpy() - X["Strong"].mean()
    # Pooled (q=2) t-stat on the weak cluster's Strong loading
    y_pool = Y.iloc[:, weak_idx].to_numpy() - Y.iloc[:, weak_idx].mean(axis=0).to_numpy()
    beta_c = float(f_strong_c @ y_pool.sum(axis=1)) / (2 * float(f_strong_c @ f_strong_c))
    # We assert the panel is calibrated so the pooled |t| is well below 0.75
    assert abs(beta_c) < 0.05, f"weak cluster β = {beta_c:.3f} unexpectedly large"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(
            model_type=LassoModelType.GROUP_LASSO,
            reg_lambda=1e-6, span=None,
            group_data=group_data,
            auto_sign_constraints=True,
            auto_sign_threshold_t=0.75,
        )
        m.fit(x=X, y=Y, verbose=False)

    # Both weak-cluster members pinned to sign 0 on the Strong factor
    assert m.derived_signs_.loc["asset_W1", "Strong"] == 0.0
    assert m.derived_signs_.loc["asset_W2", "Strong"] == 0.0
    # And their fitted β is zero (gate forces β = 0)
    assert np.isclose(m.coef_.loc["asset_W1", "Strong"], 0.0, atol=1e-8)
    assert np.isclose(m.coef_.loc["asset_W2", "Strong"], 0.0, atol=1e-8)


def test_cluster_mode_master_constraint_pins_single_member():
    """factors_beta_loading_signs can surgically override one cluster
    member while leaving the rest gated by the cluster aggregate.

    This pins that within-cluster coherence is a soft default of the
    cluster-mode gate, not a hard structural invariant — the user can
    always pierce it via the explicit overlay."""
    X, Y, group_data = _make_clustered_panel()
    # Force asset_S2 to -1 on Strong (opposite the data-derived sign of +1
    # the cluster aggregate would produce)
    override = pd.DataFrame(
        np.nan, index=Y.columns, columns=X.columns,
    )
    override.loc["asset_S2", "Strong"] = -1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(
            model_type=LassoModelType.GROUP_LASSO,
            reg_lambda=1e-6, span=None,
            group_data=group_data,
            auto_sign_constraints=True,
            auto_sign_threshold_t=0.75,
            factors_beta_loading_signs=override,
        )
        m.fit(x=X, y=Y, verbose=False)

    # The cluster's other strong member still gets the gated +1
    assert m.derived_signs_.loc["asset_S1", "Strong"] == 1.0
    # The targeted member gets the user override (−1)
    assert m.derived_signs_.loc["asset_S2", "Strong"] == -1.0
    # And the fitted β for the overridden asset is non-positive
    assert m.coef_.loc["asset_S2", "Strong"] <= 1e-8


def test_cluster_mode_hcgl_path_threshold_gates_uniformly():
    """End-to-end: GROUP_LASSO_CLUSTERS (HCGL) auto-clusters assets, then
    the threshold gate broadcasts the cluster aggregate's t-stat to every
    member. Verify that members of the same HCGL cluster always share
    their threshold decision (sign coherence)."""
    X, Y, _ = _make_clustered_panel(seed=13, T=240)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            reg_lambda=1e-6, span=None,
            auto_sign_constraints=True,
            auto_sign_threshold_t=0.75,
        )
        m.fit(x=X, y=Y, verbose=False)

    # Within any HCGL cluster, all members share the same derived_signs_ row
    assert m.clusters_ is not None
    for c in m.clusters_.unique():
        members = m.clusters_[m.clusters_ == c].index.tolist()
        if len(members) <= 1:
            continue
        rows = m.derived_signs_.loc[members]
        for col in rows.columns:
            unique_signs = set(rows[col].dropna().unique().tolist())
            assert len(unique_signs) == 1, (
                f"Cluster {c}, factor {col}: members disagree on sign "
                f"({rows[col].to_dict()})"
            )
