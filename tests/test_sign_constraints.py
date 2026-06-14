"""
Integration tests for :func:`factorlasso.derive_sign_constraints` and
:func:`factorlasso.validate_cluster_signs`, exercised against the real
``LassoModel`` and ``LassoModelCV``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from factorlasso import (
    LassoModel,
    LassoModelCV,
    LassoModelType,
    derive_sign_constraints,
    validate_cluster_signs,
)

# ───────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_factor_data(seed: int, T: int = 400, k_per_cluster: int = 5,
                      idio_scale: float = 0.10, noise_sigma: float = 1.5):
    """
    Build a factor-return-like dataset:
      * 3 latent factors (f1+, f2−, f3 zero-effect)
      * k_per_cluster noisy realizations of each factor
      * 3 independent regressors (noise)
      * 1 direct driver
      * 4 response columns y0..y3 with shared β across responses
    """
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(T)
    f2 = rng.standard_normal(T)
    f3 = rng.standard_normal(T)

    cols, cluster_ids, names = [], [], []
    for fi, f in enumerate([f1, f2, f3]):
        for k in range(k_per_cluster):
            cols.append(f + idio_scale * rng.standard_normal(T))
            cluster_ids.append(fi)
            names.append(f"f{fi}_{k}")
    for j in range(3):
        cols.append(rng.standard_normal(T))
        cluster_ids.append(3 + j)
        names.append(f"indep_{j}")
    direct = rng.standard_normal(T)
    cols.append(direct)
    cluster_ids.append(6)
    names.append("direct")

    X = pd.DataFrame(np.column_stack(cols), columns=names,
                     index=pd.RangeIndex(T, name="t"))
    M = X.shape[1]
    cluster_ids = np.array(cluster_ids)

    beta_true = np.zeros(M)
    beta_true[0:k_per_cluster] = 0.40       # f1 +
    beta_true[k_per_cluster:2*k_per_cluster] = -0.30  # f2 −
    beta_true[-1] = 1.00                    # direct +

    # 4 response columns, identical β
    N = 4
    asset_names = [f"y{i}" for i in range(N)]
    Y_arr = np.column_stack([
        X.values @ beta_true + noise_sigma * rng.standard_normal(T)
        for _ in range(N)
    ])
    Y = pd.DataFrame(Y_arr, columns=asset_names, index=X.index)
    return X, Y, beta_true, cluster_ids


# ───────────────────────────────────────────────────────────────────────
# Basic shape and convention compatibility with LassoModel
# ───────────────────────────────────────────────────────────────────────

def test_output_shape_matches_factors_beta_loading_signs():
    """Output is (N × M) DataFrame with rows = y.columns, cols = x.columns."""
    X, Y, _, _ = _make_factor_data(seed=0)
    signs = derive_sign_constraints(X, Y)
    assert isinstance(signs, pd.DataFrame)
    assert signs.shape == (Y.shape[1], X.shape[1])
    assert list(signs.columns) == list(X.columns)
    assert list(signs.index) == list(Y.columns)
    # All rows identical (pooled estimator)
    for i in range(1, signs.shape[0]):
        np.testing.assert_array_equal(signs.iloc[0].values, signs.iloc[i].values)


def test_ndarray_inputs_return_1d_vector():
    """Pure ndarray inputs return (M,) for backward compatibility / efficiency."""
    X, Y, _, _ = _make_factor_data(seed=0)
    signs = derive_sign_constraints(X.values, Y.values)
    assert isinstance(signs, np.ndarray)
    assert signs.shape == (X.shape[1],)


def test_signs_only_take_legal_values():
    """Every entry is in {-1, 0, +1} (no NaN unless master_constraints set it)."""
    X, Y, _, _ = _make_factor_data(seed=0)
    signs = derive_sign_constraints(X, Y)
    assert np.isin(signs.values, [-1.0, 0.0, 1.0]).all()


# ───────────────────────────────────────────────────────────────────────
# Drop-in compatibility with LassoModel
# ───────────────────────────────────────────────────────────────────────

def test_signs_dataframe_dropped_into_lasso_model_fits_cleanly():
    """The signs DataFrame is accepted by LassoModel and the fit respects it."""
    X, Y, beta_true, _ = _make_factor_data(seed=1, noise_sigma=0.5)
    signs = derive_sign_constraints(X, Y)
    model = LassoModel(
        model_type=LassoModelType.LASSO,
        reg_lambda=1e-4,
        factors_beta_loading_signs=signs,
    ).fit(x=X, y=Y)

    # Every fitted coefficient with derived sign +1 must be ≥ 0;
    # every fitted coefficient with derived sign -1 must be ≤ 0.
    coef = model.coef_.values  # (N, M)
    s = signs.values
    tol = 1e-6
    assert ((s == 1) <= (coef >= -tol)).all()
    assert ((s == -1) <= (coef <= tol)).all()
    assert ((s == 0) <= (np.abs(coef) <= tol)).all()


def test_master_constraint_zeros_a_factor_in_lasso_fit():
    """master_constraints={'direct': 0} forces direct's coefficient to zero
    in the downstream LassoModel fit, even though it has strong univariate
    signal."""
    X, Y, _, _ = _make_factor_data(seed=2, noise_sigma=0.5)
    signs = derive_sign_constraints(X, Y, master_constraints={"direct": 0})
    model = LassoModel(
        model_type=LassoModelType.LASSO,
        reg_lambda=1e-4,
        factors_beta_loading_signs=signs,
    ).fit(x=X, y=Y)
    direct_col = X.columns.get_loc("direct")
    assert np.allclose(model.coef_.values[:, direct_col], 0.0, atol=1e-6)


def test_nan_master_release_leaves_column_unconstrained():
    """Setting master_constraints={'f0_0': None} releases that regressor from
    any constraint — LassoModel should treat it as unconstrained (NaN)."""
    X, Y, _, _ = _make_factor_data(seed=3)
    signs = derive_sign_constraints(X, Y, master_constraints={"f0_0": None})
    assert np.isnan(signs.loc[:, "f0_0"]).all()


# ───────────────────────────────────────────────────────────────────────
# Cluster-level mode — the within-cluster coherence guarantee
# ───────────────────────────────────────────────────────────────────────

def test_singleton_clusters_equal_column_level():
    X, Y, _, _ = _make_factor_data(seed=4)
    M = X.shape[1]
    sm_col = derive_sign_constraints(X, Y)
    sm_clu = derive_sign_constraints(X, Y, clusters=np.arange(M))
    pd.testing.assert_frame_equal(sm_col, sm_clu, check_dtype=False)


def test_cluster_level_enforces_within_cluster_coherence():
    """All members of any cluster get the same sign — guaranteed by construction."""
    X, Y, _, clusters = _make_factor_data(seed=5)
    signs = derive_sign_constraints(X, Y, clusters=clusters)
    sign_row = signs.iloc[0].values  # one row, all identical
    for c in np.unique(clusters):
        members = sign_row[clusters == c]
        nonzero = members[members != 0]
        if len(nonzero) > 0:
            assert len(set(nonzero.tolist())) == 1, (
                f"Cluster {c} has mixed signs in derived output: {nonzero}"
            )


def test_cluster_constrained_fit_has_no_within_cluster_alternations():
    """End-to-end with LassoModel: cluster-constrained fit produces zero
    within-cluster sign alternations on tightly-collinear factor structure."""
    X, Y, _, clusters = _make_factor_data(
        seed=6, idio_scale=0.05, noise_sigma=2.0  # tight clusters, heavy noise
    )
    signs = derive_sign_constraints(X, Y, clusters=clusters)
    model = LassoModel(
        model_type=LassoModelType.LASSO,
        reg_lambda=1e-5,  # low λ to keep features in
        factors_beta_loading_signs=signs,
    ).fit(x=X, y=Y)

    # Count opposite-sign pairs within each cluster, summed across responses
    coef = model.coef_.values  # (N, M)
    alt_pairs = 0
    for c in np.unique(clusters):
        members = coef[:, clusters == c]  # (N, k_c)
        nz = (np.abs(members) > 1e-6)
        for i in range(members.shape[0]):
            row_nz = members[i, nz[i]]
            if len(row_nz) >= 2:
                alt_pairs += int((row_nz > 0).sum()) * int((row_nz < 0).sum())
    assert alt_pairs == 0


# ───────────────────────────────────────────────────────────────────────
# CV compatibility — signs flow through LassoModelCV via base_model
# ───────────────────────────────────────────────────────────────────────

def test_signs_propagate_through_lasso_model_cv():
    """When base_model carries signs, the CV-refitted best_model_ does too,
    and the fitted coefficients respect them."""
    X, Y, _, clusters = _make_factor_data(seed=7, noise_sigma=0.7)
    signs = derive_sign_constraints(X, Y, clusters=clusters)
    base = LassoModel(
        model_type=LassoModelType.LASSO,
        factors_beta_loading_signs=signs,
    )
    cv = LassoModelCV(
        lambdas=[1e-5, 1e-4, 1e-3, 1e-2],
        n_splits=3,
        base_model=base,
    ).fit(x=X, y=Y)

    fitted = cv.best_model_
    assert fitted is not None
    assert fitted.factors_beta_loading_signs is not None
    pd.testing.assert_frame_equal(fitted.factors_beta_loading_signs, signs)
    # And the fitted coefficients respect the signs
    coef = fitted.coef_.values
    s = signs.values
    tol = 1e-6
    assert ((s == 1) <= (coef >= -tol)).all()
    assert ((s == -1) <= (coef <= tol)).all()


# ───────────────────────────────────────────────────────────────────────
# Misspecified-cluster diagnostic helper
# ───────────────────────────────────────────────────────────────────────

def test_validate_detects_misspecified_cluster_and_warns():
    """When clusters wrongly group regressors of opposite economic effect,
    validate_cluster_signs flags them and emits a UserWarning."""
    rng = np.random.default_rng(8)
    T = 400
    f1 = rng.standard_normal(T)
    # 'good' regressors track f1, 'bad' regressor tracks -f1 — wrongly clustered
    X = pd.DataFrame({
        "good_a": f1 + 0.1 * rng.standard_normal(T),
        "good_b": f1 + 0.1 * rng.standard_normal(T),
        "bad":   -f1 + 0.1 * rng.standard_normal(T),
    })
    Y = pd.DataFrame({"y0": f1 + 0.3 * rng.standard_normal(T)})
    clusters = np.array([0, 0, 0])  # deliberately misspecified

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        disagreements = validate_cluster_signs(X, Y, clusters)
        warning_msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]

    assert len(disagreements) >= 1
    assert any("'bad'" in m or "bad" in m for m in warning_msgs)


def test_validate_silent_when_clusters_consistent():
    """No warning when every cluster member has the same univariate sign."""
    X, Y, _, clusters = _make_factor_data(seed=9, noise_sigma=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        disagreements = validate_cluster_signs(X, Y, clusters)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0
    assert len(disagreements) == 0


# ───────────────────────────────────────────────────────────────────────
# Stress test — alternation elimination vs vanilla LassoModel
# ───────────────────────────────────────────────────────────────────────

def test_stress_alternation_elimination_vs_vanilla_lasso_model():
    """Across multiple seeds with tight collinearity and heavy noise:
    cluster-constrained LassoModel produces ZERO within-cluster alternations;
    vanilla LassoModel (no signs) produces a meaningfully positive count."""
    n_seeds = 8
    alt_vanilla_total = 0
    alt_constrained_total = 0

    for seed in range(n_seeds):
        X, Y, _, clusters = _make_factor_data(
            seed=100 + seed, T=300, k_per_cluster=5,
            idio_scale=0.05, noise_sigma=2.0,
        )
        signs = derive_sign_constraints(X, Y, clusters=clusters)
        reg_lambda = 1e-6  # low λ → vanilla keeps many features → alternations possible

        m_v = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=reg_lambda,
        ).fit(x=X, y=Y)
        m_c = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=reg_lambda,
            factors_beta_loading_signs=signs,
        ).fit(x=X, y=Y)

        for model, accum_attr in [(m_v, "v"), (m_c, "c")]:
            coef = model.coef_.values
            pairs = 0
            for c in np.unique(clusters):
                members = coef[:, clusters == c]
                nz = np.abs(members) > 1e-6
                for i in range(members.shape[0]):
                    row_nz = members[i, nz[i]]
                    if len(row_nz) >= 2:
                        pairs += int((row_nz > 0).sum()) * int((row_nz < 0).sum())
            if accum_attr == "v":
                alt_vanilla_total += pairs
            else:
                alt_constrained_total += pairs

    assert alt_constrained_total == 0, (
        f"Constrained should have zero alternations across all seeds; "
        f"got {alt_constrained_total}."
    )
    assert alt_vanilla_total >= n_seeds, (
        f"Expected vanilla to produce ≥{n_seeds} alternations across "
        f"{n_seeds} stress seeds; got {alt_vanilla_total}. SNR may be too "
        f"high for this stress config."
    )


# ───────────────────────────────────────────────────────────────────────
# Edge cases & error handling
# ───────────────────────────────────────────────────────────────────────

def test_x_y_row_mismatch_raises():
    X = pd.DataFrame(np.random.randn(50, 3), columns=list("abc"))
    Y = pd.DataFrame(np.random.randn(40, 2), columns=["y0", "y1"])
    with pytest.raises(ValueError, match="rows"):
        derive_sign_constraints(X, Y)


def test_clusters_length_mismatch_raises():
    X, Y, _, _ = _make_factor_data(seed=10)
    with pytest.raises(ValueError, match="clusters length"):
        derive_sign_constraints(X, Y, clusters=np.array([0, 1, 2]))  # wrong M


def test_master_invalid_value_raises():
    X, Y, _, _ = _make_factor_data(seed=11)
    with pytest.raises(ValueError, match="must be in"):
        derive_sign_constraints(X, Y, master_constraints={"f0_0": 2})


def test_master_name_on_unnamed_x_raises():
    X, Y, _, _ = _make_factor_data(seed=12)
    with pytest.raises(ValueError, match="no column names"):
        derive_sign_constraints(X.values, Y.values,
                                master_constraints={"f0_0": 1})


def test_master_unknown_column_raises():
    X, Y, _, _ = _make_factor_data(seed=13)
    with pytest.raises(KeyError, match="not found"):
        derive_sign_constraints(X, Y, master_constraints={"not_a_column": 1})


# ───────────────────────────────────────────────────────────────────────
# NaN-agnostic behavior at the function boundary
# ───────────────────────────────────────────────────────────────────────

def test_nan_in_y_does_not_poison_slopes(rng):
    """A single NaN in y must not produce all-NaN slopes. The univariate
    slope on (x, y) is invariant to zero-filling matched NaN positions, so
    the function zero-fills at entry and returns slopes that equal those
    computed on the valid rows only."""
    T, M = 100, 4
    x = rng.standard_normal((T, M))
    beta_true = np.array([1.5, -1.0, 0.0, 0.4])
    y = (x @ beta_true + 0.2 * rng.standard_normal(T)).reshape(-1, 1)

    # Reference: signs from the clean arrays
    signs_clean = derive_sign_constraints(x, y)

    # Inject NaN into a few y rows
    y_nan = y.copy()
    y_nan[10:20] = np.nan
    signs_nan = derive_sign_constraints(x, y_nan)

    # No NaN in the output, and the strong-signal sign directions match
    assert not np.isnan(signs_nan).any()
    np.testing.assert_array_equal(
        signs_nan[beta_true != 0], signs_clean[beta_true != 0]
    )


def test_nan_in_x_does_not_poison_other_columns(rng):
    """NaN in one x column must not bleed across columns. The other columns'
    slopes should match the clean reference."""
    T, M = 100, 4
    x = rng.standard_normal((T, M))
    beta_true = np.array([1.5, -1.0, 0.5, 0.4])
    y = (x @ beta_true + 0.2 * rng.standard_normal(T)).reshape(-1, 1)

    signs_clean = derive_sign_constraints(x, y)

    x_nan = x.copy()
    x_nan[5:15, 1] = np.nan
    signs_nan = derive_sign_constraints(x_nan, y)

    # Other columns unaffected
    other_cols = [0, 2, 3]
    np.testing.assert_array_equal(
        signs_nan[other_cols], signs_clean[other_cols]
    )


def test_dataframe_with_nan_returns_valid_signs(rng):
    """External API: a DataFrame with NaN entries must not produce all-NaN
    signs — the function silent-failure mode that existed before the
    NaN-agnostic fix."""
    T = 100
    X = pd.DataFrame(rng.standard_normal((T, 3)), columns=['a', 'b', 'c'])
    Y = pd.DataFrame(
        (X.values @ np.array([1.0, -1.0, 0.5]) + 0.2 * rng.standard_normal(T)).reshape(-1, 1),
        columns=['y0'],
    )
    Y.iloc[20:30] = np.nan  # block of missing observations

    signs = derive_sign_constraints(X, Y)
    assert not signs.isna().any().any()
    # Correct direction recovered despite the missingness
    assert (signs.iloc[0] == [1.0, -1.0, 1.0]).all()


# ───────────────────────────────────────────────────────────────────────
# Internal auto_sign_constraints path — derivation inside LassoModel.fit()
# ───────────────────────────────────────────────────────────────────────

def test_auto_sign_constraints_basic_fit():
    """auto_sign_constraints=True derives signs internally from the
    EWMA-demeaned, NaN-masked arrays the solver actually consumes."""
    X, Y, _, _ = _make_factor_data(seed=20, noise_sigma=0.5)
    model = LassoModel(
        model_type=LassoModelType.LASSO,
        reg_lambda=1e-4,
        auto_sign_constraints=True,
    ).fit(x=X, y=Y)

    assert model.derived_signs_ is not None
    assert model.derived_signs_.shape == (Y.shape[1], X.shape[1])
    # Fitted coefficients must respect the derived signs
    s = model.derived_signs_.values
    coef = model.coef_.values
    tol = 1e-6
    assert ((s == 1) <= (coef >= -tol)).all()
    assert ((s == -1) <= (coef <= tol)).all()


def test_auto_sign_constraints_lasso_mode_is_per_y_column():
    """In LASSO mode each y-column gets its own univariate sign derivation,
    so the rows of derived_signs_ are produced independently — they may
    differ across responses (in particular on noise factors)."""
    X, Y, beta_true, _ = _make_factor_data(seed=21, noise_sigma=0.5)
    model = LassoModel(
        model_type=LassoModelType.LASSO, reg_lambda=1e-4,
        auto_sign_constraints=True,
    ).fit(x=X, y=Y)

    s = model.derived_signs_
    # On strongly-signaled factors, every per-column fit picks up the same
    # sign by construction of the DGP (β_true shared across responses).
    active = beta_true != 0
    for j in np.where(active)[0]:
        col_vals = s.iloc[:, j].values
        assert len(set(col_vals.tolist())) == 1, (
            f"Active factor {X.columns[j]} should have uniform sign across "
            f"per-column LASSO derivations; got {col_vals}"
        )


def test_auto_sign_constraints_group_lasso_uses_group_data():
    """In GROUP_LASSO mode signs are pooled within each asset group from
    ``self.group_data`` — every member of a group shares the same row."""
    # Build a 4-asset universe with two groups of two
    X, Y, _, _ = _make_factor_data(seed=22, noise_sigma=0.5)
    group_data = pd.Series(
        ["A", "A", "B", "B"],
        index=Y.columns, name="grp",
    )

    model = LassoModel(
        model_type=LassoModelType.GROUP_LASSO, reg_lambda=1e-4,
        group_data=group_data,
        auto_sign_constraints=True,
    ).fit(x=X, y=Y)

    s = model.derived_signs_
    # Every group's member rows must be identical (signs pooled within group)
    for grp in group_data.unique():
        members = group_data[group_data == grp].index
        rows = s.loc[members]
        for col in s.columns:
            assert len(set(rows[col].values.tolist())) == 1


def test_auto_sign_constraints_group_lasso_clusters_uses_hcgl():
    """In GROUP_LASSO_CLUSTERS mode signs are pooled within each HCGL cluster
    (the same asset-side clustering the group solver uses)."""
    X, Y, _, _ = _make_factor_data(seed=23, T=300, noise_sigma=0.5)
    model = LassoModel(
        model_type=LassoModelType.GROUP_LASSO_CLUSTERS, reg_lambda=1e-4,
        auto_sign_constraints=True,
    ).fit(x=X, y=Y)

    s = model.derived_signs_
    cl = model.clusters_  # asset-side HCGL output (length N)
    assert s is not None and cl is not None
    # Within each HCGL cluster, all member rows of derived_signs_ are equal
    for c in cl.unique():
        members = cl[cl == c].index
        rows = s.loc[members]
        for col in s.columns:
            assert len(set(rows[col].values.tolist())) == 1


def test_auto_signs_with_explicit_overlay_for_asset_specific_overrides():
    """factors_beta_loading_signs can be combined with auto_sign_constraints
    to apply asset-specific per-cell overrides on top of auto-derived signs.

    Use case: most factors should follow data-derived sign, but a specific
    asset must have a particular factor forced to zero."""
    X, Y, _, _ = _make_factor_data(seed=24, noise_sigma=0.5)
    N, M = Y.shape[1], X.shape[1]

    # NaN-filled override matrix — only one cell carries a real constraint:
    # asset_0 forced to zero loading on 'direct' factor
    override = pd.DataFrame(
        np.full((N, M), np.nan), index=Y.columns, columns=X.columns,
    )
    override.iloc[0, X.columns.get_loc("direct")] = 0

    model = LassoModel(
        model_type=LassoModelType.LASSO, reg_lambda=1e-4,
        auto_sign_constraints=True,
        factors_beta_loading_signs=override,
    ).fit(x=X, y=Y)

    # The single overridden cell must be 0 (forced zero)
    assert model.derived_signs_.iloc[0, X.columns.get_loc("direct")] == 0
    # All other 'direct'-column cells take the auto-derived sign (+1 — strong
    # positive marginal effect from DGP)
    for i in range(1, N):
        assert model.derived_signs_.iloc[i, X.columns.get_loc("direct")] == 1
    # Fitted coefficient honors the override
    assert np.isclose(model.coef_.iloc[0]["direct"], 0.0, atol=1e-6)


def test_auto_sign_constraints_uses_demeaned_data():
    """Signs are derived from get_x_y_np outputs (demeaned), not raw inputs.

    Crucially, demeaning can flip the sign of zero-effect/noise predictors
    because their tiny marginal correlations are sensitive to the mean
    subtraction. This is *desired* behavior — the internal path operates on
    exactly the same arrays the solver consumes."""
    X, Y, beta_true, _ = _make_factor_data(seed=25, noise_sigma=0.3)

    model_auto = LassoModel(
        model_type=LassoModelType.LASSO, reg_lambda=1e-4,
        auto_sign_constraints=True, demean=True,
    ).fit(x=X, y=Y)

    # External derivation per y-column (since LASSO mode is per-column too)
    active = beta_true != 0
    for k, col in enumerate(Y.columns):
        external_k = derive_sign_constraints(X, Y[[col]])
        np.testing.assert_array_equal(
            model_auto.derived_signs_.iloc[k].values[active],
            external_k.iloc[0].values[active],
            err_msg=f"Active-factor signs for {col} should agree internal vs external",
        )


def test_auto_sign_constraints_propagate_through_cv():
    """LassoModelCV with auto_sign_constraints=True derives signs per fold,
    and the refit on the full training set produces signs consistent with a
    per-y-column external derivation."""
    X, Y, beta_true, _ = _make_factor_data(seed=26, noise_sigma=0.7)

    base = LassoModel(
        model_type=LassoModelType.LASSO,
        auto_sign_constraints=True,
    )
    cv = LassoModelCV(
        lambdas=[1e-5, 1e-4, 1e-3], n_splits=3, base_model=base,
    ).fit(x=X, y=Y)

    assert cv.best_model_ is not None
    assert cv.best_model_.derived_signs_ is not None
    # Active-factor signs agree per row between CV refit and per-column external
    active = beta_true != 0
    for k, col in enumerate(Y.columns):
        external_k = derive_sign_constraints(X, Y[[col]])
        np.testing.assert_array_equal(
            cv.best_model_.derived_signs_.iloc[k].values[active],
            external_k.iloc[0].values[active],
        )
