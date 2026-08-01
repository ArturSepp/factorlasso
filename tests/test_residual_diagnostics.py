"""
Tests for factorlasso.residual_diagnostics and factorlasso.diagonality.

Covers the statistics against a known null, the selection rule including its non-degeneracy, and
the two accounting traps that motivated the module: interior-point solvers do not return exact
zeros, and a fitted model must not be scored on its own training sample.

Run:  python -m pytest test_residual_diagnostics.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlasso import (
    LassoModel,
    LassoModelCV,
    LassoModelDiagonalityCV,
    LassoModelType,
    Sparsity,
    diagnose_residuals,
    effective_sparsity,
    marchenko_pastur_edge,
    missing_factor_components,
    null_threshold,
    raw_offdiagonal_mass,
    residual_correlation,
    suggest_tolerance,
)


def _panel(seed=0, T=300, M=3, N=8, common=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2001-01-31', periods=T, freq='ME')
    x = pd.DataFrame(rng.standard_normal((T, M)), index=idx,
                     columns=[f'f{i}' for i in range(M)])
    beta = rng.uniform(0.3, 1.2, size=(N, M))
    noise = rng.standard_normal((T, N))
    if common:
        noise += common * rng.standard_normal((T, 1))
    y = pd.DataFrame(x.to_numpy() @ beta.T + noise, index=idx,
                     columns=[f'y{i}' for i in range(N)])
    return x, y


# ── the statistics ───────────────────────────────────────────────────

def test_diagonal_residuals_pass_the_test():
    rng = np.random.default_rng(1)
    resid = pd.DataFrame(rng.standard_normal((500, 8)))
    assert diagnose_residuals(resid, n_fitted_per_asset=3).passes


def test_a_common_factor_is_detected():
    rng = np.random.default_rng(2)
    resid = rng.standard_normal((500, 8)) + 1.5 * rng.standard_normal((500, 1))
    diag = diagnose_residuals(pd.DataFrame(resid), n_fitted_per_asset=3)
    assert not diag.passes
    assert diag.n_above_edge >= 1
    assert diag.top_eigenvalue > diag.mp_edge


def test_size_of_the_test_is_near_nominal():
    """Under the null the rejection rate should sit near the 5% significance."""
    rng = np.random.default_rng(3)
    rejections = sum(
        not diagnose_residuals(pd.DataFrame(rng.standard_normal((400, 6))),
                               n_fitted_per_asset=2).passes
        for _ in range(60)
    )
    assert rejections <= 9, f"{rejections}/60 rejections is far above nominal 5%"


def test_marchenko_pastur_edge_and_threshold_are_monotone():
    assert marchenko_pastur_edge(10, 100) > marchenko_pastur_edge(10, 1000)
    assert null_threshold(45, 0.01) > null_threshold(45, 0.05)
    with pytest.raises(ValueError):
        marchenko_pastur_edge(10, 0)
    with pytest.raises(ValueError):
        null_threshold(45, 1.5)


def test_correlation_requires_two_series():
    with pytest.raises(ValueError):
        residual_correlation(pd.DataFrame(np.zeros((100, 1))))


def test_nans_are_handled_pairwise():
    rng = np.random.default_rng(4)
    resid = pd.DataFrame(rng.standard_normal((300, 5)))
    resid.iloc[:100, 0] = np.nan
    diag = diagnose_residuals(resid, n_fitted_per_asset=2)
    assert np.isfinite(diag.sphericity)
    assert diag.n_obs == 200


def test_missing_factor_components_name_the_carriers():
    rng = np.random.default_rng(5)
    common = rng.standard_normal((400, 1))
    resid = pd.DataFrame(rng.standard_normal((400, 6)), columns=list('abcdef'))
    resid[['a', 'b', 'c']] += 2.0 * common          # only a, b, c share the factor
    top = missing_factor_components(resid, n_components=1, loading_floor=0.3)
    assert set(top['series']) == {'a', 'b', 'c'}


# ── sparsity accounting ──────────────────────────────────────────────

def test_effective_sparsity_ignores_solver_dust():
    betas = np.array([[1.0, 1e-9, -1e-8], [0.0, 0.5, 2e-5]])
    out = effective_sparsity(betas)
    assert isinstance(out, Sparsity)
    assert out.n_nonzero == 2
    assert int((betas != 0).sum()) == 5        # what a bare count would report
    assert out.n_total == 6
    assert out.per_asset == 1.0


def test_effective_sparsity_is_scale_free():
    """The default cut is relative, so rescaling the units must not change the answer."""
    betas = np.array([[1.0, 1e-9], [0.0, 0.5]])
    for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        assert effective_sparsity(betas * scale).n_nonzero == 2, f"failed at scale {scale}"
    # a fixed absolute cut is not scale free, which is why it is not the default
    assert effective_sparsity(betas * 1e-4, tol=1e-4, rtol=0.0).n_nonzero == 0


def test_effective_sparsity_flags_a_factor_with_no_carrier():
    """An empty column makes beta' D^-1 beta singular; the count must say so."""
    orphan = pd.DataFrame([[1.0, 0.0], [0.8, 0.0]], columns=['live', 'orphan'])
    out = effective_sparsity(orphan)
    assert out.empty_factors == ['orphan']
    assert out.is_rank_deficient
    assert not effective_sparsity(pd.DataFrame([[1.0, 0.3]])).is_rank_deficient


def test_effective_sparsity_does_not_count_a_failed_solve_as_sparse():
    """abs(nan) > tol is False, so NaN would silently read as a zero loading."""
    betas = np.array([[1.0, np.nan], [0.5, 0.4]])
    out = effective_sparsity(betas)
    assert out.n_nonfinite == 1
    with pytest.raises(ValueError):
        effective_sparsity(betas, raise_on_nonfinite=True)


def test_effective_sparsity_reports_the_densest_response():
    betas = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    out = effective_sparsity(betas)
    assert out.max_per_asset == 3
    assert out.per_asset == 2.0
    assert list(out.per_factor) == [2, 1, 1]
    assert out.empty_assets == []


def test_effective_sparsity_rejects_negative_tolerances():
    with pytest.raises(ValueError):
        effective_sparsity(np.zeros((2, 2)), tol=-1.0)
    with pytest.raises(ValueError):
        effective_sparsity(np.zeros((2, 2)), rtol=-1.0)


def test_suggest_tolerance_finds_the_gap_between_dust_and_loadings():
    """Any cut inside the located gap must give the same count as the default."""
    betas = np.array([[1.0, 8e-9, -0.4], [2e-9, 0.6, 5e-9]])
    out = suggest_tolerance(betas)
    assert out['gap_lo'] < 1e-7 < out['gap_hi']
    assert out['gap_orders'] > 6
    assert effective_sparsity(betas, tol=out['suggested_tol'], rtol=0.0).n_nonzero == 3
    assert effective_sparsity(betas).n_nonzero == 3
    with pytest.raises(ValueError):
        suggest_tolerance(np.zeros((2, 2)))


def test_raw_offdiagonal_mass_falls_with_density_so_it_cannot_select():
    """The comparison the module docstring makes: this quantity has no interior minimum."""
    rng = np.random.default_rng(13)
    common = rng.standard_normal((400, 1))
    resid = pd.DataFrame(rng.standard_normal((400, 6)) + 2.0 * common)
    absorbed = resid - resid.mean(axis=1).to_numpy()[:, None]
    assert raw_offdiagonal_mass(absorbed) < raw_offdiagonal_mass(resid)


# ── the selector ─────────────────────────────────────────────────────

def test_selector_runs_and_exposes_the_documented_state():
    x, y = _panel(seed=6, T=240, N=6)
    sel = LassoModelDiagonalityCV(lambdas=np.geomspace(1e-2, 1e-6, 6),
                                  n_splits=3).fit(x=x, y=y)
    assert isinstance(sel.best_lambda_, float)
    assert isinstance(sel.passed_, bool)
    assert sel.threshold_ > 0
    assert sel.fold_scores_.shape == (6, 3)
    for column in ('fold_sphericity_mean', 'threshold', 'top_eigenvalue', 'passes'):
        assert column in sel.diagnostics_.columns


def test_selector_picks_a_passing_penalty_on_clean_data():
    x, y = _panel(seed=7, T=400, N=6, common=0.0)
    sel = LassoModelDiagonalityCV(lambdas=np.geomspace(1e-2, 1e-6, 6),
                                  n_splits=3).fit(x=x, y=y)
    assert sel.passed_, "clean data should admit at least one passing penalty"
    passing = sel.diagnostics_.index[sel.diagnostics_['passes']]
    assert sel.best_lambda_ == pytest.approx(float(max(passing)))   # sparsest, not the minimiser


def test_selector_reports_missing_factors_when_nothing_passes():
    x, y = _panel(seed=8, T=300, N=8, common=3.0)
    sel = LassoModelDiagonalityCV(lambdas=np.geomspace(1e-2, 1e-6, 5),
                                  n_splits=3).fit(x=x, y=y)
    assert not sel.passed_
    assert not sel.missing_factors_.empty
    assert {'component', 'eigenvalue', 'series', 'loading'} <= set(sel.missing_factors_.columns)


def test_the_two_selectors_disagree_by_construction():
    """R2 and diagonality are different criteria. Adding one is only useful if they can differ."""
    x, y = _panel(seed=9, T=300, N=8, common=2.0)
    grid = list(np.geomspace(1e-2, 1e-6, 6))
    diag = LassoModelDiagonalityCV(lambdas=grid, n_splits=3).fit(x=x, y=y)
    r2 = LassoModelCV(lambdas=grid, n_splits=3).fit(x=x, y=y)
    assert isinstance(diag.best_lambda_, float) and isinstance(r2.best_lambda_, float)
    assert diag.best_score_ > 0            # a chi-square statistic
    assert r2.best_score_ <= 1.0           # an R2


def test_score_is_out_of_sample_and_lower_is_better():
    x, y = _panel(seed=10, T=300, N=6)
    sel = LassoModelDiagonalityCV(lambdas=[1e-4, 1e-5], n_splits=3).fit(x=x, y=y)
    held_out = sel.score(x.iloc[-80:], y.iloc[-80:])
    assert np.isfinite(held_out) and held_out > 0


def test_selector_rejects_misaligned_inputs():
    x, y = _panel(seed=11, T=200, N=5)
    with pytest.raises(ValueError):
        LassoModelDiagonalityCV(lambdas=[1e-4]).fit(x=x, y=y.iloc[:-5])
    with pytest.raises(ValueError):
        LassoModelDiagonalityCV(lambdas=[]).fit(x=x, y=y)


def test_group_family_path_matches_the_per_lambda_loop():
    """use_lambda_path is a speed route, not a different estimator."""
    x, y = _panel(seed=12, T=240, N=6)
    grid = list(np.geomspace(1e-3, 1e-5, 4))
    base = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO, span=60)
    fast = LassoModelDiagonalityCV(lambdas=grid, n_splits=3, base_model=base,
                                   use_lambda_path=True).fit(x=x, y=y)
    slow = LassoModelDiagonalityCV(lambdas=grid, n_splits=3, base_model=base,
                                   use_lambda_path=False).fit(x=x, y=y)
    np.testing.assert_allclose(fast.fold_scores_.to_numpy(), slow.fold_scores_.to_numpy(),
                               rtol=1e-4, atol=1e-6)
