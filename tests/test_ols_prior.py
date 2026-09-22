"""Weighted OLS priors: exact rule, fit-span weighting, signs and causal paths."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from factorlasso import LassoModel, LassoModelCV, LassoModelType
from factorlasso.beta_priors import _compute_ols_prior
from factorlasso.lasso_estimator import _compute_solver_weights


def panel():
    """Return a seeded panel with different marginal and joint factor exposures."""
    rng = np.random.default_rng(642)
    x = pd.DataFrame(rng.normal(size=(100, 3)), columns=['Rates', 'Credit', 'PE'])
    y = pd.DataFrame({'A': 1.2 * x.Rates + .6 * x.Credit + rng.normal(0, .1, 100),
                      'B': -.4 * x.Rates + 1.8 * x.PE + rng.normal(0, .1, 100)})
    return x, y


def reference(x, y, span):
    """Independent weighted-design least squares with an explicit intercept."""
    w = np.ones(len(x)) if span is None else (1. - 2. / (span + 1.)) ** np.arange(
        len(x) - 1, -1, -1)
    betas = np.full((y.shape[1], x.shape[1]), np.nan)
    r2 = betas.copy()
    for i in range(y.shape[1]):
        for j in range(x.shape[1]):
            keep = np.isfinite(x[:, j]) & np.isfinite(y[:, i])
            xx, yy, ww = x[keep, j], y[keep, i], w[keep]
            if len(xx) < 3 or len(np.unique(xx)) < 2 or len(np.unique(yy)) < 2:
                continue
            design = np.column_stack([np.ones(len(xx)), xx])
            params = np.linalg.lstsq(design * np.sqrt(ww[:, None]),
                                     yy * np.sqrt(ww), rcond=None)[0]
            betas[i, j] = params[1]
            r2[i, j] = 1 - np.sum(ww * (yy - design @ params) ** 2) / np.sum(
                ww * (yy - np.average(yy, weights=ww)) ** 2)
    return betas, r2


@pytest.mark.parametrize('span', [None, 12., 36., 7.5])
def test_bulk_matches_weighted_ols_with_intercept_and_missing_rows(span):
    """Pairwise masks retain time-grid age and slopes match an independent solve."""
    x, y = panel()
    x.iloc[4:12, 1] = np.nan
    y.iloc[:16, 0] = np.nan
    y.iloc[35:40, 1] = np.nan
    x['constant'] = 6.
    y['constant'] = 3.
    beta, r2, prior = _compute_ols_prior(x.to_numpy(), y.to_numpy(), span)
    expected_beta, expected_r2 = reference(x.to_numpy(), y.to_numpy(), span)
    np.testing.assert_allclose(beta, expected_beta, atol=1e-12, rtol=0, equal_nan=True)
    np.testing.assert_allclose(r2, expected_r2, atol=1e-12, rtol=0, equal_nan=True)
    assert (prior[-1] == 0).all() and (prior[:, -1] == 0).all()


def test_default_prior_selects_only_highest_r2_full_slope():
    """The default gives the best-explaining factor its full beta regardless of units."""
    t = np.arange(120)
    x = np.column_stack([np.sin(2 * np.pi * t / 12), .1 * np.cos(2 * np.pi * t / 12)])
    y = np.column_stack([2 * x[:, 0] + 5 * x[:, 1], 3 * x[:, 0]])
    beta, r2, prior = _compute_ols_prior(x, y, None)
    assert np.argmax(r2[0]) == 0 and np.argmax(np.abs(beta[0])) == 1
    np.testing.assert_allclose(prior, [[2., 0.], [3., 0.]], atol=1e-12, rtol=0)
    assert LassoModel().prior_selection_type == 'highest_r2'
    model = LassoModel(apply_ols_prior=True, span=None).fit(
        pd.DataFrame(x), pd.DataFrame(y))
    np.testing.assert_allclose(model.ols_beta_prior_, prior, atol=1e-12, rtol=0)


def test_degenerate_and_tied_inputs():
    """Invalid statistics give no prior; exact ties choose input column order."""
    x = np.arange(50.)[:, None]
    _, _, prior = _compute_ols_prior(np.repeat(x, 2, axis=1), 2 * x, 36)
    np.testing.assert_allclose(prior, [[2., 0.]], atol=1e-12)
    for span in [None, 1.]:
        a = np.full((50, 1), np.nan)
        assert not _compute_ols_prior(a, x, span)[2].any()
    assert not _compute_ols_prior(x, 2 * x, 1.)[2].any()
    assert not _compute_ols_prior(x[:2], x[:2], None)[2].any()
    assert not _compute_ols_prior(x, 2 * x, 36, min_periods=51)[2].any()
    with pytest.raises(ValueError, match='nonempty'):
        _compute_ols_prior(np.empty((0, 1)), np.empty((0, 1)), 36)


def test_effective_loss_span_controls_priors_instead_of_cluster_span():
    """Per-call beta span wins over constructor and clustering span settings."""
    x, y = panel()
    y.iloc[-15:] += 2 * x.iloc[-15:].Rates.to_numpy()[:, None]
    model = LassoModel(apply_ols_prior=True, span=80, cluster_correlation_span=100,
                       reg_lambda=.001).fit(x, y, span=12)
    expected, r2 = reference(x.to_numpy(), y.to_numpy(), 12)
    np.testing.assert_allclose(model.ols_betas_, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.ols_r2_, r2, atol=1e-12, rtol=0)
    assert model.ols_prior_span_ == model.effective_span_ == 12
    assert model.effective_cluster_correlation_span_ == 100
    solver_weights = _compute_solver_weights(len(x), 1, 12, np.ones((len(x), 1)))[:, 0]
    np.testing.assert_allclose(solver_weights ** 2, (1 - 2 / 13) ** np.arange(99, -1, -1))


@pytest.mark.parametrize('sign,beta', [(0., 2.), (1., -2.), (-1., 2.)])
def test_sign_conflict_is_zero_without_reselection(sign, beta):
    """A prohibited PE winner loses its prior; another factor does not inherit it."""
    x, _ = panel()
    y = pd.DataFrame({'Fund': beta * x.PE})
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    signs.loc['Fund', 'PE'] = sign
    model = LassoModel(
        apply_ols_prior=True, span=36, reg_lambda=.01,
        factors_beta_loading_signs=signs).fit(x, y)
    np.testing.assert_allclose(model.ols_beta_prior_.loc['Fund', 'PE'], beta, atol=1e-12)
    assert (model.effective_beta_prior_.to_numpy() == 0).all()


def test_compatible_negative_and_unconstrained_priors_are_retained():
    """Nonpositive and free factors keep correctly signed raw empirical priors."""
    x, _ = panel()
    y = pd.DataFrame({'Fund': -2 * x.PE})
    for sign in [-1., np.nan]:
        signs = pd.DataFrame(sign, index=y.columns, columns=x.columns)
        m = LassoModel(apply_ols_prior=True,
                       factors_beta_loading_signs=signs).fit(x, y)
        assert m.effective_beta_prior_.loc['Fund', 'PE'] == pytest.approx(-2.)
    m = LassoModel(apply_ols_prior=True, nonneg=True).fit(x, y)
    assert (m.effective_beta_prior_.to_numpy() == 0).all()


def test_auto_sign_zero_gate_also_zeros_prior():
    """The actual final solver constraints filter the selected prior."""
    x, y = panel()
    m = LassoModel(apply_ols_prior=True, span=36,
                   auto_sign_constraints=True, auto_sign_threshold_t=1e9).fit(x, y)
    assert np.count_nonzero(m.ols_beta_prior_) > 0
    assert not m.effective_beta_prior_.to_numpy().any()


def test_manual_overrides_nan_fallback_zero_and_sign_conflicts():
    """Overrides stay immutable; explicit zero wins and incompatible values are zeroed."""
    x, y = panel()
    override = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    override.loc['A', 'Rates'] = 0.
    override.loc['A', 'Credit'] = .75
    override.loc['B', 'PE'] = 9.
    saved = override.copy()
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    signs.loc['B', 'PE'] = 0.
    m = LassoModel(apply_ols_prior=True,
                   factors_beta_prior=override, factors_beta_loading_signs=signs).fit(x, y)
    assert m.effective_beta_prior_.loc['A', 'Rates'] == 0.
    assert m.effective_beta_prior_.loc['A', 'Credit'] == .75
    assert m.effective_beta_prior_.loc['B', 'PE'] == 0.
    pd.testing.assert_frame_equal(override, saved)
    override.loc['A', 'Credit'] = np.inf
    with pytest.raises(ValueError, match='finite or NaN'):
        replace(m, factors_beta_prior=override).fit(x, y)


def test_disabled_mode_and_repeated_fit_reset_diagnostics():
    """The default does not call the helper and preserves explicit-prior semantics."""
    x, y = panel()
    enabled = LassoModel(apply_ols_prior=True, span=36).fit(x, y)
    assert clone(enabled).apply_ols_prior is True
    assert clone(enabled).ols_betas_ is None
    assert 'ols_betas_' not in enabled.get_params()
    disabled = enabled.set_params(apply_ols_prior=False).fit(x, y)
    reference_model = LassoModel(span=36).fit(x, y)
    np.testing.assert_array_equal(disabled.coef_, reference_model.coef_)
    assert disabled.ols_betas_ is None and disabled.effective_beta_prior_ is None
    assert disabled.ols_prior_span_ is None


@pytest.mark.parametrize('kind', [LassoModelType.LASSO, LassoModelType.GROUP_LASSO,
    LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
    LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
    LassoModelType.COOPERATIVE_GROUP_LASSO, LassoModelType.COOPERATIVE_CLUSTER_GROUP_LASSO])
def test_prior_reaches_each_supported_solver_and_lambda_path(kind):
    """Automatic and explicitly supplied centres yield equivalent direct/path fits."""
    x, y = panel()
    kwargs = dict(model_type=kind, span=36, reg_lambda=.01)
    if kind in (LassoModelType.GROUP_LASSO, LassoModelType.COOPERATIVE_GROUP_LASSO):
        kwargs['group_data'] = pd.Series(1, index=y.columns)
    model = LassoModel(apply_ols_prior=True, **kwargs).fit(x, y)
    manual = LassoModel(factors_beta_prior=model.effective_beta_prior_, **kwargs).fit(x, y)
    np.testing.assert_array_equal(model.coef_, manual.coef_)
    path = LassoModel(apply_ols_prior=True,
                      **kwargs).fit_reg_lambda_path(x, y, [.01, .02])
    for item in path:
        direct = replace(model, reg_lambda=item.reg_lambda).fit(x, y)
        np.testing.assert_allclose(item.coef_, direct.coef_, atol=1e-4, rtol=0)
        pd.testing.assert_frame_equal(item.effective_beta_prior_, direct.effective_beta_prior_)
        assert item.ols_prior_span_ == 36
    path[0].ols_betas_.iloc[0, 0] = 999
    assert path[1].ols_betas_.iloc[0, 0] != 999


def test_group_prior_is_per_asset_not_pooled():
    """Assets in the same group retain different individual prior centres."""
    x, y = panel()
    m = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                   n_clusters=1, apply_ols_prior=True, span=36).fit(x, y)
    expected, _ = reference(x.to_numpy(), y.to_numpy(), 36)
    np.testing.assert_allclose(m.ols_betas_, expected, atol=1e-12)
    assert m.ols_beta_prior_.loc['A'].idxmax() == 'Rates'
    assert m.ols_beta_prior_.loc['B'].idxmax() == 'PE'


def test_cv_builds_priors_from_each_training_fold_once(monkeypatch):
    """Path preparation is fold-local and shared across lambdas."""
    x, y = panel()
    calls = []
    original = LassoModel._prepare_fit

    def capture(self, *args, **kwargs):
        """Inspect the actual prior and training sample supplied to the path."""
        result = original(self, *args, **kwargs)
        expected, _ = reference(kwargs['x'].to_numpy(), kwargs['y'].to_numpy(),
                                 kwargs['eff_span'])
        np.testing.assert_allclose(self.ols_betas_, expected, atol=1e-12)
        calls.append(len(kwargs['x']))
        return result

    monkeypatch.setattr(LassoModel, '_prepare_fit', capture)
    LassoModelCV(base_model=LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        apply_ols_prior=True, span=36),
        lambdas=[.001, .01], n_splits=3, refit=False, use_lambda_path=True).fit(x, y)
    assert len(calls) == 3 and max(calls) < len(x)
    assert calls == sorted(calls)


@pytest.mark.parametrize('bad', [1, 'true', None])
def test_flag_validation(bad):
    """Only a boolean flag is accepted, including after set_params."""
    with pytest.raises(ValueError, match='boolean'):
        LassoModel(apply_ols_prior=bad)
    x, y = panel()
    with pytest.raises(ValueError, match='boolean'):
        LassoModel().set_params(apply_ols_prior=bad).fit(x, y)


def test_unilasso_rejects_an_unsupported_prior_flag():
    """UniLasso has no beta-prior objective and must not silently ignore the flag."""
    with pytest.raises(ValueError, match='UNILASSO'):
        LassoModel(model_type=LassoModelType.UNILASSO, apply_ols_prior=True)


@pytest.mark.parametrize('span', [None, 12., 36., 7.5])
def test_highest_r2_uses_full_slope_not_largest_absolute_beta(span):
    """The sole winner receives its full slope, checked with residual-based WLS R2."""
    t = np.arange(120)
    x = np.column_stack([np.sin(2 * np.pi * t / 12), .01 * np.cos(2 * np.pi * t / 12)])
    y = np.column_stack([2 * x[:, 0] + 50 * x[:, 1], -3 * x[:, 0]])
    beta, r2, prior = _compute_ols_prior(x, y, span, prior_selection_type='highest_r2')
    expected_beta, expected_r2 = reference(x, y, span)
    expected = np.zeros_like(expected_beta)
    winners = expected_r2.argmax(axis=1)
    expected[np.arange(2), winners] = expected_beta[np.arange(2), winners]
    assert winners[0] == 0 and np.argmax(np.abs(expected_beta[0])) == 1
    np.testing.assert_allclose(beta, expected_beta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(r2, expected_r2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(prior, expected, atol=1e-12, rtol=0)


def test_highest_r2_selection_uses_effective_ewma_span_and_is_scale_invariant():
    """Recent evidence chooses the winner; changing factor units cannot change it."""
    rng = np.random.default_rng(986)
    x = pd.DataFrame(rng.normal(size=(400, 2)), columns=['Old', 'Recent'])
    y = pd.DataFrame({'A': 2 * x.Old})
    y.iloc[-60:, 0] = 3 * x.Recent.iloc[-60:]
    y.iloc[100:110] = np.nan
    model = LassoModel(apply_ols_prior=True, prior_selection_type='highest_r2',
                      span=100, cluster_correlation_span=200).fit(x, y, span=12)
    beta, r2 = reference(x.to_numpy(), y.to_numpy(), 12)
    _, uniform_r2 = reference(x.to_numpy(), y.to_numpy(), None)
    assert r2.argmax(axis=1)[0] == 1 and uniform_r2.argmax(axis=1)[0] == 0
    np.testing.assert_allclose(model.ols_beta_prior_, [[0, beta[0, 1]]], atol=1e-12)
    assert model.ols_prior_span_ == model.effective_span_ == 12
    scaled = x * pd.Series({'Old': .01, 'Recent': 100.})
    other = replace(model).fit(scaled, y, span=12)
    np.testing.assert_allclose(other.ols_r2_, model.ols_r2_, atol=1e-12)
    np.testing.assert_allclose(other.ols_beta_prior_ * [0.01, 100.],
                               model.ols_beta_prior_, atol=1e-12)
    assert clone(model).prior_selection_type == 'highest_r2'


def test_highest_r2_ties_and_unestimable_pairs():
    """Ties use factor order; constants, absent data and warmup produce zero priors."""
    x = np.arange(50.)[:, None]
    inputs = np.column_stack([x, x, np.ones_like(x)])
    _, _, prior = _compute_ols_prior(inputs, 2*x, 36, prior_selection_type='highest_r2')
    np.testing.assert_allclose(prior, [[2., 0., 0.]], atol=1e-12)
    for span in [None, 1.]:
        assert not _compute_ols_prior(inputs, np.full_like(x, np.nan), span,
            prior_selection_type='highest_r2')[2].any()
    assert not _compute_ols_prior(inputs, 2*x, 36, min_periods=51,
        prior_selection_type='highest_r2')[2].any()


@pytest.mark.parametrize('bad', [None, '', 'largest_beta', 'highest_r2_and_abs_beta',
                                 1, ['highest_r2']])
def test_prior_selection_validation_at_construction_and_fit(bad):
    """Invalid selection modes fail clearly, including sklearn-style mutation."""
    with pytest.raises(ValueError, match='prior_selection_type'):
        LassoModel(prior_selection_type=bad)
    x, y = panel()
    with pytest.raises(ValueError, match='prior_selection_type'):
        LassoModel().set_params(prior_selection_type=bad).fit(x, y)
    with pytest.raises(ValueError, match='prior_selection_type'):
        _compute_ols_prior(x.to_numpy(), y.to_numpy(), 36, prior_selection_type=bad)
