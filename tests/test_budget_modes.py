"""Expert-input modes checked against independent OLS and explicit peer means."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType
from factorlasso.priors._budget_modes import weighted_max_vif


def panel(*, rho: float = .35, funds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(716)
    raw = rng.normal(size=(180, 3))
    raw -= raw.mean(axis=0)
    orthogonal = np.linalg.qr(raw)[0] * np.sqrt(len(raw))
    x = pd.DataFrame(orthogonal, columns=['a', 'b', 'c'])
    x['b'] = rho * x.a + np.sqrt(1 - rho**2) * x.b
    y = pd.DataFrame({f'y{i}': (1 + i * .1) * x.a + .4 * x.b for i in range(funds)})
    return x, y


def model_for(*, y: pd.DataFrame, **kwargs: object) -> LassoModel:
    settings = dict(model_type=LassoModelType.GROUP_LASSO,
                    group_data=pd.Series('cluster', index=y.columns), apply_ols_prior=True,
                    factor_for_prior={name: 'a' for name in y.columns},
                    expert_prior_bound_n_std=1., expert_prior_hac_lags=3,
                    loss_normalization='weight_sum', reg_lambda=0., solver='CLARABEL')
    settings.update(kwargs)
    return LassoModel(**settings)


@pytest.mark.parametrize('selection', [None, {'y0': 'a'}, {'y0': ('a', 'c')}])
def test_table_data_rule_rows(selection: dict | None) -> None:
    x, y = panel()
    fitted = model_for(y=y, factor_for_prior=selection).fit(x=x, y=y)
    expected = (x.cov().loc['a', 'b'] * .4 + x.a.var()) / x.a.var()
    assert fitted.ols_beta_prior_.at['y0', 'a'] == pytest.approx(expected, abs=1e-12)
    diagnostic = fitted.prior_bound_contributions_
    assert diagnostic.budget_source.eq('n_std').all()
    assert diagnostic.contribution.isna().all()
    for row in diagnostic.itertuples():
        if abs(row.gamma) > 1e-8:
            assert row.implied_one_se_budget == pytest.approx(row.se_hac / abs(row.gamma))


def test_partial_declarations_and_explicit_zero_have_distinct_sources() -> None:
    x, y = panel()
    budget = pd.DataFrame({'b': [.2, 0.]}, index=['y0', 'y1'])
    with pytest.warns(UserWarning, match='undeclared_zero'):
        fitted = model_for(y=y, expert_prior_exposure_budget=budget).fit(x=x, y=y)
    d = fitted.prior_bound_contributions_.set_index(['response', 'competitor'])
    assert d.at[('y0', 'b'), 'budget_source'] == 'declared'
    assert d.at[('y1', 'b'), 'budget_source'] == 'declared_zero'
    assert d.at[('y0', 'c'), 'budget_source'] == 'undeclared_zero'
    assert d.at[('y2', 'b'), 'budget_source'] == 'n_std'
    assert d.at[('y0', 'b'), 'slack'] == pytest.approx(.35 * .2, abs=1e-12)
    for response in ['y0', 'y1']:
        rows = d.loc[response]
        assert rows.contribution.sum() == pytest.approx(rows.slack.iloc[0], abs=1e-12)


def test_pair_demotion_preserves_first_and_exposes_second_as_competitor() -> None:
    x, y = panel(rho=.96)
    chosen = {'y0': ('a', 'b')}
    with pytest.warns(UserWarning, match='named_set_demoted') as caught:
        fitted = model_for(y=y, factor_for_prior=chosen).fit(x=x, y=y)
    assert len(caught) == 1
    assert chosen == {'y0': ('a', 'b')}
    assert fitted.factor_for_prior == chosen
    assert fitted.effective_factor_for_prior_['y0'] == ('a',)
    assert fitted.named_set_diagnostics_.at['y0', 'largest_vif'] == pytest.approx(
        1 / (1 - .96**2), abs=1e-10)
    assert fitted.ols_beta_prior_.at['y0', 'a'] == pytest.approx(1 + .96 * .4)
    assert fitted.ols_beta_prior_.at['y0', 'b'] == 0
    assert 'b' in fitted.prior_bound_contributions_.query("response == 'y0'").competitor.values
    legacy = model_for(y=y, factor_for_prior=chosen, named_set_max_vif=None).fit(x=x, y=y)
    np.testing.assert_allclose(legacy.ols_beta_prior_.loc['y0', ['a', 'b']], [1, .4], atol=1e-12)


def test_vif_uses_response_valid_rows_and_original_calendar_weights() -> None:
    x, y = panel(rho=.6)
    y.loc[:30, 'y0'] = np.nan
    y.loc[80:87, 'y0'] = np.nan
    valid = y.y0.notna().to_numpy()
    weights = (59 / 61)**np.arange(len(x)-1, -1, -1)
    weights = weights[valid] / weights[valid].sum()
    array = x.loc[valid, ['a', 'b']].to_numpy()
    centred = array - np.average(array, axis=0, weights=weights)
    reference = 1 / (1 - np.sum(weights * centred[:, 0] * centred[:, 1])**2 /
                     np.prod(np.sum(weights[:, None] * centred**2, axis=0)))
    assert weighted_max_vif(x=x[['a', 'b']], y=y.y0, span=60, min_periods=3) == pytest.approx(
        reference, abs=1e-10)


def test_three_named_factors_rejected_with_response_and_count() -> None:
    with pytest.raises(ValueError, match="'fund'.*3 factors"):
        LassoModel(apply_ols_prior=True, factor_for_prior={'fund': ('a', 'b', 'c')})


@pytest.mark.parametrize('parameter,value', [('max_named_factors', 3),
    ('max_named_factors', True), ('named_set_max_vif', .5),
    ('named_set_max_vif', np.inf), ('expert_prior_budget_rule', 'median')])
def test_mode_validation(parameter: str, value: object) -> None:
    with pytest.raises(ValueError, match=parameter):
        LassoModel(**{parameter: value})


def test_cluster_mean_recovers_peer_loading_and_additive_slack() -> None:
    x, y = panel()
    fitted = model_for(y=y, expert_prior_budget_rule='cluster_mean').fit(x=x, y=y)
    np.testing.assert_allclose(fitted.budget_first_pass_coef_.b, .4, atol=1e-6)
    np.testing.assert_allclose(fitted.resolved_budgets_.b, .4, atol=1e-6)
    for response in y:
        reference = fitted.budget_first_pass_coef_.drop(index=response).abs().mean()
        np.testing.assert_allclose(fitted.resolved_budgets_.loc[response], reference, atol=1e-12)
    for _, rows in fitted.prior_bound_contributions_.groupby(['response', 'named_factor']):
        assert rows.contribution.sum() == pytest.approx(rows.slack.iloc[0], abs=1e-10)
        assert rows.slack.iloc[0] == pytest.approx(.35 * .4, abs=1e-6)
    assert fitted.budget_rule_status_.eq('cluster_mean').all()


def test_cluster_declarations_override_only_given_cells() -> None:
    x, y = panel()
    declared = pd.DataFrame({'b': [0.0]}, index=['y0'])
    fitted = model_for(y=y, expert_prior_budget_rule='cluster_mean',
                       expert_prior_exposure_budget=declared).fit(x=x, y=y)
    assert fitted.resolved_budgets_.at['y0', 'b'] == 0
    assert fitted.budget_sources_.at['y0', 'b'] == 'declared_zero'
    assert fitted.budget_sources_.at['y0', 'c'] == 'cluster_mean'
    assert fitted.resolved_budgets_.at['y1', 'b'] == pytest.approx(.4, abs=1e-6)


def test_cluster_two_falls_back_and_plain_lasso_rejects() -> None:
    x, y = panel(funds=2)
    fitted = model_for(y=y, expert_prior_budget_rule='cluster_mean',
                       expert_prior_bound_n_std=None).fit(x=x, y=y)
    reference = model_for(y=y).fit(x=x, y=y)
    pd.testing.assert_frame_equal(fitted.prior_lower_bounds_, reference.prior_lower_bounds_)
    assert fitted.budget_rule_status_.eq('cluster_too_small').all()
    assert fitted.prior_bound_contributions_.budget_source.eq('n_std').all()
    with pytest.raises(ValueError, match='requires clusters'):
        LassoModel(apply_ols_prior=True, expert_prior_budget_rule='cluster_mean')


def test_refit_resets_budget_state_and_path_recomputes_first_pass() -> None:
    x, y = panel()
    model = model_for(y=y, expert_prior_budget_rule='cluster_mean')
    path = model.fit_reg_lambda_path(x=x, y=y, reg_lambdas=[0., .1])
    for penalty, fit in zip([0., .1], path):
        reference = model.copy(kwargs={'reg_lambda': penalty}).fit(x=x, y=y)
        pd.testing.assert_frame_equal(fit.budget_first_pass_coef_,
                                      reference.budget_first_pass_coef_)
    assert not np.allclose(path[0].resolved_budgets_, path[1].resolved_budgets_)
    model.fit(x=x, y=y)
    model.set_params(expert_prior_budget_rule='n_std', expert_prior_bound_n_std=None)
    model.fit(x=x, y=y)
    assert model.budget_first_pass_coef_ is None
    assert model.prior_bound_contributions_ is None


def test_internal_first_pass_adapter_rejects_stale_panel_and_changed_penalty() -> None:
    from factorlasso.utils._panel import get_x_y_np
    x, y = panel()
    model = model_for(y=y, expert_prior_budget_rule='cluster_mean')
    first = model.copy(kwargs=dict(expert_prior_budget_rule='n_std',
        expert_prior_bound_n_std=None, expert_prior_exposure_budget=None,
        apply_ols_prior=False, factor_for_prior=None, factors_beta_prior=None)).fit(x=x, y=y)
    xp, yp, valid = get_x_y_np(x=x, y=y, span=None, demean=model.demean)
    kwargs = dict(x=x, y=y, x_np=xp, y_np=yp, valid_mask=valid,
                  eff_span=None, eff_cluster_correlation_span=None, budget_first_pass=first)
    prepared = model._prepare_fit(**kwargs)
    native = model_for(y=y, expert_prior_budget_rule='cluster_mean').fit(x=x, y=y)
    np.testing.assert_allclose(prepared.lower_bounds_np, native.prior_lower_bounds_, equal_nan=True)
    first.reg_lambda = 1.
    with pytest.raises(ValueError, match='reg_lambda'):
        model._prepare_fit(**kwargs)
    first.reg_lambda = 0.
    first.y_ = first.y_ + 1.
    with pytest.raises(ValueError, match='different panel'):
        model._prepare_fit(**kwargs)


def test_bound_diagnostics_refresh_declarations_between_calls() -> None:
    from factorlasso.linear_model._restrictions import expert_prior_bounds
    x, y = panel()
    model = model_for(y=y).fit(x=x, y=y)
    for limit in [.2, .4]:
        model.expert_prior_exposure_budget = pd.DataFrame(0., index=y.columns, columns=x.columns)
        model.expert_prior_exposure_budget['b'] = limit
        _, _, diagnostic = expert_prior_bounds(
            model, x, y, None, model.ols_r2_, model.ols_beta_prior_, model.effective_beta_prior_)
        np.testing.assert_allclose(diagnostic.slack, .35 * limit, atol=1e-12)
