"""Exposure-budget floors, floor-centred penalties and the selector scope of expert floors.

The references are computed independently of the package: weighted normal equations for the
omitted-variable slopes, and a direct solver call with a zero centre for the floor-centred
LASSO (Sepp 2026, Proposition 5(i): on b >= l >= 0, |b - l| = |b| - l).
"""
import numpy as np
import pandas as pd
import pytest
from factorlasso import LassoModel, LassoModelType, get_x_y_np
from factorlasso.linear_model._solvers.lasso import solve_lasso_cvx_problem

SPAN = 60


def panel(noise=.01, seed=11):
    """Three correlated factors and two responses with exposure to all of them."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(0, .03, (180, 3)), columns=['Equity', 'Credit', 'PE'])
    x['Credit'] += 1.5 * x.Equity
    x['PE'] += .8 * x.Equity
    y = pd.DataFrame({
        'asset': .9 * x.Equity + .5 * x.Credit + .3 * x.PE + rng.normal(0, noise, len(x)),
        'other': -(.7 * x.Equity + .4 * x.Credit) + rng.normal(0, noise, len(x)),
    })
    return x, y


def slopes_on(x, y, factor):
    """Weighted OLS coefficient of every other factor on ``factor``, on the rows of ``y``."""
    w = (1 - 2 / (SPAN + 1)) ** np.arange(len(x) - 1, -1, -1) * y.notna().to_numpy()
    z = np.column_stack([np.ones(len(x)), x[factor].to_numpy()])
    gram = z.T @ (w[:, None] * z)
    return pd.Series({other: np.linalg.solve(gram, z.T @ (w * x[other].to_numpy()))[1]
                      for other in x.columns.drop(factor)})


def model(**kwargs):
    """The LASSO configuration shared by the tests."""
    settings = dict(apply_ols_prior=True, factor_for_prior={'asset': 'Equity', 'other': 'Equity'},
                    span=SPAN, reg_lambda=1e-5, loss_normalization='weight_sum')
    settings.update(kwargs)
    return LassoModel(**settings)


def test_reference_slopes_satisfy_the_omitted_variable_identity():
    """Without noise the marginal slope equals the loading plus sum_j Gamma_j * beta_j."""
    x, y = panel(noise=0.)
    fitted = model().fit(x, y)
    gamma = slopes_on(x, y.asset, 'Equity')
    expected = .9 + gamma['Credit'] * .5 + gamma['PE'] * .3
    assert fitted.ols_betas_.at['asset', 'Equity'] == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize('signs', ['free', 'hard'])
def test_budget_slack_matches_independent_slopes(signs):
    """The floor is |target| minus the sign-aware sum of slope times budget."""
    x, y = panel()
    budget = pd.DataFrame({'Credit': [.2], 'PE': [.5]}, index=['asset'])
    hard = None
    if signs == 'hard':
        hard = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
        hard.loc['asset', ['Credit', 'PE']] = [1., -1.]
    fitted = model(expert_prior_exposure_budget=budget,
                   factors_beta_loading_signs=hard).fit(x, y)
    gamma = slopes_on(x, y.asset, 'Equity')
    assert gamma['Credit'] > 0 and gamma['PE'] > 0
    if signs == 'free':
        slack = abs(gamma['Credit']) * .2 + abs(gamma['PE']) * .5
    else:
        # A negative competitor loading lowers the marginal slope and adds no floor slack.
        slack = max(gamma['Credit'], 0) * .2 + max(-gamma['PE'], 0) * .5
    row = fitted.prior_bound_diagnostics_.set_index('asset').loc['asset']
    target = fitted.ols_betas_.at['asset', 'Equity']
    assert row.slack_source == 'budget'
    assert row.slack == pytest.approx(slack, rel=1e-9)
    assert fitted.prior_lower_bounds_.at['asset', 'Equity'] == pytest.approx(target - slack)
    assert fitted.coef_.at['asset', 'Equity'] >= target - slack - 2e-5
    assert row.competitor == 'Credit'
    assert row.competitor_slope == pytest.approx(gamma['Credit'], rel=1e-9)
    assert row.implied_budget == pytest.approx(slack / gamma['Credit'], rel=1e-9)


def test_negative_target_orients_the_slack_and_caps_the_loading():
    """A negative selected slope gets an upper bound with the slack oriented by its sign."""
    x, y = panel()
    hard = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    hard.loc['other', 'Credit'] = -1.
    budget = pd.DataFrame({'Credit': [.3]}, index=['other'])
    fitted = model(expert_prior_exposure_budget=budget,
                   factors_beta_loading_signs=hard).fit(x, y)
    gamma = slopes_on(x, y.other, 'Equity')['Credit']
    target = fitted.ols_betas_.at['other', 'Equity']
    assert target < 0
    slack = max(-1. * gamma * -1., 0.) * .3
    assert fitted.prior_upper_bounds_.at['other', 'Equity'] == pytest.approx(-(abs(target) - slack))
    assert fitted.coef_.at['other', 'Equity'] <= -(abs(target) - slack) + 2e-5
    assert fitted.prior_lower_bounds_.loc['other'].isna().all()


@pytest.mark.parametrize('n_std', [None, 1.])
def test_responses_without_a_budget_fall_back_to_standard_errors(n_std):
    """Budget rows take precedence; other responses use n_std or carry no floor."""
    x, y = panel()
    budget = pd.DataFrame({'Credit': [.2], 'PE': [np.nan]}, index=['asset'])
    fitted = model(expert_prior_exposure_budget=budget, expert_prior_bound_n_std=n_std,
                   expert_prior_hac_lags=3).fit(x, y)
    rows = fitted.prior_bound_diagnostics_.set_index('asset')
    assert rows.at['asset', 'slack_source'] == 'budget'
    # The missing PE budget counts as zero in a declared row.
    gamma = slopes_on(x, y.asset, 'Equity')['Credit']
    assert rows.at['asset', 'slack'] == pytest.approx(abs(gamma) * .2, rel=1e-9)
    if n_std is None:
        assert rows.at['other', 'status'] == 'no_exposure_budget'
        assert fitted.prior_upper_bounds_.loc['other'].isna().all()
    else:
        assert rows.at['other', 'slack_source'] == 'n_std'
        assert rows.at['other', 'slack'] == pytest.approx(rows.at['other', 'se_hac'])
        assert fitted.prior_upper_bounds_.loc['other'].notna().any()


def test_zero_budget_floors_the_loading_at_its_target():
    """A declared zero exposure leaves no slack below the selected slope."""
    x, y = panel()
    budget = pd.DataFrame(0., index=['asset'], columns=['Credit', 'PE'])
    fitted = model(expert_prior_exposure_budget=budget).fit(x, y)
    target = fitted.ols_betas_.at['asset', 'Equity']
    assert fitted.prior_lower_bounds_.at['asset', 'Equity'] == pytest.approx(target)
    assert fitted.coef_.at['asset', 'Equity'] >= target - 2e-5


def test_named_scope_leaves_automatic_winners_unfloored():
    """Only explicit selections are floored; automatic winners are recorded as such."""
    x, y = panel()
    fitted = model(factor_for_prior={'asset': 'Equity'}, expert_prior_bound_n_std=1.,
                   expert_prior_bound_scope='named').fit(x, y)
    rows = fitted.prior_bound_diagnostics_.set_index('asset')
    assert rows.at['other', 'status'] == 'automatic_selection_not_floored'
    assert fitted.prior_lower_bounds_.loc['other'].isna().all()
    assert fitted.prior_upper_bounds_.loc['other'].isna().all()
    assert rows.at['asset', 'status'] == 'imposed'
    everything = model(factor_for_prior={'asset': 'Equity'}, expert_prior_bound_n_std=1.).fit(x, y)
    assert everything.prior_bound_diagnostics_.set_index('asset').at['other', 'status'] == 'imposed'


def solve_with_centre(fitted, x, y, centre):
    """Solve the fitted LASSO problem again with a supplied penalty centre."""
    x_np, y_np, valid = get_x_y_np(x, y, span=SPAN, demean=fitted.demean)
    return solve_lasso_cvx_problem(
        x=x_np, y=y_np, valid_mask=valid, reg_lambda=fitted.reg_lambda, span=SPAN,
        solver='CLARABEL', loss_normalization=fitted.loss_normalization,
        factors_beta_prior=centre,
        beta_lower_bounds=fitted.prior_lower_bounds_.to_numpy(),
        beta_upper_bounds=fitted.prior_upper_bounds_.to_numpy()).estimated_beta


def test_floor_centre_equals_the_zero_centred_lasso_on_the_floored_cells():
    """Proposition 5(i): centring at the floor removes the pull towards the prior."""
    x, y = panel()
    budget = pd.DataFrame({'Credit': [.15], 'PE': [.15]}, index=['asset', 'other'])
    floor = model(expert_prior_exposure_budget=budget, expert_prior_bound_centre='floor',
                  reg_lambda=2e-4).fit(x, y)
    prior = model(expert_prior_exposure_budget=budget, reg_lambda=2e-4).fit(x, y)
    floored = (floor.prior_lower_bounds_.notna() | floor.prior_upper_bounds_.notna()).to_numpy()
    assert floored.sum() == 2
    zero_centre = floor.effective_beta_prior_.to_numpy().copy()
    zero_centre[floored] = 0.
    np.testing.assert_allclose(floor.coef_.to_numpy(), solve_with_centre(floor, x, y, zero_centre),
                               atol=2e-6)
    prior_centre = prior.effective_beta_prior_.to_numpy()
    np.testing.assert_allclose(prior.coef_.to_numpy(), solve_with_centre(prior, x, y, prior_centre),
                               atol=2e-6)
    assert np.abs(floor.coef_.to_numpy() - prior.coef_.to_numpy()).max() > 1e-3


@pytest.mark.parametrize('mode', [LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                                  LassoModelType.FACTOR_CLUSTER_GROUP_LASSO])
def test_group_modes_receive_the_floor_as_centre(mode, monkeypatch):
    """Only the floored cells change centre; every other solver input is unchanged."""
    from factorlasso.linear_model import _dispatch
    x, y = panel()
    captured = {}
    original = _dispatch.solve_group_lasso_cvx_problem

    def capture(**kwargs):
        captured.setdefault('calls', []).append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(_dispatch, 'solve_group_lasso_cvx_problem', capture)
    budget = pd.DataFrame({'Credit': [.15]}, index=['asset'])
    for centre in ('prior', 'floor'):
        model(model_type=mode, expert_prior_exposure_budget=budget, expert_prior_bound_n_std=1.,
              expert_prior_bound_centre=centre).fit(x, y)
    prior_call, floor_call = captured['calls']
    lower = prior_call['beta_lower_bounds']
    upper = prior_call['beta_upper_bounds']
    expected = prior_call['factors_beta_prior'].copy()
    expected[np.isfinite(lower)] = lower[np.isfinite(lower)]
    expected[np.isfinite(upper)] = upper[np.isfinite(upper)]
    assert np.isfinite(lower).sum() + np.isfinite(upper).sum() == 2
    np.testing.assert_array_equal(floor_call['factors_beta_prior'], expected)
    for key in ('beta_lower_bounds', 'beta_upper_bounds'):
        np.testing.assert_array_equal(floor_call[key], prior_call[key])


@pytest.mark.parametrize('kwargs, error', [
    (dict(expert_prior_exposure_budget=pd.DataFrame({'Credit': [-.1]}, index=['asset'])),
     ValueError),
    (dict(expert_prior_exposure_budget=pd.DataFrame({'Credit': [np.inf]}, index=['asset'])),
     ValueError),
    (dict(expert_prior_exposure_budget={'asset': {'Credit': .1}}), TypeError),
    (dict(expert_prior_exposure_budget=pd.DataFrame({'Credit': [.1]}, index=['asset']),
          apply_ols_prior=False, factor_for_prior=None), ValueError),
    (dict(expert_prior_bound_centre='middle'), ValueError),
    (dict(expert_prior_bound_scope='automatic'), ValueError),
])
def test_invalid_budget_settings_fail_closed(kwargs, error):
    """Malformed budgets and unknown policies raise at construction."""
    with pytest.raises(error):
        model(**kwargs)


def test_unknown_budget_factor_raises_at_fit_and_clone_keeps_the_budget():
    """Budget columns must name fitted factors; cloning keeps the frame unchanged."""
    from sklearn.base import clone
    x, y = panel()
    bad = model(expert_prior_exposure_budget=pd.DataFrame({'Rates': [.1]}, index=['asset']))
    with pytest.raises(ValueError, match='unknown factors'):
        bad.fit(x, y)
    budget = pd.DataFrame({'Credit': [.2]}, index=['asset'])
    copy = clone(model(expert_prior_exposure_budget=budget, expert_prior_bound_centre='floor',
                       expert_prior_bound_scope='named'))
    pd.testing.assert_frame_equal(copy.expert_prior_exposure_budget, budget)
    assert copy.expert_prior_bound_centre == 'floor'
    assert copy.expert_prior_bound_scope == 'named'
