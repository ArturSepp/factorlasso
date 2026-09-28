"""Individual expert-OLS bounds, independent HAC reference and solver propagation."""
import numpy as np
import pandas as pd
import pytest
from factorlasso import LassoModel, LassoModelType


def panel():
    """Correlated factors let a selected marginal exposure migrate in a full fit."""
    rng = np.random.default_rng(923)
    x = pd.DataFrame(rng.normal(0, .03, (200, 3)), columns=['Equity', 'Credit', 'PE'])
    x['Credit'] += 2 * x.Equity
    y = pd.DataFrame({'asset': x.Credit + rng.normal(0, .007, len(x)),
                      'other': -x.Credit + rng.normal(0, .008, len(x))})
    return x, y


def test_expert_and_automatic_priors_both_receive_individual_bounds():
    """A partial expert mapping bounds both its explicit target and automatic fallback."""
    x, y = panel()
    model = LassoModel(apply_ols_prior=True, factor_for_prior={'asset': 'Equity'},
                       expert_prior_bound_n_std=1.0, expert_prior_hac_lags=3,
                       span=60, reg_lambda=1e-5, loss_normalization='weight_sum').fit(x, y)
    floor = model.prior_lower_bounds_.loc['asset', 'Equity']
    assert floor > 1.5
    assert model.coef_.loc['asset', 'Equity'] >= floor - 2e-5
    assert model.prior_lower_bounds_.loc['other'].isna().all()
    ceiling = model.prior_upper_bounds_.loc['other', 'Credit']
    assert ceiling < -.5
    assert model.coef_.loc['other', 'Credit'] <= ceiling + 2e-5
    assert model.prior_upper_bounds_.loc['asset'].isna().all()
    assert model.solver == 'CLARABEL'


@pytest.mark.parametrize('mode', [LassoModelType.LASSO, LassoModelType.GROUP_LASSO,
    LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, LassoModelType.FACTOR_CLUSTER_GROUP_LASSO])
@pytest.mark.parametrize('mapping', [None, {}, {'asset': None, 'other': np.nan}])
def test_automatic_winners_receive_the_same_bounds_as_expert_selections(mode, mapping):
    """Missing expert choices use the winning univariate slope and identical HAC bound."""
    x, y = panel()
    y = y.copy()
    y.iloc[:20] = np.nan
    y.loc[50:53, 'other'] = np.nan
    grouping = pd.Series('one', index=y.columns) if mode == LassoModelType.GROUP_LASSO else None
    model = LassoModel(model_type=mode, group_data=grouping, apply_ols_prior=True,
        factor_for_prior=mapping, expert_prior_bound_n_std=1, expert_prior_hac_lags=3,
        span=60, reg_lambda=1e-5, loss_normalization='weight_sum')
    for fitted in model.fit_reg_lambda_path(x, y, [1e-5, .0001]):
        assert fitted.ols_r2_.idxmax(axis=1).eq('Credit').all()
        explicit = model.copy({'factor_for_prior': dict.fromkeys(y.columns, 'Credit'),
                               'reg_lambda': fitted.reg_lambda}).fit(x, y)
        pd.testing.assert_frame_equal(fitted.prior_lower_bounds_, explicit.prior_lower_bounds_)
        pd.testing.assert_frame_equal(fitted.prior_upper_bounds_, explicit.prior_upper_bounds_)
        np.testing.assert_allclose(fitted.coef_, explicit.coef_, atol=2e-5, rtol=2e-4)
        assert len(fitted.prior_bound_diagnostics_) == 2
        for asset in y:
            beta, se = reference(x[['Credit']], y[asset], 60, 3)
            floor = max(0, abs(beta[0])-se[0])
            assert floor > .5
            if beta[0] > 0:
                assert fitted.prior_lower_bounds_.at[asset, 'Credit'] == pytest.approx(floor)
                assert fitted.coef_.at[asset, 'Credit'] >= floor-2e-5
            else:
                assert fitted.prior_upper_bounds_.at[asset, 'Credit'] == pytest.approx(-floor)
                assert fitted.coef_.at[asset, 'Credit'] <= -floor+2e-5


def test_automatic_winner_is_not_replaced_after_a_hard_sign_conflict():
    """A forbidden winning prior is recorded and suppressed without picking a runner-up."""
    x, y = panel()
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    signs.loc['other', 'Credit'] = 1
    model = LassoModel(apply_ols_prior=True, expert_prior_bound_n_std=1, span=60,
        factors_beta_loading_signs=signs).fit(x, y)
    row = model.prior_bound_diagnostics_.set_index('asset').loc['other']
    assert row.factor == 'Credit'
    assert row.status == 'hard_constraint_has_precedence'
    assert model.prior_lower_bounds_.loc['other'].isna().all()
    assert model.prior_upper_bounds_.loc['other'].isna().all()


def reference(x, y, span, lag):
    """Independent unscaled normal equations and full sandwich, including missing dates."""
    valid = x.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
    w = np.ones(len(x)) if span is None else (1-2/(span+1))**np.arange(len(x)-1, -1, -1)
    w *= valid
    z = np.column_stack([np.ones(len(x)), x.fillna(0).to_numpy()])
    yy = y.fillna(0).to_numpy()
    bread = np.linalg.inv(z.T @ (w[:, None] * z))
    b = bread @ z.T @ (w * yy)
    scores = w[:, None] * z * (yy-z @ b)[:, None]
    meat = scores.T @ scores
    for k in range(1, lag+1):
        cross = scores[k:].T @ scores[:-k]
        meat += (1-k/(lag+1)) * (cross+cross.T)
    covariance = bread @ meat @ bread * valid.sum() / (valid.sum()-z.shape[1])
    return b[1:], np.sqrt(np.diag(covariance)[1:])


@pytest.mark.parametrize('span', [None, 24, 60])
@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('gaps', [False, True])
def test_statistics_match_independent_sandwich(span, joint, gaps):
    """EWMA squared scores, joint conditioning and original calendar agree independently."""
    from factorlasso import compute_expert_prior_statistics
    x, y = panel()
    x = x[['Equity', 'Credit']] if joint else x[['Equity']]
    yy = y.asset.copy()
    yy.iloc[:50] = np.nan
    if gaps:
        yy.iloc[80:86] = np.nan
        x = x.copy()
        x.iloc[121, 0] = np.nan
    result = compute_expert_prior_statistics(x, yy, span=span, hac_lags=3)
    beta, se = reference(x, yy, span, 3)
    np.testing.assert_allclose(result.reference_beta, beta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.se_hac, se, rtol=1e-10, atol=1e-12)
    assert result.status.eq('ok').all()
    assert result.calendar_gaps.eq(7 if gaps else 0).all()
    # Leading pre-inception dates do not shrink the prior or its uncertainty.
    shorter = compute_expert_prior_statistics(x.iloc[50:], yy.iloc[50:], span, 3)
    pd.testing.assert_frame_equal(result, shorter, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('mode', [LassoModelType.LASSO, LassoModelType.GROUP_LASSO,
    LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, LassoModelType.FACTOR_CLUSTER_GROUP_LASSO])
def test_negative_floor_joint_prior_and_path_match_direct(mode):
    """Negative expert slopes create upper bounds; direct/path fits preserve every bound."""
    from sklearn.base import clone
    x, y = panel()
    grouping = pd.Series('one', index=y.columns) if mode == LassoModelType.GROUP_LASSO else None
    model = LassoModel(model_type=mode, group_data=grouping, apply_ols_prior=True,
        factor_for_prior={'asset': ('Equity', 'Credit'), 'other': 'Equity'},
        expert_prior_bound_n_std=1, expert_prior_hac_lags=3, span=60,
        reg_lambda=1e-5, loss_normalization='weight_sum')
    original = model.get_params()
    for fitted in model.fit_reg_lambda_path(x, y, [1e-5, .0001]):
        direct = model.copy({'reg_lambda': fitted.reg_lambda}).fit(x, y)
        pd.testing.assert_frame_equal(fitted.prior_upper_bounds_, direct.prior_upper_bounds_)
        pd.testing.assert_frame_equal(fitted.prior_lower_bounds_, direct.prior_lower_bounds_)
        pd.testing.assert_frame_equal(
            fitted.prior_bound_diagnostics_, direct.prior_bound_diagnostics_)
        np.testing.assert_allclose(fitted.coef_, direct.coef_, atol=2e-5, rtol=2e-4)
        assert fitted.prior_upper_bounds_.loc['other', 'Equity'] < -1.5
        ceiling = fitted.prior_upper_bounds_.loc['other', 'Equity']
        assert fitted.coef_.loc['other', 'Equity'] <= ceiling + 2e-5
        assert clone(fitted).prior_bound_diagnostics_ is None
    assert model.get_params()['expert_prior_bound_n_std'] == original['expert_prior_bound_n_std']


@pytest.mark.parametrize('policy', ['manual', 'hard_zero', 'opposite', 'nonneg'])
def test_manual_targets_and_hard_signs_take_precedence(policy):
    """Finite manual priors stay soft; explicit zero/opposite signs suppress floors."""
    x, y = panel()
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    numeric = signs.copy()
    kwargs = {}
    if policy == 'manual':
        numeric.loc['other', 'Equity'] = .3
        kwargs['factors_beta_prior'] = numeric
    elif policy in ('hard_zero', 'opposite'):
        signs.loc['other', 'Equity'] = 0 if policy == 'hard_zero' else 1
        kwargs['factors_beta_loading_signs'] = signs
    else:
        kwargs['nonneg'] = True
    model = LassoModel(apply_ols_prior=True, factor_for_prior={'other':'Equity'},
        expert_prior_bound_n_std=1, span=60, **kwargs).fit(x, y)
    assert model.prior_upper_bounds_.loc['other'].isna().all()
    assert model.prior_lower_bounds_.loc['other'].isna().all()
    assert not model.prior_bound_diagnostics_.set_index('asset').loc['other', 'imposed']
    assert model.prior_lower_bounds_.loc['asset', 'Credit'] > 0


def test_individual_bounds_are_independent_of_clusters_and_reset_on_refit():
    """Changing the cluster assignment cannot change the individually estimated floor."""
    x, y = panel()
    model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        apply_ols_prior=True, factor_for_prior={'asset':'Equity'}, span=60,
        expert_prior_bound_n_std=1, expert_prior_hac_lags=3)
    model.fit(x, y, external_clusters=pd.Series([0, 0], index=y.columns))
    expected = model.prior_lower_bounds_.copy()
    model.fit(x, y, external_clusters=pd.Series([0, 1], index=y.columns))
    pd.testing.assert_frame_equal(model.prior_lower_bounds_, expected)
    model.set_params(expert_prior_bound_n_std=None).fit(x, y)
    assert model.prior_lower_bounds_ is None
    assert model.prior_bound_diagnostics_ is None


@pytest.mark.parametrize('explicit', [False, True])
def test_cv_recomputes_floors_inside_each_training_fold(monkeypatch, explicit):
    """Every path reuses only its training-window OLS/HAC calculation."""
    from factorlasso import LassoModelCV
    x, y = panel()
    lengths = []
    original = LassoModel._prepare_fit
    def observe(self, *args, **kwargs):
        """Independently audit the raw window presented to each fold."""
        result = original(self, *args, **kwargs)
        factor = 'Equity' if explicit else self.ols_r2_.loc['asset'].idxmax()
        beta, se = reference(kwargs['x'][[factor]], kwargs['y'].asset, 60, 3)
        floor = max(0, beta[0]-se[0])
        assert self.prior_lower_bounds_.loc['asset', factor] == pytest.approx(floor)
        lengths.append(len(kwargs['x']))
        return result
    monkeypatch.setattr(LassoModel, '_prepare_fit', observe)
    model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        apply_ols_prior=True, factor_for_prior={'asset':'Equity'} if explicit else None, span=60,
        expert_prior_bound_n_std=1, expert_prior_hac_lags=3)
    LassoModelCV(base_model=model, lambdas=[1e-5, 1e-4], n_splits=3,
                 refit=False, use_lambda_path=True).fit(x, y)
    assert len(lengths) == 3 and max(lengths) < len(x)


@pytest.mark.parametrize('kind', ['constant', 'short', 'rank', 'span_one', 'noisy'])
def test_nonidentifiable_or_insignificant_prior_never_creates_spurious_floor(kind):
    """Insufficient information does not become an arbitrary hard exposure."""
    x, y = panel()
    factors, span, n_std = ['Equity'], 60, 1
    if kind == 'constant':
        x['Equity'] = 1
    elif kind == 'short':
        y.iloc[:-3] = np.nan
    elif kind == 'rank':
        x['Credit'] = x.Equity
        factors = ['Equity', 'Credit']
    elif kind == 'span_one':
        from factorlasso import compute_expert_prior_statistics
        stats = compute_expert_prior_statistics(x, y.asset, span=1)
        assert stats.status.eq('insufficient_observations').all()
        return
    else:
        n_std = 1e6
    model = LassoModel(apply_ols_prior=True, factor_for_prior={'asset':factors},
        expert_prior_bound_n_std=n_std, span=span).fit(x, y)
    assert model.prior_lower_bounds_.loc['asset'].isna().all()
    assert model.prior_upper_bounds_.loc['asset'].isna().all()


@pytest.mark.parametrize('kwargs', [
    {'expert_prior_bound_n_std': -1}, {'expert_prior_bound_n_std': True},
    {'expert_prior_bound_n_std': np.nan}, {'expert_prior_hac_lags': 1.5},
    {'expert_prior_hac_lags': -1}, {'expert_prior_hac_lags_freq_dict': {}},
    {'expert_prior_hac_lags_freq_dict': {'ME': -1}},
])
def test_invalid_bound_configuration_fails_closed(kwargs):
    """Do not silently ignore malformed confidence sizes or frequency maps."""
    with pytest.raises(ValueError):
        LassoModel(apply_ols_prior=True, **kwargs)
