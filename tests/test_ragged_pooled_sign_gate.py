"""Observed-cell joint slopes and date-clustered HC1 verified independently."""
import numpy as np
import pandas as pd
import pytest

import factorlasso as fl


def ragged_panel():
    """Provide alternating histories with no complete shared response date."""
    rng = np.random.default_rng(2026110301)
    dates = pd.date_range('2000-01-31', periods=61, freq='ME')
    x = pd.DataFrame(rng.normal(size=(61, 3)), index=dates, columns=['market','rates','credit'])
    shocks = rng.normal(size=(61, 1)) + .3 * rng.normal(size=(61, 3))
    y = pd.DataFrame(x.to_numpy() @ np.array([[.6,-.2,.3],[.5,-.3,.2],[.7,-.1,.4]]).T
                     + shocks, index=dates, columns=['new','old','other'])
    for col in range(3):
        y.iloc[np.arange(len(y)) % 3 == col, col] = np.nan
    groups = pd.Series(['g'] * 3, index=y.columns)
    weights = pd.Series(.94 ** np.arange(len(x)-1,-1,-1), index=dates)
    return x, y, groups, weights


def stacked_reference(x, y, weights):
    """Fit explicitly repeated observed cells, then aggregate scores by date."""
    rows, response, loss_weights, times = [], [], [], []
    for t in range(len(x)):
        if not np.isfinite(x.iloc[t]).all():
            continue
        for value in y.iloc[t]:
            if np.isfinite(value):
                rows.append(np.r_[1., x.iloc[t].to_numpy()])
                response.append(value)
                loss_weights.append(weights.iloc[t])
                times.append(t)
    design, response, w = np.asarray(rows), np.asarray(response), np.asarray(loss_weights)
    beta = np.linalg.lstsq(design * np.sqrt(w[:,None]), response * np.sqrt(w), rcond=None)[0]
    bread = np.linalg.inv(design.T @ (w[:,None] * design))
    cell_scores = w[:,None] * design * (response-design@beta)[:,None]
    date_scores = np.zeros((len(x), design.shape[1]))
    np.add.at(date_scores, times, cell_scores)
    n = len(np.unique(times))
    covariance = bread @ (date_scores.T @ date_scores) @ bread * n/(n-design.shape[1])
    return beta[1:], np.sqrt(np.diag(covariance)[1:]), n


def test_observed_cells_without_any_common_complete_date():
    """The original MAC blocker must not require complete shared histories."""
    x, y, groups, weights = ragged_panel()
    assert y.dropna().empty
    result = fl.compute_ragged_pooled_sign_gate(x, y, groups, weights=weights, threshold=1.)
    beta, se, n = stacked_reference(x, y, weights)
    np.testing.assert_allclose(result.slopes, np.tile(beta, (3,1)), atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(result.standard_errors, np.tile(se, (3,1)), atol=1e-12, rtol=1e-10)
    assert result.diagnostics.loc['g','observations'] == n == len(x)
    assert result.diagnostics.loc['g','response_observations'] == int(y.notna().to_numpy().sum())
    assert result.diagnostics.loc['g','status'] == 'ok'
    np.testing.assert_array_equal(result.observation_mask, y.notna())


@pytest.mark.parametrize('action', ['exclude','release','zero_only'])
def test_balanced_result_matches_existing_complete_gate(action):
    """The additive ragged API agrees with the existing complete-panel contract."""
    x, y, groups, weights = ragged_panel()
    y = y.fillna(.01)
    complete = fl.compute_pooled_sign_gate(x, y, groups, weights=weights, action=action)
    ragged = fl.compute_ragged_pooled_sign_gate(x, y, groups, weights=weights, action=action)
    for attr in ('slopes','standard_errors','t_stats','signs','restrictions'):
        np.testing.assert_allclose(getattr(ragged, attr), getattr(complete, attr),
                                   atol=1e-12, rtol=1e-10, equal_nan=True)


def test_gaps_keep_original_weights_and_missing_factors_remove_whole_date():
    """No compressed EWMA calendar, factor-wise fit, return zero-fill or input mutation."""
    x, y, groups, weights = ragged_panel()
    y.iloc[8:13] = np.nan
    x.iloc[18:21,1] = np.nan
    originals = [v.copy(deep=True) for v in (x,y,groups,weights)]
    result = fl.compute_ragged_pooled_sign_gate(x, y, groups, weights=weights)
    beta, se, n = stacked_reference(x, y, weights)
    np.testing.assert_allclose(result.slopes.iloc[0], beta, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(result.standard_errors.iloc[0], se, atol=1e-12, rtol=1e-10)
    assert result.diagnostics.loc['g','observations'] == n
    assert result.diagnostics.loc['g','hc1_correction'] == pytest.approx(n/(n-4))
    assert not result.observation_mask.iloc[18:21].to_numpy().any()
    for original, saved in zip((x,y,groups,weights), originals):
        assert original.equals(saved)
    np.testing.assert_allclose(result.date_weights, weights/weights.mean())


@pytest.mark.parametrize('kind', ['empty','too_short','rank'])
def test_unidentified_defaults_to_error_and_explicit_release_warns(kind):
    """Unavailable evidence is not a weak-zero gate or a marginal fallback."""
    x, y, groups, weights = ragged_panel()
    if kind == 'empty':
        y[:] = np.nan
    elif kind == 'too_short':
        y.iloc[4:] = np.nan
    else:
        x.iloc[:,2] = x.iloc[:,0]
    with pytest.raises(ValueError, match='unidentified'):
        fl.compute_ragged_pooled_sign_gate(x, y, groups, weights=weights)
    with pytest.warns(UserWarning, match='unidentified.*released'):
        result = fl.compute_ragged_pooled_sign_gate(
            x, y, groups, weights=weights, on_unidentified='release')
    assert result.restrictions.isna().all().all()
    assert result.signs.isna().all().all()
    assert result.slopes.isna().all().all()
    assert result.diagnostics.loc['g','status'] in ('insufficient_observations','rank_deficient')


@pytest.mark.parametrize('column', ['x','y','weights'])
def test_infinite_values_are_not_treated_as_missing(column):
    """NaNs are permitted data gaps, but infinity remains an invalid input."""
    x, y, groups, weights = ragged_panel()
    target = {'x':x,'y':y,'weights':weights}[column]
    if isinstance(target,pd.DataFrame):
        target.iloc[2,0] = np.inf
    else:
        target.iloc[2] = np.inf
    with pytest.raises(ValueError):
        fl.compute_ragged_pooled_sign_gate(x, y, groups, weights=weights)


def test_original_complete_api_still_rejects_missing_cells():
    """Extending validation internally must not weaken old public defaults."""
    x, y, groups, weights = ragged_panel()
    with pytest.raises(ValueError, match='complete finite'):
        fl.compute_pooled_sign_gate(x, y, groups, weights=weights)


def test_release_is_local_to_the_unidentified_group_and_preserves_group_identity():
    """A missing group does not erase supported evidence or merge mixed ID types."""
    x, y, _, weights = ragged_panel()
    y['missing'] = np.nan
    groups = pd.Series([7,'7',7,'absent'], index=y.columns)
    with pytest.warns(UserWarning, match='unidentified.*released'):
        result = fl.compute_ragged_pooled_sign_gate(
            x,y,groups,weights=weights,on_unidentified='release')
    assert result.diagnostics.index.tolist() == [7,'7','absent']
    assert result.diagnostics['status'].tolist() == ['ok','ok','insufficient_observations']
    assert result.slopes.loc[['new','old','other']].notna().all().all()
    assert result.slopes.loc['missing'].isna().all()
    assert not result.observation_mask['missing'].any()


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_weight_rescaling_and_result_ownership(scale):
    """Loss rescaling cancels and returned mutable objects do not alias inputs."""
    x,y,groups,weights = ragged_panel()
    base = fl.compute_ragged_pooled_sign_gate(x,y,groups,weights=weights)
    scaled = fl.compute_ragged_pooled_sign_gate(x,y,groups,weights=weights*scale)
    np.testing.assert_allclose(base.slopes,scaled.slopes,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(base.standard_errors,scaled.standard_errors,rtol=1e-10,atol=1e-12)
    mask = base.observation_mask.copy()
    base.slopes.iloc[0,0] = 999.
    base.observation_mask.iloc[:,:] = False
    base.date_weights.iloc[0] = 0.
    base.groups.iloc[0] = 'edited'
    assert not (y == 999.).any().any()
    assert mask.equals(y.notna())
    assert (weights > 0).all() and groups.iloc[0] == 'g'
    assert not (base.signs == 999.).any().any()


@pytest.mark.parametrize('argument,value', [
    ('threshold', True), ('threshold', np.inf), ('threshold', -1),
    ('action', 'ignore'), ('on_unidentified', 'marginal'), ('on_unidentified', True),
])
def test_explicit_policies_reject_invalid_values(argument,value):
    """Invalid choices cannot silently select a numerical failure policy."""
    x,y,groups,weights = ragged_panel()
    with pytest.raises(ValueError):
        fl.compute_ragged_pooled_sign_gate(x,y,groups,weights=weights,**{argument:value})


def test_varying_member_counts_require_observed_cell_loss_weights():
    """Equal date weighting is wrong when the observed cluster mix varies."""
    x,y,groups,weights = ragged_panel()
    y.iloc[:17,0] = np.nan
    y.iloc[28:44,1] = np.nan
    assert y.notna().sum(axis=1).nunique() > 1
    result = fl.compute_ragged_pooled_sign_gate(x,y,groups,weights=weights)
    beta,se,_ = stacked_reference(x,y,weights)
    np.testing.assert_allclose(result.slopes.iloc[0],beta,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(result.standard_errors.iloc[0],se,rtol=1e-10,atol=1e-12)
    mean_response = y.mean(axis=1).to_frame('mean')
    wrong = fl.compute_pooled_sign_gate(x,mean_response,
                pd.Series(['g'],index=mean_response.columns),weights=weights)
    assert np.max(np.abs(wrong.slopes.iloc[0]-result.slopes.iloc[0])) > 1e-3
    imputed = fl.compute_pooled_sign_gate(x,y.fillna(0),groups,weights=weights)
    assert np.max(np.abs(imputed.slopes.iloc[0]-result.slopes.iloc[0])) > 1e-2
