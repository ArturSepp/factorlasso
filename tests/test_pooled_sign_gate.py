"""Independent conditional pooled HC1, strict inputs and gate composition checks."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import factorlasso.inference._pooled_signs as pooled_module
from factorlasso.inference._pooled_signs import compute_pooled_sign_gate
from factorlasso.utils._pooled_inputs import _validate_pooled_inputs, _weighted_center


def panel_fixture():
    """Make heteroskedastic dependent members with mixed-type unsorted IDs."""
    rng = np.random.default_rng(2026110105)
    index = pd.date_range('2001-01-01', periods=71, freq='D')
    x = pd.DataFrame(rng.normal(size=(71, 3)) + [1.3, -.7, 2.],
                     index=index, columns=['value', 'market', 'carry'])
    beta = np.array([[.8, -.4, .1], [.1, .9, -.5], [.4, -.3, .3],
                     [-.4, -.7, .2], [.2, .3, .9]])
    errors = rng.normal(size=(71, 5)) + rng.normal(size=(71, 1))
    errors *= .2 + .1 * np.abs(x.iloc[:, [0]].to_numpy())
    y = pd.DataFrame(x.to_numpy() @ beta.T + errors + np.arange(5) * .7,
                     index=index, columns=['b', 'a', 'e', 'c', 'd'])
    groups = pd.Series([7, '7', 7, 'solo', 'solo'], index=y.columns, name='group')
    weights = pd.Series(np.exp(np.linspace(-2., 0., len(x))), index=index, name='date_weight')
    return x, y, groups, weights


def direct_reference(x, y, groups, weights):
    """Use an explicit intercept normal solve and date-score outer products."""
    design = np.column_stack([np.ones(len(x)), np.asarray(x)])
    w = np.ones(len(x)) if weights is None else np.asarray(weights)
    bread = np.linalg.inv(design.T @ (w[:, None] * design))
    slopes = np.zeros((y.shape[1], x.shape[1]))
    errors = np.zeros_like(slopes)
    for group in dict.fromkeys(groups):
        members = np.flatnonzero(np.asarray([value == group for value in groups]))
        response = np.asarray(y)[:, members].mean(axis=1)
        beta = np.linalg.solve(design.T @ (w[:, None] * design), design.T @ (w * response))
        score = w[:, None] * design * (response - design @ beta)[:, None]
        meat = sum(np.outer(row, row) for row in score)
        covariance = bread @ meat @ bread * len(x) / (len(x) - design.shape[1])
        slopes[members] = beta[1:]
        errors[members] = np.sqrt(np.diag(covariance)[1:])
    return slopes, errors


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('action', ['exclude', 'release', 'zero_only'])
def test_gate_against_intercepted_normal_sandwich(weighted, action):
    """Independent date-score covariance checks the wrapper and reused engine."""
    x, y, groups, weights = panel_fixture()
    weights = weights if weighted else None
    result = compute_pooled_sign_gate(x, y, groups, weights=weights, action=action)
    beta, se = direct_reference(x, y, groups, weights)
    np.testing.assert_allclose(result.slopes, beta, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(result.standard_errors, se, atol=1e-12, rtol=1e-10)
    statistic = beta / se
    expected_sign = np.where((beta != 0) & (np.abs(statistic) >= 2), np.sign(beta), 0)
    np.testing.assert_array_equal(result.signs, expected_sign)
    expected = expected_sign.astype(float)
    if action == 'release':
        expected[expected_sign == 0] = np.nan
    elif action == 'zero_only':
        expected[expected_sign != 0] = np.nan
    np.testing.assert_array_equal(result.restrictions, expected)
    assert result.slopes.index.equals(y.columns)
    assert result.slopes.columns.equals(x.columns)
    assert result.groups.equals(groups)
    assert result.date_weights.index.equals(x.index)
    assert result.date_weights.mean() == pytest.approx(1)
    assert result.observations == 71 and result.residual_df == 67
    assert result.hc1_correction == pytest.approx(71 / 67)
    assert result.evidence == 'conditional' and result.variance_estimator == 'HC1_date'
    assert result.hac_lags == 0 and result.fit_intercept is True


def test_member_score_pooling_not_average_variance_or_kish():
    """Dependent members require cross-score terms and ordinary HC1 df."""
    x, y, groups, weights = panel_fixture()
    result = compute_pooled_sign_gate(x, y, groups, weights=weights)
    singleton = pd.Series(np.arange(y.shape[1]), index=y.columns)
    _, member_se = direct_reference(x, y, singleton, weights)
    incorrect = np.sqrt((member_se[[0, 2]]**2).sum(axis=0)) / 2
    assert np.max(np.abs(result.standard_errors.iloc[0] - incorrect)) > .01
    neff = weights.sum()**2 / (weights @ weights)
    assert result.effective_n == pytest.approx(neff)
    assert abs(result.hc1_correction - neff / (neff-1)) > .01
    np.testing.assert_array_equal(result.slopes.iloc[0], result.slopes.iloc[2])
    assert not np.allclose(result.slopes.iloc[0], result.slopes.iloc[1])


def test_original_date_weights_scaling_and_coordinate_conversion():
    """Irregular original-date subsets retain age, translations and factor units."""
    x, y, groups, weights = panel_fixture()
    selected = np.array([i for i in range(len(x)) if i % 5 in (0, 1, 3)])
    x, y, weights = x.iloc[selected], y.iloc[selected], weights.iloc[selected]
    original = compute_pooled_sign_gate(x, y, groups, weights=weights)
    tiny = compute_pooled_sign_gate(x, y, groups, weights=weights * 1e-200)
    huge = compute_pooled_sign_gate(x, y, groups, weights=weights * 1e200)
    for scaled in (tiny, huge):
        np.testing.assert_allclose(scaled.slopes, original.slopes, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(scaled.standard_errors, original.standard_errors,
                                   atol=1e-12, rtol=1e-10)
    converted = compute_pooled_sign_gate(x * [2., .3, 4.] + 1000., y * 3. - 400., groups,
                                        weights=weights)
    np.testing.assert_allclose(converted.slopes, original.slopes * 3 / [2., .3, 4.],
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(converted.t_stats, original.t_stats, rtol=1e-10, atol=1e-10)
    restarted = pd.Series(np.exp(np.arange(len(x)) * (2 / 70)), index=x.index)
    wrong = compute_pooled_sign_gate(x, y, groups, weights=restarted)
    assert np.max(np.abs(wrong.slopes.to_numpy() - original.slopes.to_numpy())) > 1e-4


@pytest.mark.parametrize('action', ['exclude', 'release', 'zero_only'])
def test_exact_threshold_and_zero_standard_error(monkeypatch, action):
    """Exercise exact boundary arithmetic independently of SVD roundoff."""
    x, y, groups, _ = panel_fixture()
    real_engine = pooled_module.compute_wls_hac_statistics

    def controlled_engine(*args, **kwargs):
        result = real_engine(*args, **kwargs)
        return replace(result, coefficients=np.array([0., 2., -3., 0.]),
                       standard_error=np.array([0., 1., 0., 0.]))

    monkeypatch.setattr(pooled_module, 'compute_wls_hac_statistics', controlled_engine)
    gate = compute_pooled_sign_gate(x, y, groups, threshold=2., action=action)
    np.testing.assert_array_equal(gate.t_stats.iloc[0], [2., -np.inf, 0.])
    np.testing.assert_array_equal(gate.signs.iloc[0], [1., -1., 0.])
    expected = {'exclude': [1., -1., 0.], 'release': [1., -1., np.nan],
                'zero_only': [np.nan, np.nan, 0.]}[action]
    np.testing.assert_array_equal(gate.restrictions.iloc[0], expected)
    zero_threshold = compute_pooled_sign_gate(x, y, groups, threshold=0.)
    assert zero_threshold.signs.iloc[0, 2] == 0


def test_exact_zero_response_and_conditional_marginal_reversal():
    """Zero response stays weak; correlated factors reverse a marginal sign."""
    x, y, groups, _ = panel_fixture()
    zero = compute_pooled_sign_gate(x, y * 0, groups, threshold=0)
    assert (zero.signs.to_numpy() == 0).all()
    assert (zero.t_stats.to_numpy() == 0).all()
    u = np.tile([-1., -1., 1., 1.], 12)
    v = np.tile([-1., 1., -1., 1.], 12)
    factors = pd.DataFrame({'x1': u, 'x2': .8 * u + .6 * v})
    response = pd.DataFrame({'asset': factors.x1 - .5 * factors.x2})
    gate = compute_pooled_sign_gate(factors, response, pd.Series(['all'], index=response.columns))
    marginal = factors.x2 @ response.asset / (factors.x2 @ factors.x2)
    assert marginal > 0 and gate.slopes.loc['asset', 'x2'] < 0
    np.testing.assert_allclose(gate.slopes, [[1., -.5]], atol=1e-12)


def test_copies_and_groups_do_not_alias_callers_or_other_fields():
    """Returned mutable frames and axes are owned despite a frozen container."""
    x, y, groups, weights = panel_fixture()
    snapshots = [obj.copy(deep=True) for obj in (x, y, groups, weights)]
    result = compute_pooled_sign_gate(x, y, groups, weights=weights)
    for original, snapshot in zip((x, y, groups, weights), snapshots):
        assert original.equals(snapshot)
    result.slopes.iloc[0, 0] = 999.
    result.groups.iloc[0] = 'edited'
    result.date_weights.iloc[0] = 0
    result.slopes.index.values[0] = 'edited-axis'
    result.slopes.columns.values[0] = 'edited-factor'
    for original, snapshot in zip((x, y, groups, weights), snapshots):
        assert original.equals(snapshot)
    assert result.signs.index[0] == y.columns[0]
    assert result.signs.columns[0] == x.columns[0]


@pytest.mark.parametrize('bad', [True, np.bool_(False), -1., np.nan, np.inf, '2', 1+0j, 10**400])
def test_invalid_threshold(bad):
    """The scalar threshold is a finite nonnegative nonboolean real."""
    x, y, groups, _ = panel_fixture()
    with pytest.raises(ValueError, match='threshold'):
        compute_pooled_sign_gate(x, y, groups, threshold=bad)


@pytest.mark.parametrize('bad', ['signs', None, 1, ['exclude']])
def test_invalid_action(bad):
    """Only the three declared action names are accepted."""
    x, y, groups, _ = panel_fixture()
    with pytest.raises(ValueError, match='action'):
        compute_pooled_sign_gate(x, y, groups, action=bad)


@pytest.mark.parametrize('which', ['x', 'y', 'groups', 'weights'])
def test_container_contract(which):
    """The new opt-in wrapper does not silently accept unlabeled arrays."""
    x, y, groups, weights = panel_fixture()
    args = {'x': x, 'y': y, 'groups': groups, 'weights': weights}
    args[which] = args[which].to_numpy()
    with pytest.raises(TypeError):
        compute_pooled_sign_gate(**args)


@pytest.mark.parametrize('case', [
    'reorder_y', 'reorder_groups', 'reorder_weights', 'duplicate_rows', 'missing_rows',
    'duplicate_x_columns', 'duplicate_y_columns', 'missing_columns', 'empty_x', 'empty_y',
    'nan_x', 'inf_y', 'complex_x', 'bool_y', 'string_x', 'zero_weight', 'negative_weight',
    'nan_weight', 'complex_weight', 'bool_weight', 'underflow_weight', 'missing_group',
    'infinite_group', 'boolean_group', 'tuple_group',
])
def test_strict_input_rejection(case):
    """Reject invalid complete-panel inputs before calling the permissive engine."""
    x, y, groups, weights = panel_fixture()
    if case == 'reorder_y':
        y = y.iloc[::-1]
    elif case == 'reorder_groups':
        groups = groups.iloc[::-1]
    elif case == 'reorder_weights':
        weights = weights.iloc[::-1]
    elif case in ('duplicate_rows', 'missing_rows'):
        labels = x.index.to_numpy(copy=True)
        labels[0] = labels[1] if case == 'duplicate_rows' else np.datetime64('NaT', 'ns')
        x.index = y.index = labels
    elif case == 'duplicate_x_columns':
        x.columns = ['a', 'a', 'b']
    elif case == 'duplicate_y_columns':
        y.columns = ['a'] * 5
        groups.index = y.columns
    elif case == 'missing_columns':
        x.columns = ['a', None, 'b']
    elif case == 'empty_x':
        x = x.iloc[:, :0]
    elif case == 'empty_y':
        y = y.iloc[:0]
    elif case == 'nan_x':
        x.iloc[2, 1] = np.nan
    elif case == 'inf_y':
        y.iloc[1, 2] = np.inf
    elif case == 'complex_x':
        x = x.astype(complex)
    elif case == 'bool_y':
        y = y > 0
    elif case == 'string_x':
        x = x.astype(str)
    elif case in ('zero_weight', 'negative_weight', 'nan_weight'):
        weights.iloc[0] = {'zero_weight': 0, 'negative_weight': -1, 'nan_weight': np.nan}[case]
    elif case == 'complex_weight':
        weights = weights.astype(complex)
    elif case == 'bool_weight':
        weights = weights > 0
    elif case == 'underflow_weight':
        weights.iloc[0], weights.iloc[-1] = 1e-300, 1e300
    elif case in ('missing_group', 'infinite_group', 'boolean_group', 'tuple_group'):
        groups.iloc[0] = {'missing_group': None, 'infinite_group': np.inf,
                          'boolean_group': True, 'tuple_group': ('a', 'b')}[case]
    with pytest.raises(ValueError):
        compute_pooled_sign_gate(x, y, groups, weights=weights)


@pytest.mark.parametrize('case', ['df', 'constant', 'collinear'])
def test_rank_and_residual_degrees_of_freedom(case):
    """Joint rank failure and no residual df never become a neutral gate."""
    x, y, groups, _ = panel_fixture()
    if case == 'df':
        x, y = x.iloc[:4], y.iloc[:4]
    elif case == 'constant':
        x.iloc[:, 0] = 1.
    else:
        x.iloc[:, 1] = 2 * x.iloc[:, 0]
    with pytest.raises(ValueError, match='residual|rank'):
        compute_pooled_sign_gate(x, y, groups)


def test_invalid_engine_output_and_overflow_are_failures(monkeypatch):
    """A finite input panel does not authorize a nonfinite numerical result."""
    x, y, groups, _ = panel_fixture()
    real_engine = pooled_module.compute_wls_hac_statistics

    def bad_engine(*args, **kwargs):
        result = real_engine(*args, **kwargs)
        return replace(result, standard_error=np.full(4, np.nan))

    monkeypatch.setattr(pooled_module, 'compute_wls_hac_statistics', bad_engine)
    with pytest.raises(RuntimeError, match='invalid slopes or standard errors'):
        compute_pooled_sign_gate(x, y, groups)
    monkeypatch.setattr(pooled_module, 'compute_wls_hac_statistics', real_engine)
    with pytest.raises(RuntimeError, match='arithmetic'):
        compute_pooled_sign_gate(x, y * 1e300, groups)


def test_shared_center_and_numeric_equal_group_ids():
    """Stable centering and integer/float equality are explicit shared contracts."""
    x, y, groups, weights = panel_fixture()
    groups.iloc[:3] = [1, 1., '1']
    panel = _validate_pooled_inputs(x, y, groups, weights)
    np.testing.assert_array_equal(panel.members[0], [0, 1])
    np.testing.assert_array_equal(panel.members[1], [2])
    values = x.to_numpy() + 1e10
    centered, means = _weighted_center(values, panel.date_weights.to_numpy())
    np.testing.assert_allclose(np.average(centered, axis=0, weights=weights), 0, atol=1e-14)
    np.testing.assert_allclose(centered + means, values, atol=2e-6, rtol=0)
    with pytest.raises(RuntimeError, match='centering'):
        _weighted_center(np.array([[1e308], [-1e308]]), np.ones(2))


def test_multiindex_names_and_large_integer_ids():
    """Valid labels do not become invalid through duplicate level names or casts."""
    x, y, groups, weights = panel_fixture()
    index = pd.MultiIndex.from_arrays([np.arange(len(x)), np.ones(len(x))], names=['t', 't'])
    x.index = y.index = weights.index = index
    groups.iloc[0] = 10**100
    result = compute_pooled_sign_gate(x, y, groups, weights=weights)
    assert result.date_weights.index.equals(index)
    assert result.groups.iloc[0] == 10**100
    assert isinstance(result.groups.iloc[0], int)


def test_finite_positive_se_statistic_overflow_is_rejected(monkeypatch):
    """Infinity is the exact-zero-SE convention, not a silent overflow fallback."""
    x, y, groups, _ = panel_fixture()
    real_engine = pooled_module.compute_wls_hac_statistics

    def overflowing_statistic(*args, **kwargs):
        result = real_engine(*args, **kwargs)
        return replace(result, coefficients=np.array([0., 1e300, 0., 0.]),
                       standard_error=np.array([1., 1e-300, 1., 1.]))

    monkeypatch.setattr(pooled_module, 'compute_wls_hac_statistics', overflowing_statistic)
    with pytest.raises(RuntimeError, match='t-statistic'):
        compute_pooled_sign_gate(x, y, groups)


def test_independent_covariance_assertion_catches_injected_wrong_hc1(monkeypatch):
    """Prove the independent numerical assertion fails under a plausible defect."""
    x, y, groups, weights = panel_fixture()
    real_engine = pooled_module.compute_wls_hac_statistics
    _, expected_se = direct_reference(x, y, groups, weights)

    def wrong_correction(*args, **kwargs):
        result = real_engine(*args, **kwargs)
        n = result.observations
        wrong = np.sqrt((result.effective_n / (result.effective_n-1)) / (n/(n-4)))
        return replace(result, standard_error=result.standard_error * wrong)

    with monkeypatch.context() as patched:
        patched.setattr(pooled_module, 'compute_wls_hac_statistics', wrong_correction)
        faulty = compute_pooled_sign_gate(x, y, groups, weights=weights)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(faulty.standard_errors, expected_se, rtol=1e-10, atol=1e-12)
    restored = compute_pooled_sign_gate(x, y, groups, weights=weights)
    np.testing.assert_allclose(restored.standard_errors, expected_se, rtol=1e-10, atol=1e-12)
