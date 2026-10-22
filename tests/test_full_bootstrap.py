"""Joint, full-estimator bootstrap contracts and independent numerical checks."""
import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType, bootstrap_factor_betas
from factorlasso.linear_model._bootstrap import _changed


def test_diagnostic_allows_a_different_number_of_clusters():
    assert _changed(np.ones((3, 13)), np.ones((4, 13))) == 52


def panel():
    rng = np.random.default_rng(819)
    x = pd.DataFrame(rng.normal(0, .04, (60, 3)), columns=['a', 'b', 'c'])
    y = pd.DataFrame(x.to_numpy() @ np.array([[.6, .2], [.0, .4], [.2, .0]])
                     + rng.normal(0, .02, (60, 2)), columns=['one', 'two'])
    y.loc[:12, 'two'] = np.nan
    return x, y


def config():
    return LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        span=36, reg_lambda=1e-5, loss_normalization='weight_sum', solver='CLARABEL',
        apply_ols_prior=True, auto_sign_constraints=True, auto_sign_adaptive_weights=True)


def test_explicit_draw_matches_independent_fresh_fit_and_keeps_masks():
    x, y = panel()
    model = config()
    original = model.copy().fit(x, y)
    z = np.random.default_rng(9).normal(size=(20, len(y)))
    result = bootstrap_factor_betas(model, x, y, draws=20,
        multipliers=pd.DataFrame(z, columns=x.index))
    # Independent construction: chronological squared EWMA weights, not the
    # implementation helper. The same time multiplier hits both residuals.
    w = (1-2/37)**np.arange(len(y)-1, -1, -1)
    residual = y-original.predict(x)
    mean = residual.mul(w, axis=0).sum() / y.notna().mul(w, axis=0).sum()
    response = original.predict(x) + (residual-mean).mul(z[0], axis=0)
    expected = model.copy().fit(x, response)
    pd.testing.assert_frame_equal(result.draws.loc[0].rename_axis(None), expected.coef_)
    pd.testing.assert_frame_equal(result.estimate, original.coef_)
    assert model.coef_ is None
    assert result.response_support.observations.to_list() == [60, 47]
    assert result.fit_diagnostics.prior_cells_changed.gt(0).all()
    assert result.fit_diagnostics.block_weights_changed.gt(0).any()
    assert result.fit_diagnostics.sign_cells_changed.gt(0).any()
    assert result.metadata['coverage'] == 'not_validated'


def test_joint_portfolio_bounds_are_quantiles_of_weighted_draws():
    x, y = panel()
    result = bootstrap_factor_betas(config(), x, y, draws=20, seed=82)
    weights = pd.Series({'two': -.4, 'one': 1.5})
    summary = result.portfolio_intervals(weights)
    manual = np.array([weights.reindex(y.columns).to_numpy() @ result.draws.loc[b].to_numpy()
                       for b in range(20)])
    np.testing.assert_allclose(summary[['lower', 'upper']].T,
                               np.quantile(manual, [.025, .975], axis=0))
    basic = result.portfolio_intervals(weights, method='basic')
    np.testing.assert_allclose(basic.lower, 2*summary.estimate-summary.upper)
    with pytest.raises(ValueError, match='every fitted asset'):
        result.portfolio_intervals(weights.drop('two'))


def test_exact_shared_response_relation_and_reproducibility():
    x, y = panel()
    y = pd.DataFrame({'one': y.one, 'two': 2*y.one})
    model = LassoModel(solver='CLARABEL', reg_lambda=0., span=36)
    result = bootstrap_factor_betas(model, x, y, draws=20, seed=82)
    again = bootstrap_factor_betas(model, x, y, draws=20, seed=82)
    pd.testing.assert_frame_equal(result.draws, again.draws)
    np.testing.assert_allclose(result.draws.xs('two', level='asset'),
                               2*result.draws.xs('one', level='asset'), atol=2e-6)


@pytest.mark.parametrize('defect', ['short', 'partial_x', 'dates', 'inf', 'calendar'])
def test_invalid_panels_fail(defect):
    x, y = panel()
    kwargs = {}
    if defect == 'short':
        y.loc[:58, 'two'] = np.nan
    elif defect == 'partial_x':
        x.iloc[5, 0] = np.nan
    elif defect == 'dates':
        y.index = y.index+1
    elif defect == 'inf':
        y.iloc[0, 0] = np.inf
    else:
        kwargs['calendar'] = np.zeros(len(x))
    with pytest.raises(ValueError):
        bootstrap_factor_betas(config(), x, y, draws=20, **kwargs)


def test_solver_failure_is_not_dropped(monkeypatch):
    x, y = panel()
    original = LassoModel.fit
    calls = []

    def fail(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise ValueError('intentional failed draw')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LassoModel, 'fit', fail)
    with pytest.raises(RuntimeError, match='refit 1 failed; no draws were discarded'):
        bootstrap_factor_betas(config(), x, y, draws=20)


def test_linear_limit_matches_dense_joint_sandwich_without_monte_carlo_error():
    # A balanced linear special case has an independent matrix answer. Balanced
    # +/- kernel-root innovations have exactly the desired covariance, avoiding
    # a statistical tolerance or an implementation-to-itself comparison.
    x, y = panel()
    # Rescale both sides to avoid a tiny objective relative to solver tolerances;
    # an unpenalised coefficient and its covariance are invariant to this scaling.
    x, y = 25*x, 25*y[['one']]
    n = len(x)
    times = np.arange(n)
    kernel = np.maximum(0., 1-np.abs(times[:, None]-times[None, :])/3.)
    eigenvalues, vectors = np.linalg.eigh(kernel)
    root = vectors*np.sqrt(np.maximum(eigenvalues, 0.))
    z = np.vstack([root.T, -root.T])*np.sqrt(n)
    model = LassoModel(solver='CLARABEL', reg_lambda=0., span=36, demean=False)
    result = bootstrap_factor_betas(model, x, y, draws=2*n,
        multipliers=pd.DataFrame(z, columns=x.index))
    design, response = x.to_numpy(), y.one.to_numpy()
    weights = (1-2/37)**np.arange(n-1, -1, -1)
    gram = design.T @ (weights[:, None]*design)
    beta = np.linalg.solve(gram, design.T @ (weights*response))
    residual = response-design @ beta
    residual -= weights @ residual/weights.sum()
    influence = np.linalg.solve(gram, design.T*(weights*residual))
    expected = influence @ kernel @ influence.T
    actual = np.cov(result.draws.to_numpy(), rowvar=False, ddof=0)
    np.testing.assert_allclose(result.estimate.iloc[0], beta, atol=2e-6)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-8)
