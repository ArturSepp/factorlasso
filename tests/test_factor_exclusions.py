"""Per-response hard exclusions across priors, solver modes and reusable fits."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from factorlasso import LassoModel, LassoModelType


def panel():
    """A strong forbidden predictor and two admissible predictors."""
    rng = np.random.default_rng(73)
    x = pd.DataFrame(rng.normal(size=(160, 3)), columns=['Rates', 'HY', 'EM'])
    x -= x.mean()
    y = pd.DataFrame({'Govt': x.Rates + 3 * x.EM + .1 * rng.normal(size=160),
                      'Credit': .3 * x.Rates + 2 * x.HY + x.EM})
    y -= y.mean()
    return x, y


@pytest.mark.parametrize('kind', [LassoModelType.LASSO,
    LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, LassoModelType.FACTOR_CLUSTER_GROUP_LASSO])
def test_exclusion_matches_independent_reduced_design(kind):
    """A hard-zero fit equals least squares on each response's allowed columns."""
    x, y = panel()
    model = LassoModel(model_type=kind, reg_lambda=0, span=None, demean=False,
                       solver='CLARABEL', warmup_period=3,
                       excluded_factors={'Govt': ('HY', 'EM')}).fit(x, y)
    assert model.coef_.loc['Govt', ['HY', 'EM']].eq(0).all()
    expected = np.linalg.lstsq(x[['Rates']], y.Govt, rcond=None)[0]
    np.testing.assert_allclose(model.coef_.loc['Govt', ['Rates']], expected, atol=1e-6)
    np.testing.assert_allclose(model.coef_.loc['Credit'],
                               np.linalg.lstsq(x, y.Credit, rcond=None)[0], atol=1e-6)


def test_automatic_prior_and_bounds_use_only_admissible_factors():
    """An excluded highest-R2 winner cannot receive a prior or exposure floor."""
    x, y = panel()
    model = LassoModel(apply_ols_prior=True, expert_prior_bound_n_std=1,
                       excluded_factors={'Govt': ['HY', 'EM'], 'Credit': ['Rates', 'HY', 'EM']},
                       span=None, solver='CLARABEL').fit(x, y)
    assert model.ols_r2_.loc['Govt'].idxmax() == 'EM'
    assert model.ols_beta_prior_.loc['Govt', 'Rates'] != 0
    assert model.ols_beta_prior_.loc['Govt', ['HY', 'EM']].eq(0).all()
    assert model.coef_.loc['Credit'].eq(0).all()
    assert model.effective_beta_prior_.loc['Credit'].eq(0).all()


@pytest.mark.parametrize('excluded,prior,match', [
    ({'Govt': ['missing']}, None, 'unknown factors'),
    ({'Govt': ['EM', 'EM']}, None, 'unique'),
    ({'Govt': 'EM'}, {'Govt': 'EM'}, 'excluded.*prior'),
])
def test_invalid_or_contradictory_configuration_fails(excluded, prior, match):
    """Typos and mutually contradictory expert inputs must not be ignored."""
    x, y = panel()
    with pytest.raises(ValueError, match=match):
        LassoModel(excluded_factors=excluded, factor_for_prior=prior,
                   apply_ols_prior=True, solver='CLARABEL').fit(x, y)


def test_clone_path_and_response_subsets_retain_exclusions():
    """Reusable estimators may hold a superset of response labels across cadences."""
    x, y = panel()
    model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO, solver='CLARABEL',
                       excluded_factors=pd.Series({'Govt': ('HY', 'EM'), 'Credit': None}))
    fitted = clone(model).fit(x, y[['Govt']])
    assert fitted.coef_.loc['Govt', ['HY', 'EM']].eq(0).all()
    for fit in model.fit_reg_lambda_path(x, y, reg_lambdas=[0.0, 0.01]):
        assert fit.coef_.loc['Govt', ['HY', 'EM']].eq(0).all()
        reference = clone(model).set_params(reg_lambda=fit.reg_lambda).fit(x, y)
        np.testing.assert_allclose(fit.coef_, reference.coef_, atol=1e-5)
    fitted.set_params(excluded_factors=None).fit(x, y)
    assert fitted.excluded_factor_mask_ is None
    assert abs(fitted.coef_.at['Govt', 'EM']) > 1


def test_exclusions_do_not_allow_nonzero_explicit_prior():
    """A numerical prior and hard exclusion cannot make opposite declarations."""
    x, y = panel()
    priors = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    priors.at['Govt', 'EM'] = 1
    with pytest.raises(ValueError, match='excluded.*prior'):
        LassoModel(excluded_factors={'Govt': 'EM'}, factors_beta_prior=priors,
                   apply_ols_prior=True).fit(x, y)


def test_unsupported_solver_rejects_exclusions():
    """A solver without hard constraints must not silently discard exclusions."""
    with pytest.raises(ValueError, match='excluded_factors'):
        LassoModel(model_type=LassoModelType.UNILASSO, excluded_factors={'Govt': 'EM'})
