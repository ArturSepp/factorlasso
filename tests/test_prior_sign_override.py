"""Prior signs override automatic detection without weakening explicit constraints."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType


def panel():
    """Make one pooled negative factor with oppositely exposed responses."""
    rng = np.random.default_rng(164)
    x = pd.DataFrame(rng.normal(size=(120, 2)), columns=['Inflation', 'Rates'])
    y = pd.DataFrame({'IL': x.Inflation + .1*x.Rates,
                      'nominal': -3*x.Inflation + .1*x.Rates})
    return x, y


def model_for(y, **kwargs):
    """Use fixed groups so sign precedence can be compared independently."""
    return LassoModel(model_type=LassoModelType.GROUP_LASSO,
                      group_data=pd.Series(1, index=y.columns), span=36,
                      auto_sign_constraints=True, reg_lambda=.02, **kwargs)


@pytest.mark.parametrize('apply_ols', [False, True])
@pytest.mark.parametrize('direction', [-1., 1.])
def test_supplied_prior_overrides_detection_and_matches_reference(apply_ols, direction):
    """The actual solver gets the prior sign, checked against an explicit-sign fit."""
    x, y = panel()
    y = y*direction
    prior = pd.DataFrame(0., index=y.columns, columns=x.columns)
    prior.loc['IL','Inflation'] = direction*.8
    before = prior.copy()
    detected = model_for(y).fit(x, y)
    assert detected.derived_signs_.loc['IL','Inflation'] == -direction
    fitted = model_for(y, apply_ols_prior=apply_ols, factors_beta_prior=prior).fit(x, y)
    assert fitted.derived_signs_.loc['IL','Inflation'] == direction
    assert fitted.derived_signs_.loc['nominal','Inflation'] == -direction
    assert direction*fitted.coef_.loc['IL','Inflation'] >= -1e-7
    reference_signs = detected.derived_signs_.copy()
    reference_signs.loc['IL','Inflation'] = direction
    reference = replace(fitted, auto_sign_constraints=False, apply_ols_prior=False,
                        factors_beta_loading_signs=reference_signs).fit(x, y)
    np.testing.assert_allclose(fitted.coef_, reference.coef_, atol=1e-9, rtol=1e-9)
    pd.testing.assert_frame_equal(prior, before)


@pytest.mark.parametrize('explicit_sign', [-1., 0., 1.])
def test_explicit_hard_sign_still_wins(explicit_sign):
    """Prior precedence applies only to detected constraints, never hard inputs."""
    x, y = panel()
    prior = pd.DataFrame(0., index=y.columns, columns=x.columns)
    prior.loc['IL','Inflation'] = .8
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    signs.loc['IL','Inflation'] = explicit_sign
    fitted = model_for(y, apply_ols_prior=True, factors_beta_prior=prior,
                       factors_beta_loading_signs=signs).fit(x, y)
    assert fitted.derived_signs_.loc['IL','Inflation'] == explicit_sign
    assert fitted.effective_beta_prior_.loc['IL','Inflation'] == (.8 if explicit_sign == 1 else 0.)


def test_nonzero_prior_overrides_automatic_gate_but_zero_and_missing_do_not():
    """A prior has direction only when finite and nonzero; excluded factors stay free."""
    x, y = panel()
    prior = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    prior.loc['IL','Inflation'] = .8
    prior.loc['IL','Rates'] = 0.
    fitted = model_for(y, factors_beta_prior=prior, auto_sign_threshold_t=1e9).fit(x,y)
    expected = pd.DataFrame(0., index=y.columns, columns=x.columns)
    expected.loc['IL','Inflation'] = 1.
    pd.testing.assert_frame_equal(fitted.derived_signs_, expected)
    excluded = replace(fitted, auto_sign_excluded_factors=['Inflation']).fit(x,y)
    assert excluded.derived_signs_['Inflation'].isna().all()
    disabled = LassoModel(factors_beta_prior=prior).fit(x,y)
    assert disabled.derived_signs_ is None


@pytest.mark.parametrize('mapping', [None, {'IL':'Inflation'}])
def test_computed_prior_and_lambda_path_use_same_precedence(mapping):
    """Automatic and mapped OLS centres can override a pooled sign on their own row."""
    x, y = panel()
    fitted = model_for(y, apply_ols_prior=True, factor_for_prior=mapping).fit(x,y)
    assert fitted.derived_signs_.loc['IL','Inflation'] == 1.
    assert fitted.derived_signs_.loc['nominal','Inflation'] == -1.
    pd.testing.assert_frame_equal(fitted.effective_beta_prior_, fitted.ols_beta_prior_)
    for item in fitted.fit_reg_lambda_path(x,y,[.02,.04]):
        direct = replace(fitted,reg_lambda=item.reg_lambda).fit(x,y)
        pd.testing.assert_frame_equal(item.derived_signs_,direct.derived_signs_)
        pd.testing.assert_frame_equal(item.effective_beta_prior_,direct.effective_beta_prior_)
        np.testing.assert_allclose(item.coef_,direct.coef_,atol=1e-4,rtol=0)


def test_adaptive_weights_are_unchanged_by_prior_sign_override(monkeypatch):
    """Only the solver sign changes; automatic pooled slopes still set penalties."""
    x, y = panel()
    captures = []
    original = LassoModel._prepare_fit

    def capture(self,*args,**kwargs):
        """Record the owner-prepared penalties for comparison."""
        prep = original(self,*args,**kwargs)
        captures.append(prep)
        return prep

    monkeypatch.setattr(LassoModel,'_prepare_fit',capture)
    prior = pd.DataFrame(0.,index=y.columns,columns=x.columns)
    prior.loc['IL','Inflation'] = .8
    base = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        n_clusters=1,span=36,auto_sign_constraints=True,auto_sign_adaptive_weights=True,
        reg_lambda=.02)
    base.fit(x,y)
    fitted = replace(base,factors_beta_prior=prior).fit(x,y)
    assert fitted.derived_signs_.loc['IL','Inflation'] == 1.
    for field in ['penalty_weights_np','row_weights_np','col_weights_np']:
        np.testing.assert_array_equal(getattr(captures[0],field),getattr(captures[1],field))
