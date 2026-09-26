"""Joint empirical priors use each fit's original observations and loss weights."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from factorlasso import LassoModel, LassoModelCV


def sample():
    """Create correlated Rates/Inflation and a stronger unrelated univariate winner."""
    rng = np.random.default_rng(2049)
    rates = rng.normal(size=120)
    x = pd.DataFrame({'Rates': rates, 'Inflation': -.8*rates + .3*rng.normal(size=120),
                      'Equity': rng.normal(size=120)})
    y = pd.DataFrame({'il': 1.1*x.Rates + .65*x.Inflation + .1*rng.normal(size=120),
                      'other': .9*x.Equity + .2*rng.normal(size=120)})
    x.loc[4:7, 'Inflation'] = np.nan
    y.loc[13:14, 'il'] = np.nan
    return x, y


def reference(x, y, span):
    """Independently solve weighted normal equations including the intercept."""
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    weight = np.ones(len(x)) if span is None else (1-2/(span+1))**np.arange(len(x)-1,-1,-1)
    design = np.column_stack([np.ones(valid.sum()), x.loc[valid]])
    return np.linalg.solve(design.T @ (weight[valid,None]*design),
                           design.T @ (weight[valid]*y.loc[valid]))[1:]


@pytest.mark.parametrize('span', [None, 24.])
def test_joint_ols_prior_and_automatic_fallback(span):
    """Named factors get joint slopes; blank assets retain their automatic priors."""
    x, y = sample()
    mapping = pd.Series({'il': ('Rates', 'Inflation'), 'other': None})
    model = LassoModel(apply_ols_prior=True, span=span, factor_for_prior=mapping).fit(x,y)
    expected = reference(x[['Rates','Inflation']],y.il,span)
    np.testing.assert_allclose(model.ols_beta_prior_.loc['il',['Rates','Inflation']],expected,
                               rtol=1e-11,atol=1e-11)
    assert model.ols_beta_prior_.loc['il','Equity'] == 0
    auto = LassoModel(apply_ols_prior=True,span=span).fit(x,y)
    pd.testing.assert_series_equal(model.ols_beta_prior_.loc['other'],auto.ols_beta_prior_.loc['other'])
    assert clone(model).factor_for_prior.equals(mapping)
    subset = clone(model).fit(x,y[['other']])
    pd.testing.assert_frame_equal(subset.ols_beta_prior_, auto.ols_beta_prior_.loc[['other']])


def test_joint_priors_override_detected_signs_but_preserve_explicit_hard_signs():
    """The selected positive Inflation prior takes precedence over detected negative signs."""
    x,y=sample()
    y=y[['il']]
    model=LassoModel(apply_ols_prior=True,auto_sign_constraints=True,auto_sign_threshold_t=0.,
                      factor_for_prior={'il':['Rates','Inflation']}).fit(x,y)
    assert model.ols_betas_.loc['il','Inflation'] < 0
    assert model.effective_beta_prior_.loc['il','Inflation'] > 0
    assert model.derived_signs_.loc['il','Inflation'] == 1
    signs=pd.DataFrame(np.nan,index=y.columns,columns=x.columns)
    signs.loc['il','Inflation']=-1.
    fixed=replace(model,factors_beta_loading_signs=signs).fit(x,y)
    assert fixed.derived_signs_.loc['il','Inflation']==-1
    assert fixed.effective_beta_prior_.loc['il','Inflation']==0


@pytest.mark.parametrize('bad', [[], ['Rates','Rates'], ['Rates',None], ['Rates',['Inflation']]])
def test_invalid_joint_selections_rejected(bad):
    """Ambiguous empty, repeated or nested selections cannot be accepted."""
    with pytest.raises((ValueError,TypeError),match='factor_for_prior'):
        LassoModel(apply_ols_prior=True,factor_for_prior={'il':bad})


@pytest.mark.parametrize('mode',['constant','collinear','few_rows'])
def test_unidentifiable_joint_prior_is_neutral(mode):
    """Do not invent a partial or minimum-norm prior when joint slopes are unidentified."""
    x,y=sample()
    if mode=='constant':
        x['Inflation']=1.
    elif mode=='collinear':
        x['Inflation']=2*x.Rates
    else:
        x.loc[:118,'Inflation']=np.nan
    model=LassoModel(apply_ols_prior=True,factor_for_prior={'il':['Rates','Inflation']}).fit(x,y)
    assert model.ols_beta_prior_.loc['il'].eq(0).all()


def test_joint_prior_recomputed_within_cv_training_windows(monkeypatch):
    """CV never estimates its named-factor prior on held-out observations."""
    x,y=sample()
    observed=[]
    original=LassoModel._prepare_fit
    def capture(self,*args,**kwargs):
        """Compare each fit to that fold's independent regression."""
        out=original(self,*args,**kwargs)
        expected=reference(kwargs['x'][['Rates','Inflation']],kwargs['y'].il,kwargs['eff_span'])
        np.testing.assert_allclose(self.ols_beta_prior_.loc['il',['Rates','Inflation']],expected,
                                   rtol=1e-10,atol=1e-10)
        observed.append(len(kwargs['x']))
        return out
    monkeypatch.setattr(LassoModel,'_prepare_fit',capture)
    model=LassoModel(apply_ols_prior=True,span=24,factor_for_prior={'il':('Rates','Inflation')})
    LassoModelCV(base_model=model,lambdas=[.01,.02],n_splits=3,refit=False,
                 use_lambda_path=True).fit(x,y)
    assert set(observed)=={30,60,90} and max(observed)<len(x)
