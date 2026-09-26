"""Explicit factor selection uses its estimated OLS slope, with automatic fallback."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from factorlasso import LassoModel, LassoModelCV, LassoModelType


def panel():
    """Make the automatic winner differ from the requested credit factor."""
    rng = np.random.default_rng(314)
    x = pd.DataFrame(rng.normal(size=(100, 3)), columns=['Rates', 'IG', 'EM'])
    y = pd.DataFrame({'bond': 1.7*x.EM + .35*x.IG + .1*rng.normal(size=100),
                      'other': 1.4*x.Rates + .1*rng.normal(size=100)})
    x.loc[8:12, 'IG'] = np.nan
    y.loc[21:23, 'bond'] = np.nan
    return x, y


def slope(x, y, span):
    """Solve an independent weighted design with intercept and pairwise masks."""
    w = (1 - 2/(span+1)) ** np.arange(len(x)-1, -1, -1)
    keep = x.notna() & y.notna()
    design = np.column_stack([np.ones(keep.sum()), x[keep]])
    return np.linalg.lstsq(design*np.sqrt(w[keep, None]),
                           y[keep]*np.sqrt(w[keep]), rcond=None)[0][1]


def test_selected_factor_uses_weighted_slope_and_preserves_automatic_fallback():
    """The chosen lower-R2 factor replaces the winner; magnitude is estimated."""
    x, y = panel()
    mapping = pd.Series({'bond': 'IG', 'other': None, 'inactive_asset': 'EM'})
    before = mapping.copy()
    auto = LassoModel(apply_ols_prior=True, span=60).fit(x, y, span=24)
    model = LassoModel(apply_ols_prior=True, span=60,
                       factor_for_prior=mapping).fit(x, y, span=24)
    assert auto.ols_r2_.loc['bond'].idxmax() == 'EM'
    expected = slope(x.IG, y.bond, 24)
    assert abs(expected-1) > .1
    assert model.ols_beta_prior_.loc['bond', 'IG'] == pytest.approx(expected)
    assert model.ols_beta_prior_.loc['bond'].drop('IG').eq(0).all()
    pd.testing.assert_series_equal(model.ols_beta_prior_.loc['other'],
                                    auto.ols_beta_prior_.loc['other'])
    pd.testing.assert_series_equal(mapping, before)
    assert clone(model).factor_for_prior.equals(mapping)
    # Full-universe maps are reusable for subsets used in cadence-group fits.
    subset = clone(model).fit(x, y[['other']], span=24)
    pd.testing.assert_frame_equal(subset.ols_beta_prior_, auto.ols_beta_prior_.loc[['other']])


def test_selected_factor_respects_signs_explicit_overrides_and_nonestimable_pairs():
    """Existing sign and finite-cell precedence still apply after factor selection."""
    x, y = panel()
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    signs.loc['bond', 'IG'] = 0
    model = LassoModel(apply_ols_prior=True, factor_for_prior={'bond': 'IG'},
                       factors_beta_loading_signs=signs).fit(x, y)
    assert model.effective_beta_prior_.loc['bond'].eq(0).all()
    override = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    override.loc['bond', 'IG'] = .6
    model = replace(model, factors_beta_loading_signs=None,
                    factors_beta_prior=override).fit(x, y)
    assert model.effective_beta_prior_.loc['bond', 'IG'] == .6
    x['IG'] = 1.
    model = replace(model, factors_beta_prior=None).fit(x, y)
    assert model.ols_beta_prior_.loc['bond'].eq(0).all()


@pytest.mark.parametrize('mapping', [[], 'IG', pd.Series(['IG','EM'], index=['bond','bond'])])
def test_bad_mapping_rejected(mapping):
    """Malformed maps must not silently ignore an intended prior selection."""
    with pytest.raises((TypeError, ValueError), match='factor_for_prior'):
        LassoModel(apply_ols_prior=True, factor_for_prior=mapping)


def test_unknown_factor_and_disabled_ols_rejected():
    """Unknown named factors and disabled OLS cannot be silently accepted."""
    x, y = panel()
    with pytest.raises(ValueError, match='factor_for_prior'):
        LassoModel(apply_ols_prior=True, factor_for_prior={'bond':'missing'}).fit(x, y)
    with pytest.raises(ValueError, match='apply_ols_prior'):
        LassoModel(factor_for_prior={'bond':'IG'})


def test_mapping_recomputed_inside_cv_and_preserved_on_lambda_paths(monkeypatch):
    """Each fit estimates its selected prior only from that training sample."""
    x, y = panel()
    observed = []
    original = LassoModel._prepare_fit

    def capture(self, *args, **kwargs):
        """Check the fitted prior against the actual training sample."""
        result = original(self, *args, **kwargs)
        expected = slope(kwargs['x'].IG, kwargs['y'].bond, kwargs['eff_span'])
        assert self.ols_beta_prior_.loc['bond','IG'] == pytest.approx(expected)
        observed.append(len(kwargs['x']))
        return result

    monkeypatch.setattr(LassoModel, '_prepare_fit', capture)
    model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                       span=24, apply_ols_prior=True, factor_for_prior={'bond':'IG'})
    LassoModelCV(base_model=model, lambdas=[.01,.02], n_splits=3,
                 refit=False, use_lambda_path=True).fit(x, y)
    assert len(observed) == 3 and max(observed) < len(x)
    for fitted in model.fit_reg_lambda_path(x, y, [.01,.02]):
        assert fitted.factor_for_prior == {'bond':'IG'}
        assert fitted.ols_beta_prior_.loc['bond','IG'] == pytest.approx(slope(x.IG,y.bond,24))
