"""Offline full-FCGL bootstrap example; teaching ranges, not a coverage claim."""
import numpy as np
import pandas as pd

from factorlasso import LassoModel, LassoModelType, bootstrap_factor_betas

rng = np.random.default_rng(401)
x = pd.DataFrame(rng.normal(0., .03, (72, 2)), columns=['Equity', 'Credit'])
y = pd.DataFrame(x.to_numpy() @ np.array([[.7, .4], [.2, .5]])
                 + rng.normal(0., .012, (72, 2)), columns=['Asset A', 'Asset B'])
y.loc[:11, 'Asset B'] = np.nan
model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
    span=36, reg_lambda=1e-5, loss_normalization='weight_sum', solver='CLARABEL',
    apply_ols_prior=True, auto_sign_constraints=True, auto_sign_adaptive_weights=True)
bootstrap = bootstrap_factor_betas(model, x, y, draws=30, seed=92, bandwidth=3.)
weights = pd.Series({'Asset A': .8, 'Asset B': .4})
table = bootstrap.portfolio_intervals(weights, confidence=.95)

# Independent aggregation in pandas, retaining same-draw cross-asset covariance.
manual = bootstrap.draws.mul(weights, level='asset', axis=0).groupby('draw').sum()
np.testing.assert_allclose(table.lower, manual.quantile(.025))
np.testing.assert_allclose(table.upper, manual.quantile(.975))
np.testing.assert_allclose(table.estimate, weights @ bootstrap.estimate)
assert bootstrap.metadata['coverage'] == 'not_validated'
assert bootstrap.response_support.observations.to_list() == [72, 60]
assert model.coef_ is None
print(table[['estimate', 'lower', 'upper', 'status']].to_string())
