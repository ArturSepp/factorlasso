"""Offline example of conditional inference using the unchanged FCGL statistic."""
import numpy as np
import pandas as pd

from factorlasso import FactorTestInversion, LassoModel, LassoModelType


rng = np.random.default_rng(42)
x = pd.DataFrame(rng.normal(0., .04, (48, 2)), columns=['Equity', 'Credit'])
y = pd.DataFrame(x.to_numpy() @ np.array([[.6, .2], [.3, .8]])
                 + rng.normal(0., .01, (48, 2)), columns=['A', 'B'])
y.loc[:7, 'B'] = np.nan
model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
    reg_lambda=1e-5, span=36, warmup_period=8, solver='CLARABEL')
baseline = model.copy().fit(x, y).coef_.copy()
contrast = pd.DataFrame(0., index=y.columns, columns=x.columns)
contrast['Equity'] = [.7, .3]

# This synthetic example has independently known iid Gaussian covariance shape.
# The unknown common variance is eliminated by conditioning.
engine = FactorTestInversion(model, x, y,
    covariance=np.eye(int(y.notna().sum().sum())), draws=39, seed=7)
point = float(baseline.Equity @ np.array([.7, .3]))
test = engine.test(contrast, point)
assert test['pvalue'] == 1.
pd.testing.assert_frame_equal(engine.estimate, baseline)
assert model.coef_ is None

# The score at the production point is zero under the independently calculated
# absolute-deviation statistic, so every simulated score is at least as extreme.
assert abs(point-float((baseline*contrast).to_numpy().sum())) < 1e-12
assert test['estimate'] == point

# With estimated covariance, independently validated calibration is compulsory.
estimated = FactorTestInversion(model, x, y, draws=39, seed=7)
try:
    estimated.intervals({'portfolio Equity': contrast})
except ValueError as error:
    assert 'passing independent calibration' in str(error)
else:
    raise AssertionError('Estimated noise must not acquire an exact-coverage label')
