"""Offline reference for conditional EWMA residual-alpha uncertainty."""
import numpy as np
import pandas as pd
from factorlasso import (
    compute_ewm, estimate_alpha_uncertainty, gaussian_quadratic_summary,
    sample_gaussian_estimates, calibrate_alpha_uncertainty,
)

calendar = np.arange(120)
rng = np.random.default_rng(20261004)
residuals = pd.DataFrame(rng.normal(.01, .1, size=(120, 2)), columns=['a', 'b'])
residuals.loc[:19, 'b'] = np.nan
result = estimate_alpha_uncertainty(residuals, 60, calendar=calendar, bandwidth=6.)
np.testing.assert_allclose(result.estimates, compute_ewm(residuals, span=60).iloc[-1])
assert result.diagnostics.loc['b', 'weight_mass'] < 1

summary = gaussian_quadratic_summary(result.estimates, result.covariance, np.eye(2))
assert summary['noise'] == np.trace(result.covariance)
assert summary['lower'] <= summary['observed'] <= summary['upper']
scenarios = sample_gaussian_estimates(result.estimates, result.covariance, draws=20000, seed=42)
np.testing.assert_allclose(np.cov(scenarios.T), result.covariance, rtol=.06, atol=2e-6)

intervals = calibrate_alpha_uncertainty(result, residuals, method='known_shape',
    covariance_shapes={name: np.eye(residuals[name].notna().sum()) for name in residuals})
np.testing.assert_allclose(intervals.estimate, result.estimates)
np.testing.assert_allclose(intervals.standard_error, result.diagnostics.standard_error)
assert intervals.status.eq('known_shape_gaussian_model').all()
assert (intervals.lower <= intervals.estimate).all()
assert (intervals.upper >= intervals.estimate).all()
