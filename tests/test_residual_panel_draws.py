"""Independent residual-panel and missing-score dependence contracts."""
import numpy as np
import pandas as pd
import pytest

from factorlasso import bootstrap_weighted_means, estimate_residual_correlation


def test_residual_panels_reconstruct_same_means_and_preserve_masks():
    """Scalar calendar multipliers reproduce every raw panel and weighted mean."""
    x = np.arange(24., dtype=float).reshape(12, 2) / 100
    x[:2, 1] = np.nan
    q = np.where(np.isfinite(x), .9**np.arange(11, -1, -1)[:, None], 0.) / 10
    result = bootstrap_weighted_means(x, q, calendar=np.arange(12), bandwidth=3,
        draws=100, seed=5, return_residuals=True, scale=[2., 3.])
    plain = bootstrap_weighted_means(x, q, calendar=np.arange(12), bandwidth=3,
        draws=100, seed=5, scale=[2., 3.])
    np.testing.assert_array_equal(result['errors'], plain['errors'])
    centre = np.nansum(q*x, axis=0)/q.sum(axis=0)
    kernel = np.maximum(1 - abs(np.arange(12)[:, None]-np.arange(12))/3., 0.)
    multipliers = np.random.default_rng(5).normal(size=(100, 12)) @ np.linalg.cholesky(kernel).T
    expected = centre + (x-centre)*multipliers[:, :, None]
    np.testing.assert_allclose(result['residual_draws'], expected, equal_nan=True)
    means = np.nansum(q*result['residual_draws'], axis=1)*[2., 3.]
    np.testing.assert_allclose(means-result['estimate'], result['errors'], atol=1e-15)
    assert 'residual_draws' not in plain
    with pytest.raises(ValueError, match='return_residuals'):
        bootstrap_weighted_means(x, q, calendar=np.arange(12), return_residuals=1)


def test_zero_innovation_correlation_matches_calendar_outer_product():
    """A common decay applied to masked scores is PSD without pairwise repair."""
    from factorlasso import compute_ewm
    dates = pd.date_range('2020-03-31', periods=16, freq='QE')
    x = pd.DataFrame(np.random.default_rng(7).normal(size=(16, 3)), index=dates)
    x.iloc[[1, 4, 9], 1] = np.nan
    metadata = pd.DataFrame({'frequency': 'QE', 'beta_span': 5.,
        'annualisation_factor': 4., 'residual_scale': 1.}, index=x.columns)
    result = estimate_residual_correlation(x, metadata, dates[-1], span=5.,
                                          missing_policy='zero_innovation')
    z = (x-compute_ewm(x, span=5.)).iloc[1:].fillna(0).to_numpy()
    moment = np.zeros((3, 3))
    for row in z:
        moment = (2/3)*moment + (1/3)*np.outer(row, row)
    vol = np.sqrt(np.diag(moment))
    np.testing.assert_allclose(result.correlation, moment/np.outer(vol, vol), atol=1e-14)
    assert np.linalg.eigvalsh(result.correlation).min() >= -1e-12
    assert result.residual_returns.isna().equals(x.isna())
    assert (result.asset_metadata.correlation_missing_policy == 'zero_innovation').all()
    with pytest.raises(ValueError, match='missing_policy'):
        estimate_residual_correlation(x, metadata, dates[-1], missing_policy='repair')
