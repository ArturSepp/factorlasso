"""Common-frequency residual correlation, intervals and as-of availability contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import factorlasso as fl


def inputs():
    """Annual-alpha-scaled monthly/quarterly residuals on their native grids."""
    dates = pd.date_range("2018-01-31", periods=66, freq="ME")
    raw_monthly = pd.Series(np.sin(np.arange(66)) * 0.02 + 0.001, index=dates, name="m")
    quarters = dates[dates.month % 3 == 0]
    raw_quarterly = pd.Series(np.cos(np.arange(len(quarters))) * 0.04,
                              index=quarters, name="q")
    residuals = pd.concat([12 * raw_monthly, 4 * raw_quarterly], axis=1)
    metadata = pd.DataFrame({"frequency": ["ME", "QE"], "beta_span": [36., 12.],
                             "annualisation_factor": [12., 4.],
                             "residual_scale": [12., 4.]}, index=["m", "q"])
    common = pd.concat([raw_monthly.resample("QE").sum(), raw_quarterly], axis=1)
    return residuals, metadata, common


def estimate(**overrides):
    """Build a prepared estimate with an explicitly known availability date."""
    residuals, metadata, _ = inputs()
    kwargs = dict(residuals=residuals, metadata=metadata,
                  estimation_date=pd.Timestamp("2023-07-15"))
    kwargs.update(overrides)
    return fl.estimate_residual_correlation(**kwargs)


def reference(panel, span):
    """Independent pandas means plus a finite normalized exponential Gram matrix."""
    z = (panel - panel.ewm(span=span, adjust=False).mean()).iloc[1:].to_numpy()
    weights = (1 - 2 / (span + 1)) ** np.arange(len(z) - 1, -1, -1)
    moment = z.T @ (weights[:, None] * z) / weights.sum()
    return moment / np.sqrt(np.outer(np.diag(moment), np.diag(moment)))


def snapshot(prepared=None):
    """An unchanged factor block and native annual-alpha residual panel."""
    residuals, metadata, _ = inputs()
    return fl.CurrentFactorCovarData(
        x_covar=pd.DataFrame([[0.04]], index=["f"], columns=["f"]),
        y_betas=pd.DataFrame({"f": [0.7, 0.2]}, index=["m", "q"]),
        y_variances=pd.DataFrame({"residual_var": [0.01, 0.02]}, index=["m", "q"]),
        residuals=residuals, estimation_date=pd.Timestamp("2023-07-15"),
        residual_metadata=metadata, residual_correlation=prepared,
    )


def test_monthly_quarterly_aggregation_and_units():
    """Aggregate raw residual log returns before estimating dimensionless quarterly correlation."""
    data = estimate()
    _, _, common = inputs()
    pd.testing.assert_frame_equal(data.residual_returns, common, check_freq=False)
    assert data.frequency == "QE"
    assert data.span == 12
    assert data.observation_date == pd.Timestamp("2023-06-30")
    assert data.estimation_date == pd.Timestamp("2023-07-15")
    np.testing.assert_allclose(data.correlation, reference(common, 12), atol=1e-15)
    assert np.linalg.eigvalsh(data.correlation).min() >= -1e-14


def test_asof_and_no_future_periods():
    """Ignore partial/future quarters and never backdate a fitted covariance."""
    residuals, metadata, _ = inputs()
    residuals.loc[pd.Timestamp("2023-09-30")] = 1e9
    data = estimate(residuals=residuals, metadata=metadata)
    np.testing.assert_allclose(data.correlation, estimate().correlation)
    with pytest.raises(ValueError, match="available"):
        data.get_corr(pd.Timestamp("2023-06-30"))
    pd.testing.assert_frame_equal(data.get_corr(pd.Timestamp("2023-08-01")), data.correlation)


def test_incomplete_last_quarter_carries_last_complete_period():
    """An incomplete latest period is not extrapolated into a quarter's return."""
    residuals, metadata, _ = inputs()
    residuals.loc[pd.Timestamp("2023-06-30"), "q"] = np.nan
    data = estimate(residuals=residuals, metadata=metadata)
    assert data.observation_date == pd.Timestamp("2023-03-31")
    assert data.residual_returns.index[-1] == pd.Timestamp("2023-03-31")


def test_internal_gap_fails():
    """An absent month must not be treated as a zero residual contribution."""
    residuals, metadata, _ = inputs()
    residuals = residuals.drop(pd.Timestamp("2020-02-29"))
    with pytest.raises(ValueError, match="incomplete"):
        estimate(residuals=residuals, metadata=metadata)


def test_explicit_coarser_grid_converts_decay():
    """Monthly span 36 keeps its calendar decay when mapped to quarters."""
    residuals, metadata, _ = inputs()
    data = estimate(residuals=residuals[["m"]], metadata=metadata.loc[["m"]],
                    frequency="QE", periods_per_year=4.)
    expected_decay = (1 - 2 / 37) ** 3
    assert 1 - 2 / (data.span + 1) == pytest.approx(expected_decay)
    np.testing.assert_allclose(data.correlation,
                               reference(data.residual_returns, data.span))


def test_prepared_assembly_and_alpha_unchanged():
    """Assembly combines correlation with current MATF risk; native annual alpha is independent."""
    data = snapshot(estimate())
    vol = np.sqrt(data.y_variances.residual_var.to_numpy())
    residual = data.residual_correlation.correlation.to_numpy() * np.outer(vol, vol)
    np.fill_diagonal(residual, data.y_variances.residual_var)
    expected = data.get_y_covar(0.) + residual
    np.testing.assert_allclose(data.get_y_covar(residual_type="empirical"), expected)
    pd.testing.assert_frame_equal(data.get_y_covar(), data.get_y_covar(residual_type="orthogonal"))
    pd.testing.assert_series_equal(data.estimate_alpha(), snapshot().estimate_alpha())
    pd.testing.assert_series_equal(
        data.estimate_alpha(alpha_span={"ME": 36, "QE": 12}),
        data.estimate_alpha(alpha_span={"ME": 36, "QE": 12},
                            asset_frequencies=data.residual_metadata.frequency),
    )


def test_rolling_asof_queries_and_residual_history():
    """Select past fitted components; hold the prepared residual matrix between fits."""
    first = snapshot(estimate())
    second_date = pd.Timestamp("2023-08-15")
    second = replace(first, estimation_date=second_date, y_betas=first.y_betas * 1.2)
    rolling = fl.RollingFactorCovarData({first.estimation_date: first, second_date: second})
    dates = pd.to_datetime(["2023-07-20", "2023-08-20"])
    covars = rolling.get_y_covars(residual_type="empirical", dates=dates)
    pd.testing.assert_frame_equal(covars[dates[0]], first.get_y_covar(residual_type="empirical"))
    pd.testing.assert_frame_equal(covars[dates[1]], second.get_y_covar(residual_type="empirical"))
    history = rolling.get_residual_correlations()
    assert len(history) == 1
    pd.testing.assert_frame_equal(
        history[first.estimation_date], first.residual_correlation.correlation
    )
    with pytest.raises(ValueError, match="available"):
        rolling.get_y_covars(dates=[pd.Timestamp("2023-07-01")])


def test_filter_and_excel_roundtrip(tmp_path):
    """Persist the prepared covariance, units, grid, spans and availability date."""
    data = snapshot(estimate())
    sub = data.filter_on_tickers({"q": "quarterly", "m": "monthly"})
    assert sub.residual_metadata.index.tolist() == ["quarterly", "monthly"]
    assert sub.residual_correlation.correlation.index.tolist() == ["quarterly", "monthly"]
    path = tmp_path / "prepared.xlsx"
    sub.save(str(path))
    loaded = fl.CurrentFactorCovarData.load(str(path))
    pd.testing.assert_frame_equal(loaded.get_y_covar(residual_type="empirical"),
                                  sub.get_y_covar(residual_type="empirical"))
    assert loaded.estimation_date == sub.estimation_date
    assert loaded.residual_correlation.estimation_date == sub.residual_correlation.estimation_date


def test_non_nested_weekly_quarterly_fails():
    """Weekly periods crossing a quarter-end cannot be prorated or silently assigned."""
    dates = pd.date_range("2020-01-01", periods=180, freq="W-WED")
    residuals = pd.DataFrame({"w": np.sin(np.arange(len(dates)))}, index=dates)
    metadata = pd.DataFrame({"frequency": ["W-WED"], "beta_span": [156.],
                             "annualisation_factor": [52.], "residual_scale": [1.]},
                            index=["w"])
    with pytest.raises(ValueError, match="boundar"):
        estimate(residuals=residuals, metadata=metadata, frequency="QE", periods_per_year=4.)


def test_asof_rejects_snapshot_fitted_after_its_dictionary_key():
    """A mislabeled snapshot cannot expose future betas through the new as-of API."""
    data = snapshot(estimate())
    rolling = fl.RollingFactorCovarData({pd.Timestamp("2023-06-30"): data})
    with pytest.raises(ValueError, match="available"):
        rolling.get_y_covars(dates=pd.to_datetime(["2023-07-01"]))


@pytest.mark.parametrize("frequency, annualisation", [("D", 365.), ("B", 260.),
                                                      ("BME", 12.)])
def test_complete_nested_calendars(frequency, annualisation):
    """Sum exact daily or business-period observations onto calendar quarters."""
    dates = pd.date_range("2020-01-01", "2022-12-31", freq=frequency)
    residuals = pd.DataFrame({"a": np.sin(np.arange(len(dates))) * 0.01}, index=dates)
    metadata = pd.DataFrame({"frequency": [frequency], "beta_span": [36.],
                             "annualisation_factor": [annualisation], "residual_scale": [1.]},
                            index=["a"])
    data = estimate(residuals=residuals, metadata=metadata, estimation_date="2022-12-31",
                    frequency="QE", periods_per_year=4., span=12.)
    pd.testing.assert_frame_equal(data.residual_returns, residuals.resample("QE").sum())


@pytest.mark.parametrize("mutation, message", [
    ("missing_asset", "every asset"), ("negative_scale", "finite and positive"),
    ("finer_grid", "no larger"), ("conflicting_units", "disagrees"),
    ("unweighted", "unweighted"), ("no_history", "available"),
    ("few_periods", "two complete"), ("missing_annualisation", "annualisation_factor"),
])
def test_invalid_units_and_history_fail(mutation, message):
    """Invalid frequency metadata and insufficient histories cannot yield a risk matrix."""
    residuals, metadata, _ = inputs()
    kwargs = dict(residuals=residuals, metadata=metadata)
    if mutation == "missing_asset":
        kwargs["metadata"] = metadata.loc[["q"]]
    elif mutation == "negative_scale":
        metadata.loc["m", "residual_scale"] = -12.
    elif mutation == "finer_grid":
        kwargs.update(frequency="ME", periods_per_year=12.)
    elif mutation == "conflicting_units":
        kwargs["periods_per_year"] = 2.
    elif mutation == "unweighted":
        metadata.loc["q", "beta_span"] = np.nan
    elif mutation == "no_history":
        kwargs["estimation_date"] = "2010-01-01"
    elif mutation == "few_periods":
        kwargs["estimation_date"] = "2018-03-31"
    else:
        kwargs["frequency"] = "YE"
    with pytest.raises(ValueError, match=message):
        estimate(**kwargs)


@pytest.mark.parametrize("frequency, annualisation", [("BME", 12.), ("BQE", 4.), ("BYE", 1.)])
def test_native_business_periods_keep_their_own_boundaries(frequency, annualisation):
    """A business period ending before a weekend remains valid on its native grid."""
    dates = pd.date_range("2018-01-01", periods=24, freq=frequency)
    residuals = pd.DataFrame({"a": np.sin(np.arange(len(dates))) * 0.01}, index=dates)
    metadata = pd.DataFrame({"frequency": [frequency], "beta_span": [12.],
                             "annualisation_factor": [annualisation], "residual_scale": [1.]},
                            index=["a"])
    data = estimate(residuals=residuals, metadata=metadata, estimation_date=dates[-1])
    pd.testing.assert_frame_equal(data.residual_returns, residuals)
    np.testing.assert_allclose(data.correlation, reference(residuals, 12.))
