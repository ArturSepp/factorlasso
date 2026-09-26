"""Native-variance empirical dependence, units, migration and rolling-state contracts."""

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



def reference(panel, span, annualisation):
    """Independent pandas means plus a finite normalized exponential Gram matrix."""
    z = (panel - panel.ewm(span=span, adjust=False).mean()).iloc[1:].to_numpy()
    weights = (1 - 2 / (span + 1)) ** np.arange(len(z) - 1, -1, -1)
    return annualisation * z.T @ (weights[:, None] * z) / weights.sum()


def snapshot():
    """An unchanged factor block and native annual-alpha residual panel."""
    residuals, metadata, _ = inputs()
    return fl.CurrentFactorCovarData(
        x_covar=pd.DataFrame([[0.04]], index=["f"], columns=["f"]),
        y_betas=pd.DataFrame({"f": [0.7, 0.2]}, index=["m", "q"]),
        y_variances=pd.DataFrame({"residual_var": [0.01, 0.02]}, index=["m", "q"]),
        residuals=residuals, estimation_date=pd.Timestamp("2023-07-15"),
        residual_metadata=metadata,
    )





def prepared(**kwargs):
    """Estimate common-quarter correlation using known native metadata."""
    residuals, metadata, _ = inputs()
    options = dict(residuals=residuals, metadata=metadata,
                   estimation_date=pd.Timestamp("2023-07-15"))
    options.update(kwargs)
    return fl.estimate_residual_correlation(**options)


def current(**kwargs):
    """Attach correlation without changing native annual marginal risk or alpha."""
    return replace(snapshot(), residual_correlation=prepared(**kwargs))


def correlation_reference(panel, span):
    """Normalize an independently weighted pandas/Gram covariance reference."""
    moment = reference(panel, span, 1.)
    return moment / np.sqrt(np.outer(np.diag(moment), np.diag(moment)))


def test_common_frequency_and_independent_correlation():
    """Native aggregation and EWMA horizon precede dimensionless normalization."""
    data = prepared()
    _, _, common = inputs()
    pd.testing.assert_frame_equal(data.residual_returns, common, check_freq=False)
    np.testing.assert_allclose(data.correlation, correlation_reference(common, 12), atol=1e-14)
    assert data.frequency == "QE" and data.span == 12
    assert data.observation_date == pd.Timestamp("2023-06-30")
    assert data.estimation_date == pd.Timestamp("2023-07-15")
    assert data.observation_count == len(common)
    assert data.schema_version == 2


@pytest.mark.parametrize("rho", [0., .25, .5, .75, 1.])
def test_independent_assembly_and_exact_diagonal(rho):
    """Correlation changes only cross entries, preserving the MATF diagonal exactly."""
    data = current()
    r = correlation_reference(data.residual_correlation.residual_returns, 12.)
    v = data.y_variances.residual_var.to_numpy()
    expected = np.diag(v)
    expected[0, 1] = expected[1, 0] = rho * r[0, 1] * np.sqrt(v[0] * v[1])
    residual = data.get_residual_covar(residual_type="empirical", residual_corr_weight=rho)
    total = data.get_y_covar(residual_type="empirical", residual_corr_weight=rho)
    np.testing.assert_allclose(residual, expected, atol=1e-15)
    np.testing.assert_allclose(total, data.get_y_covar(0) + expected, atol=1e-15)
    np.testing.assert_array_equal(np.diag(residual), v)
    np.testing.assert_array_equal(np.diag(total), np.diag(data.get_y_covar()))
    assert np.linalg.eigvalsh(residual).min() >= -1e-14
    if rho == 0:
        pd.testing.assert_frame_equal(total, data.get_y_covar(), check_exact=True)


def test_positive_per_asset_scaling_and_alpha_invariance():
    """Different constant scale factors cancel after complete native aggregation."""
    residuals, metadata, _ = inputs()
    base = current()
    scale = pd.Series({"m": 100., "q": 4.})
    scaled = prepared(residuals=residuals * scale, metadata=metadata)
    np.testing.assert_allclose(
        scaled.correlation, base.residual_correlation.correlation, atol=1e-14
    )
    changed = replace(base, residual_correlation=scaled)
    np.testing.assert_allclose(changed.get_y_covar(residual_type="empirical"),
                               base.get_y_covar(residual_type="empirical"), atol=1e-14)
    pd.testing.assert_series_equal(base.estimate_alpha(), snapshot().estimate_alpha())
    pd.testing.assert_frame_equal(base.residuals, snapshot().residuals)


def test_held_correlation_uses_current_variances_and_asof_dates():
    """A monthly update changes covariance while retaining the last quarterly dependence."""
    first = current()
    v = first.y_variances.copy()
    v.loc["m", "residual_var"] *= 4
    second = replace(first, estimation_date=pd.Timestamp("2023-08-15"), y_variances=v)
    rolling = fl.RollingFactorCovarData(
        {first.estimation_date: first, second.estimation_date: second}
    )
    dates = pd.to_datetime(["2023-07-20", "2023-08-20"])
    covars = rolling.get_residual_covars(residual_type="empirical", dates=dates)
    assert covars[dates[1]].loc["m", "q"] == pytest.approx(2 * covars[dates[0]].loc["m", "q"])
    assert covars[dates[1]].loc["m", "m"] == .04
    assert len(rolling.get_residual_correlations()) == 1
    assert len(rolling.get_residual_covars()) == 2
    for date, expected in zip(dates, [first, second]):
        pd.testing.assert_frame_equal(
            rolling.get_y_covars(residual_type="empirical", dates=dates)[date],
            expected.get_y_covar(residual_type="empirical"),
        )
    with pytest.raises(ValueError, match="available"):
        rolling.get_residual_covars(dates=[pd.Timestamp("2023-07-01")])


def test_filter_and_excel_roundtrip(tmp_path):
    """Persist correlation state directly, with no covariance schema or conversion."""
    data = current().filter_on_tickers({"q": "quarterly", "m": "monthly"})
    path = tmp_path / "correlation.xlsx"
    data.save(str(path))
    restored = fl.CurrentFactorCovarData.load(str(path))
    pd.testing.assert_frame_equal(restored.get_y_covar(residual_type="empirical"),
                                  data.get_y_covar(residual_type="empirical"))
    assert restored.residual_correlation.schema_version == 2


def test_weight_and_zero_retention_are_separate():
    """Variance weight still scales the complete residual risk block."""
    data = current()
    full = data.get_residual_covar(residual_type="empirical", residual_corr_weight=.5)
    half = data.get_residual_covar(.5, residual_type="empirical", residual_corr_weight=.5)
    np.testing.assert_allclose(half, .5 * full)
    no_state = snapshot()
    pd.testing.assert_frame_equal(
        no_state.get_y_covar(residual_type="empirical", residual_corr_weight=0),
        no_state.get_y_covar(),
    )
    pd.testing.assert_frame_equal(
        no_state.get_y_covar(0, residual_type="empirical"), no_state.get_y_covar(0)
    )


@pytest.mark.parametrize("rho", [-.1, 1.1, np.nan, np.inf])
def test_invalid_correlation_weight_fails_even_on_empty_rolling(rho):
    """Invalid mixtures cannot be swallowed by an empty output schedule."""
    with pytest.raises(ValueError, match="residual_corr_weight"):
        current().get_y_covar(residual_type="empirical", residual_corr_weight=rho)
    with pytest.raises(ValueError, match="residual_corr_weight"):
        fl.RollingFactorCovarData().get_y_covars(
            residual_type="empirical", residual_corr_weight=rho
        )


def test_only_current_assembly_options_are_accepted():
    """The revised empirical path has no getter unit conversion or span override."""
    for options in [{"beta_span":36}, {"residual_covar_scale":12}]:
        with pytest.raises(TypeError, match="unexpected keyword"):
            current().get_y_covar(residual_type="empirical", **options)
    with pytest.raises(ValueError, match="orthogonal"):
        current().get_y_covar(residual_corr_weight=.5)
    with pytest.raises(ValueError, match="prepared"):
        snapshot().get_y_covar(residual_type="empirical")


def test_no_lookahead_or_undefined_correlations():
    """Do not backdate residual dependence or invent correlation for constant residuals."""
    data = current()
    with pytest.raises(ValueError, match="available"):
        replace(data, estimation_date=pd.Timestamp("2023-06-30")).get_y_covar(
            residual_type="empirical"
        )
    residuals, metadata, _ = inputs()
    residuals.loc[pd.Timestamp("2023-09-30")] = 1e9
    np.testing.assert_allclose(prepared(residuals=residuals).correlation, prepared().correlation)
    residuals["m"] = 0.
    with pytest.raises(ValueError, match="variance"):
        prepared(residuals=residuals, metadata=metadata)


def test_explicit_coarser_grid_and_invalid_schema():
    """Common-grid decay and persistence type remain explicit."""
    residuals, metadata, _ = inputs()
    data = prepared(residuals=residuals[["m"]], metadata=metadata.loc[["m"]],
                    frequency="QE", periods_per_year=4.)
    assert 1 - 2 / (data.span + 1) == pytest.approx((1 - 2 / 37) ** 3)
    sheets = data.to_sheets()
    sheets["residual_corr_info"].loc["schema_version"] = 99
    with pytest.raises(ValueError, match="schema"):
        fl.ResidualCorrelationData.from_sheets(sheets)


@pytest.mark.parametrize("weight", [-1., np.nan, np.inf])
def test_invalid_variance_weight(weight):
    """Negative or nonfinite empirical risk multipliers cannot imply valid PSD risk."""
    with pytest.raises(ValueError, match="residual_var_weight"):
        current().get_y_covar(weight, residual_type="empirical")


@pytest.mark.parametrize("variance", [-.1, np.nan, np.inf])
def test_invalid_matf_variance(variance):
    """Empirical dependence must not hide invalid native marginal risk."""
    data = current()
    v = data.y_variances.copy()
    v.iloc[0, 0] = variance
    with pytest.raises(ValueError, match="variances"):
        replace(data, y_variances=v).get_residual_covar(residual_type="empirical")


def test_zero_matf_variance_and_label_alignment():
    """Zero native variance produces a zero row; reorder empirical state by labels."""
    data = current()
    v = data.y_variances.copy()
    v.iloc[0, 0] = 0.
    actual = replace(data, y_variances=v).get_residual_covar(residual_type="empirical")
    np.testing.assert_array_equal(actual.loc["m"], [0., 0.])
    reordered = replace(
        data,
        residual_correlation=data.residual_correlation.filter_on_tickers(["q", "m"]),
    )
    pd.testing.assert_frame_equal(reordered.get_y_covar(residual_type="empirical"),
                                  data.get_y_covar(residual_type="empirical"))
    with pytest.raises(ValueError, match="index disagree"):
        replace(data, y_variances=v.iloc[::-1]).get_residual_covar()


def test_public_surface_contains_only_correlation_state():
    """Development cleanup removes obsolete state, factory, migration and diagnostic APIs."""
    import inspect
    assert not hasattr(fl, "ResidualCovarianceData")
    assert not hasattr(fl, "estimate_residual_covariance")
    assert "residual_covariance" not in fl.CurrentFactorCovarData.__dataclass_fields__
    assert not hasattr(fl.CurrentFactorCovarData, "migrate_residual_covariance")
    assert not hasattr(fl.CurrentFactorCovarData, "get_legacy_empirical_residual_covar")
    for method in [fl.CurrentFactorCovarData.get_y_covar,
                   fl.CurrentFactorCovarData.get_residual_covar,
                   fl.RollingFactorCovarData.get_y_covars]:
        parameters = inspect.signature(method).parameters
        assert "beta_span" not in parameters and "residual_covar_scale" not in parameters


@pytest.mark.parametrize("span", [0., -1., np.nan, np.inf])
def test_invalid_preparation_span(span):
    """Invalid spans fail during preparation, before any correlation can be attached."""
    with pytest.raises(ValueError, match="span"):
        prepared(span=span)


@pytest.mark.parametrize(
    "defect", ["duplicate_dates", "unsorted", "duplicate_assets", "index", "empty"]
)
def test_invalid_native_panels(defect):
    """Reject invalid axes before aggregating native return intervals."""
    residuals, _, _ = inputs()
    if defect == "duplicate_dates":
        residuals.index = [residuals.index[0]] * len(residuals)
    elif defect == "unsorted":
        residuals = residuals.iloc[::-1]
    elif defect == "duplicate_assets":
        residuals.columns = ["m", "m"]
    elif defect == "index":
        residuals.index = pd.RangeIndex(len(residuals))
    else:
        residuals = residuals.iloc[:0]
    with pytest.raises(ValueError, match="sorted unique"):
        prepared(residuals=residuals)


@pytest.mark.parametrize(
    "defect", ["asymmetry", "indefinite", "diagonal", "nan", "labels", "dates", "count"]
)
def test_invalid_persisted_correlation(defect):
    """Fail malformed or impossible saved correlation rather than repairing its meaning."""
    data = prepared()
    sheets = data.to_sheets()
    sheets = {name: frame.copy() for name, frame in sheets.items()}
    corr = sheets["residual_corr"]
    if defect == "asymmetry":
        corr.iloc[0, 1] = .5
    elif defect == "indefinite":
        corr.iloc[0, 1] = corr.iloc[1, 0] = 1.2
    elif defect == "diagonal":
        corr.iloc[0, 0] = .5
    elif defect == "nan":
        corr.iloc[0, 1] = np.nan
    elif defect == "labels":
        sheets["residual_corr_metadata"] = sheets["residual_corr_metadata"].iloc[::-1]
    elif defect == "dates":
        sheets["residual_corr_info"].loc["estimation_date"] = "2018-01-01"
    else:
        sheets["residual_corr_info"].loc["observation_count"] = 999
    with pytest.raises(ValueError, match="Residual correlation"):
        fl.ResidualCorrelationData.from_sheets(sheets)
