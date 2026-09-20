"""Prepared residual correlation on complete common return intervals.

Input residuals are additive log-return residuals at explicitly declared native
frequencies. Stored multipliers are undone before summation. A common-grid causal
EWMA mean and normalized second moment use the existing factorlasso EWMA kernels.
No factor-residual cross term, pairwise deletion, interpolation or extrapolation
is used. The fit/availability date is distinct from the last observed period:
residual history recomputed with fitted betas must never be backdated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Dict, Optional, Union

import numpy as np
import pandas as pd

from factorlasso.ewm_utils import _validate_span, compute_ewm, compute_ewm_covar


def _compatible_boundary(date: pd.Timestamp, offset: pd.DateOffset) -> bool:
    """Recognize exact native boundaries, including business calendar period ends."""
    if (offset.is_on_offset(date)
            or isinstance(offset, (pd.offsets.Day, pd.offsets.BusinessDay))):
        return True
    if isinstance(offset, (pd.offsets.BusinessMonthEnd, pd.offsets.BQuarterEnd,
                           pd.offsets.BYearEnd)):
        return offset.rollback(date) + pd.offsets.MonthEnd(0) == date
    return False


def _aggregate_residuals(residuals: pd.DataFrame, metadata: pd.DataFrame,
                        frequency: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Sum complete native log-return intervals into nested common periods.

    Expected native dates are explicit: absent rows and NaNs cannot become zero
    returns. Business-day panels use the declared pandas business-day calendar;
    callers with other calendars must supply a compatible aligned return panel.
    Weekly periods crossing a target boundary require exact reaggregation from
    finer source data before calling this function; no prorating is attempted.
    """
    target = pd.tseries.frequencies.to_offset(frequency)
    ends = pd.date_range(residuals.index[0], cutoff, freq=target)
    common = pd.DataFrame(np.nan, index=ends, columns=residuals.columns, dtype=float)
    for native, group in metadata.groupby("frequency", sort=False):
        source = pd.tseries.frequencies.to_offset(native)
        columns = group.index
        raw = residuals.loc[:, columns].div(group["residual_scale"], axis="columns")
        for end in ends:
            start = end - target
            if not (_compatible_boundary(start, source) and _compatible_boundary(end, source)):
                raise ValueError(
                    f"{native} periods cross {frequency} boundaries; reconstruct exact "
                    "common-period residuals from finer prices/returns before estimation"
                )
            expected = pd.date_range(start, end, freq=source, inclusive="right")
            if expected.empty:
                raise ValueError("Common residual frequency must not be finer than native returns")
            block = raw.reindex(expected)
            values = block.to_numpy(dtype=float)
            valid = np.isfinite(values).all(axis=0)
            common.loc[end, columns[valid]] = values[:, valid].sum(axis=0)
    complete = np.isfinite(common.to_numpy()).all(axis=1)
    positions = np.flatnonzero(complete)
    if len(positions) < 2:
        raise ValueError("Residual covariance requires at least two complete common periods")
    common = common.iloc[positions[0]:positions[-1] + 1]
    if not np.isfinite(common.to_numpy()).all():
        raise ValueError(
            "Residual history contains an incomplete common period; exclude gapped assets"
        )
    return common


def _prepare_common_residuals(residuals, metadata, estimation_date, frequency, span,
                              annualisation_factor):
    """Validate native units and select complete nested intervals and common decay."""
    if (not isinstance(residuals.index, pd.DatetimeIndex)
            or not residuals.index.is_monotonic_increasing or not residuals.index.is_unique
            or residuals.index.hasnans or residuals.empty or not residuals.columns.is_unique):
        raise ValueError("Residuals require a nonempty sorted unique date grid and asset labels")
    required = {"frequency", "beta_span", "annualisation_factor", "residual_scale"}
    if (not required.issubset(metadata.columns) or not metadata.index.is_unique
            or not residuals.columns.isin(metadata.index).all()):
        raise ValueError("Residual metadata must describe every asset's frequency, span and scale")
    metadata = metadata.loc[residuals.columns].copy()
    for column in ["annualisation_factor", "residual_scale"]:
        values = metadata[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Residual metadata {column} must be finite and positive")
    lowest = metadata["annualisation_factor"].min()
    coarse = metadata.loc[metadata["annualisation_factor"] == lowest]
    if frequency is None:
        if coarse["frequency"].nunique() != 1:
            raise ValueError("Choose an explicit compatible residual covariance frequency")
        frequency = str(coarse["frequency"].iloc[0])
    matching = metadata.loc[metadata["frequency"] == frequency, "annualisation_factor"].unique()
    if annualisation_factor is None:
        if len(matching) != 1:
            raise ValueError("Supply annualisation_factor for an explicit common frequency")
        annualisation_factor = float(matching[0])
    if (not np.isfinite(annualisation_factor) or annualisation_factor <= 0
            or annualisation_factor > lowest):
        raise ValueError("Common annualisation_factor must be positive and no larger than native")
    if len(matching) and not np.allclose(matching, annualisation_factor):
        raise ValueError("Common annualisation_factor disagrees with native frequency metadata")
    if span is None:
        spans = coarse["beta_span"].unique()
        if len(spans) != 1 or pd.isna(spans[0]):
            raise ValueError(
                "Supply a residual covariance span for differing or unweighted beta spans"
            )
        _validate_span(float(spans[0]), name="beta_span")
        decay = (1 - 2 / (float(spans[0]) + 1)) ** (lowest / annualisation_factor)
        span = (1 + decay) / (1 - decay)
    _validate_span(span, name="residual_covar_span")
    cutoff = pd.Timestamp(estimation_date)
    native = residuals.loc[:cutoff]
    if native.empty:
        raise ValueError("No residual observations were available at estimation_date")
    common = _aggregate_residuals(native, metadata, frequency, cutoff)
    return common, metadata, frequency, float(span), float(annualisation_factor), cutoff


def _moment_to_correlation(moment: pd.DataFrame) -> pd.DataFrame:
    """Normalize a complete moment matrix without changing any marginal risk estimate."""
    values = moment.to_numpy(dtype=float)
    variances = np.diag(values)
    if not np.isfinite(values).all() or (variances <= 0).any():
        raise ValueError("Residual correlation requires finite positive residual variance")
    vol = np.sqrt(variances)
    corr = values / vol[:, None] / vol[None, :]
    corr = (corr + corr.T) * 0.5
    np.fill_diagonal(corr, 1.)
    return pd.DataFrame(corr, index=moment.index, columns=moment.columns)


@dataclass(frozen=True)
class ResidualCorrelationData:
    """Dimensionless common-period dependence, independently of current MATF variances.

    Native residual metadata and common raw returns document aggregation and alpha
    units. No annual covariance scale is applied to this correlation. Availability
    is the fit date, never the last observation date of a retrospectively fitted panel.
    """

    correlation: pd.DataFrame
    residual_returns: pd.DataFrame
    asset_metadata: pd.DataFrame
    frequency: str
    span: float
    observation_date: pd.Timestamp
    estimation_date: pd.Timestamp
    schema_version: ClassVar[int] = 2

    def __post_init__(self):
        """Reject malformed or indefinite dependence rather than silently repairing it."""
        corr = self.correlation
        values = corr.to_numpy(dtype=float)
        if (corr.empty or not corr.index.is_unique or not corr.index.equals(corr.columns)
                or not np.isfinite(values).all()
                or not np.allclose(values, values.T, rtol=0, atol=1e-12)
                or not np.allclose(np.diag(values), 1., rtol=0, atol=1e-12)
                or np.linalg.eigvalsh(values).min() < -1e-8):
            raise ValueError(
                "Residual correlation must be finite, symmetric, PSD with unit diagonal"
            )
        if (not corr.index.equals(self.residual_returns.columns)
                or not corr.index.equals(self.asset_metadata.index)):
            raise ValueError("Residual correlation, returns and metadata asset labels must agree")
        if self.observation_date > self.estimation_date:
            raise ValueError(
                "Residual correlation observation was not available at estimation_date"
            )
        _validate_span(self.span, name="residual_covar_span")

    @property
    def observation_count(self) -> int:
        """Number of complete common periods, including the EWMA initialization anchor."""
        return len(self.residual_returns)

    def get_corr(self, date: Optional[pd.Timestamp] = None,
                 assets: Optional[pd.Index] = None) -> pd.DataFrame:
        """Return dependence only at or after its fit/availability date."""
        if date is not None and pd.Timestamp(date) < self.estimation_date:
            raise ValueError("Residual correlation was not available at the requested date")
        return (self.correlation.copy() if assets is None
                else self.correlation.loc[assets, assets].copy())

    def filter_on_tickers(self, assets: Union[list, pd.Index, dict]) -> ResidualCorrelationData:
        """Subset and optionally rename all asset-indexed state together."""
        keys = list(assets)
        rename = assets if isinstance(assets, dict) else {}
        return replace(
            self, correlation=self.correlation.loc[keys, keys].rename(index=rename, columns=rename),
            residual_returns=self.residual_returns[keys].rename(columns=rename),
            asset_metadata=self.asset_metadata.loc[keys].rename(index=rename),
        )

    def to_sheets(self) -> Dict[str, pd.DataFrame]:
        """Persist dimensionless correlation with its grid and availability metadata."""
        info = pd.Series({
            "schema_version": self.schema_version,
            "frequency": self.frequency, "span": self.span,
            "observation_date": self.observation_date.isoformat(),
            "estimation_date": self.estimation_date.isoformat(),
            "observation_count": self.observation_count,
        }, name="value").to_frame()
        return {"residual_corr": self.correlation, "common_residuals": self.residual_returns,
                "residual_corr_metadata": self.asset_metadata, "residual_corr_info": info}

    @classmethod
    def from_sheets(cls, sheets: Dict[str, pd.DataFrame]) -> ResidualCorrelationData:
        """Restore the declared correlation schema and verify its observation count."""
        info = sheets["residual_corr_info"].iloc[:, 0]
        if info["schema_version"] != cls.schema_version:
            raise ValueError("Unsupported residual correlation schema version")
        data = cls(
            correlation=sheets["residual_corr"], residual_returns=sheets["common_residuals"],
            asset_metadata=sheets["residual_corr_metadata"], frequency=str(info["frequency"]),
            span=float(info["span"]),
            observation_date=pd.Timestamp(info["observation_date"]),
            estimation_date=pd.Timestamp(info["estimation_date"]),
        )
        if data.observation_count != int(info["observation_count"]):
            raise ValueError("Residual correlation schema observation count disagrees with returns")
        return data


def estimate_residual_correlation(
    residuals: pd.DataFrame,
    metadata: pd.DataFrame,
    estimation_date: pd.Timestamp,
    frequency: Optional[str] = None,
    span: Optional[float] = None,
    periods_per_year: Optional[float] = None,
) -> ResidualCorrelationData:
    """Prepare common-period EWMA correlation, keeping native MATF marginal variances.

    Metadata has frequency, beta_span, annualisation_factor and residual_scale per
    asset. Undo stored residual multipliers and sum only complete nested log-return
    periods. The lowest compatible native grid and its beta span are defaults.
    For a coarser explicit grid, periods_per_year converts decay, not covariance
    units. An explicit span is measured in common-grid observations. Causal EWMA
    means remove the initialization anchor before the shared second-moment kernel.

    Positive constant per-asset scaling cancels. Undefined zero-variance residuals,
    gaps, nonnested boundaries and insufficient observations fail explicitly.
    Alpha continues to use the unchanged native residual panel.
    """
    common, metadata, frequency, span, _, cutoff = _prepare_common_residuals(
        residuals, metadata, estimation_date, frequency, span, periods_per_year,
    )
    centered = (common - compute_ewm(common, span=span)).iloc[1:].to_numpy()
    # The finite EWMA mass is common to every entry and cancels in correlation.
    moment = pd.DataFrame(compute_ewm_covar(centered, span=span),
                          index=common.columns, columns=common.columns)
    return ResidualCorrelationData(
        correlation=_moment_to_correlation(moment), residual_returns=common,
        asset_metadata=metadata, frequency=frequency, span=span,
        observation_date=common.index[-1], estimation_date=cutoff,
    )
