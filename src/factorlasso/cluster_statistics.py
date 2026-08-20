"""Causal stability statistics for rolling operating partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from factorlasso.cluster_smoothing import compute_co_association_panel


def _normalise_frequency(value: object) -> str:
    """Return the supported canonical estimation-frequency label."""
    raw = getattr(value, "value", value)
    label = str(raw).upper().replace("_", "-")
    if label in {"M", "ME", "MONTHLY"} or label.startswith("ME-"):
        return "ME"
    if label in {"Q", "QE", "QUARTERLY"} or label.startswith(("Q-", "QE-")):
        return "QE"
    raise ValueError(f"unsupported partition frequency {value!r}; expected ME or QE")


def _infer_partition_frequency(dates: list[pd.Timestamp]) -> str:
    """Infer ME or QE from a regular month-end partition schedule."""
    if len(dates) < 2:
        raise ValueError("at least two partition dates are required to infer frequency")
    month_numbers = np.array([date.year * 12 + date.month for date in dates], dtype=int)
    month_steps = np.diff(month_numbers)
    if np.all(month_steps == 1):
        return "ME"
    if np.all(month_steps == 3):
        return "QE"
    inferred = pd.infer_freq(pd.DatetimeIndex(dates)) if len(dates) >= 3 else None
    if inferred is not None:
        return _normalise_frequency(inferred)
    raise ValueError("partition dates must form a regular ME or QE schedule")


def _validate_span_map(span_by_freq: Mapping[object, int]) -> dict[str, int]:
    """Validate and canonicalise the explicit per-frequency span map."""
    spans = {}
    for frequency, span in span_by_freq.items():
        key = _normalise_frequency(frequency)
        if isinstance(span, bool) or not isinstance(span, (int, np.integer)) or span <= 0:
            raise ValueError(f"span for {key} must be a positive integer, got {span!r}")
        spans[key] = int(span)
    if not spans:
        raise ValueError("span_by_freq must contain at least one span")
    return spans


def _cluster_weight_panel(
    partitions: Mapping[pd.Timestamp, pd.Series],
    asset_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate asset stability to each date's current clusters."""
    rows = {}
    for date, assignment in partitions.items():
        active = assignment.dropna()
        weights = asset_weights.loc[date].reindex(active.index)
        rows[date] = {
            cluster_id: float(weights.reindex(members).mean())
            for cluster_id, members in active.groupby(active).groups.items()
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index().sort_index(axis=1)


def _coverage_frame(
    partitions: Mapping[pd.Timestamp, pd.Series],
    asset_weights: pd.DataFrame,
    cluster_weights: pd.DataFrame,
    min_history: int,
) -> pd.DataFrame:
    """Summarise asset and cluster stability coverage by partition date."""
    rows = []
    for position, (date, assignment) in enumerate(partitions.items()):
        active = assignment.dropna()
        active_clusters = pd.Index(active.unique())
        observed_assets = int(asset_weights.loc[date].reindex(active.index).notna().sum())
        observed_clusters = int(
            cluster_weights.loc[date].reindex(active_clusters).notna().sum()
        )
        rows.append(
            {
                "date": date,
                "active_assets": len(active),
                "observed_weights": observed_assets,
                "coverage": observed_assets / len(active) if len(active) else np.nan,
                "active_clusters": len(active_clusters),
                "observed_cluster_weights": observed_clusters,
                "cluster_coverage": (
                    observed_clusters / len(active_clusters) if len(active_clusters) else np.nan
                ),
                "short_history_fallback": position + 1 < min_history,
            }
        )
    return pd.DataFrame(rows).set_index("date")


@dataclass(frozen=True)
class ClusterStabilityStatistics:
    """Precomputed EWMA stability weights and diagnostics for one partition panel."""

    partitions: Dict[pd.Timestamp, pd.Series]
    frequency: str
    span: int
    min_history: int
    w_i: pd.DataFrame
    w_g: pd.DataFrame
    coverage: pd.DataFrame

    def _long_frame(self, membership: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return date-asset stability records used by the public diagnostics."""
        prior_membership = membership.shift(1) if membership is not None else None
        rows = []
        for position, (date, assignment) in enumerate(self.partitions.items()):
            active = assignment.dropna()
            for cluster_id, members in active.groupby(active).groups.items():
                members = pd.Index(members)
                cluster_weight = self.w_g.loc[date, cluster_id]
                for asset in members:
                    derived = (
                        membership.loc[date, asset]
                        if membership is not None
                        and date in membership.index
                        and asset in membership.columns
                        else np.nan
                    )
                    prior = (
                        prior_membership.loc[date, asset]
                        if prior_membership is not None
                        and date in prior_membership.index
                        and asset in prior_membership.columns
                        else np.nan
                    )
                    rows.append(
                        {
                            "date": date,
                            "asset": asset,
                            "cluster_id": cluster_id,
                            "cluster_size": len(members),
                            "w_asset": self.w_i.loc[date, asset],
                            "w_cluster": cluster_weight,
                            "derived_cluster_id": derived,
                            "reassigned": bool(
                                pd.notna(derived) and pd.notna(prior) and derived != prior
                            ),
                            "short_history_fallback": position + 1 < self.min_history,
                        }
                    )
        return pd.DataFrame(rows)

    def boundary_statistics(self, membership: pd.DataFrame) -> pd.DataFrame:
        """Return reassignment rates for the bottom and top stability quartiles."""
        long = self._long_frame(membership)
        estimated = long.loc[long["w_asset"] < 1.0].copy()
        if estimated.empty:
            estimated = long.copy()
        lower = estimated["w_asset"].quantile(0.25)
        upper = estimated["w_asset"].quantile(0.75)
        return pd.DataFrame(
            [
                {
                    "frequency": self.frequency,
                    "span": self.span,
                    "mean_w_reassigned": estimated.loc[
                        estimated["reassigned"], "w_asset"
                    ].mean(),
                    "mean_w_stable": estimated.loc[
                        ~estimated["reassigned"], "w_asset"
                    ].mean(),
                    "reassignment_rate_bottom_w_quartile": estimated.loc[
                        estimated["w_asset"] <= lower, "reassigned"
                    ].mean(),
                    "reassignment_rate_top_w_quartile": estimated.loc[
                        estimated["w_asset"] >= upper, "reassigned"
                    ].mean(),
                    "bottom_w_cut": lower,
                    "top_w_cut": upper,
                }
            ]
        )

    def size_vs_w_correlation(self) -> pd.DataFrame:
        """Return the cross-cluster size-versus-stability correlation by date."""
        rows = []
        for date, assignment in self.partitions.items():
            active = assignment.dropna()
            groups = pd.DataFrame(
                [
                    {
                        "cluster_id": cluster_id,
                        "cluster_size": len(members),
                        "cluster_weight": self.w_g.loc[date, cluster_id],
                    }
                    for cluster_id, members in active.groupby(active).groups.items()
                ]
            )
            correlation = (
                groups["cluster_size"].corr(groups["cluster_weight"])
                if len(groups) >= 2
                else np.nan
            )
            rows.append(
                {
                    "date": date,
                    "clusters": len(groups),
                    "size_w_correlation": correlation,
                    "abs_correlation_gt_0_5": bool(
                        pd.notna(correlation) and abs(correlation) > 0.5
                    ),
                }
            )
        return pd.DataFrame(rows)

    def within_cluster_asset_w_dispersion(self) -> pd.DataFrame:
        """Return within-cluster cross-asset stability dispersion by date."""
        long = self._long_frame()
        estimated = long.loc[~long["short_history_fallback"]]
        return (
            estimated.groupby(["date", "cluster_id"], as_index=False)
            .agg(
                cluster_size=("cluster_size", "first"),
                w_asset_std=("w_asset", "std"),
            )
        )


def compute_cluster_stability_statistics(
    partitions: Mapping[pd.Timestamp, pd.Series],
    span_by_freq: Mapping[object, int],
    min_history: int = 12,
) -> ClusterStabilityStatistics:
    """Compute one causal EWMA stability object from rolling partitions.

    Parameters
    ----------
    partitions : mapping of pandas.Timestamp to pandas.Series
        Rolling operating partitions for a single regular estimation cadence.
    span_by_freq : mapping
        Explicit EWMA spans keyed by ``ME`` and/or ``QE``. The active span is
        resolved from the partition schedule, never from an estimator model.
    min_history : int
        Observed partition dates required before estimated weights are used.

    Returns
    -------
    ClusterStabilityStatistics
        Shared asset/cluster weights, coverage, and diagnostic methods.
    """
    if not partitions:
        raise ValueError("partitions must not be empty")
    if (
        isinstance(min_history, bool)
        or not isinstance(min_history, (int, np.integer))
        or min_history <= 0
    ):
        raise ValueError(f"min_history must be a positive integer, got {min_history!r}")
    ordered = {
        pd.Timestamp(date): assignment.copy()
        for date, assignment in sorted(partitions.items())
    }
    if any(not isinstance(assignment, pd.Series) for assignment in ordered.values()):
        raise ValueError("every partition must be a pandas Series")
    spans = _validate_span_map(span_by_freq)
    frequency = _infer_partition_frequency(list(ordered))
    if frequency not in spans:
        raise ValueError(f"span_by_freq has no entry for inferred frequency {frequency}")
    span = spans[frequency]
    asset_weights = compute_co_association_panel(
        ordered,
        min_history=int(min_history),
        span=span,
        adjust=True,
    )
    cluster_weights = _cluster_weight_panel(ordered, asset_weights)
    coverage = _coverage_frame(
        ordered,
        asset_weights,
        cluster_weights,
        int(min_history),
    )
    return ClusterStabilityStatistics(
        partitions=ordered,
        frequency=frequency,
        span=span,
        min_history=int(min_history),
        w_i=asset_weights,
        w_g=cluster_weights,
        coverage=coverage,
    )
