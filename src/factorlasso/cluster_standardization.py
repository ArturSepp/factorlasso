"""Stability-pooled cross-sectional standardisation within rolling clusters.

The optional pooling modes shrink a cluster variance toward the contemporaneous
global variance using a causal co-cluster stability weight ``w``.  The default
``NONE`` mode is the unpooled within-cluster z-score.  Small groups retain the
global fallback and therefore never use pooled moments.
"""

from __future__ import annotations

from bisect import bisect_right
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd


class StabilityPoolingType(Enum):
    """Opt-in stability-pooling variants for cluster standardisation."""

    NONE = "none"
    CLUSTER_VARIANCE = "cluster_variance"
    ASSET_VARIANCE = "asset_variance"


def _last_on_or_before(dates: list[pd.Timestamp], date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the latest available date no later than ``date``."""
    position = bisect_right(dates, date)
    return dates[position - 1] if position else None


def _global_zscore(row_values: pd.Series, columns: list[str]) -> pd.Series:
    """Standardise selected values against their own cross-section."""
    if len(columns) < 2:
        return pd.Series(0.0, index=columns)
    values = row_values[columns].dropna()
    if len(values) < 2 or values.std() == 0:
        return pd.Series(0.0, index=columns)
    return (row_values[columns] - values.mean()) / values.std()


def _global_panel_score(raw_signal: pd.DataFrame) -> pd.DataFrame:
    """Apply the ungrouped fallback independently to every signal date."""
    return pd.DataFrame(
        [
            _global_zscore(raw_signal.loc[date], raw_signal.columns.tolist()).rename(date)
            for date in raw_signal.index
        ]
    )


def _stability_row(
    stability_weights: pd.DataFrame,
    stability_dates: list[pd.Timestamp],
    cluster_date: pd.Timestamp,
    columns: list[str],
) -> pd.Series:
    """Resolve and validate the causal stability weights for one partition."""
    weight_date = _last_on_or_before(stability_dates, cluster_date)
    if weight_date is None:
        return pd.Series(1.0, index=columns)
    weights = stability_weights.loc[weight_date].reindex(columns).astype(float).fillna(1.0)
    invalid = ~weights.between(0.0, 1.0)
    if invalid.any():
        raise ValueError(
            "stability weights must lie in [0, 1]; "
            f"invalid assets={weights.index[invalid].tolist()!r}"
        )
    return weights


def _pooled_cluster_score(
    row_values: pd.Series,
    columns: list[str],
    global_variance: float,
    cluster_weights: pd.Series,
    pooling_type: StabilityPoolingType,
) -> pd.Series:
    """Score one sufficiently large cluster under the requested pooled moments."""
    cluster_values = row_values[columns].dropna()
    if len(cluster_values) < 2:
        return pd.Series(0.0, index=columns)
    cluster_mean = float(cluster_values.mean())
    cluster_std = float(cluster_values.std())
    if cluster_std <= 0.0:
        return pd.Series(0.0, index=columns)

    if pooling_type == StabilityPoolingType.ASSET_VARIANCE:
        weights = cluster_weights.reindex(columns).fillna(1.0)
    else:
        weight = float(cluster_weights.mean()) if len(cluster_weights) else 1.0
        weights = pd.Series(weight, index=columns)

    if (weights == 1.0).all():
        return (row_values[columns] - cluster_mean) / cluster_std

    cluster_variance = float(cluster_values.var())
    pooled_variance = weights * cluster_variance + (1.0 - weights) * global_variance
    denominator = np.sqrt(pooled_variance)
    mean = pd.Series(cluster_mean, index=columns)
    valid = denominator > 0.0
    scored = pd.Series(0.0, index=columns)
    scored.loc[valid] = (row_values[columns].loc[valid] - mean.loc[valid]) / denominator.loc[valid]
    return scored


def score_with_stability_pooled_clusters(
    raw_signal: pd.DataFrame,
    rolling_clusters: Dict[pd.Timestamp, pd.Series],
    stability_weights: Optional[pd.DataFrame] = None,
    min_cluster_size: int = 3,
    pooling_type: StabilityPoolingType = StabilityPoolingType.NONE,
) -> pd.DataFrame:
    """Score a signal within rolling clusters with optional stability pooling.

    Parameters
    ----------
    raw_signal : pandas.DataFrame
        Dates by assets raw signal values.
    rolling_clusters : dict of pandas.Timestamp to pandas.Series
        Point-in-time cluster assignments. The latest assignment no later than
        each signal date is used.
    stability_weights : pandas.DataFrame, optional
        Dates by assets causal co-cluster weights ``w`` in ``[0, 1]``. Required
        for every pooling mode other than ``NONE``. Missing entrant weights use
        one, preserving the unpooled score until stability evidence exists.
    min_cluster_size : int
        Clusters with size at or below this threshold use global moments and
        bypass pooling.
    pooling_type : StabilityPoolingType
        Variance-only cluster weights (V1), variance-only asset weights (V2),
        or the unpooled V0 default.

    Returns
    -------
    pandas.DataFrame
        Cross-sectional scores with the input index and column order.
    """
    pooling_type = StabilityPoolingType(pooling_type)
    if pooling_type != StabilityPoolingType.NONE and stability_weights is None:
        raise ValueError("stability_weights are required when stability pooling is enabled")
    if not rolling_clusters:
        return _global_panel_score(raw_signal)

    cluster_dates = sorted(pd.Timestamp(date) for date in rolling_clusters)
    stability_dates = (
        sorted(pd.Timestamp(date) for date in stability_weights.index)
        if stability_weights is not None
        else []
    )
    all_columns = raw_signal.columns.tolist()
    scores = []
    for date in raw_signal.index:
        row_values = raw_signal.loc[date]
        cluster_date = _last_on_or_before(cluster_dates, pd.Timestamp(date))
        if cluster_date is None:
            scores.append(pd.Series(0.0, index=all_columns, name=date))
            continue
        clusters = rolling_clusters[cluster_date].dropna()
        valid_columns = [column for column in clusters.index if column in all_columns]
        clusters = clusters.loc[valid_columns]
        if len(clusters) < 2 or clusters.nunique() < 2:
            scored = _global_zscore(row_values, valid_columns)
            scores.append(scored.reindex(all_columns).fillna(0.0).rename(date))
            continue

        global_values = row_values[valid_columns].dropna()
        if len(global_values) >= 2:
            global_mean = float(global_values.mean())
            global_std = float(global_values.std())
            global_variance = float(global_values.var())
        else:
            global_mean = 0.0
            global_std = 1.0
            global_variance = 1.0
        weights = (
            _stability_row(
                stability_weights,
                stability_dates,
                cluster_date,
                valid_columns,
            )
            if stability_weights is not None
            else pd.Series(1.0, index=valid_columns)
        )
        scored_row = pd.Series(0.0, index=all_columns, name=date)
        for _, tickers in clusters.groupby(clusters).groups.items():
            columns = [column for column in tickers if column in row_values.index]
            if len(columns) <= min_cluster_size:
                if global_std > 0.0:
                    scored_row[columns] = (row_values[columns] - global_mean) / global_std
            elif pooling_type == StabilityPoolingType.NONE:
                cluster_values = row_values[columns].dropna()
                if len(cluster_values) >= 2:
                    cluster_std = cluster_values.std()
                    if cluster_std > 0.0:
                        scored_row[columns] = (
                            row_values[columns] - cluster_values.mean()
                        ) / cluster_std
            else:
                scored_row[columns] = _pooled_cluster_score(
                    row_values=row_values,
                    columns=columns,
                    global_variance=global_variance,
                    cluster_weights=weights.reindex(columns),
                    pooling_type=pooling_type,
                )
        scores.append(scored_row)
    return pd.DataFrame(scores)
