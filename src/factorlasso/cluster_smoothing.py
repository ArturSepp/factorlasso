"""Causal temporal smoothing for rolling hierarchical risk clusters.

The routines in this module operate on response-return histories and clustering
configuration only.  Every partition at date ``t`` is a deterministic function
of observations no later than ``t``.  They do not estimate factor loadings and
do not change the clustering distance, linkage, or dendrogram-cut conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as spc
from scipy.spatial.distance import squareform

from factorlasso.cluster_utils import (
    ClusterCorrelationTransform,
    _corr_to_distance,
    apply_cluster_correlation_transform,
    compute_clusters_from_corr_matrix,
)
from factorlasso.dependence_utils import DependenceMeasure, compute_dependence_matrix

if TYPE_CHECKING:
    from factorlasso.lasso_estimator import LassoModel


class ClusterSmootherType(Enum):
    """Supported causal temporal smoothers for discovered partitions."""

    NONE = 1
    HOLD = 2
    PARTITION_BONUS = 3
    SIMILARITY_EWMA = 4


@dataclass(frozen=True)
class RollingClusterData:
    """Rolling partition, dendrogram, cutoff, and optional confidence data.

    Parameters
    ----------
    clusters : dict of pandas.Timestamp to pandas.Series
        Per-date asset cluster assignments.
    linkages : dict of pandas.Timestamp to numpy.ndarray
        Per-date scipy linkage matrices.
    cutoffs : dict of pandas.Timestamp to float
        Per-date dendrogram cut distances.
    co_association : pandas.DataFrame, optional
        Dates by assets trailing six-date co-cluster confidence.
    """

    clusters: Dict[pd.Timestamp, pd.Series]
    linkages: Dict[pd.Timestamp, np.ndarray]
    cutoffs: Dict[pd.Timestamp, float]
    co_association: Optional[pd.DataFrame] = None


def smooth_similarity_ewma(
    corr: pd.DataFrame,
    prev_smoothed: Optional[pd.DataFrame],
    smoother_lambda: float,
) -> pd.DataFrame:
    """Update a causal EWMA state on a correlation matrix.

    Parameters
    ----------
    corr : pandas.DataFrame
        Current square correlation matrix.
    prev_smoothed : pandas.DataFrame, optional
        Previous smoothed state. Missing new-asset pairs are initialized from
        ``corr`` so no unavailable history is fabricated.
    smoother_lambda : float
        Weight on the prior state, in ``[0, 1)``.

    Returns
    -------
    pandas.DataFrame
        Updated symmetric similarity matrix with unit diagonal.
    """
    if not 0.0 <= smoother_lambda < 1.0:
        raise ValueError(
            f"smoother_lambda must lie in [0, 1), got {smoother_lambda!r}"
        )
    if prev_smoothed is None:
        return corr.copy()
    prior = prev_smoothed.reindex(index=corr.index, columns=corr.columns)
    state = (1.0 - smoother_lambda) * corr + smoother_lambda * prior.fillna(corr)
    values = state.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def apply_partition_distance_bonus(
    dist: np.ndarray,
    prev_clusters: pd.Series,
    delta: float,
) -> np.ndarray:
    """Discount distances for pairs co-clustered in the prior partition.

    Parameters
    ----------
    dist : numpy.ndarray
        Current square distance matrix ordered like ``prev_clusters``.
    prev_clusters : pandas.Series
        Previous labels reindexed to the current asset order. NaN denotes an
        entering asset and receives no bonus.
    delta : float
        Non-negative distance discount.

    Returns
    -------
    numpy.ndarray
        A copied distance matrix floored at zero after the discount.
    """
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative, got {delta!r}")
    values = np.asarray(dist, dtype=float).copy()
    if values.shape != (len(prev_clusters), len(prev_clusters)):
        raise ValueError(
            f"dist shape {values.shape} does not match {len(prev_clusters)} cluster labels"
        )
    labels = prev_clusters.to_numpy()
    valid = pd.notna(labels)
    same = valid[:, None] & valid[None, :] & (labels[:, None] == labels[None, :])
    values[same] = np.maximum(values[same] - delta, 0.0)
    np.fill_diagonal(values, 0.0)
    return values


def _cluster_distance_matrix(
    distance: np.ndarray,
    corr: pd.DataFrame,
    lasso_model: "LassoModel",
) -> Tuple[pd.Series, np.ndarray, float]:
    """Apply the model's unchanged linkage and cut to a square distance matrix."""
    condensed = squareform(distance, checks=False)
    linkage = spc.linkage(condensed, method=lasso_model.linkage_method)
    if lasso_model.n_clusters is None:
        cutoff = float(lasso_model.cutoff_fraction * np.max(condensed))
        labels = spc.fcluster(linkage, cutoff, criterion="distance")
    else:
        count = min(int(lasso_model.n_clusters), len(corr))
        labels = spc.fcluster(linkage, count, criterion="maxclust")
        n_merges = len(corr) - len(np.unique(labels))
        cutoff = float(linkage[n_merges - 1, 2]) if n_merges > 0 else 0.0
    return pd.Series(labels, index=corr.index), linkage, cutoff


def _correlation_input(y: pd.DataFrame, lasso_model: "LassoModel") -> pd.DataFrame:
    """Reproduce the response correlation used by ``LassoModel._prepare_fit``."""
    from factorlasso.lasso_estimator import get_x_y_np

    dummy_x = pd.DataFrame(0.0, index=y.index, columns=["__cluster_dummy__"])
    _, y_np, valid_mask = get_x_y_np(
        x=dummy_x,
        y=y,
        span=lasso_model.span,
        demean=lasso_model.demean,
    )
    y_for_corr = np.where(valid_mask > 0, y_np, np.nan)
    corr = compute_dependence_matrix(
        a=y_for_corr,
        dependence_measure=lasso_model.dependence_measure,
        span=lasso_model.span,
        gerber_threshold=lasso_model.gerber_threshold,
    )
    return pd.DataFrame(corr, index=y.columns, columns=y.columns)


def _iter_correlation_inputs(
    y: pd.DataFrame,
    dates: List[pd.Timestamp],
    lasso_model: "LassoModel",
) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame]]:
    """Yield in-fit correlations, using an exact O(TN²) Pearson recursion."""
    measure = DependenceMeasure(lasso_model.dependence_measure)
    if measure != DependenceMeasure.PEARSON or lasso_model.span is None:
        for date in dates:
            yield date, _correlation_input(y.loc[:date], lasso_model)
        return

    from factorlasso.lasso_estimator import get_x_y_np

    y_limit = y.loc[:dates[-1]]
    dummy_x = pd.DataFrame(0.0, index=y_limit.index, columns=["__cluster_dummy__"])
    _, y_np, valid_mask = get_x_y_np(
        x=dummy_x,
        y=y_limit,
        span=lasso_model.span,
        demean=lasso_model.demean,
    )
    observations = np.where(valid_mask > 0, y_np, np.nan)
    observation_index = y_limit.index[1:] if lasso_model.demean else y_limit.index
    ewm_lambda = 1.0 - 2.0 / (lasso_model.span + 1.0)
    lam1 = 1.0 - ewm_lambda
    covariance = np.zeros((len(y.columns), len(y.columns)))
    position = 0

    for date in dates:
        while position < len(observation_index) and observation_index[position] <= date:
            outer = np.outer(observations[position], observations[position])
            updated = lam1 * outer + ewm_lambda * covariance
            covariance = np.where(np.isfinite(updated), updated, covariance)
            position += 1
        diagonal = np.diag(covariance)
        if np.nansum(diagonal) > 1e-10:
            positive = diagonal > 1e-12
            inverse_vol = np.zeros_like(diagonal)
            inverse_vol[positive] = 1.0 / np.sqrt(diagonal[positive])
            corr = covariance * np.outer(inverse_vol, inverse_vol)
            np.fill_diagonal(corr, np.where(positive, np.diag(corr), 1.0))
        else:
            corr = np.identity(len(y.columns))
        yield date, pd.DataFrame(corr, index=y.columns, columns=y.columns)


def _is_recluster_date(
    date: pd.Timestamp,
    schedule_dates: List[pd.Timestamp],
    frequency: str,
) -> bool:
    """Return whether a schedule date is the last observation before an anchor."""
    anchors = pd.date_range(schedule_dates[0], schedule_dates[-1], freq=frequency)
    prior_anchor = schedule_dates[0] - pd.Timedelta(nanoseconds=1)
    for anchor in anchors:
        candidates = [item for item in schedule_dates if prior_anchor < item <= anchor]
        if candidates and date == candidates[-1]:
            return True
        prior_anchor = anchor
    return False


def _join_entrants(
    held: pd.Series,
    corr: pd.DataFrame,
) -> pd.Series:
    """Assign assets absent from a held partition by current mean correlation."""
    assigned = held.reindex(corr.index).copy()
    for asset in assigned[assigned.isna()].index:
        means = {}
        for label, members in held.groupby(held).groups.items():
            available = corr.columns.intersection(pd.Index(members))
            means[label] = float(corr.loc[asset, available].mean())
        assigned.loc[asset] = max(means, key=means.get)
    return assigned


def _co_association_panel(
    clusters: Dict[pd.Timestamp, pd.Series],
    window: int = 6,
) -> pd.DataFrame:
    """Compute trailing peer co-cluster frequency for each current assignment."""
    dates = sorted(clusters)
    rows = {}
    for i, date in enumerate(dates):
        current = clusters[date].dropna()
        history = dates[max(0, i - window + 1):i + 1]
        row = {}
        for asset, label in current.items():
            peers = current.index[(current == label) & (current.index != asset)]
            if len(peers) == 0:
                row[asset] = 1.0
                continue
            observations = []
            for prior_date in history:
                prior = clusters[prior_date]
                if asset not in prior.index or pd.isna(prior.get(asset)):
                    continue
                observations.extend(
                    prior.get(peer) == prior.get(asset)
                    for peer in peers
                    if peer in prior.index and pd.notna(prior.get(peer))
                )
            row[asset] = float(np.mean(observations)) if observations else np.nan
        rows[date] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)


def _ewma_co_association_panel(
    clusters: Dict[pd.Timestamp, pd.Series],
    span: int,
    adjust: bool,
) -> pd.DataFrame:
    """Compute EWMA peer co-cluster frequency for each current assignment."""
    dates = sorted(clusters)
    assignments = pd.DataFrame.from_dict(clusters, orient="index").reindex(dates)
    available = assignments.notna().to_numpy()
    missing = object()
    labels = assignments.astype(object).where(assignments.notna(), missing).to_numpy()
    column_positions = {asset: position for position, asset in enumerate(assignments.columns)}
    decay = 1.0 - 2.0 / (span + 1.0)
    rows = {}
    for position, date in enumerate(dates):
        current = clusters[date].dropna()
        row = {}
        for _, members in current.groupby(current).groups.items():
            members = list(members)
            if len(members) == 1:
                row[members[0]] = 1.0
                continue
            for asset in members:
                asset_position = column_positions[asset]
                peer_positions = [
                    column_positions[peer] for peer in members if peer != asset
                ]
                valid = (
                    available[:position + 1, asset_position, np.newaxis]
                    & available[:position + 1, peer_positions]
                )
                counts = valid.sum(axis=1)
                observed_dates = counts > 0
                if not observed_dates.any():
                    row[asset] = np.nan
                    continue
                matches = (
                    labels[:position + 1, peer_positions]
                    == labels[:position + 1, asset_position, np.newaxis]
                ) & valid
                observations = np.full(position + 1, np.nan, dtype=float)
                observations[observed_dates] = (
                    matches[observed_dates].sum(axis=1) / counts[observed_dates]
                )
                if adjust:
                    weights = decay ** np.arange(position, -1, -1, dtype=float)
                    row[asset] = float(
                        np.dot(observations[observed_dates], weights[observed_dates])
                        / weights[observed_dates].sum()
                    )
                else:
                    row[asset] = float(
                        pd.Series(observations).ewm(span=span, adjust=False).mean().iloc[-1]
                    )
        rows[date] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1).clip(0.0, 1.0)


def compute_co_association_panel(
    clusters: Dict[pd.Timestamp, pd.Series],
    window: int = 6,
    min_history: int = 1,
    *,
    span: Optional[int] = None,
    adjust: bool = True,
) -> pd.DataFrame:
    """Return causal trailing or EWMA peer co-cluster frequencies.

    Parameters
    ----------
    clusters : dict of pandas.Timestamp to pandas.Series
        Point-in-time asset cluster assignments.
    window : int
        Number of trailing partition dates, including the current date.
    min_history : int
        Number of available partition dates required before estimated weights
        are used. Earlier active assignments receive one, preserving unpooled
        scoring during short history.
    span : int, optional
        EWMA span over observed partition dates through each current date. If
        supplied, it replaces the flat trailing ``window`` calculation.
    adjust : bool
        Whether EWMA weights are renormalised over available history. The
        default is the pandas ``adjust=True`` convention.

    Returns
    -------
    pandas.DataFrame
        Dates by assets co-cluster frequencies in ``[0, 1]`` where observed.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    if (
        isinstance(min_history, bool)
        or not isinstance(min_history, (int, np.integer))
        or min_history <= 0
    ):
        raise ValueError(f"min_history must be a positive integer, got {min_history!r}")
    if span is not None and (
        isinstance(span, bool) or not isinstance(span, (int, np.integer)) or span <= 0
    ):
        raise ValueError(f"span must be a positive integer or None, got {span!r}")
    panel = (
        _ewma_co_association_panel(clusters, span=int(span), adjust=adjust)
        if span is not None
        else _co_association_panel(clusters, window=int(window))
    )
    for position, date in enumerate(sorted(clusters)):
        if position + 1 >= min_history:
            break
        active = clusters[date].dropna().index.intersection(panel.columns)
        panel.loc[date, active] = 1.0
    return panel


def compute_rolling_smoothed_clusters(
    y: pd.DataFrame,
    estimation_dates: List[pd.Timestamp],
    lasso_model: "LassoModel",
    eligibility: Optional[pd.DataFrame] = None,
) -> RollingClusterData:
    """Compute causal rolling clusters from response-return history.

    Parameters
    ----------
    y : pandas.DataFrame
        Response returns. Each date is truncated to ``y.loc[:date]`` before
        any correlation or smoother state is computed.
    estimation_dates : list of pandas.Timestamp
        Ordered evaluation dates. They are sorted internally for deterministic
        state transitions.
    lasso_model : LassoModel
        Declarative clustering and smoother configuration. The model is not
        fitted or mutated.  An optional ``recluster_freq`` on
        ``PARTITION_BONUS`` or ``SIMILARITY_EWMA`` updates smoother state only
        at those anchors and holds the resulting partition between them.
    eligibility : pandas.DataFrame, optional
        Exact point-in-time asset eligibility. It must have Boolean values,
        contain every estimation date, and have exactly the columns of ``y``.
        It is intersected with the model's data-warmup mask. No membership is
        forward-filled or inferred. The asset submatrix is selected before an
        optional common-mode transform and before temporal smoothing.

    Returns
    -------
    RollingClusterData
        Per-date partitions and dendrogram metadata plus trailing confidence.
    """
    dates = [pd.Timestamp(date) for date in sorted(estimation_dates)]
    if eligibility is not None:
        if not isinstance(eligibility, pd.DataFrame):
            raise ValueError("eligibility must be a pandas DataFrame or None")
        if not eligibility.columns.equals(y.columns):
            raise ValueError("eligibility must have exactly the columns of y in the same order")
        if not eligibility.index.is_unique:
            raise ValueError("eligibility dates must be unique")
        missing_dates = [date for date in dates if date not in eligibility.index]
        if missing_dates:
            raise ValueError(
                f"eligibility is missing {len(missing_dates)} estimation dates: "
                f"{missing_dates[:3]!r}"
            )
        if eligibility.isna().any().any() or not all(
            pd.api.types.is_bool_dtype(dtype) for dtype in eligibility.dtypes
        ):
            raise ValueError("eligibility values must be non-missing Boolean values")
    if not dates:
        return RollingClusterData(
            clusters={}, linkages={}, cutoffs={}, co_association=pd.DataFrame()
        )

    clusters: Dict[pd.Timestamp, pd.Series] = {}
    linkages: Dict[pd.Timestamp, np.ndarray] = {}
    cutoffs: Dict[pd.Timestamp, float] = {}
    previous_clusters: Optional[pd.Series] = None
    previous_similarity: Optional[pd.DataFrame] = None
    held: Optional[Tuple[pd.Series, np.ndarray, float]] = None
    held_eligible: Optional[pd.Series] = None

    transform = ClusterCorrelationTransform(
        lasso_model.cluster_correlation_transform
    )
    restrict_universe = (
        eligibility is not None or transform != ClusterCorrelationTransform.NONE
    )

    for date, corr in _iter_correlation_inputs(y, dates, lasso_model):
        y_current = y.loc[:date]
        if y_current.empty:
            raise ValueError(f"no response observations at or before {date!r}")
        if lasso_model.warmup_period is None:
            eligible = corr.index
        else:
            eligible = y_current.columns[
                y_current.notna().sum() >= lasso_model.warmup_period
            ]
        if eligibility is not None:
            eligible_at_date = eligibility.loc[date]
            eligible = pd.Index(
                [asset for asset in eligible if bool(eligible_at_date.loc[asset])]
            )
        if len(eligible) == 0:
            raise ValueError(f"no eligible response assets at {date!r}")
        if restrict_universe:
            corr = corr.reindex(index=eligible, columns=eligible)
        corr = apply_cluster_correlation_transform(corr, transform=transform)
        smoother = ClusterSmootherType(lasso_model.cluster_smoother_type)
        is_scheduled = lasso_model.recluster_freq is not None
        update_partition = not is_scheduled or held is None or _is_recluster_date(
            date, dates, str(lasso_model.recluster_freq)
        )

        if not update_partition:
            assert held is not None
            assert held_eligible is not None
            partition = held[0].copy()
            assigned = _join_entrants(
                held_eligible,
                corr.reindex(index=eligible, columns=eligible),
            )
            if restrict_universe:
                partition = assigned
            else:
                partition.loc[assigned.index] = assigned
            bundle = partition, held[1], held[2]
            held = bundle
            held_eligible = assigned
        elif smoother == ClusterSmootherType.HOLD:
            bundle = compute_clusters_from_corr_matrix(
                corr,
                cutoff_fraction=lasso_model.cutoff_fraction,
                linkage_method=lasso_model.linkage_method,
                distance_transform=lasso_model.distance_transform,
                n_clusters=lasso_model.n_clusters,
            )
        elif smoother == ClusterSmootherType.PARTITION_BONUS and previous_clusters is not None:
            distance = _corr_to_distance(
                corr.fillna(0.0).to_numpy(),
                distance_transform=lasso_model.distance_transform,
            )
            distance = apply_partition_distance_bonus(
                distance,
                previous_clusters.reindex(corr.index),
                lasso_model.smoother_delta,
            )
            bundle = _cluster_distance_matrix(distance, corr, lasso_model)
        elif smoother == ClusterSmootherType.SIMILARITY_EWMA:
            previous_similarity = smooth_similarity_ewma(
                corr, previous_similarity, lasso_model.smoother_lambda
            )
            bundle = compute_clusters_from_corr_matrix(
                previous_similarity,
                cutoff_fraction=lasso_model.cutoff_fraction,
                linkage_method=lasso_model.linkage_method,
                distance_transform=lasso_model.distance_transform,
                n_clusters=lasso_model.n_clusters,
            )
        else:
            bundle = compute_clusters_from_corr_matrix(
                corr,
                cutoff_fraction=lasso_model.cutoff_fraction,
                linkage_method=lasso_model.linkage_method,
                distance_transform=lasso_model.distance_transform,
                n_clusters=lasso_model.n_clusters,
            )

        if is_scheduled and update_partition:
            held = bundle
            held_eligible = bundle[0].reindex(eligible)
        clusters[date], linkages[date], cutoffs[date] = bundle
        previous_clusters = bundle[0].reindex(eligible)

    confidence = _co_association_panel(clusters)
    return RollingClusterData(
        clusters=clusters,
        linkages=linkages,
        cutoffs=cutoffs,
        co_association=confidence,
    )
