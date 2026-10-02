"""Shared kernels of response clustering: prepared-panel dependence, and linkage with a cut.

The in-fit clustering of :class:`~factorlasso.LassoModel` and the causal rolling partitions of
:func:`~factorlasso.compute_rolling_smoothed_clusters` compute the same quantities from
differently prepared observations. The fit masks a response observation when its whole factor
row is missing; the rolling path has no factors and masks only missing responses. Each caller
keeps its own preparation and passes the prepared panel and its validity mask here, so equal
inputs give equal outputs and intentionally different inputs stay different.

Two small settings records carry what the kernels need. They use the attribute names of
:class:`~factorlasso.LassoModel`, so a model can be passed wherever a record is expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as spc
from scipy.spatial.distance import squareform

from factorlasso.cluster._dependence import compute_dependence_matrix


@dataclass(frozen=True)
class _ResponseDependenceSettings:
    """How a response panel is prepared and its dependence measured."""
    span: Optional[float]
    cluster_correlation_span: Optional[float]
    demean: bool
    dependence_measure: str
    gerber_threshold: float

    @classmethod
    def from_model(cls, model) -> "_ResponseDependenceSettings":
        """The settings of an estimator (or of another record with the same attributes)."""
        return cls(span=model.span, cluster_correlation_span=model.cluster_correlation_span,
                   demean=model.demean, dependence_measure=model.dependence_measure,
                   gerber_threshold=model.gerber_threshold)


@dataclass(frozen=True)
class _ClusterGeometry:
    """Distance transform, linkage and cut of a partition."""
    cutoff_fraction: float
    linkage_method: str
    distance_transform: object
    n_clusters: Optional[int]

    @classmethod
    def from_model(cls, model) -> "_ClusterGeometry":
        """The geometry of an estimator (or of another record with the same attributes)."""
        return cls(cutoff_fraction=model.cutoff_fraction, linkage_method=model.linkage_method,
                   distance_transform=model.distance_transform, n_clusters=model.n_clusters)

    def as_kwargs(self) -> dict:
        """Keyword arguments of :func:`~factorlasso.compute_clusters_from_corr_matrix`."""
        return dict(cutoff_fraction=self.cutoff_fraction, linkage_method=self.linkage_method,
                    distance_transform=self.distance_transform, n_clusters=self.n_clusters)


def prepared_response_dependence(
    y_np: np.ndarray,
    valid_mask: np.ndarray,
    columns: pd.Index,
    dependence_measure,
    span: Optional[float],
    gerber_threshold: float,
) -> pd.DataFrame:
    """Dependence matrix of a prepared, zero-filled response panel.

    Masked cells are restored to NaN first: a zero-filled cell is a solver convenience, not a
    zero return, and the NaN-aware recursions then estimate each pair over its valid window.
    """
    observations = np.where(valid_mask > 0, y_np, np.nan)
    corr = compute_dependence_matrix(
        a=observations,
        dependence_measure=dependence_measure,
        span=span,
        gerber_threshold=gerber_threshold,
    )
    return pd.DataFrame(corr, index=columns, columns=columns)


def linkage_and_cut(
    distance: np.ndarray,
    n_assets: int,
    linkage_method: str,
    cutoff_fraction: float,
    n_clusters: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Agglomerate a square distance matrix and cut the dendrogram.

    Without ``n_clusters`` the tree is cut at ``cutoff_fraction`` of the largest pairwise
    distance. With it, SciPy's ``maxclust`` criterion yields at most ``n_clusters`` groups
    (callers clamp the count to the universe), and the reported cutoff is the height of the
    last accepted merge, so both branches report comparable heights.

    Returns
    -------
    labels : numpy.ndarray
        1-indexed cluster labels in the order of ``distance``.
    linkage : numpy.ndarray
        SciPy linkage matrix.
    cutoff : float
        The cut height: ``cutoff_fraction * max(pdist)`` (a NumPy scalar) or the last merge
        height.
    """
    condensed = squareform(distance, checks=False)
    linkage = spc.linkage(condensed, method=linkage_method)
    if n_clusters is None:
        cutoff = cutoff_fraction * np.max(condensed)
        labels = spc.fcluster(linkage, cutoff, criterion="distance")
    else:
        labels = spc.fcluster(linkage, n_clusters, criterion="maxclust")
        # With k realised clusters over n assets, exactly n - k merges were accepted, and
        # ``linkage`` is ordered by increasing height.
        n_merges = n_assets - len(np.unique(labels))
        cutoff = float(linkage[n_merges - 1, 2]) if n_merges > 0 else 0.0
    return labels, linkage, cutoff
