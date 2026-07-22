"""
Clustering utilities for HCGL factor covariance estimation.

This module houses pure utility functions around hierarchical clustering
of asset universes: distance-matrix construction, Ward's-method
clustering, and helpers for splitting and reconstructing the flat
per-freq-prefixed containers that ``CurrentFactorCovarData`` uses for
persistence.

Functions
---------
compute_clusters_from_corr_matrix
    Ward's hierarchical clustering from a correlation matrix. The
    primitive used by HCGL (``LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO``)
    but callable independently for any group-discovery workflow.

DistanceTransform
    String enum selecting the correlation-to-distance transform applied
    before linkage: ``ONE_MINUS_RHO`` (``1 - rho``, default), ``CHORD``
    (``sqrt(2(1 - rho))``, the Euclidean chord under which Ward's variance
    criterion is exact), or ``ARCCOS`` (``arccos(rho)``, the geodesic arc).

get_linkage_array
    Extract the scipy linkage ndarray for a single frequency from the
    stacked linkage DataFrame stored in ``CurrentFactorCovarData``.

get_clusters_by_freq, get_linkages_by_freq, get_cutoffs_by_freq
    Split the flat per-freq-prefixed clusters / linkages / cutoffs
    containers back into per-frequency dicts. Primary use case:
    extracting clusters from a reference-currency CMA run to feed as
    ``precomputed_clusters`` into a subsequent CMA run (e.g., USD-anchored
    clustering for non-USD reference currencies).

Design note
-----------
These functions live in their own module because they are useful
independently of any particular LassoModel or CurrentFactorCovarData
workflow. ``lasso_estimator.py`` and ``factor_covar.py`` previously
each held one of these functions, which created cross-dependencies
and blurred the boundary between "solver" and "diagnostic utility".
The module-level split is cleaner and keeps the import graph acyclic.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import squareform

# ═══════════════════════════════════════════════════════════════════════
# Clustering from correlation
# ═══════════════════════════════════════════════════════════════════════


class DistanceTransform(str, Enum):
    """correlation-to-distance transform for hierarchical clustering.

    All three transforms are strictly decreasing monotone functions of the
    correlation ``rho``, so single and complete linkage (rank-based) build
    the identical merge tree under any of them. Ward, average, centroid,
    and median linkage read distance magnitudes, so the tree and the
    partition react to the choice.

    - ``ONE_MINUS_RHO``: ``d = 1 - rho``. The correlation dissimilarity and
      the package default. Not a proper metric (the triangle inequality can
      fail), applied under Ward as a stable clustering heuristic.
    - ``CHORD``: ``d = sqrt(2 (1 - rho))``. The exact Euclidean distance
      between unit-norm standardised return vectors
      (``||u_i - u_j||^2 = 2 (1 - rho_ij)``), so Ward's variance criterion
      is exact under this transform (Mantegna 1999).
    - ``ARCCOS``: ``d = arccos(rho)``. The great-circle (geodesic) angle
      between the same vectors. A proper metric, but not
      Euclidean-embeddable in general, so under Ward it is a heuristic like
      ``ONE_MINUS_RHO``.
    """
    ONE_MINUS_RHO = 'one_minus_rho'   # d = 1 - rho          (dissimilarity, default)
    CHORD = 'chord'                   # d = sqrt(2(1 - rho)) (Euclidean chord, Ward-valid)
    ARCCOS = 'arccos'                 # d = arccos(rho)      (geodesic arc)


# Correlation-to-distance transform applied before linkage. ONE_MINUS_RHO
# is the package-wide default and reproduces the pre-0.9.0 hardcoded
# ``1 - rho`` path exactly. The cutoff calibration note in
# ``compute_clusters_from_corr_matrix`` applies when switching.
DEFAULT_DISTANCE_TRANSFORM: DistanceTransform = DistanceTransform.ONE_MINUS_RHO


def _corr_to_distance(
    corr: np.ndarray,
    distance_transform: DistanceTransform = DEFAULT_DISTANCE_TRANSFORM,
) -> np.ndarray:
    """map a correlation matrix to a square distance matrix under the chosen transform.

    Parameters
    ----------
    corr : np.ndarray, shape (N, N)
        Correlation matrix. Entries are clipped to ``[-1, 1]`` first, which
        guards against floating-point overshoot at ``rho = 1`` and keeps
        ``arccos`` inside its domain.
    distance_transform : DistanceTransform
        ``ONE_MINUS_RHO`` -> ``1 - rho``; ``CHORD`` -> ``sqrt(2 (1 - rho))``;
        ``ARCCOS`` -> ``arccos(rho)``.

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric distance matrix with an exact zero diagonal, ready for
        ``scipy.spatial.distance.squareform``.

    Raises
    ------
    ValueError
        If ``distance_transform`` is not a member of ``DistanceTransform``.
    """
    try:
        distance_transform = DistanceTransform(distance_transform)
    except ValueError:
        raise ValueError(
            f"distance_transform must be one of "
            f"{[t.value for t in DistanceTransform]}, "
            f"got {distance_transform!r}"
        ) from None
    rho = np.clip(corr, -1.0, 1.0)
    if distance_transform == DistanceTransform.ONE_MINUS_RHO:
        d = 1.0 - rho
    elif distance_transform == DistanceTransform.CHORD:
        d = np.sqrt(np.clip(2.0 * (1.0 - rho), 0.0, None))
    else:  # DistanceTransform.ARCCOS
        d = np.arccos(rho)
    np.fill_diagonal(d, 0.0)
    return d

# Default fraction of ``max(pdist)`` at which to cut the dendrogram.
# 0.5 is the package-wide convention — half the maximum pairwise
# correlation-distance — and empirically produces ~15–25 clusters on
# a 150-asset multi-asset universe.  Callers that need a different
# granularity can override via ``cutoff_fraction``.
DEFAULT_CUTOFF_FRACTION: float = 0.5

# Agglomerative linkage method passed to ``scipy.cluster.hierarchy.linkage``.
# Ward is the package default and the setting used throughout the MATF-CMA
# pipeline.  The Euclidean-oriented methods (``ward``, ``centroid``,
# ``median``) are applied here to a correlation dissimilarity as stable
# clustering heuristics, not as exact variance minimisation in Euclidean
# space (see the Notes of ``compute_clusters_from_corr_matrix``).
DEFAULT_LINKAGE_METHOD: str = 'ward'
VALID_LINKAGE_METHODS: Tuple[str, ...] = (
    'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward',
)


def compute_clusters_from_corr_matrix(
    corr_matrix: pd.DataFrame,
    cutoff_fraction: float = DEFAULT_CUTOFF_FRACTION,
    linkage_method: str = DEFAULT_LINKAGE_METHOD,
    distance_transform: Union[DistanceTransform, str] = DEFAULT_DISTANCE_TRANSFORM,
) -> Tuple[pd.Series, np.ndarray, float]:
    """
    Hierarchical clustering from a correlation matrix (Ward's method).

    Converts correlation to distance under ``distance_transform``
    (default ``1 − corr``), applies Ward's agglomerative clustering, and
    cuts the dendrogram at ``cutoff_fraction × max(pairwise distance)``.

    Parameters
    ----------
    corr_matrix : pd.DataFrame, shape (N, N)
        Square correlation matrix.
    cutoff_fraction : float, default 0.5
        Fraction of ``max(pdist)`` at which to cut the dendrogram.
        Must lie in ``(0, 1]``.  Smaller values produce more, tighter
        clusters; larger values produce fewer, looser clusters.  The
        default ``0.5`` is the canonical setting used throughout the
        MATF-CMA pipeline and typically yields ~15–25 clusters on a
        150-asset multi-asset universe.
    linkage_method : str, default 'ward'
        Agglomerative linkage method passed to
        ``scipy.cluster.hierarchy.linkage``.  One of ``'single'``,
        ``'complete'``, ``'average'``, ``'weighted'``, ``'centroid'``,
        ``'median'``, or ``'ward'``.  The default ``'ward'`` reproduces the
        prior behaviour exactly.
    distance_transform : DistanceTransform or str, default ONE_MINUS_RHO
        Correlation-to-distance transform applied before linkage.  One of
        ``DistanceTransform.ONE_MINUS_RHO`` (``d = 1 - rho``),
        ``DistanceTransform.CHORD`` (``d = sqrt(2 (1 - rho))``, the exact
        Euclidean chord under which Ward's variance criterion is valid), or
        ``DistanceTransform.ARCCOS`` (``d = arccos(rho)``, the geodesic
        arc).  Plain strings (``'chord'``) are accepted.  The default
        reproduces the pre-0.9.0 behaviour exactly.

        .. warning::
            ``cutoff_fraction`` is calibrated per transform and does not
            port across transforms.  The transforms remap merge heights
            nonlinearly, so a shared fraction changes the partition
            granularity — on mostly-positive correlation panels,
            ``CHORD`` or ``ARCCOS`` at the ``ONE_MINUS_RHO``-calibrated
            ``0.5`` can shatter the partition into near-singletons.  To
            preserve the implied pairwise merge threshold when switching
            from ``ONE_MINUS_RHO`` at fraction ``f``, use
            ``sqrt(f)`` under ``CHORD`` (panel-independent), and
            ``arccos(rho*) / arccos(rho_min)`` with
            ``rho* = 1 - f (1 - rho_min)`` under ``ARCCOS``
            (panel-dependent through the minimum off-diagonal
            correlation ``rho_min``).  At matched granularity the three
            transforms typically produce identical partitions on block
            correlation structures.

    Returns
    -------
    clusters : pd.Series
        Cluster labels (1-indexed) for each column.
    linkage : np.ndarray
        Scipy linkage matrix.
    cutoff : float
        Absolute distance threshold used for cutting
        (``cutoff_fraction × max(pdist)``).

    Notes
    -----
    The condensed distance vector is built via ``squareform(1 - C)``,
    which is the standard correlation-dissimilarity path from a
    correlation matrix to the condensed
    pairwise-distance vector that ``scipy.cluster.hierarchy.linkage``
    expects. A previous implementation passed ``pdist(1 - corr)``, which
    treated rows of ``(1 - corr)`` as observations in N-dimensional space
    and computed Euclidean distances between those rows — a different
    (non-semantic) metric that conflated correlation structure with
    higher-order geometric relationships.

    Note that the default ``1 - rho`` is a correlation dissimilarity, not
    the Euclidean chord distance ``sqrt(2(1 - rho))`` between standardised
    return vectors. Under the default, Ward linkage is applied as a stable
    correlation-clustering heuristic, not as exact Ward variance
    minimisation in Euclidean space. Pass
    ``distance_transform=DistanceTransform.CHORD`` for the metric-exact
    Ward criterion (mind the ``cutoff_fraction`` calibration warning
    above).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from factorlasso import compute_clusters_from_corr_matrix
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 6))
    >>> X[:, 1] = X[:, 0] + 0.1 * rng.standard_normal(200)  # 0 & 1 correlated
    >>> C = pd.DataFrame(np.corrcoef(X, rowvar=False),
    ...                  columns=list("abcdef"), index=list("abcdef"))
    >>> clusters, _, cutoff = compute_clusters_from_corr_matrix(C)
    >>> clusters.loc['a'] == clusters.loc['b']
    True
    """
    if not (0.0 < cutoff_fraction <= 1.0):
        raise ValueError(
            f"cutoff_fraction must lie in (0, 1], got {cutoff_fraction!r}"
        )
    if linkage_method not in VALID_LINKAGE_METHODS:
        raise ValueError(
            f"linkage_method must be one of {VALID_LINKAGE_METHODS}, "
            f"got {linkage_method!r}"
        )
    try:
        distance_transform = DistanceTransform(distance_transform)
    except ValueError:
        raise ValueError(
            f"distance_transform must be one of "
            f"{[t.value for t in DistanceTransform]}, "
            f"got {distance_transform!r}"
        ) from None
    corr_matrix = corr_matrix.fillna(0.0)
    # A single asset is trivially its own cluster. SciPy's squareform/linkage
    # are undefined for one observation — the condensed pairwise-distance
    # vector is empty and ``spc.linkage`` then raises "The number of
    # observations cannot be determined on an empty distance matrix". Short-
    # circuit instead of letting that propagate: return the lone asset in
    # cluster 1, an empty linkage (scipy's ``(n - 1, 4)`` linkage format has
    # zero rows for n = 1), and a zero cutoff. This keeps the function total —
    # callers that build group loadings receive a valid one-element Series
    # rather than an exception or None — which unblocks any fit whose
    # frequency bucket holds exactly one asset (e.g. a mandate whose sole
    # quarterly-rebalanced sleeve is a single hedge-fund proxy).
    if corr_matrix.shape[0] == 1:
        clusters = pd.Series(1, index=corr_matrix.columns)
        return clusters, np.empty((0, 4)), 0.0
    # ``_corr_to_distance`` maps the correlation matrix to a square
    # distance matrix under the chosen transform (default ``1 - rho``,
    # numerically identical to the pre-0.9.0 hardcoded path: clipping
    # ``rho`` to [-1, 1] before ``1 - rho`` equals clipping ``1 - rho``
    # to [0, 2]). It clips, zeroes the diagonal as ``squareform``
    # requires, and validates the transform.
    dist_square = _corr_to_distance(
        corr_matrix.to_numpy(), distance_transform=distance_transform,
    )
    pdist = squareform(dist_square, checks=False)
    linkage = spc.linkage(pdist, method=linkage_method)
    cutoff = cutoff_fraction * np.max(pdist)
    idx = spc.fcluster(linkage, cutoff, 'distance')
    clusters = pd.Series(idx, index=corr_matrix.columns)
    return clusters, linkage, cutoff


# ═══════════════════════════════════════════════════════════════════════
# Flat-container helpers for CurrentFactorCovarData
# ═══════════════════════════════════════════════════════════════════════


def get_linkage_array(
    linkages: pd.DataFrame,
    freq: str,
) -> np.ndarray:
    """
    Slice the stacked linkage DataFrame back to a scipy-compatible ndarray.

    ``CurrentFactorCovarData.linkages`` stacks per-freq linkage matrices
    into one DataFrame with a freq-prefixed index (e.g. ``"ME:step_0"``,
    ``"QE:step_0"``). SciPy's dendrogram / fcluster routines expect a raw
    ``(K, 4)`` ndarray. This helper extracts the rows for one frequency
    and returns them as a plain array, suitable for passing directly to
    ``scipy.cluster.hierarchy.dendrogram`` or ``fcluster``.

    Parameters
    ----------
    linkages : pd.DataFrame
        Stacked linkage matrix with columns
        ``['left', 'right', 'distance', 'n_samples']`` and freq-prefixed
        index. Typically ``covar_data.linkages``.
    freq : str
        Frequency code to extract, e.g. ``'ME'`` or ``'QE'``.

    Returns
    -------
    np.ndarray, shape (K, 4)
        Linkage matrix for the requested frequency.

    Raises
    ------
    KeyError
        If no rows match the given freq prefix.
    """
    mask = linkages.index.astype(str).str.startswith(f"{freq}:")
    if not mask.any():
        available = sorted(
            set(linkages.index.astype(str).str.split(':').str[0])
        )
        raise KeyError(
            f"No linkage rows found for freq={freq!r}. "
            f"Available: {available}"
        )
    return linkages.loc[mask, ['left', 'right', 'distance', 'n_samples']].to_numpy()


def get_clusters_by_freq(
    clusters: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Split a flat freq-prefixed cluster Series into per-frequency dicts.

    ``CurrentFactorCovarData.clusters`` stores cluster labels as a flat
    asset-indexed Series with values like ``'ME:12'``, ``'QE:4'``. To use
    these as ``precomputed_clusters`` in a subsequent call to
    ``estimate_lasso_factor_covar_data``, they must be split back into
    per-frequency dicts with the freq prefix stripped.

    Parameters
    ----------
    clusters : pd.Series
        Asset-indexed Series of freq-prefixed cluster IDs
        (e.g. ``{'MSCI USA': 'ME:5', 'PE MSCI': 'QE:4', ...}``).

    Returns
    -------
    Dict[str, pd.Series]
        Dict mapping each frequency code to the subset of assets with
        that frequency, with cluster labels stripped of the freq prefix.
        E.g. ``{'ME': {'MSCI USA': '5', ...}, 'QE': {'PE MSCI': '4', ...}}``.

    Examples
    --------
    >>> clusters = pd.Series({
    ...     'asset_a': 'ME:1', 'asset_b': 'ME:2', 'asset_c': 'QE:1'
    ... })
    >>> by_freq = get_clusters_by_freq(clusters)
    >>> by_freq['ME']
    asset_a    1
    asset_b    2
    dtype: object
    """
    clusters = clusters.dropna().astype(str)
    freq_codes = clusters.str.split(':').str[0].unique()

    by_freq: Dict[str, pd.Series] = {}
    for freq in freq_codes:
        prefix = f"{freq}:"
        mask = clusters.str.startswith(prefix)
        sub = clusters[mask].str.slice(start=len(prefix))
        by_freq[freq] = sub
    return by_freq


def get_linkages_by_freq(
    linkages: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Split a stacked linkage DataFrame into per-frequency ndarrays.

    ``CurrentFactorCovarData.linkages`` stores per-freq linkage matrices
    stacked into a single DataFrame with a freq-prefixed merge-step
    index. This helper splits them back into per-frequency scipy-compatible
    ndarrays, suitable for passing as ``precomputed_linkages`` in a
    subsequent fit.

    Parameters
    ----------
    linkages : pd.DataFrame
        Stacked linkage DataFrame with columns
        ``['left', 'right', 'distance', 'n_samples']`` and freq-prefixed
        index (e.g. ``'ME:step_0'``, ``'QE:step_3'``).

    Returns
    -------
    Dict[str, np.ndarray]
        Dict mapping each frequency code to its ``(K, 4)`` linkage ndarray.

    See Also
    --------
    get_linkage_array : Extract a single frequency's linkage.
    """
    index_prefixes = linkages.index.astype(str).str.split(':').str[0].unique()
    by_freq: Dict[str, np.ndarray] = {}
    for freq in index_prefixes:
        by_freq[freq] = get_linkage_array(linkages, freq)
    return by_freq


def get_cutoffs_by_freq(
    cutoffs: pd.Series,
) -> Dict[str, float]:
    """
    Convert a freq-indexed cutoff Series into a plain dict.

    ``CurrentFactorCovarData.cutoffs`` is already keyed by frequency code
    as its index, so this is a thin wrapper that returns
    ``cutoffs.to_dict()`` with float conversion. It exists for API
    symmetry with :func:`get_clusters_by_freq` and
    :func:`get_linkages_by_freq`.

    Parameters
    ----------
    cutoffs : pd.Series
        Series indexed by frequency code with float cutoff values.

    Returns
    -------
    Dict[str, float]
        Plain dict mapping each frequency code to its dendrogram cut distance.
    """
    return {freq: float(cutoffs.loc[freq]) for freq in cutoffs.index}
