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
    primitive used by HCGL (``LassoModelType.GROUP_LASSO_CLUSTERS``)
    but callable independently for any group-discovery workflow.

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

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spc

# ═══════════════════════════════════════════════════════════════════════
# Clustering from correlation
# ═══════════════════════════════════════════════════════════════════════

# Default fraction of ``max(pdist)`` at which to cut the dendrogram.
# 0.5 is the package-wide convention — half the maximum pairwise
# correlation-distance — and empirically produces ~15–25 clusters on
# a 150-asset multi-asset universe.  Callers that need a different
# granularity can override via ``cutoff_fraction``.
DEFAULT_CUTOFF_FRACTION: float = 0.5


def compute_clusters_from_corr_matrix(
    corr_matrix: pd.DataFrame,
    cutoff_fraction: float = DEFAULT_CUTOFF_FRACTION,
) -> Tuple[pd.Series, np.ndarray, float]:
    """
    Hierarchical clustering from a correlation matrix (Ward's method).

    Converts correlation to distance ``(1 − corr)``, applies Ward's
    agglomerative clustering, and cuts the dendrogram at
    ``cutoff_fraction × max(pairwise distance)``.

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
    which is the correct path from a correlation matrix to the condensed
    pairwise-distance vector that ``scipy.cluster.hierarchy.linkage``
    expects. A previous implementation passed ``pdist(1 - corr)``, which
    treated rows of ``(1 - corr)`` as observations in N-dimensional space
    and computed Euclidean distances between those rows — a different
    (non-semantic) metric that conflated correlation structure with
    higher-order geometric relationships.

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
    corr_matrix = corr_matrix.fillna(0.0)
    # squareform(1 - C) is the correct conversion from a correlation matrix
    # to scipy's condensed pairwise-distance vector. Clip guards against
    # tiny negative values from floating-point noise; fill diagonal with
    # exact zeros on the diagonal as squareform requires.
    dist_square = np.clip(1.0 - corr_matrix.to_numpy(), 0.0, 2.0)
    np.fill_diagonal(dist_square, 0.0)
    pdist = spc.distance.squareform(dist_square, checks=False)
    linkage = spc.linkage(pdist, method='ward')
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
