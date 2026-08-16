"""
Dependence measures for the clustering correlation matrix.

The hierarchical clustering that drives HCGL/FCGL consumes a correlation
matrix. Which dependence measure produces that matrix is a modelling
choice, separate from the correlation-to-distance transform applied
afterwards (see :mod:`factorlasso.cluster_utils`).

Three measures are available:

- ``PEARSON`` (default): the linear product-moment correlation, and the
  behaviour of every version before 0.10.0.
- ``SPEARMAN``: Pearson correlation of the ranks. Robust to outliers and
  to monotone non-linearity, at the cost of discarding magnitudes.
- ``GERBER``: the Gerber statistic of Gerber et al. (2022), which counts
  the proportion of co-movements exceeding asset-specific thresholds.
  Insensitive both to extreme moves and to sub-threshold noise.

All three are *signed*, which is a hard requirement here: the partition
feeds cluster-pooled sign derivation, so a measure that discards the sign
of the relationship (``|rho|``, distance correlation, mutual information)
would pool assets carrying opposite factor exposures. Unsigned dependence
measures are therefore inadmissible for this package, whatever their
merits elsewhere.

Observation weighting
---------------------
Every measure honours the same ``span`` contract as the estimator loss:
``span=None`` weights observations uniformly, and a finite ``span``
applies EWMA weights ``w_t ∝ λ^(T-t)`` with ``λ = 1 - 2/(span+1)``.
Keeping the clustering correlation on the same weighting as the solver
loss is deliberate — see the block comment at the call site in
:meth:`factorlasso.LassoModel.fit`.

For the Gerber statistic the EWMA generalisation is exact rather than
approximate. Both the numerator and the denominator are counts of
indicator variables, so weighting the counts recovers the published
statistic as ``span → ∞`` (deviation is O(1/span)) and admits the
first-order recursion

    N_t = λ N_{t-1} + (1-λ) s_{i,t} s_{j,t}
    D_t = λ D_{t-1} + (1-λ) [not both-noise]_t
    g_t = N_t / D_t

which makes roll-forward updates O(1) per new observation.

Functions
---------
compute_dependence_matrix
    Dispatch to the chosen measure, honouring the ``span`` weighting.
compute_gerber_matrix
    The (optionally EWMA-weighted) Gerber statistic on its own.

References
----------
Gerber, S., B. Javid, H. Markowitz, P. Sargen, and D. Starer (2022).
    The Gerber statistic: a robust co-movement measure for portfolio
    optimization. *The Journal of Portfolio Management* 48(2), 87-102.
Gerber, S., H. Markowitz, P. Ernst, Y. Miao, B. Javid, and P. Sargen
    (2023). Proofs that the Gerber statistic is positive semidefinite.
    arXiv:2305.05663.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd

from factorlasso.ewm_utils import NanBackfill, compute_ewm_covar

# ═══════════════════════════════════════════════════════════════════════
# Dependence measures
# ═══════════════════════════════════════════════════════════════════════


class DependenceMeasure(str, Enum):
    """dependence measure used to build the clustering correlation matrix.

    - ``PEARSON``: linear product-moment correlation. The default, and the
      behaviour of all versions before 0.10.0.
    - ``SPEARMAN``: Pearson correlation of ranks. Invariant to any
      monotone transform of the marginals.
    - ``GERBER``: proportion of concordant minus discordant co-movements
      among observations piercing ``gerber_threshold * sigma`` on both
      legs (Gerber et al. 2022).

    All three are signed, as cluster-pooled sign derivation requires.
    """
    PEARSON = 'pearson'
    SPEARMAN = 'spearman'
    GERBER = 'gerber'


# Dependence measure applied before the distance transform. PEARSON is the
# package-wide default and reproduces the pre-0.10.0 behaviour exactly.
DEFAULT_DEPENDENCE_MEASURE: DependenceMeasure = DependenceMeasure.PEARSON

# Gerber threshold c, in units of each asset's own standard deviation.
# Gerber et al. (2022) advocate at least 0.5 so that sub-threshold noise
# is excluded; 0.5 is their headline setting.
DEFAULT_GERBER_THRESHOLD: float = 0.5


def _normalised_ewm_weights(n_obs: int, span: Optional[float]) -> np.ndarray:
    """observation weights summing to one: uniform if span is None, else EWMA.

    Parameters
    ----------
    n_obs : int
        Number of observations T.
    span : float, optional
        EWMA span; converts to ``λ = 1 - 2/(span+1)``. None gives uniform
        weights.

    Returns
    -------
    np.ndarray, shape (T,)
        Weights in chronological order, most recent last, summing to one.
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs!r}")
    if span is None:
        return np.full(n_obs, 1.0 / n_obs)
    ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    age = np.arange(n_obs - 1, -1, -1, dtype=float)   # 0 for the last observation
    weights = ewm_lambda ** age
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"degenerate EWMA weights for span={span!r}")
    return weights / total


def compute_gerber_matrix(a: np.ndarray,
                          span: Optional[float] = None,
                          gerber_threshold: float = DEFAULT_GERBER_THRESHOLD,
                          ) -> np.ndarray:
    """Gerber co-movement matrix, optionally EWMA-weighted.

    Implements

        g_ij = (n_UU + n_DD - n_UD - n_DU) / (T - n_NN)

    where an observation pierces on leg ``i`` when ``|a_ti| >= c sigma_i``,
    ``UU`` / ``DD`` count observations piercing in the same direction on
    both legs, ``UD`` / ``DU`` count opposite directions, and ``NN`` counts
    observations piercing on neither leg. With ``span`` set, each count
    becomes a weighted count under the EWMA weights, which recovers the
    published statistic as ``span → ∞``.

    Both counts are computed through matrix products rather than an
    ``(T, N, N)`` intermediate, so memory stays O(N^2).

    Parameters
    ----------
    a : np.ndarray, shape (T, N)
        Return panel. NaN entries are excluded pairwise: an observation
        contributes to the pair ``(i, j)`` only when both legs are finite.
    span : float, optional
        EWMA span. None applies uniform weights, giving the published
        equal-weight statistic.
    gerber_threshold : float, default 0.5
        Threshold ``c`` in [0, 1], applied as ``c * sigma_i`` per leg.
        Larger values discard more observations as noise.

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric matrix with unit diagonal and entries in [-1, 1]. Pairs
        with no qualifying observation are set to zero.

    Raises
    ------
    ValueError
        If ``a`` is not two-dimensional, or ``gerber_threshold`` lies
        outside [0, 1].

    Notes
    -----
    Unlike the Pearson correlation, the Gerber statistic is **not**
    invariant to centring: the thresholds apply to levels, so demeaning
    the panel changes the result. The estimator passes whatever panel the
    solver loss sees, which is demeaned when ``demean=True``.

    The equal-weight Gerber matrix is positive semi-definite (Gerber et
    al. 2023). The EWMA-weighted matrix is not covered by that proof,
    although it is PSD across the panels tested. Positive semi-definiteness
    matters here only for the ``CHORD`` distance transform, whose exact
    Euclidean interpretation assumes a Gram matrix.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"a must be 2-dimensional (T, N), got shape {a.shape}")
    if not 0.0 <= gerber_threshold <= 1.0:
        raise ValueError(
            f"gerber_threshold must lie in [0, 1], got {gerber_threshold!r}")

    n_obs = a.shape[0]
    weights = _normalised_ewm_weights(n_obs, span)
    is_valid = np.isfinite(a)
    a_filled = np.where(is_valid, a, 0.0)
    valid_f = is_valid.astype(float)

    # Weighted standard deviation per asset, over its own valid observations.
    weight_sum = weights @ valid_f
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(weight_sum > 0.0,
                        (weights @ (a_filled * valid_f)) / weight_sum, 0.0)
        centred = (a_filled - mean[None, :]) * valid_f
        variance = np.where(weight_sum > 0.0,
                            (weights @ (centred ** 2)) / weight_sum, 0.0)
    sigma = np.sqrt(np.maximum(variance, 0.0))
    threshold = gerber_threshold * sigma

    is_up = is_valid & (a_filled >= threshold[None, :])
    is_down = is_valid & (a_filled <= -threshold[None, :])
    # sign is 0 on invalid observations, so an invalid leg contributes
    # nothing to the numerator of any pair containing it.
    sign = is_up.astype(float) - is_down.astype(float)
    # noise = valid but piercing on neither side
    is_noise = (is_valid & ~is_up & ~is_down).astype(float)

    numerator = (sign * weights[:, None]).T @ sign
    # T - n_NN, restricted to pairwise-valid observations:
    #   sum_t w_t [valid_i valid_j] - sum_t w_t [noise_i noise_j]
    pair_weight = (valid_f * weights[:, None]).T @ valid_f
    both_noise = (is_noise * weights[:, None]).T @ is_noise
    denominator = pair_weight - both_noise

    with np.errstate(invalid='ignore', divide='ignore'):
        gerber = np.where(denominator > 0.0, numerator / denominator, 0.0)
    gerber = 0.5 * (gerber + gerber.T)        # kill floating-point asymmetry
    np.fill_diagonal(gerber, 1.0)
    return np.clip(gerber, -1.0, 1.0)


def compute_dependence_matrix(a: np.ndarray,
                              dependence_measure: Union[DependenceMeasure, str] = (
                                  DEFAULT_DEPENDENCE_MEASURE),
                              span: Optional[float] = None,
                              gerber_threshold: float = DEFAULT_GERBER_THRESHOLD,
                              nan_backfill: NanBackfill = NanBackfill.FFILL,
                              ) -> np.ndarray:
    """dependence matrix under the chosen measure and observation weighting.

    Parameters
    ----------
    a : np.ndarray, shape (T, N)
        Return panel, NaN allowed.
    dependence_measure : DependenceMeasure or str, default PEARSON
        One of ``'pearson'``, ``'spearman'``, ``'gerber'``.
    span : float, optional
        EWMA span. None weights observations uniformly.
    gerber_threshold : float, default 0.5
        Threshold for ``GERBER``; ignored by the other measures.
    nan_backfill : NanBackfill, default FFILL
        NaN policy for the EWMA recursion; ignored when ``span`` is None
        (the uniform paths use pairwise-complete observations) and by
        ``GERBER`` (which excludes NaN pairwise by construction).

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric dependence matrix with unit diagonal.

    Raises
    ------
    ValueError
        If ``dependence_measure`` is not a member of ``DependenceMeasure``.

    Notes
    -----
    ``SPEARMAN`` ranks the panel once and then reuses the Pearson
    machinery, so its EWMA form is the EWMA correlation of the ranks.
    Ranks depend on the whole sample, so a rolling re-estimation must
    re-rank; unlike ``PEARSON`` and ``GERBER``, the Spearman path is not
    a pure recursion and costs O(T log T) per update.
    """
    try:
        dependence_measure = DependenceMeasure(dependence_measure)
    except ValueError:
        raise ValueError(
            f"dependence_measure must be one of "
            f"{[m.value for m in DependenceMeasure]}, "
            f"got {dependence_measure!r}"
        ) from None

    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"a must be 2-dimensional (T, N), got shape {a.shape}")

    if dependence_measure == DependenceMeasure.GERBER:
        return compute_gerber_matrix(
            a=a, span=span, gerber_threshold=gerber_threshold)

    if dependence_measure == DependenceMeasure.SPEARMAN:
        # ranks keep NaN as NaN, so the downstream paths treat missing
        # observations exactly as they do for the raw panel
        a = pd.DataFrame(a).rank().to_numpy(dtype=float)

    if span is None:
        return pd.DataFrame(a).corr().to_numpy()
    return compute_ewm_covar(a=a, span=span, is_corr=True, nan_backfill=nan_backfill)
