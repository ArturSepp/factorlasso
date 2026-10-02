"""
Residual diagnostics for fitted factor models.

A sparse factor model asserts a *strict factor structure*: after the factors are removed, what
remains is idiosyncratic, so the residual covariance is diagonal. Nothing in the estimation
enforces that assertion, and a penalty set too high leaves common variation in the residual while
the fit still looks healthy on R². These diagnostics test the assertion directly.

The statistics are estimator-agnostic — they take a residual panel, not a
:class:`~factorlasso.LassoModel` — so they apply equally to loadings from LASSO, from a
time-series regression, or from observed characteristics.

Statistics
----------
Let ``R`` be the sample correlation matrix of the ``p`` residual series, and let
``nu = n - k - 1`` be the degrees of freedom left after fitting ``k`` loadings per asset on
``n`` observations. Under the null that the residual covariance is exactly diagonal,

    S = nu * sum_{i<j} r_ij^2   ~   chi^2 with p (p - 1) / 2 degrees of freedom,

and the largest eigenvalue of ``R`` is bounded above by the Marchenko-Pastur edge
``(1 + sqrt(p / nu))^2``. Eigenvalues above that edge count the factors the model does not carry,
and their eigenvectors say which series would define them.

Why not simply minimise off-diagonal mass
-----------------------------------------
Because every added loading absorbs some common variation, raw off-diagonal mass falls
monotonically in model density until it flattens, so its minimum sits in a flat region and moves
with sampling noise. :func:`raw_offdiagonal_mass` is provided for that comparison, but selection
should compare a statistic against the FIXED null threshold of :func:`null_threshold`, which is
what :class:`~factorlasso.LassoModelDiagonalityCV` does.

Which regime the calibration assumes
------------------------------------
``S`` is calibrated for small ``p`` against large ``nu``, the classical regime, where the
chi-square limit is the right one. The Marchenko-Pastur edge is an asymptotic result in both
dimensions and is a crude bound at small ``p``, so on a short cross-section read ``n_above_edge``
as descriptive and put the weight on ``sphericity``. When ``p`` and ``nu`` are comparable, or when
``nu`` is the smaller of the two, the references below give calibrations built for that corner and
this module's chi-square threshold is not the right instrument.

Partition share of cross-sectional variance
-------------------------------------------
:func:`partition_variance_share` asks whether a given partition of the series, for example the
HCGL clusters, still groups co-moving series. For each date it reports the share of the
cross-sectional variance that demeaning within groups removes, the floor ``(K - 1) / (N - 1)``
that any partition into ``K`` groups removes by construction, and the adjusted share that nets
the floor out. Applied to factor-model residuals it complements
:func:`missing_factor_components`: the spectral test looks for an omitted common direction, the
partition share measures block structure left among the grouped series. On raw returns it
separates an informative partition from the mechanical effect of grouping.

References
----------
None of these statistics originates here.

- Schott, J. R. (2005), "Testing for complete independence in high dimensions", Biometrika 92(4),
  951-956. The sum of squared sample correlations as a test of complete independence. ``S`` is its
  fixed-``p`` chi-square limit; the ``nu = n - k - 1`` charge for fitted loadings is a heuristic
  correction and not part of that result.
- Marchenko, V. A. and Pastur, L. A. (1967), for the spectral edge. Laloux, L., Cizeau, P.,
  Bouchaud, J.-P. and Potters, M. (1999), "Noise dressing of financial correlation matrices", for
  its use on financial correlation matrices.
- Gagliardini, P., Ossola, E. and Scaillet, O. (2019), "A diagnostic criterion for approximate
  factor structure", Journal of Econometrics 212(2), 503-521. Reads the largest eigenvalue of a
  residual covariance as a test for an omitted common factor, and selects the factor count as the
  smallest ``k`` whose penalised eigenvalue turns negative. That is the published form of what
  :func:`missing_factor_components` reports and of the selection rule in
  :class:`~factorlasso.LassoModelDiagonalityCV`, which differs only in indexing a penalty rather
  than a factor count. Their calibration is built for panels with many more series than periods,
  and it accounts for the loadings being estimated, which this module does not.
- Onatski, A. (2009), Econometrica 77(5), 1447-1479, and Ahn, S. C. and Horenstein, A. R. (2013),
  Econometrica 81(3), 1203-1227, reach a factor count from the same residual eigenvalues under
  proportional asymptotics.
- Pitman, E. J. G. (1938), "Significance tests which may be applied to samples from any
  populations. III. The analysis of variance test", Biometrika 29(3-4), 322-335. The permutation
  distribution of the analysis-of-variance ratio. :func:`partition_variance_share` uses only its
  first moment, derived in that function's notes, and attaches no test.
- Zhu, L., He, Y. and Cucuringu, M. (2026), "Quantifying the contributions of clustering to
  statistical arbitrage", working paper, September 2026. Compare the variance share that
  demeaning within a partition of equity residuals removes against the floor ``(K - 1)/(N - 1)``
  of a random partition. They draw the random partition i.i.d. uniform, where the floor holds
  up to empty groups. This module conditions on the realised group sizes, where it is exact,
  and adds the adjusted share for comparing partitions with different ``K``.

Examples
--------
>>> import numpy as np, pandas as pd
>>> rng = np.random.default_rng(0)
>>> resid = pd.DataFrame(rng.standard_normal((400, 6)))       # truly diagonal
>>> d = diagnose_residuals(resid, n_fitted_per_asset=3)
>>> d.passes
True
>>> common = rng.standard_normal((400, 1))                     # inject a common factor
>>> d2 = diagnose_residuals(resid + common, n_fitted_per_asset=3)
>>> d2.passes
False
>>> d2.n_above_edge >= 1
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

ArrayLike = Union[np.ndarray, pd.DataFrame]

__all__ = [
    "ResidualDiagnostics",
    "diagnose_residuals",
    "marchenko_pastur_edge",
    "missing_factor_components",
    "null_threshold",
    "raw_offdiagonal_mass",
    "residual_correlation",
    "effective_sparsity",
    "Sparsity",
    "suggest_tolerance",
    "partition_variance_share",
]


# ═══════════════════════════════════════════════════════════════════════
# Container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ResidualDiagnostics:
    """
    Outcome of a residual-diagonality test.

    Attributes
    ----------
    sphericity : float
        ``nu * sum_{i<j} r_ij^2``, distributed chi-square with ``n_pairs`` degrees of freedom
        under exact diagonality. Lower is closer to diagonal.
    threshold : float
        Critical value of that chi-square at the chosen significance.
    top_eigenvalue : float
        Largest eigenvalue of the residual correlation matrix.
    mp_edge : float
        Marchenko-Pastur upper edge for a pure-noise panel of the same shape.
    n_above_edge : int
        Eigenvalues above the edge. A count of the factors the model does not carry.
    mean_abs_offdiag, max_abs_offdiag : float
        Location and worst case of the off-diagonal correlations.
    raw_offdiag_ss : float
        ``sum_{i<j} r_ij^2`` without the degrees-of-freedom scaling. Reported for comparison
        only; see the module docstring on why minimising it does not select a penalty.
    nu : float
        Degrees of freedom used, ``n - k - 1``.
    n_obs, n_series, n_pairs : int
        Panel shape.
    passes : bool
        True when the sphericity statistic is at or below ``threshold`` AND no eigenvalue
        exceeds the edge.
    """

    sphericity: float
    threshold: float
    top_eigenvalue: float
    mp_edge: float
    n_above_edge: int
    mean_abs_offdiag: float
    max_abs_offdiag: float
    raw_offdiag_ss: float
    nu: float
    n_obs: int
    n_series: int
    n_pairs: int
    correlation: pd.DataFrame = field(repr=False)

    @property
    def passes(self) -> bool:
        """Whether the residual covariance is indistinguishable from diagonal."""
        return bool(
            self.sphericity <= self.threshold and self.top_eigenvalue <= self.mp_edge
        )

    def to_dict(self) -> dict:
        """Flat mapping for tabulating a path or a grid."""
        return {
            "sphericity": self.sphericity,
            "threshold": self.threshold,
            "top_eigenvalue": self.top_eigenvalue,
            "mp_edge": self.mp_edge,
            "n_above_edge": self.n_above_edge,
            "mean_abs_offdiag": self.mean_abs_offdiag,
            "max_abs_offdiag": self.max_abs_offdiag,
            "raw_offdiag_ss": self.raw_offdiag_ss,
            "nu": self.nu,
            "n_obs": self.n_obs,
            "n_series": self.n_series,
            "passes": self.passes,
        }


# ═══════════════════════════════════════════════════════════════════════
# Primitives
# ═══════════════════════════════════════════════════════════════════════

def residual_correlation(
    resid: ArrayLike, min_periods: int = 30,
) -> pd.DataFrame:
    """
    Pairwise-complete correlation matrix of a residual panel.

    Parameters
    ----------
    resid : np.ndarray or pd.DataFrame, shape (T, p)
        Residual panel. NaNs are permitted and handled pairwise.
    min_periods : int, default 30
        Minimum overlapping observations for a pair to be estimated. Pairs below the threshold
        come back NaN and are excluded from the statistics.

    Returns
    -------
    pd.DataFrame, shape (p, p)

    Raises
    ------
    ValueError
        If the panel has fewer than two series.
    """
    frame = resid if isinstance(resid, pd.DataFrame) else pd.DataFrame(resid)
    if frame.shape[1] < 2:
        raise ValueError(
            f"residual panel needs at least 2 series, got {frame.shape[1]}"
        )
    return frame.corr(min_periods=min_periods)


def marchenko_pastur_edge(n_series: int, nu: float) -> float:
    """
    Upper edge of the Marchenko-Pastur spectrum, ``(1 + sqrt(p / nu))^2``.

    The largest eigenvalue of a correlation matrix built from ``p`` independent series over
    ``nu`` effective observations concentrates below this value. An eigenvalue above it is
    evidence of genuine common structure rather than estimation noise.

    Marchenko and Pastur (1967); Laloux et al. (1999) for the financial application. The edge is
    asymptotic in both dimensions, so at small ``p`` it is a bound rather than a calibrated
    critical value, and it does not account for the loadings having been estimated. See the module
    references.
    """
    if nu <= 0:
        raise ValueError(f"nu must be positive, got {nu}")
    return float((1.0 + np.sqrt(n_series / nu)) ** 2)


def null_threshold(n_pairs: int, significance: float = 0.05) -> float:
    """Chi-square critical value for the sphericity statistic under exact diagonality.

    The fixed-``p`` limit of Schott (2005). See the module references.
    """
    if not 0.0 < significance < 1.0:
        raise ValueError(f"significance must lie in (0, 1), got {significance}")
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")
    return float(stats.chi2.ppf(1.0 - significance, n_pairs))


def raw_offdiagonal_mass(resid: ArrayLike, min_periods: int = 30) -> float:
    """
    ``sum_{i<j} r_ij^2`` with no degrees-of-freedom scaling.

    Provided for comparison with :func:`diagnose_residuals`. Minimising this quantity does not
    select a penalty: it falls monotonically in model density until it flattens, so its minimum
    is not identified. See the module docstring.
    """
    corr = residual_correlation(resid, min_periods=min_periods).to_numpy()
    off = corr[np.triu_indices_from(corr, 1)]
    off = off[np.isfinite(off)]
    return float(np.sum(off ** 2))


# ═══════════════════════════════════════════════════════════════════════
# The test
# ═══════════════════════════════════════════════════════════════════════

def diagnose_residuals(
    resid: ArrayLike,
    n_fitted_per_asset: float = 0.0,
    significance: float = 0.05,
    min_periods: int = 30,
) -> ResidualDiagnostics:
    """
    Test whether a residual panel is consistent with a diagonal covariance.

    Parameters
    ----------
    resid : np.ndarray or pd.DataFrame, shape (T, p)
        Residual panel. NaNs permitted.
    n_fitted_per_asset : float, default 0.0
        Loadings estimated per series. Enters the degrees of freedom as ``nu = n - k - 1``, so a
        denser model is held to the same threshold on fewer effective observations. Pass the
        model's average non-zero loading count; see :func:`effective_sparsity`.
    significance : float, default 0.05
        Size of the test.
    min_periods : int, default 30
        Forwarded to :func:`residual_correlation`.

    Returns
    -------
    ResidualDiagnostics

    Raises
    ------
    ValueError
        If no pair of series has enough overlap, or if the fit leaves no degrees of freedom.

    Notes
    -----
    Computed on the same sample the model was fitted on, the statistic is biased toward passing.
    :class:`~factorlasso.LassoModelDiagonalityCV` removes that bias by evaluating on held-out
    folds.
    """
    frame = resid if isinstance(resid, pd.DataFrame) else pd.DataFrame(resid)
    corr = residual_correlation(frame, min_periods=min_periods)
    values = corr.to_numpy()
    n_series = values.shape[0]
    upper = np.triu_indices(n_series, 1)
    off = values[upper]
    off = off[np.isfinite(off)]
    if off.size == 0:
        raise ValueError(
            f"no pair of the {n_series} series has {min_periods} overlapping observations"
        )

    n_obs = int(frame.notna().sum().min())
    nu = n_obs - float(n_fitted_per_asset) - 1.0
    if nu <= 1.0:
        raise ValueError(
            f"fit leaves nu={nu:.1f} degrees of freedom: {n_obs} observations against "
            f"{n_fitted_per_asset:.1f} loadings per series"
        )

    filled = np.nan_to_num(values, nan=0.0)
    np.fill_diagonal(filled, 1.0)
    eigenvalues = np.sort(np.linalg.eigvalsh(filled))[::-1]
    edge = marchenko_pastur_edge(n_series, nu)

    return ResidualDiagnostics(
        sphericity=float(nu * np.sum(off ** 2)),
        threshold=null_threshold(off.size, significance=significance),
        top_eigenvalue=float(eigenvalues[0]),
        mp_edge=edge,
        n_above_edge=int(np.sum(eigenvalues > edge)),
        mean_abs_offdiag=float(np.mean(np.abs(off))),
        max_abs_offdiag=float(np.max(np.abs(off))),
        raw_offdiag_ss=float(np.sum(off ** 2)),
        nu=float(nu),
        n_obs=n_obs,
        n_series=int(n_series),
        n_pairs=int(off.size),
        correlation=corr,
    )


def missing_factor_components(
    resid: ArrayLike,
    n_components: Optional[int] = None,
    loading_floor: float = 0.25,
    min_periods: int = 30,
    n_fitted_per_asset: float = 0.0,
) -> pd.DataFrame:
    """
    Principal components of the residual correlation, read as a specification test.

    Each component above the Marchenko-Pastur edge names a factor the model does not carry, and
    its loadings say which series would define it. When no penalty setting passes
    :func:`diagnose_residuals`, this is the actionable output: the remedy is to extend the factor
    set, not to retune the penalty.

    Gagliardini, Ossola and Scaillet (2019) give the published form of this diagnostic, with a
    calibration that accounts for the loadings being estimated. Prefer their criterion over the
    raw edge comparison when the count itself carries the conclusion. See the module references.

    Parameters
    ----------
    resid : np.ndarray or pd.DataFrame, shape (T, p)
    n_components : int, optional
        Components to report. Default: the number above the edge, at least one.
    loading_floor : float, default 0.25
        Report loadings at or above this absolute value.
    min_periods, n_fitted_per_asset
        As in :func:`diagnose_residuals`.

    Returns
    -------
    pd.DataFrame
        Columns ``component``, ``eigenvalue``, ``series``, ``loading``, sorted by component then
        by absolute loading. Each component's sign is fixed so its largest loading is positive.
    """
    frame = resid if isinstance(resid, pd.DataFrame) else pd.DataFrame(resid)
    diag = diagnose_residuals(
        frame, n_fitted_per_asset=n_fitted_per_asset, min_periods=min_periods,
    )
    values = np.nan_to_num(diag.correlation.to_numpy(), nan=0.0)
    np.fill_diagonal(values, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    order = np.argsort(eigenvalues)[::-1]
    count = n_components if n_components is not None else max(diag.n_above_edge, 1)

    names = list(frame.columns)
    rows = []
    for rank in range(min(count, len(order))):
        idx = order[rank]
        vector = eigenvectors[:, idx]
        if vector[np.argmax(np.abs(vector))] < 0:
            vector = -vector
        for position in np.argsort(-np.abs(vector)):
            if abs(vector[position]) >= loading_floor:
                rows.append({
                    "component": rank + 1,
                    "eigenvalue": float(eigenvalues[idx]),
                    "series": names[position],
                    "loading": float(vector[position]),
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Sparsity accounting
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Sparsity:
    """
    Loadings a fit actually kept, counted at a stated tolerance.

    Attributes
    ----------
    n_nonzero, n_total : int
        Kept loadings and the size of the coefficient matrix.
    density : float
        ``n_nonzero / n_total``.
    per_asset : float
        Mean kept loadings per response. This is the quantity that enters the degrees of
        freedom of :func:`diagnose_residuals`.
    max_per_asset : int
        Densest single response. A response near the factor count has no residual degrees of
        freedom left.
    per_factor : pd.Series
        Kept loadings by regressor, indexed by regressor name where available.
    empty_factors : list
        Regressors no response loads on. **Each one makes the information matrix
        ``beta' D^-1 beta`` singular**, so ``beta_F^-1`` does not exist and any quantity built
        from it needs a rank-safe form.
    empty_assets : list
        Responses that load on nothing. Their fitted values are the intercept alone.
    n_nonfinite : int
        NaN or infinite coefficients, typically a response whose solve failed. These are NOT
        counted as zero: a failed solve is not a sparse one.
    tol_used : float
        The absolute tolerance actually applied, after the relative floor.
    """

    n_nonzero: int
    n_total: int
    density: float
    per_asset: float
    max_per_asset: int
    per_factor: pd.Series = field(repr=False)
    empty_factors: list = field(repr=False)
    empty_assets: list = field(repr=False)
    n_nonfinite: int
    tol_used: float

    @property
    def is_rank_deficient(self) -> bool:
        """True when some regressor has no carrier, hence ``beta_F`` is singular."""
        return len(self.empty_factors) > 0

    def to_dict(self) -> dict:
        """Flat mapping for tabulating a path or a grid."""
        return {
            "n_nonzero": self.n_nonzero,
            "n_total": self.n_total,
            "density": self.density,
            "per_asset": self.per_asset,
            "max_per_asset": self.max_per_asset,
            "n_empty_factors": len(self.empty_factors),
            "n_empty_assets": len(self.empty_assets),
            "n_nonfinite": self.n_nonfinite,
            "tol_used": self.tol_used,
        }


def suggest_tolerance(betas: ArrayLike) -> dict:
    """
    Locate the gap between solver dust and live loadings.

    Sorts the non-zero magnitudes and finds the largest multiplicative step between neighbours.
    Interior-point output is bimodal — dust many orders below the coefficients the fit meant to
    keep — so that step is the natural place to cut, and any tolerance inside it gives the same
    sparsity count.

    Parameters
    ----------
    betas : np.ndarray or pd.DataFrame, shape (N, M)

    Returns
    -------
    dict
        ``gap_lo`` and ``gap_hi`` bound the empty interval, ``gap_orders`` is its width in
        decades, ``suggested_tol`` is the geometric midpoint, and ``suggested_rtol`` expresses
        it relative to the largest absolute loading. ``gap_orders`` below about 1 means the two
        populations are not separated and the count is sensitive to the tolerance.

    Examples
    --------
    >>> import numpy as np
    >>> betas = np.array([[1.0, 1e-9], [0.4, 1e-8]])
    >>> out = suggest_tolerance(betas)
    >>> out['gap_orders'] > 7
    True
    """
    frame = betas if isinstance(betas, pd.DataFrame) else pd.DataFrame(np.asarray(betas))
    values = np.abs(frame.to_numpy(dtype=float).ravel())
    values = np.sort(values[np.isfinite(values) & (values > 0)])
    if values.size < 2:
        raise ValueError(f"need at least 2 non-zero loadings, got {values.size}")
    steps = np.log10(values[1:]) - np.log10(values[:-1])
    k = int(np.argmax(steps))
    lo, hi = float(values[k]), float(values[k + 1])
    largest = float(values[-1])
    suggested = float(np.sqrt(lo * hi))
    return {
        "gap_lo": lo,
        "gap_hi": hi,
        "gap_orders": float(steps[k]),
        "suggested_tol": suggested,
        "suggested_rtol": suggested / largest if largest else float("nan"),
    }


def effective_sparsity(
    betas: ArrayLike,
    tol: float = 0.0,
    rtol: float = 1e-4,
    raise_on_nonfinite: bool = False,
) -> Sparsity:
    """
    Count loadings the fit actually kept, at a scale-aware tolerance.

    Interior-point solvers return numerically-zero loadings as small non-zero values rather than
    exact zeros, so a bare ``(betas != 0).sum()`` reports every cell as occupied and any sparsity
    statement built on it is vacuous. This counts at a tolerance and reports which tolerance.

    The tolerance is scale-aware by default. A fixed absolute cut is not portable: loadings scale
    with the units of the regressors and responses, so ``1e-4`` means one thing on decimal monthly
    returns and something else on percentage points. The applied cut is

        tol_used = max(tol, rtol * max |beta|),

    so the default behaviour is relative to the largest loading in the matrix and independent of
    units. Pass ``rtol=0.0`` for a purely absolute cut.

    The relative cut presumes that the largest loading is a live one. When the penalty has driven
    every loading to zero, the largest magnitude is itself solver dust, the cut scales down with
    it, and the dust is counted as kept loadings: a fully collapsed fit reads as a dense one.
    Wherever a fit may have collapsed, for example at the top of a penalty grid, pass an absolute
    ``tol`` at the scale of a meaningful loading, with ``rtol=0.0`` for a purely absolute cut, or
    inspect :func:`suggest_tolerance`.

    Parameters
    ----------
    betas : np.ndarray or pd.DataFrame, shape (N, M)
        Fitted loadings, e.g. ``LassoModel.estimated_betas``.
    tol : float, default 0.0
        Absolute floor on the tolerance.
    rtol : float, default 1e-4
        Relative tolerance, applied to the largest absolute loading. Interior-point solvers place
        their dust many orders of magnitude below the live coefficients, and the two populations
        are separated by a wide empty gap in the magnitude distribution, so any cut inside that
        gap gives the same count. On a production factor-cluster group-lasso fit the dust sits
        below 2e-6 relative and the live loadings above 2e-2, and every ``rtol`` from 1e-4 to
        1e-2 returns the same answer. Call :func:`suggest_tolerance` to locate the gap on a
        particular fit rather than trusting the default.
    raise_on_nonfinite : bool, default False
        Raise when any coefficient is NaN or infinite instead of reporting the count. A failed
        solve otherwise looks like a sparse one, since ``abs(nan) > tol`` is False.

    Returns
    -------
    Sparsity

    Raises
    ------
    ValueError
        If ``tol`` or ``rtol`` is negative, if ``betas`` is not two-dimensional, or if
        ``raise_on_nonfinite`` is set and the matrix carries non-finite entries.

    Examples
    --------
    >>> import numpy as np
    >>> betas = np.array([[1.0, 1e-9], [0.0, 0.5]])
    >>> s = effective_sparsity(betas)
    >>> s.n_nonzero
    2
    >>> int((betas != 0).sum())          # what a bare count reports
    3
    >>> s.is_rank_deficient              # every factor still has a carrier
    False

    A scaled copy of the same matrix gives the same answer, which a fixed absolute cut would not:

    >>> effective_sparsity(betas * 1e-4).n_nonzero
    2
    >>> effective_sparsity(betas * 1e-4, tol=1e-4, rtol=0.0).n_nonzero   # absolute cut misreads
    0
    >>> suggest_tolerance(betas)['gap_orders'] > 5    # dust and loadings are far apart
    True

    A collapsed fit defeats the relative cut, because its largest magnitude is dust as well:

    >>> dust = np.array([[3e-9, 1e-9], [2e-9, 4e-9]])
    >>> effective_sparsity(dust).n_nonzero               # the relative cut counts the dust
    4
    >>> effective_sparsity(dust, tol=1e-6).n_nonzero     # an absolute floor does not
    0

    A factor with no carrier is flagged, because it makes ``beta_F`` singular:

    >>> orphan = np.array([[1.0, 0.0], [0.8, 0.0]])
    >>> effective_sparsity(orphan).empty_factors
    [1]
    """
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")
    if rtol < 0:
        raise ValueError(f"rtol must be non-negative, got {rtol}")

    frame = betas if isinstance(betas, pd.DataFrame) else pd.DataFrame(np.asarray(betas))
    values = frame.to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError(f"betas must be two-dimensional, got shape {values.shape}")

    finite = np.isfinite(values)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite and raise_on_nonfinite:
        raise ValueError(
            f"{n_nonfinite} of {values.size} coefficients are not finite; "
            f"a failed solve is not a sparse one"
        )

    magnitude = np.abs(np.where(finite, values, 0.0))
    tol_used = float(max(tol, rtol * magnitude.max())) if magnitude.size else float(tol)
    kept = magnitude > tol_used

    per_factor = pd.Series(kept.sum(axis=0), index=frame.columns)
    per_row = kept.sum(axis=1)
    n_total = int(values.size)
    n_nonzero = int(kept.sum())
    return Sparsity(
        n_nonzero=n_nonzero,
        n_total=n_total,
        density=n_nonzero / n_total if n_total else float("nan"),
        per_asset=n_nonzero / values.shape[0] if values.shape[0] else float("nan"),
        max_per_asset=int(per_row.max()) if per_row.size else 0,
        per_factor=per_factor,
        empty_factors=[c for c, n in per_factor.items() if n == 0],
        empty_assets=[r for r, n in zip(frame.index, per_row) if n == 0],
        n_nonfinite=n_nonfinite,
        tol_used=tol_used,
    )


# ═══════════════════════════════════════════════════════════════════════
# Partition share of cross-sectional variance
# ═══════════════════════════════════════════════════════════════════════

_PARTITION_SHARE_COLUMNS = ["share", "floor", "adjusted_share", "n_names", "n_groups"]


def _partition_share_row(values: np.ndarray, groups: np.ndarray) -> tuple:
    """Share, permutation floor and adjusted share for one filtered cross-section."""
    n_names = int(values.shape[0])
    codes, uniques = pd.factorize(groups)
    n_groups = int(uniques.shape[0])
    nan = float("nan")
    if n_names < 2:
        return nan, nan, nan, n_names, n_groups
    deviation = values - values.mean()
    total = float(deviation @ deviation)
    if not total > 0.0:
        return nan, nan, nan, n_names, n_groups
    group_sums = np.bincount(codes, weights=deviation, minlength=n_groups)
    group_sizes = np.bincount(codes, minlength=n_groups)
    share = float(np.sum(group_sums ** 2 / group_sizes) / total)
    floor = (n_groups - 1) / (n_names - 1)
    if n_names > n_groups:
        adjusted = 1.0 - (1.0 - share) * (n_names - 1) / (n_names - n_groups)
    else:
        adjusted = nan
    return share, floor, adjusted, n_names, n_groups


def partition_variance_share(
    returns: Union[pd.DataFrame, pd.Series],
    labels: Union[pd.DataFrame, pd.Series],
) -> pd.DataFrame:
    r"""
    Share of cross-sectional variance removed by demeaning within groups, against its floor.

    For one cross-section ``x`` of ``N`` names that carry both a value and a label, grouped
    into ``K`` non-empty groups of sizes ``n_k``,

        share = SSB / SST,   SSB = sum_k n_k (xbar_k - xbar)^2,   SST = sum_i (x_i - xbar)^2.

    Under a uniformly random permutation of the labels over the names, with the group sizes
    held fixed, ``E[share] = (K - 1) / (N - 1)`` for every ``x`` and every size profile. That
    expectation is the ``floor``: what any partition into ``K`` groups removes by construction.
    The adjusted share

        adjusted_share = (share - floor) / (1 - floor) = 1 - (1 - share) (N - 1) / (N - K)

    is the adjusted R^2 of the one-way analysis of variance of ``x`` on group indicators. It is
    zero in expectation under permutation and one when the groups explain ``x`` exactly, so
    partitions with different ``K`` are compared on it rather than on ``share``.

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N), or pd.Series, shape (N,)
        Cross-sections to decompose, one row per date. Any panel works: raw or excess
        returns, factor-model residuals, or signal scores. A Series is a single cross-section.
    labels : pd.DataFrame, shape (T, N), or pd.Series, shape (N,)
        Group labels of any hashable type. A DataFrame is a date-varying partition, aligned to
        ``returns`` on both index and columns, so a date absent from ``labels`` yields an empty
        cross-section. A Series is a static partition aligned on names. A missing label or a
        non-finite value drops the name from that date's cross-section.

    Returns
    -------
    pd.DataFrame
        One row per row of ``returns`` (a single row for a Series), with columns

        - ``share``: SSB / SST, in [0, 1].
        - ``floor``: (K - 1) / (N - 1).
        - ``adjusted_share``: (share - floor) / (1 - floor).
        - ``n_names``, ``n_groups``: N and K after dropping missing values and labels.

        ``share``, ``floor`` and ``adjusted_share`` are NaN when N < 2 or the cross-section is
        constant. ``adjusted_share`` is also NaN when every name is its own group (N = K).

    Raises
    ------
    ValueError
        If ``returns`` is not a DataFrame or Series, if its columns repeat, if a Series
        cross-section comes with DataFrame labels, or if ``returns`` and ``labels`` share no
        names.

    Notes
    -----
    The floor follows from sampling without replacement. Under permutation the mean of a
    group of ``n_k`` names is the mean of ``n_k`` values drawn without replacement from the
    ``N`` values, so ``n_k E[(xbar_k - xbar)^2] = (SST / N)(N - n_k)/(N - 1)``. Summing over the
    ``K`` groups gives ``E[SSB] = SST (K - 1)/(N - 1)``.

    The floor is a first moment, not a critical value, and no test is attached. Singletons
    count as groups: each removes its own deviation, and the floor charges for it through
    ``K``. Group means are taken over the names present on the date, so a name with a missing
    value does not enter its group's mean.

    Examples
    --------
    >>> import pandas as pd
    >>> x = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    >>> z = pd.Series(["a", "a", "a", "b", "b", "b"])
    >>> out = partition_variance_share(x, z)
    >>> round(float(out["share"].iloc[0]), 4), float(out["floor"].iloc[0])
    (0.9681, 0.2)
    """
    if isinstance(returns, pd.Series):
        if not isinstance(labels, pd.Series):
            raise ValueError(
                f"a Series cross-section needs Series labels, got {type(labels).__name__}"
            )
        returns = returns.to_frame().T
    elif not isinstance(returns, pd.DataFrame):
        raise ValueError(
            f"returns must be a pandas DataFrame or Series, got {type(returns).__name__}"
        )
    if returns.columns.has_duplicates:
        duplicated = returns.columns[returns.columns.duplicated()].unique().tolist()
        raise ValueError(f"returns has repeated names: {duplicated[:5]}")

    if isinstance(labels, pd.Series):
        common = returns.columns.intersection(labels.index)
        static = labels.reindex(returns.columns).to_numpy(dtype=object)
        label_values = np.tile(static, (returns.shape[0], 1))
    elif isinstance(labels, pd.DataFrame):
        common = returns.columns.intersection(labels.columns)
        label_values = labels.reindex(
            index=returns.index, columns=returns.columns
        ).to_numpy(dtype=object)
    else:
        raise ValueError(
            f"labels must be a pandas DataFrame or Series, got {type(labels).__name__}"
        )
    if common.empty:
        raise ValueError("returns and labels share no names")

    values = returns.to_numpy(dtype=float)
    rows = []
    for row_values, row_labels in zip(values, label_values):
        keep = np.isfinite(row_values) & ~pd.isna(row_labels)
        rows.append(_partition_share_row(row_values[keep], row_labels[keep]))
    out = pd.DataFrame(rows, index=returns.index, columns=_PARTITION_SHARE_COLUMNS)
    return out.astype({"n_names": int, "n_groups": int})
