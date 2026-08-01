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
    """
    if nu <= 0:
        raise ValueError(f"nu must be positive, got {nu}")
    return float((1.0 + np.sqrt(n_series / nu)) ** 2)


def null_threshold(n_pairs: int, significance: float = 0.05) -> float:
    """Chi-square critical value for the sphericity statistic under exact diagonality."""
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
