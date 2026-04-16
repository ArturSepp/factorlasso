"""
Exponentially Weighted Moving Average (EWMA) utilities.

Standalone, pure-NumPy implementations of EWMA mean and covariance.
The recursion, initialisation, and NaN handling match
``qis.models.linear.ewm`` exactly — but without the numba JIT,
keeping factorlasso's dependency footprint to numpy / pandas / scipy / cvxpy.

Decay factor convention::

    λ = 1 − 2 / (span + 1)
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd

# ── Enums (1-to-1 with qis) ───────────────────────────────────────────

class InitType(Enum):
    """How to derive the initial state of the EWMA recursion."""
    ZERO = 1   # y[0] = 0
    X0 = 2     # y[0] = x[0]   (or 0 if x[0] is NaN)
    MEAN = 3   # y[0] = nanmean(x)
    VAR = 4    # y[0] = nanvar(x)


class NanBackfill(Enum):
    """How to fill NaN observations during the recursion."""
    FFILL = 1            # carry the previous EWMA value forward
    DEFLATED_FFILL = 2   # λ × previous EWMA value (decays through gaps)
    ZERO_FILL = 3        # treat the NaN observation as zero contribution
    NAN_FILL = 4         # like DEFLATED_FFILL but mark output as NaN


# ── Internal helpers ──────────────────────────────────────────────────

def _to_np(data, fill_value: float = np.nan) -> np.ndarray:
    """Convert pandas/array-like to float64 ndarray; replace ±inf with fill_value."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        a = data.to_numpy(dtype=np.float64, na_value=fill_value, copy=True)
    else:
        a = np.asarray(data, dtype=np.float64).copy()
    return np.where(np.isfinite(a), a, fill_value)


def set_init_dim1(data, init_type: InitType = InitType.X0) -> Union[float, np.ndarray]:
    """Initial value for 1-D EWMA recursion. Matches qis.set_init_dim1."""
    x = _to_np(data, fill_value=np.nan)
    x0 = x[0]
    if init_type == InitType.ZERO:
        return np.zeros_like(x0)
    if init_type == InitType.X0:
        # NaN at t=0 → start from 0; otherwise start from the observation.
        # ``~np.isnan(x0)`` (not ``not np.isnan(x0)``) because x0 is a
        # scalar for 1-D inputs and a vector for 2-D inputs.
        return np.where(~np.isnan(x0), x0, 0.0)
    if init_type == InitType.MEAN:
        m = np.nanmean(x, axis=0)
        return np.where(~np.isnan(m), m, 0.0)
    if init_type == InitType.VAR:
        return np.nanvar(x, axis=0)
    raise ValueError(f"Unsupported init_type {init_type!r}")


# ── Core recursion (pure NumPy) ───────────────────────────────────────

def ewm_recursion(a: np.ndarray,
                  init_value: Union[float, np.ndarray],
                  span: Optional[float] = None,
                  ewm_lambda: float = 0.94,
                  is_start_from_first_nonan: bool = True,
                  nan_backfill: NanBackfill = NanBackfill.FFILL,
                  ) -> np.ndarray:
    """
    EWMA recursion ``y[t] = λ y[t-1] + (1-λ) x[t]``.

    Pure-NumPy reimplementation of ``qis.models.linear.ewm.ewm_recursion``;
    produces identical outputs to within machine epsilon.

    Parameters
    ----------
    a : np.ndarray, shape (T,) or (T, N)
        Input array.  NaN entries are handled per ``nan_backfill``.
    init_value : float or ndarray
        Initial state at t = 0 (or at the first non-NaN observation when
        ``is_start_from_first_nonan`` is True).
    span : float, optional
        EWMA span; converts to ``λ = 1 − 2/(span+1)``.
    ewm_lambda : float, default 0.94
        Decay factor.  Used only if ``span`` is None.
    is_start_from_first_nonan : bool, default True
        If True, the recursion only "starts" at the first non-NaN
        observation; earlier rows stay NaN. Recommended — avoids
        contaminating early output with the placeholder ``init_value``
        when the series begins with missing data.
    nan_backfill : NanBackfill, default FFILL
        How to handle NaN observations mid-stream.
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    lam1 = 1.0 - ewm_lambda
    is_1d = (a.ndim == 1)

    ewm = np.full_like(a, np.nan, dtype=np.float64)

    # t = 0
    if is_start_from_first_nonan:
        if is_1d:
            ewm[0] = init_value if np.isfinite(a[0]) else np.nan
        else:
            ewm[0] = np.where(np.isfinite(a[0]), init_value, np.nan)
    else:
        ewm[0] = init_value
    last = ewm[0]

    for t in range(1, a.shape[0]):
        a_t = a[t]

        # First non-NaN after a leading NaN stretch: jump-start from init_value
        if is_start_from_first_nonan:
            if is_1d:
                if not np.isfinite(last) and np.isfinite(a_t):
                    last = init_value
            else:
                start = np.logical_and(~np.isfinite(last), np.isfinite(a_t))
                if np.any(start):
                    last = np.where(start, init_value, last)

        current = ewm_lambda * last + lam1 * a_t

        # NaN observation → ``current`` is non-finite; fall back per policy
        if is_1d:
            if not np.isfinite(current):
                if nan_backfill == NanBackfill.FFILL:
                    current = last
                elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                    current = ewm_lambda * last
                else:  # ZERO_FILL or NAN_FILL (NAN_FILL post-processed below)
                    current = 0.0
        else:
            if nan_backfill == NanBackfill.FFILL:
                fill = last
            elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                fill = ewm_lambda * last
            else:
                fill = np.zeros_like(last)
            current = np.where(np.isfinite(current), current, fill)

        ewm[t] = last = current

    return ewm


# ── Observation weighting ─────────────────────────────────────────────

def compute_expanding_power(n: int,
                            power_lambda: float,
                            reverse_columns: bool = False,
                            ) -> np.ndarray:
    """
    Geometric power sequence ``[1, λ, λ², …, λ^(n−1)]``.

    Used to construct observation weights for EWMA-weighted objectives.

    Parameters
    ----------
    n : int
        Length of the sequence.
    power_lambda : float
        Base of the geometric sequence.
    reverse_columns : bool, default False
        If True, reverse so that the most recent observation has weight 1.

    Returns
    -------
    np.ndarray
        1-D array of length *n*.
    """
    a = np.log(power_lambda) * np.ones(n)
    a[0] = 0.0
    b = np.exp(np.cumsum(a))
    if reverse_columns:
        b = b[::-1]
    return b


# ── EWMA mean ────────────────────────────────────────────────────────

def compute_ewm(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                span: Optional[float] = None,
                ewm_lambda: float = 0.94,
                init_value: Union[float, np.ndarray, None] = None,
                init_type: InitType = InitType.X0,
                nan_backfill: NanBackfill = NanBackfill.FFILL,
                ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    EWMA mean.

    Drop-in replacement for ``qis.models.linear.ewm.compute_ewm`` —
    same recursion, same initialisation, same NaN handling, no numba.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, or np.ndarray, shape (T,) or (T, N)
    span : float, optional
        EWMA span; ``λ = 1 − 2/(span+1)``.  Overrides ``ewm_lambda``.
    ewm_lambda : float, default 0.94
        Decay factor (used only if ``span`` is None).
    init_value : float or ndarray, optional
        Override for the recursion's initial state.  Derived from
        ``init_type`` if not provided.
    init_type : InitType, default X0
    nan_backfill : NanBackfill, default FFILL

    Returns
    -------
    Same type and shape as ``data``.
    """
    a = _to_np(data, fill_value=np.nan)

    if init_value is None:
        init_value = set_init_dim1(a, init_type=init_type)

    # qis quirk preserved for parity: scalarise for 1-D inputs
    if isinstance(data, pd.Series) or (isinstance(data, np.ndarray) and data.ndim == 1):
        ewm_lambda = float(ewm_lambda)
        if isinstance(init_value, np.ndarray):
            init_value = float(init_value)

    ewm = ewm_recursion(a=a,
                        span=span,
                        ewm_lambda=ewm_lambda,
                        init_value=init_value,
                        nan_backfill=nan_backfill)

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(ewm, index=data.index, columns=data.columns)
    if isinstance(data, pd.Series):
        return pd.Series(ewm, index=data.index, name=data.name)
    return ewm


# ── EWMA covariance (qis-aligned, no numba) ──────────────────────────

def compute_ewm_covar(a: np.ndarray,
                      b: Optional[np.ndarray] = None,
                      span: Optional[int] = None,
                      ewm_lambda: float = 0.94,
                      covar0: Optional[np.ndarray] = None,
                      is_corr: bool = False,
                      nan_backfill: NanBackfill = NanBackfill.FFILL,
                      ) -> np.ndarray:
    """
    EWMA covariance (or correlation) matrix at the last observation.

    Pure-NumPy reimplementation of ``qis.models.linear.ewm.compute_ewm_covar``
    — same recursion, same NaN handling, same correlation normalisation
    **with one deliberate deviation**: when ``is_corr=True`` and some
    diagonal element of the EWMA covariance is zero (e.g. an all-NaN
    column, or a constant column in the window), this function zeroes
    out the corresponding row/column of the correlation matrix and sets
    the diagonal to 1, rather than producing ``inf`` from
    ``1/sqrt(0)``.  The qis reference propagates ``inf`` and ``NaN``
    into the output, which in practice (a) floods the console with
    ``RuntimeWarning`` and (b) silently corrupts downstream computation
    such as HCGL clustering via Ward linkage.  The deviation only
    affects inputs that qis cannot handle cleanly anyway.

    Recursion: ``Σ[t] = λ Σ[t−1] + (1−λ) a[t] ⊗ b[t]``

    Parameters
    ----------
    a : np.ndarray, shape (T,) or (T, N)
        Returns matrix (typically demeaned).
    b : np.ndarray, optional
        Cross matrix; defaults to ``a``. Must have the same shape as ``a``.
    span : int, optional
        EWMA span; ``λ = 1 − 2/(span+1)``.
    ewm_lambda : float, default 0.94
        Decay factor (used only if ``span`` is None).
    covar0 : np.ndarray, optional
        Initial covariance matrix; defaults to zeros. Non-finite entries
        are zeroed.
    is_corr : bool, default False
        If True, normalise the final matrix to a correlation matrix.
        See the note above on zero-variance handling.
    nan_backfill : NanBackfill, default FFILL
        How to handle rows where the outer product is non-finite.

    Returns
    -------
    np.ndarray, shape (N, N)
    """
    if b is None:
        b = a
    else:
        assert a.shape[0] == b.shape[0]
        if a.ndim == 2:
            assert a.shape[1] == b.shape[1]

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    # Note: λ enters the covariance recursion *linearly* on the outer
    # product a ⊗ b.  The matched quadratic-loss weights in
    # ``lasso_estimator._compute_solver_weights`` therefore use
    # ``sqrt(λ)`` so that the effective observation weight inside
    # ``sum_squares(...)`` is λ^s — consistent with this convention.
    lam1 = 1.0 - ewm_lambda

    n = a.shape[0] if a.ndim == 1 else a.shape[1]

    if covar0 is None:
        covar = np.zeros((n, n))
    else:
        covar = np.where(np.isfinite(covar0), covar0, 0.0)
    last_covar = covar

    if a.ndim == 1:
        # Single-shot update (one observation row)
        r_ij = np.outer(a, b)
        new = lam1 * r_ij + ewm_lambda * last_covar
        if nan_backfill == NanBackfill.FFILL:
            fill = last_covar
        elif nan_backfill == NanBackfill.DEFLATED_FFILL:
            fill = ewm_lambda * last_covar
        else:
            fill = np.zeros_like(last_covar)
        covar = last_covar = np.where(np.isfinite(new), new, fill)
    else:
        # Time loop
        for t in range(a.shape[0]):
            r_ij = np.outer(a[t], b[t])
            new = lam1 * r_ij + ewm_lambda * last_covar
            if nan_backfill == NanBackfill.FFILL:
                fill = last_covar
            elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                fill = ewm_lambda * last_covar
            else:
                fill = np.zeros_like(last_covar)
            last_covar = np.where(np.isfinite(new), new, fill)

        if is_corr:
            d = np.diag(last_covar)
            if np.nansum(d) > 1e-10:
                # Per-element guard: invert sqrt only where variance is
                # strictly positive. Assets with zero/negative diagonal
                # (all-NaN column, no variance in the EWMA window) get
                # a zero row/column in the correlation matrix — not an
                # inf poisoning every other correlation. Diagonal is
                # then forced back to 1 so the result remains a valid
                # correlation matrix.
                pos = d > 1e-12
                inv_vol = np.zeros_like(d)
                inv_vol[pos] = 1.0 / np.sqrt(d[pos])
                covar = last_covar * np.outer(inv_vol, inv_vol)
                np.fill_diagonal(
                    covar, np.where(pos, np.diag(covar), 1.0),
                )
            else:
                covar = np.identity(n)
        else:
            covar = last_covar

    return covar


# ── Group loadings ───────────────────────────────────────────────────

def set_group_loadings(group_data: pd.Series,
                       group_order: Optional[List[str]] = None,
                       ) -> pd.DataFrame:
    """
    Convert group-membership Series to a binary loading matrix.

    Parameters
    ----------
    group_data : pd.Series
        Index = item names, values = group labels.
    group_order : list of str, optional
        Column order.  Defaults to ``group_data.unique()``.

    Returns
    -------
    pd.DataFrame, shape (N_items, G)
        Binary indicator matrix.
    """
    if not isinstance(group_data, pd.Series):
        raise ValueError(f"Expected pd.Series, got {type(group_data)}")
    if group_order is None:
        group_order = list(group_data.unique())
    return pd.DataFrame(
        {g: pd.Series(np.where(group_data == g, 1.0, 0.0), index=group_data.index)
         for g in group_order}
    )