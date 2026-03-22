"""
Exponentially Weighted Moving Average (EWMA) utilities.

Standalone implementations of EWMA mean, covariance, and helper functions.
These remove the dependency on the ``qis`` package so that ``factorlasso``
requires only standard scientific Python (numpy, pandas, scipy, cvxpy).

All functions accept an optional ``span`` parameter converted to the
EWMA decay factor via::

    λ = 1 − 2 / (span + 1)
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

# ── Observation weighting ─────────────────────────────────────────────

def compute_expanding_power(n: int,
                            power_lambda: float,
                            reverse_columns: bool = False
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
                ewm_lambda: float = 0.94
                ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    EWMA mean via the recursion ``ewm[t] = (1 − λ) x[t] + λ ewm[t−1]``.

    Parameters
    ----------
    data : array-like
        Input of shape ``(T,)`` or ``(T, N)``.
    span : float, optional
        EWMA span.  Overrides *ewm_lambda* if given.
    ewm_lambda : float, default 0.94
        Decay factor.

    Returns
    -------
    Same type and shape as *data*.
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.ewm(alpha=1.0 - ewm_lambda, adjust=False).mean()

    # --- numpy path ---
    a = np.asarray(data, dtype=np.float64)
    lam1 = 1.0 - ewm_lambda
    result = np.empty_like(a, dtype=np.float64)

    if a.ndim == 1:
        result[0] = a[0] if np.isfinite(a[0]) else 0.0
        for t in range(1, len(a)):
            x_t = a[t] if np.isfinite(a[t]) else result[t - 1]
            result[t] = ewm_lambda * result[t - 1] + lam1 * x_t
    else:
        result[0] = np.where(np.isfinite(a[0]), a[0], 0.0)
        for t in range(1, a.shape[0]):
            x_t = np.where(np.isfinite(a[t]), a[t], result[t - 1])
            result[t] = ewm_lambda * result[t - 1] + lam1 * x_t

    return result


# ── EWMA covariance ──────────────────────────────────────────────────

def compute_ewm_covar(a: np.ndarray,
                      b: Optional[np.ndarray] = None,
                      span: Optional[int] = None,
                      ewm_lambda: float = 0.94,
                      is_corr: bool = False
                      ) -> np.ndarray:
    """
    EWMA covariance (or correlation) matrix at the last observation.

    Recursion: ``Σ[t] = λ Σ[t−1] + (1 − λ) a[t] ⊗ b[t]``

    Parameters
    ----------
    a : np.ndarray, shape (T, N)
        Demeaned return matrix.
    b : np.ndarray, optional
        Cross matrix (T, N).  Defaults to *a*.
    span : int, optional
        EWMA span.
    ewm_lambda : float, default 0.94
        Decay factor.
    is_corr : bool, default False
        If True, normalise to correlations.

    Returns
    -------
    np.ndarray, shape (N, N)
    """
    if b is None:
        b = a
    assert a.shape[0] == b.shape[0]

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    lam1 = 1.0 - ewm_lambda

    n = a.shape[1] if a.ndim == 2 else a.shape[0]
    covar = np.zeros((n, n))

    if a.ndim == 2:
        for t in range(a.shape[0]):
            r_ij = np.outer(a[t], b[t])
            new = ewm_lambda * covar + lam1 * r_ij
            covar = np.where(np.isfinite(new), new, covar)
    else:
        covar = lam1 * np.outer(a, b) + ewm_lambda * covar

    if is_corr:
        d = np.diag(covar)
        if np.nansum(d) > 1e-10:
            # zero-variance assets get inv_vol=0 (zeroes their correlation row/column)
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_vol = np.where(d > 1e-16, np.reciprocal(np.sqrt(d)), 0.0)
            covar = covar * np.outer(inv_vol, inv_vol)
        else:
            covar = np.identity(n)

    return covar


# ── Group loadings ───────────────────────────────────────────────────

def set_group_loadings(group_data: pd.Series,
                       group_order: Optional[List[str]] = None
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
    loadings = {}
    for group in group_order:
        loadings[group] = pd.Series(
            np.where(group_data == group, 1.0, 0.0), index=group_data.index
        )
    return pd.DataFrame.from_dict(loadings, orient='columns')
