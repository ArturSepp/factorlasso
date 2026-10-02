"""Panel preparation for the solvers: index alignment, validity masks and demeaning."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from factorlasso.utils._ewm import _validate_span, compute_ewm


def get_x_y_np(
    x: Union[pd.DataFrame, pd.Series],
    y: Union[pd.DataFrame, pd.Series],
    span: Optional[float] = None,
    demean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare numpy arrays from regressor/response DataFrames with NaN masking.

    Parameters
    ----------
    x : pd.DataFrame or pd.Series, shape (T, N) or (T,)
        Regressor data.  May have all-NaN rows.
    y : pd.DataFrame or pd.Series, shape (T, N) or (T,)
        Response data.  May contain NaNs (different history lengths).
        Series is converted to single-column DataFrame.
    span : float, optional
        EWMA span for demeaning.  ``None`` uses simple mean.  Must be ≥ 1
        when provided.  Float accepted — the recursion math does not
        require an integer span.
    demean : bool, default True
        If True, subtract (rolling) mean before estimation.

    Returns
    -------
    x_np : np.ndarray, shape (T', M)
    y_np : np.ndarray, shape (T', N)
    valid_mask : np.ndarray, shape (T', N)
        ``T' = T − 1`` when EWMA demeaning is used.
    """
    _validate_span(span)
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if isinstance(y, pd.Series):
        y = y.to_frame()
    if not x.index.equals(y.index):
        raise ValueError(
            f"x and y must share the same index: "
            f"x has {len(x.index)} rows, y has {len(y.index)} rows"
        )

    nan_mask_y = y.isna().to_numpy().copy()
    x_all_nan = x.isna().all(axis=1).to_numpy()
    if np.any(x_all_nan):
        nan_mask_y[x_all_nan, :] = True

    # Demean on NaN-PRESERVED arrays, zero-fill afterwards. Versions before
    # 0.5.1 zero-filled first, so the per-column mean was diluted by the
    # zero-filled cells: an asset with valid fraction f was demeaned by
    # f·μ instead of μ, injecting a constant offset (1 − f)·μ into the
    # solver response on its valid window. The offset biased β at second
    # order (the demeaned X is not orthogonal to a constant on the valid
    # sub-window) and deflated the residual-variance and R² diagnostics at
    # first order. Computing the mean on NaN-preserved data removes the
    # dilution: ``np.nanmean`` ranges over valid observations only, and
    # ``compute_ewm`` runs its NaN-aware recursion (leading-NaN start from
    # the first observation, FFILL through mid-stream gaps). Cells that are
    # missing in the input are zero-filled AFTER demeaning — equivalent to
    # "at the running mean" rather than "a zero return below it" — and are
    # excluded from the loss by the validity mask in either case.
    x_np = x.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)

    if demean:
        if span is None:
            with warnings.catch_warnings():
                # all-NaN columns produce a harmless 'Mean of empty slice'
                warnings.simplefilter("ignore", category=RuntimeWarning)
                x_np = x_np - np.nanmean(x_np, axis=0)
                y_np = y_np - np.nanmean(y_np, axis=0)
        else:
            x_np = x_np - compute_ewm(x_np, span=span)
            y_np = y_np - compute_ewm(y_np, span=span)
            x_np = x_np[1:, :]
            y_np = y_np[1:, :]
            nan_mask_y = nan_mask_y[1:, :]

    x_np = np.nan_to_num(x_np, nan=0.0)
    y_np = np.nan_to_num(y_np, nan=0.0)

    return x_np, y_np, (~nan_mask_y).astype(float)
