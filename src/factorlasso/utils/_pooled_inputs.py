"""Strict labeled inputs shared by the opt-in pooled estimators.

These helpers do not inherit the older estimators' missing-data masks, infer
date weights, standardize columns, discover groups, or impose a rank policy.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


@dataclass(frozen=True)
class _PooledInputs:
    """Owned numerical panels, labels, normalized weights and memberships."""

    x: np.ndarray
    y: np.ndarray
    groups: pd.Series
    date_weights: pd.Series
    members: tuple[np.ndarray, ...]
    row_index: pd.Index
    factor_columns: pd.Index
    response_columns: pd.Index


def _validate_axis(index: pd.Index, name: str) -> None:
    """Reject duplicate and missing labels, including missing MultiIndex levels."""
    missing = (any(np.any(codes < 0) for codes in index.codes)
               if isinstance(index, pd.MultiIndex) else index.hasnans)
    if not index.is_unique or missing:
        raise ValueError(f'{name} must have unique nonmissing labels')


def _numeric_values(data: pd.DataFrame | pd.Series, name: str, *,
                    allow_missing: bool = False) -> np.ndarray:
    """Copy real numeric columns without accepting coercible strings or booleans."""
    dtypes = data.dtypes if isinstance(data, pd.DataFrame) else [data.dtype]
    if any(not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype)
           for dtype in dtypes):
        raise ValueError(f'{name} must contain real numeric, nonboolean columns')
    try:
        values = data.to_numpy(dtype=float, na_value=np.nan, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must contain representable real numeric values') from exc
    if np.isinf(values).any() or (not allow_missing and np.isnan(values).any()):
        raise ValueError(f'{name} must contain complete finite values')
    return values


def _validate_pooled_inputs(
    x: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    weights: pd.Series | None,
    *,
    allow_missing: bool = False,
) -> _PooledInputs:
    """Validate complete ordered panels and normalize strictly positive weights."""
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise TypeError('x and y must be pandas DataFrames')
    if not isinstance(groups, pd.Series):
        raise TypeError('groups must be a pandas Series')
    if weights is not None and not isinstance(weights, pd.Series):
        raise TypeError('weights must be a pandas Series or None')
    if x.empty or y.empty:
        raise ValueError('x and y must have nonempty rows and columns')
    for axis, name in ((x.index, 'x.index'), (y.index, 'y.index'),
                       (x.columns, 'x.columns'), (y.columns, 'y.columns')):
        _validate_axis(axis, name)
    if not x.index.equals(y.index):
        raise ValueError('x and y must have identical ordered row indexes')
    if not groups.index.equals(y.columns):
        raise ValueError('groups.index must equal y.columns in the same order')
    xx = _numeric_values(x, 'x', allow_missing=allow_missing)
    yy = _numeric_values(y, 'y', allow_missing=allow_missing)
    membership = {}
    for position, label in enumerate(groups):
        valid = isinstance(label, (str, Integral)) and not isinstance(label, (bool, np.bool_))
        if isinstance(label, Real) and not isinstance(label, (Integral, bool, np.bool_)):
            try:
                valid = bool(np.isfinite(label))
            except (TypeError, OverflowError):
                valid = False
        if not valid:
            raise ValueError('groups must contain nonmissing strings or finite real scalar IDs')
        membership.setdefault(label, []).append(position)
    if weights is None:
        w = np.ones(len(x), dtype=float)
    else:
        if not weights.index.equals(x.index):
            raise ValueError('weights.index must equal x.index in the same order')
        w = _numeric_values(weights, 'weights')
        if np.any(w <= 0):
            raise ValueError('weights must be strictly positive')
        with np.errstate(under='ignore'):
            w /= w.max()
            w /= w.mean()
        if not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError('weights lose positive support after normalization')
    copied_groups = groups.copy(deep=True)
    copied_groups.index = y.columns.copy(deep=True)
    return _PooledInputs(
        x=xx, y=yy, groups=copied_groups,
        date_weights=pd.Series(w.copy(), index=x.index.copy(deep=True),
                               name=None if weights is None else weights.name),
        members=tuple(np.asarray(indices, dtype=int) for indices in membership.values()),
        row_index=x.index.copy(deep=True), factor_columns=x.columns.copy(deep=True),
        response_columns=y.columns.copy(deep=True),
    )


def _weighted_center(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return shifted-offset weighted centering and means, rejecting overflow."""
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            shifted = values - values[-1]
            offset = np.average(shifted, axis=0, weights=weights)
            centered = shifted - offset
            mean = values[-1] + offset
    except (FloatingPointError, OverflowError) as exc:
        raise RuntimeError('weighted centering is not representable in floating point') from exc
    if not np.isfinite(centered).all() or not np.isfinite(mean).all():
        raise RuntimeError('weighted centering produced nonfinite values')
    return centered, mean
