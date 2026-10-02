"""Causal nowcast of responses from realised factors and residual alpha.

See :meth:`factorlasso.LassoModel.nowcast` for the contract. The calculation reads only the
fitted snapshots recorded by ``fit`` (betas, the original-unit residual panel, the validity
mask and the recorded spans), never the caller-owned ``x_``/``y_`` frames.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from factorlasso.linear_model._solvers.common import _compute_solver_weights
from factorlasso.linear_model._types import LassoNowcastResult
from factorlasso.utils._ewm import _validate_span, compute_ewm


def nowcast(model, x: pd.DataFrame, alpha_span: Optional[float] = None) -> LassoNowcastResult:
    """Validate the fitted provenance and the targets, then assemble the nowcast."""
    fitted_state = (
        model.coef_,
        model.estimation_result_,
        model.alpha_const_,
        model.valid_mask_,
        model.fit_demeaned_,
        model.nowcast_residuals_,
        model.nowcast_factors_complete_,
        model.nowcast_final_response_complete_,
    )
    if any(value is None for value in fitted_state):
        raise RuntimeError("Model not fitted. Call fit() first.")
    if model.fit_demeaned_ is not True:
        raise ValueError("nowcast requires a model fitted with demean=True")
    _validate_span(alpha_span, name="alpha_span")
    if model.nowcast_factors_complete_ is not True:
        raise ValueError("nowcast requires complete fitted factor rows")
    if model.nowcast_final_response_complete_ is not True:
        raise ValueError("nowcast requires a fully observed final response row")

    betas = model.coef_.copy(deep=True)
    residuals = model.nowcast_residuals_.copy(deep=True)
    if not np.isfinite(betas.to_numpy(dtype=float)).all():
        raise ValueError("nowcast requires finite fitted betas")
    residual_values = residuals.to_numpy(dtype=float)
    if np.isinf(residual_values).any():
        raise ValueError("nowcast residual history cannot contain infinite values")
    if not np.isfinite(residual_values[-1]).all():
        raise ValueError("nowcast requires finite terminal residuals")

    if not isinstance(x, pd.DataFrame):
        raise TypeError("x must be a pandas DataFrame")
    if x.empty:
        raise ValueError("x must contain at least one target factor row")
    if not x.columns.equals(betas.columns):
        raise ValueError(
            "x columns must exactly match fitted factor columns in the same order"
        )
    if not isinstance(residuals.index, pd.DatetimeIndex):
        raise TypeError("fitted data must have a DatetimeIndex for nowcast")
    if not residuals.index.is_monotonic_increasing or not residuals.index.is_unique:
        raise ValueError("fitted DatetimeIndex must be sorted and unique")
    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError("x must have a DatetimeIndex")
    if not x.index.is_monotonic_increasing or not x.index.is_unique:
        raise ValueError("x DatetimeIndex must be sorted and unique")
    if residuals.index.tz != x.index.tz:
        raise ValueError("x and fitted DatetimeIndex values must use the same timezone")
    if bool(np.any(x.index <= residuals.index[-1])):
        raise ValueError("every x target date must be strictly after the fit cutoff")

    target_factors = x.copy(deep=True)
    if not np.isfinite(target_factors.to_numpy(dtype=float)).all():
        raise ValueError("x target factor values must all be finite")

    resolved_alpha_span = model.effective_span_ if alpha_span is None else alpha_span
    stat_alpha = pd.Series(index=betas.index.copy(), dtype=float, name="stat_alpha")
    if resolved_alpha_span is None:
        stat_alpha.loc[:] = residuals.mean(axis=0, skipna=True)
    else:
        for response in residuals.columns:
            first_valid = residuals[response].first_valid_index()
            history = residuals.loc[first_valid:, response]
            stat_alpha.loc[response] = compute_ewm(
                history, span=resolved_alpha_span
            ).iloc[-1]

    factor_component = target_factors @ betas.T
    prediction = factor_component.add(stat_alpha, axis="columns")

    sqrt_solver_weights = _compute_solver_weights(
        t=model.valid_mask_.shape[0],
        n_y=len(betas.index),
        span=model.effective_span_,
        valid_mask=model.valid_mask_,
    )
    loss_weights = np.square(sqrt_solver_weights)
    weight_sums = np.sum(loss_weights, axis=0)
    squared_weight_sums = np.sum(np.square(loss_weights), axis=0)
    effective_n_obs = np.divide(
        np.square(weight_sums),
        squared_weight_sums,
        out=np.full_like(weight_sums, np.nan),
        where=squared_weight_sums > 0.0,
    )

    fit_start_dates = []
    for response in residuals.columns:
        fit_start_dates.append(residuals[response].first_valid_index())
    result = model.estimation_result_
    diagnostics = pd.DataFrame(
        {
            "fit_start_date": fit_start_dates,
            "fit_end_date": residuals.index[-1],
            "n_response_obs_to_cutoff": residuals.notna().sum(axis=0).to_numpy(),
            "n_obs_used": model.valid_mask_.sum(axis=0).astype(int),
            "effective_n_obs": effective_n_obs,
            "beta_span": model.effective_span_,
            "alpha_span": resolved_alpha_span,
            "fit_demeaned": model.fit_demeaned_,
            "stat_alpha": stat_alpha.to_numpy(copy=True),
            "alpha_const": model.alpha_const_.to_numpy(copy=True),
            "factorlasso_fit_ss_total_ewma_demeaned": result.ss_total.copy(),
            "factorlasso_fit_ss_res_ewma_demeaned": result.ss_res.copy(),
            "factorlasso_fit_r2_ewma_demeaned": result.r2.copy(),
            "n_nonzero_betas": np.count_nonzero(
                ~np.isclose(betas.to_numpy(dtype=float), 0.0), axis=1
            ),
        },
        index=betas.index.copy(),
    )
    return LassoNowcastResult(
        prediction=prediction.copy(deep=True),
        factor_component=factor_component.copy(deep=True),
        target_factors=target_factors.copy(deep=True),
        stat_alpha=stat_alpha.copy(deep=True),
        betas=betas.copy(deep=True),
        residuals=residuals.copy(deep=True),
        diagnostics=diagnostics.copy(deep=True),
    )
