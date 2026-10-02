"""Fitted state of a solved fit: warmup zeroing, coefficients, intercepts and bookkeeping.

A fitted model's state has two parts. The preparation state (signs, priors, bounds, adaptive
weights; :data:`~factorlasso.linear_model._preparation.PREPARATION_STATE`) does not depend on
``reg_lambda``; :func:`preparation_state` copies it from the model that was prepared.
:func:`fitted_state` enumerates the solve-dependent attributes. :func:`install_fitted_state`
stores a record. A single fit and every point of a regularisation path share this
post-processing, so a path model carries exactly the diagnostics of a fresh fit.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from factorlasso.linear_model._preparation import PREPARATION_STATE
from factorlasso.linear_model._solvers.common import _compute_solver_weights
from factorlasso.linear_model._types import LassoEstimationResult, LassoModelType


def fitted_state(
    model,
    result: LassoEstimationResult,
    x: pd.DataFrame,
    y: pd.DataFrame,
    valid_mask: np.ndarray,
    eff_span: Optional[float],
    eff_cluster_correlation_span: Optional[float],
    asset_clusters: Optional[pd.Series],
    linkage,
    cutoff,
    stacklevel: int = 3,
) -> Dict[str, Any]:
    """The solve-dependent fitted attributes, in the order they are stored.

    Responses with fewer than ``warmup_period`` valid observations are zeroed, in ``result``
    itself, which becomes ``estimation_result_``. ``stacklevel`` attributes the warmup
    warning to the estimator method that requested the state.
    """
    # Zero out betas for variables with insufficient history
    est_beta = result.estimated_beta
    short_assets: Optional[pd.Index] = None
    if model.warmup_period is not None:
        n_valid = np.count_nonzero(~np.isnan(y.to_numpy()), axis=0)
        short = n_valid < model.warmup_period
        if np.any(short):
            est_beta[short, :] = 0.0
            # Capture the zeroed assets for the warning below and for the cluster
            # assignment at the end, which must drop these same assets so clusters_,
            # coef_, and the per-asset diagnostics stay mutually consistent. Without
            # this the zeroed-beta assets would still receive spurious singleton
            # cluster labels that inflate downstream n_clusters and pollute
            # cluster-based risk attribution / regime diagnostics.
            short_assets = y.columns[short]
            zeroed = ', '.join(map(str, short_assets[:10]))
            if len(short_assets) > 10:
                zeroed += f", ... (+{len(short_assets) - 10} more)"
            warnings.warn(
                f"factorlasso: {int(np.sum(short))} of "
                f"{len(short)} assets had fewer than "
                f"warmup_period={model.warmup_period} valid "
                f"observations and were zeroed: {zeroed}",
                stacklevel=stacklevel,
            )
            for attr in ('alpha', 'ss_total', 'ss_res', 'r2'):
                getattr(result, attr)[short] = np.nan

    state: Dict[str, Any] = {"x_": x, "y_": y, "valid_mask_": valid_mask}
    n_loss_rows = valid_mask.shape[0]
    state["n_loss_rows_"] = n_loss_rows
    loss_weights = _compute_solver_weights(
        n_loss_rows, len(y.columns), eff_span, valid_mask)
    if model.model_type == LassoModelType.UNILASSO:
        # UniLasso's stage-two objective is unweighted on each valid window.
        mass = np.sum(valid_mask, axis=0)
        denominator = mass.copy()
    else:
        mass = np.sum(np.square(loss_weights), axis=0)
        denominator = (mass.copy() if model.loss_normalization == 'weight_sum'
                       else np.full(len(y.columns), n_loss_rows, dtype=float))
    state["loss_weight_mass_"] = pd.Series(mass, index=y.columns, name='loss_weight_mass')
    state["loss_denominator_"] = pd.Series(denominator, index=y.columns,
                                           name='loss_denominator')
    state["effective_span_"] = eff_span
    state["effective_cluster_correlation_span_"] = eff_cluster_correlation_span
    coef = pd.DataFrame(
        est_beta, index=y.columns, columns=x.columns,
    )
    state["coef_"] = coef
    # Capture one original-unit T x N residual panel for causal nowcasts.
    # Existing x_/y_ intentionally preserve their historical aliasing
    # contract, so eligibility and alpha cannot be reconstructed from
    # those mutable caller-owned frames after fit().
    x_snapshot = x.to_numpy(dtype=float, copy=True)
    y_snapshot = y.to_numpy(dtype=float, copy=True)
    beta_snapshot = coef.to_numpy(dtype=float, copy=True)
    state["fit_demeaned_"] = bool(model.demean)
    state["nowcast_residuals_"] = pd.DataFrame(
        y_snapshot - x_snapshot @ beta_snapshot.T,
        index=y.index.copy(),
        columns=y.columns.copy(),
        copy=True,
    )
    state["nowcast_factors_complete_"] = bool(np.isfinite(x_snapshot).all())
    state["nowcast_final_response_complete_"] = bool(
        len(y_snapshot) > 0 and np.isfinite(y_snapshot[-1]).all()
    )
    # intercept_ : preserved from v0.3.3 — the raw solver output, namely
    # the EWMA-weighted residual mean on the demeaned data. This is the
    # mechanical artefact of fitting a no-intercept model on centered
    # inputs (see :class:`LassoEstimationResult` docstring on ``alpha``).
    # It is NOT the regression intercept in the original
    # ``y = α + Xβ + ε`` representation; for span=None it is identically
    # zero by construction. Kept under this name for back-compat with
    # any analytics that read ``model.intercept_``.
    state["intercept_"] = pd.Series(
        result.alpha, index=y.columns, name='intercept',
    )
    state["alpha_const_"] = economic_intercept(
        x, y, est_beta, eff_span, short_assets,
    )
    state["estimation_result_"] = result
    # asset_clusters is already populated by the upstream dispatch (HCGL
    # output for HIERARCHICAL_CLUSTER_GROUP_LASSO, group_data for GROUP_LASSO, None
    # for plain LASSO). Filter out cluster labels for ghost assets —
    # assets whose betas were zeroed above because they had fewer than
    # ``warmup_period`` valid observations. Dropping them here keeps
    # ``clusters_`` consistent with ``coef_`` (zeroed) and per-asset
    # diagnostics (NaN), so downstream consumers that count or analyse
    # clusters see only assets that actually contributed to the fit.
    # Without this, pre-launch / short-history assets receive
    # placeholder singleton labels that inflate ``n_clusters`` in early
    # history (observed: 83 raw vs 31 real on a 160-asset multi-asset
    # universe at 2002-12-31).
    if asset_clusters is not None and short_assets is not None and len(short_assets) > 0:
        asset_clusters = asset_clusters.drop(short_assets, errors='ignore')
    state["clusters_"] = asset_clusters
    state["linkage_"] = linkage
    state["cutoff_"] = cutoff
    return state


def economic_intercept(
    x: pd.DataFrame,
    y: pd.DataFrame,
    est_beta: np.ndarray,
    eff_span: Optional[float],
    short_assets: Optional[pd.Index],
) -> pd.Series:
    """``alpha_const_``: the economic intercept α of ``y = α + Xβ + ε``.

    Reconstructed from the same weighting that produced β: for ``span=None`` (uniform
    weights) this is the sample-mean reconstruction ``α = ȳ - x̄·β``; for ``span=integer``
    it uses EWMA-weighted means with the same weights factorlasso applies in the loss
    function. This guarantees the (α, β) pair is internally consistent — both are
    estimators on the same weighted objective. Using sample means with EWMA-weighted β
    would mix two different estimators and the resulting α would not be the
    weighted-residual-mean that pairs with β.

    For ``span=None`` and unconstrained coefficients this equals the OLS intercept exactly.
    """
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    T_full, n_x_full = x_arr.shape
    n_y_full = y_arr.shape[1]
    # Per-response valid mask of full (pre-demean) length T_full.
    # NaN in y[:, j] or all-NaN in x[t] makes row invalid for response j.
    y_valid = (~np.isnan(y_arr)).astype(float)
    x_row_valid = (~np.isnan(x_arr).all(axis=1)).astype(float)
    valid_full = y_valid * x_row_valid[:, None]
    # Weights aligned with the original T_full rows. Match factorlasso's
    # loss function: w_t² = (1 - 2/(span+1))^(T-1-t) for EWMA, uniform
    # for span=None.
    if eff_span is None:
        w_sq_full = np.ones(T_full)
    else:
        lam = 1.0 - 2.0 / (float(eff_span) + 1.0)
        w_sq_full = lam ** np.arange(T_full - 1, -1, -1)
    x_safe = np.nan_to_num(x_arr)
    y_safe = np.nan_to_num(y_arr)
    x_means = np.zeros((n_y_full, n_x_full))
    y_means = np.zeros(n_y_full)
    for j in range(n_y_full):
        w_j = w_sq_full * valid_full[:, j]
        tot = w_j.sum()
        if tot > 0.0:
            w_j_norm = w_j / tot
            x_means[j] = w_j_norm @ x_safe
            y_means[j] = w_j_norm @ y_safe[:, j]
        else:
            x_means[j] = np.nan
            y_means[j] = np.nan
    beta_arr = np.where(np.isnan(est_beta), 0.0, est_beta)
    alpha_const_arr = y_means - np.einsum('ij,ij->i', beta_arr, x_means)
    alpha_const_ser = pd.Series(
        alpha_const_arr, index=y.columns, name='alpha_const',
    )
    if short_assets is not None:
        alpha_const_ser.loc[short_assets] = np.nan
    return alpha_const_ser


def owned(value):
    """An independent copy of a mutable fitted value; immutable values are returned as is."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy(deep=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def preparation_state(prepared_model) -> Dict[str, Any]:
    """Independent copies of every preparation diagnostic of ``prepared_model``."""
    return {name: owned(getattr(prepared_model, name)) for name in PREPARATION_STATE}


def install_fitted_state(model, state: Dict[str, Any]) -> None:
    """Store a fitted-state record on ``model``."""
    for name, value in state.items():
        setattr(model, name, value)


def fitted_attribute_names(model) -> Tuple[str, ...]:
    """The fitted attributes of an estimator: its dataclass fields with a trailing underscore."""
    return tuple(field.name for field in fields(model) if field.name.endswith("_"))


@contextmanager
def fitted_state_transaction(model, keep: bool = True) -> Iterator[None]:
    """Restore ``model``'s fitted attributes unless the block completes and ``keep`` is set.

    Preparation stores its diagnostics on the model as it derives them, which estimator
    hooks may observe. A fit that raises therefore restores every fitted attribute to the
    object it held before (an attribute never set returns to the class default); constructor
    parameters, including ones changed by ``set_params``, are not touched. With
    ``keep=False`` the attributes are restored after a successful block as well, which keeps
    a regularisation-path template unchanged.
    """
    names = fitted_attribute_names(model)
    namespace = vars(model)
    saved = {name: namespace[name] for name in names if name in namespace}

    def restore() -> None:
        for name in names:
            if name in saved:
                namespace[name] = saved[name]
            else:
                namespace.pop(name, None)

    try:
        yield
    except BaseException:
        restore()
        raise
    if not keep:
        restore()
