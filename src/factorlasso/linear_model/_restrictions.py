"""Loading restrictions of one fit: signs, prior centres, expert bounds and penalty weights.

Each function computes one layer from the prepared arrays and the estimator settings and
returns NumPy arrays; none of them stores fitted state. The precedence between the layers is
the estimator's documented contract:

1. automatic signs from univariate slopes, pooled within clusters for the group modes;
2. ``auto_sign_excluded_factors`` removes columns from the automatic constraint layer only;
3. explicit ``factors_beta_loading_signs`` cells win over the automatic layer;
4. an OLS prior (optionally restricted by ``factor_for_prior``), overlaid by explicit
   ``factors_beta_prior`` cells;
5. a nonzero prior overrides a conflicting *automatically detected* sign, never an explicit
   one or an excluded column;
6. adaptive penalty weights from the detected slope magnitudes (Zou 2006), aggregated to rows
   (Wang & Leng 2008) or to cluster-by-factor blocks.

The sign-derivation helpers are imported from their module at call time, so patching that
module's functions changes every subsequent fit.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from factorlasso.beta_priors import _compute_joint_ols_prior, _compute_ols_prior
from factorlasso.linear_model._settings import _selected_prior_factors
from factorlasso.linear_model._types import LassoModelType
from factorlasso.prior_bounds import _compute_expert_prior_bounds
from factorlasso.utils._ewm import set_group_loadings

#: Sign-derivation diagnostics and the fitted attribute that stores each one.
SIGN_DIAGNOSTIC_ATTRIBUTES = (
    ('sign_t_stats_', 't_stats'), ('sign_effective_n_', 'effective_n'),
    ('sign_valid_counts_', 'n_obs'),
)


def automatic_signs(
    x: pd.DataFrame,
    x_np: np.ndarray,
    y_np: np.ndarray,
    valid_mask: np.ndarray,
    asset_clusters: Optional[pd.Series],
    sign_span: Optional[float],
    threshold_t: float,
    variance: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Univariate-slope signs, slopes and their diagnostics (N x M each).

    Slopes are computed on the EWMA-demeaned, NaN-masked arrays the solver consumes. Pooling
    mirrors the solver's structural assumption: with a partition, the responses of each
    cluster are pooled and every member shares the cluster's signs; without one (plain LASSO
    or a single response) each response is fitted on its own.
    """
    from factorlasso.sign_constraints import (
        _compute_sign_matrix_per_response, _compute_sign_vector,
    )
    n, m = y_np.shape[1], x_np.shape[1]
    # Solver arrays remain zero-filled; signs must see the original masks.
    offset = len(x) - len(x_np)
    sign_x = np.where(x.iloc[offset:].notna().to_numpy(), x_np, np.nan)
    sign_y = np.where(valid_mask > 0, y_np, np.nan)
    sign_kwargs = dict(auto_sign_threshold_t=threshold_t,
                       ewma_span=sign_span, variance_estimator=variance,
                       return_diagnostics=True)
    if asset_clusters is not None:
        auto_signs_np = np.empty((n, m))
        auto_slopes_np = np.empty((n, m))
        diagnostics = {key: np.empty((n, m))
                       for key in ('t_stats', 'effective_n', 'n_obs')}
        cluster_vals = np.asarray(asset_clusters)
        for c in np.unique(cluster_vals):
            members_idx = np.where(cluster_vals == c)[0]
            signs, slopes, diag = _compute_sign_vector(
                x_arr=sign_x, y_arr=sign_y[:, members_idx], **sign_kwargs)
            auto_signs_np[members_idx] = signs
            auto_slopes_np[members_idx] = slopes
            for key in diagnostics:
                diagnostics[key][members_idx] = diag[key]
    else:
        auto_signs_np, auto_slopes_np, diagnostics = _compute_sign_matrix_per_response(
            x_arr=sign_x, y_arr=sign_y, **sign_kwargs)
    return auto_signs_np, auto_slopes_np, diagnostics


def combine_signs(
    auto_signs_np: Optional[np.ndarray],
    explicit_signs_np: Optional[np.ndarray],
    excluded,
    x_columns: pd.Index,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """The automatic constraint layer after exclusions, and the explicit overlay on it.

    Returns ``(auto_constraint_signs, signs)``. NaN cells of the explicit matrix use the
    automatic layer; any non-NaN explicit cell wins. The original pooled detections stay
    available to the caller for adaptive weights.
    """
    auto_constraint_signs = auto_signs_np
    if auto_signs_np is not None and excluded:
        auto_constraint_signs = auto_signs_np.copy()
        auto_constraint_signs[:, x_columns.get_indexer(excluded)] = np.nan
    signs_np = None
    if auto_constraint_signs is not None and explicit_signs_np is not None:
        # Overlay: explicit per-cell value wins where non-NaN
        signs_np = np.where(
            np.isnan(explicit_signs_np), auto_constraint_signs, explicit_signs_np
        )
    elif auto_constraint_signs is not None:
        signs_np = auto_constraint_signs
    elif explicit_signs_np is not None:
        signs_np = explicit_signs_np
    return auto_constraint_signs, signs_np


def ols_prior(model, x: pd.DataFrame, y: pd.DataFrame,
              eff_span: Optional[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS betas, their R-squared and the prior centres, with ``factor_for_prior`` applied.

    A response named in ``factor_for_prior`` keeps a prior only on its selected factors: one
    factor takes its univariate OLS beta, several take their joint OLS betas.
    """
    ols_beta, ols_r2, prior_np = _compute_ols_prior(
        x.to_numpy(dtype=float), y.to_numpy(dtype=float), span=eff_span,
        min_periods=max(3, model.warmup_period or 0),
        prior_selection_type=model.prior_selection_type,
    )
    if model.factor_for_prior is not None:
        for response, selection in model.factor_for_prior.items():
            factors = _selected_prior_factors(selection)
            if not factors:
                continue
            missing = [factor for factor in factors if factor not in x.columns]
            if missing:
                raise ValueError(f'factor_for_prior names unknown factors {missing!r}')
            if response not in y.columns:
                continue
            i = y.columns.get_loc(response)
            columns = x.columns.get_indexer(factors)
            prior_np[i, :] = 0.0
            if len(factors) == 1:
                j = columns[0]
                prior_np[i, j] = ols_beta[i, j] if np.isfinite(ols_beta[i, j]) else 0.0
            else:
                prior_np[i, columns] = _compute_joint_ols_prior(
                    x.iloc[:, columns].to_numpy(dtype=float),
                    y.iloc[:, i].to_numpy(dtype=float), span=eff_span,
                    min_periods=max(3, model.warmup_period or 0))
    return ols_beta, ols_r2, prior_np


def overlay_explicit_prior(model, x: pd.DataFrame, y: pd.DataFrame,
                           prior_np: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Explicit ``factors_beta_prior`` cells: overrides of an OLS prior, or the prior itself."""
    if model.factors_beta_prior is not None:
        explicit_prior = model.factors_beta_prior.loc[y.columns, x.columns].to_numpy()
        if model.apply_ols_prior:
            if np.isinf(explicit_prior).any():
                raise ValueError('OLS prior overrides must be finite or NaN')
            prior_np = np.where(np.isnan(explicit_prior), prior_np, explicit_prior)
        else:
            prior_np = explicit_prior
    return prior_np


def override_detected_signs(
    signs_np: np.ndarray,
    auto_constraint_signs: np.ndarray,
    prior_np: np.ndarray,
    explicit_signs_np: Optional[np.ndarray],
) -> np.ndarray:
    """Let a nonzero prior's direction replace a conflicting automatically detected sign.

    Prior direction overrides only data-detected constraints, including the automatic zero
    gate. Explicit hard signs and excluded columns retain their precedence; zero/missing
    priors carry no direction.
    """
    prior_signs = np.sign(prior_np)
    override = (np.isfinite(prior_np) & (prior_np != 0.0)
                & np.isfinite(auto_constraint_signs)
                & (prior_signs != auto_constraint_signs))
    if explicit_signs_np is not None:
        override &= np.isnan(explicit_signs_np)
    signs_np = signs_np.copy()
    signs_np[override] = prior_signs[override]
    return signs_np


def expert_prior_bounds(model, x: pd.DataFrame, y: pd.DataFrame, eff_span: Optional[float],
                        ols_r2: pd.DataFrame, ols_beta_prior: pd.DataFrame,
                        effective_beta_prior: pd.DataFrame):
    """Individual floors and caps around the selected OLS priors, with diagnostics.

    A response without an explicit ``factor_for_prior`` selection uses the same raw
    highest-R-squared winner as the OLS prior, chosen before manual overlays or hard signs.
    It is never reselected after a constraint removes it; ties retain input factor order.
    """
    selections = {}
    for asset in y.columns:
        explicit = (None if model.factor_for_prior is None
                    else model.factor_for_prior.get(asset))
        factors = _selected_prior_factors(explicit)
        if not factors:
            scores = ols_r2.loc[asset].dropna()
            factors = (scores.idxmax(),) if not scores.empty else ()
        selections[asset] = factors
    return _compute_expert_prior_bounds(
        x, y, selections, ols_beta_prior, effective_beta_prior,
        model.factors_beta_prior, eff_span, model.expert_prior_bound_n_std,
        model.expert_prior_hac_lags, max(3, model.warmup_period or 0))


def adaptive_penalty_weights(
    model,
    auto_signs_np: Optional[np.ndarray],
    auto_slopes_np: np.ndarray,
    asset_clusters: Optional[pd.Series],
    is_lasso_mode: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Per-cell adaptive L1 weights and their row and block aggregates.

    Only the automatically derived layer contributes; explicit sign overrides carry no
    magnitude information. The cell weights enter the L1 term (Zou 2006) when
    ``l1_weight > 0``; the row aggregates enter the group-L2 term (Wang & Leng 2008), which
    gives the adaptive flag its effect in the production HCGL configuration with
    ``l1_weight = 0``. Cluster-by-factor block weights exist only for FCGL, and not after
    a single-response fit has reduced to plain LASSO.
    """
    from factorlasso.sign_constraints import (
        _adaptive_penalty_weights,
        _aggregate_to_block_weights,
        _aggregate_to_row_weights,
    )
    penalty_weights_np = _adaptive_penalty_weights(
        slopes=auto_slopes_np,
        signs=auto_signs_np if auto_signs_np is not None
        else np.sign(auto_slopes_np),
        gamma=model.auto_sign_adaptive_gamma,
        floor=model.auto_sign_adaptive_floor,
    )
    row_weights_np = _aggregate_to_row_weights(
        cell_weights=penalty_weights_np,
        signs=auto_signs_np if auto_signs_np is not None
        else np.sign(auto_slopes_np),
    )
    col_weights_np = None
    if (model.model_type == LassoModelType.FACTOR_CLUSTER_GROUP_LASSO
            and not is_lasso_mode):
        col_weights_np = _aggregate_to_block_weights(
            cell_weights=penalty_weights_np,
            signs=auto_signs_np if auto_signs_np is not None
            else np.sign(auto_slopes_np),
            group_loadings=set_group_loadings(
                group_data=asset_clusters
            ).to_numpy(),
        )
    return penalty_weights_np, row_weights_np, col_weights_np
