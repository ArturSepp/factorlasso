"""The ``reg_lambda``-independent preparation of one fit.

Clustering, the sign matrix, the prior centres, the expert bounds and the adaptive penalty
weights do not depend on ``reg_lambda``. :func:`prepare_fit` derives them once from the
prepared panel, so a regularisation path reuses them across the grid, and returns the solver
inputs as a :class:`_PreparedFit`.

The preparation diagnostics (:data:`PREPARATION_STATE`) are stored on the model as they are
derived, in the order and at the points where earlier releases stored them; a fit that raises
part-way therefore leaves the same partial state as before. Every one of them is reset first,
so a refit never keeps a diagnostic of an earlier configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from factorlasso.priors._ols import _zero_incompatible_priors
from factorlasso.cluster._hierarchical import (
    apply_cluster_correlation_transform, compute_clusters_from_corr_matrix,
)
from factorlasso.cluster._response import _ClusterGeometry, prepared_response_dependence
from factorlasso.linear_model._restrictions import (
    SIGN_DIAGNOSTIC_ATTRIBUTES, adaptive_penalty_weights, automatic_signs, combine_signs,
    expert_prior_bounds, ols_prior, overlay_explicit_prior, override_detected_signs,
)
from factorlasso.linear_model._settings import (
    validate_excluded_factors, validate_fit_modes, validate_sign_settings,
)
from factorlasso.linear_model._types import _mode_spec
from factorlasso.utils._panel import get_x_y_np

#: Prior and bound diagnostics, reset at the start of every preparation.
PRIOR_DIAGNOSTICS = (
    'ols_betas_', 'ols_r2_', 'ols_beta_prior_', 'effective_beta_prior_', 'ols_prior_span_',
    'prior_lower_bounds_', 'prior_upper_bounds_', 'prior_bound_diagnostics_',
    'effective_prior_hac_lags_',
)

#: Sign diagnostics, reset after the sign settings have been validated. ``derived_signs_``
#: is the solver-facing sign matrix; a fit that enforces no signs leaves it ``None``.
SIGN_DIAGNOSTICS = (
    'detected_signs_', 'sign_slopes_', 'sign_t_stats_', 'sign_effective_n_',
    'sign_valid_counts_', 'sign_penalty_weights_', 'sign_block_weights_', 'derived_signs_',
)

#: Every fitted attribute that the preparation derives (independent of ``reg_lambda``).
PREPARATION_STATE = PRIOR_DIAGNOSTICS + ('effective_sign_span_',) + SIGN_DIAGNOSTICS


@dataclass(frozen=True)
class _PreparedFit:
    """reg_lambda-independent solver inputs derived once by ``_prepare_fit``."""
    asset_clusters: Optional[pd.Series]
    linkage: Optional[Any]
    cutoff: Optional[float]
    is_lasso_mode: bool
    signs_np: Optional[np.ndarray]
    prior_np: Optional[np.ndarray]
    penalty_weights_np: Optional[np.ndarray]
    row_weights_np: Optional[np.ndarray]
    col_weights_np: Optional[np.ndarray]
    lower_bounds_np: Optional[np.ndarray]
    upper_bounds_np: Optional[np.ndarray]


def discover_response_clusters(
    model,
    x: pd.DataFrame,
    y: pd.DataFrame,
    y_np: np.ndarray,
    valid_mask: np.ndarray,
    eff_span: Optional[float],
    eff_cluster_correlation_span: Optional[float],
) -> Tuple[pd.Series, np.ndarray, float]:
    """Partition the responses from their dependence matrix (HCGL, FCGL, cooperative HCGL).

    When the two spans are equal, the solver-ready panel is reused so the default path stays
    numerically identical. A distinct clustering span must also own EWMA demeaning;
    changing only the final dependence weights would still leak the beta span into cluster
    discovery through the transformed response panel.

    ``get_x_y_np`` zero-fills missing responses so the CVXPY quadratic losses see a finite
    array (``valid_mask`` removes those cells from the loss). A dependence estimator would
    read a zero-filled cell as a zero return rather than as no observation, shrinking the
    correlations of assets with longer leading-NaN prefixes towards zero and propagating
    into linkage distances, cuts and the group estimates. The missing cells are therefore
    restored to NaN first, and the NaN-aware recursions estimate each pair over its valid
    window.

    By default the dependence uses the same observation weighting as the solver loss: the
    sample Pearson correlation (pairwise-complete) when the span is ``None``, the EWMA(span)
    correlation otherwise. Versions before 0.5.1 always used a 0.94 EWMA (about a 32-period
    RiskMetrics span), contradicting the documented Pearson ``corr(Y)``. Pearson and
    Spearman are invariant to centring; Gerber is not, and sees the same demeaned panel as
    the loss.
    """
    if eff_cluster_correlation_span == eff_span:
        clustering_y_np = y_np
        clustering_valid_mask = valid_mask
    else:
        _, clustering_y_np, clustering_valid_mask = get_x_y_np(
            x=x,
            y=y,
            span=eff_cluster_correlation_span,
            demean=model.demean,
        )
    corr_df = prepared_response_dependence(
        clustering_y_np, clustering_valid_mask, y.columns, model.dependence_measure,
        eff_cluster_correlation_span, model.gerber_threshold,
    )
    corr_df = apply_cluster_correlation_transform(
        corr_df, transform=model.cluster_correlation_transform
    )
    return compute_clusters_from_corr_matrix(
        corr_df, **_ClusterGeometry.from_model(model).as_kwargs(),
    )


def asset_partition(
    model,
    x: pd.DataFrame,
    y: pd.DataFrame,
    y_np: np.ndarray,
    valid_mask: np.ndarray,
    eff_span: Optional[float],
    eff_cluster_correlation_span: Optional[float],
    external_clusters: Optional[pd.Series] = None,
    external_linkage: Optional[np.ndarray] = None,
    external_cutoff: Optional[float] = None,
):
    """The response partition of the group modes and whether the fit reduces to LASSO.

    Returns ``(asset_clusters, linkage, cutoff, is_lasso_mode)``. The partition is ``None``
    for plain LASSO, UniLasso and a single response under a row- or block-grouped penalty;
    the user groups for GROUP_LASSO and COOPERATIVE_GROUP_LASSO; the supplied external
    partition, or the discovered one, for the cluster modes.
    """
    spec = _mode_spec(model.model_type)
    asset_clusters: Optional[pd.Series] = None
    linkage = None
    cutoff = None
    is_lasso_mode = (
        spec.solver == "lasso"
        or (y_np.shape[1] == 1 and spec.single_response_lasso)
    )
    if is_lasso_mode:
        # asset_clusters stays None → per-y-column sign derivation
        pass
    elif spec.grouping == "user":
        asset_clusters = model.group_data[y.columns]
    elif spec.grouping == "discovered":
        if external_clusters is not None:
            asset_clusters = external_clusters.reindex(y.columns)
            if asset_clusters.isna().any():
                missing = asset_clusters[asset_clusters.isna()].index.tolist()
                raise ValueError(
                    f"external_clusters is missing assignments for {missing!r}"
                )
            linkage = external_linkage
            cutoff = external_cutoff
        else:
            asset_clusters, linkage, cutoff = discover_response_clusters(
                model, x, y, y_np, valid_mask, eff_span, eff_cluster_correlation_span,
            )
    return asset_clusters, linkage, cutoff, is_lasso_mode


def prepare_fit(
    model,
    x: pd.DataFrame,
    y: pd.DataFrame,
    x_np: np.ndarray,
    y_np: np.ndarray,
    valid_mask: np.ndarray,
    eff_span: Optional[float],
    eff_cluster_correlation_span: Optional[float],
    external_clusters: Optional[pd.Series] = None,
    external_linkage: Optional[np.ndarray] = None,
    external_cutoff: Optional[float] = None,
) -> _PreparedFit:
    """Derive the solver inputs of one fit and store its preparation diagnostics on ``model``.

    The partition is computed once and shared by the sign derivation and the solver. Signs,
    priors, bounds and adaptive weights follow the precedence documented in
    :mod:`factorlasso.linear_model._restrictions`.
    """
    validate_fit_modes(model)
    for name in PRIOR_DIAGNOSTICS:
        setattr(model, name, None)
    validate_sign_settings(model)
    model.effective_sign_span_ = None
    for name in SIGN_DIAGNOSTICS:
        setattr(model, name, None)
    validate_excluded_factors(model, x)
    excluded = model.auto_sign_excluded_factors

    asset_clusters, linkage, cutoff, is_lasso_mode = asset_partition(
        model, x, y, y_np, valid_mask, eff_span, eff_cluster_correlation_span,
        external_clusters, external_linkage, external_cutoff,
    )

    # ── Signs: automatic layer (with diagnostics) and explicit overlay ──
    auto_signs_np = None
    auto_slopes_np = None
    explicit_signs_np = None
    if model.auto_sign_constraints:
        sign_span = eff_span if model.auto_sign_use_fit_span else model.auto_sign_ewma_span
        model.effective_sign_span_ = sign_span
        auto_signs_np, auto_slopes_np, diagnostics = automatic_signs(
            x, x_np, y_np, valid_mask, asset_clusters, sign_span,
            model.auto_sign_threshold_t, model.auto_sign_variance,
        )
        model.detected_signs_ = pd.DataFrame(auto_signs_np, index=y.columns, columns=x.columns)
        model.sign_slopes_ = pd.DataFrame(auto_slopes_np, index=y.columns, columns=x.columns)
        for attr, key in SIGN_DIAGNOSTIC_ATTRIBUTES:
            setattr(model, attr, pd.DataFrame(
                diagnostics[key], index=y.columns, columns=x.columns))
    if model.factors_beta_loading_signs is not None:
        explicit_signs_np = model.factors_beta_loading_signs.loc[
            y.columns, x.columns
        ].to_numpy()
    auto_constraint_signs, signs_np = combine_signs(
        auto_signs_np, explicit_signs_np, excluded, x.columns,
    )

    # ── Prior centres ──
    prior_np = None
    if model.apply_ols_prior:
        ols_beta, ols_r2, prior_np = ols_prior(model, x, y, eff_span)
        model.ols_betas_ = pd.DataFrame(ols_beta, index=y.columns, columns=x.columns)
        model.ols_r2_ = pd.DataFrame(ols_r2, index=y.columns, columns=x.columns)
        model.ols_beta_prior_ = pd.DataFrame(
            prior_np.copy(), index=y.columns, columns=x.columns,
        )
        model.ols_prior_span_ = eff_span
    prior_np = overlay_explicit_prior(model, x, y, prior_np)
    if auto_constraint_signs is not None and prior_np is not None:
        signs_np = override_detected_signs(
            signs_np, auto_constraint_signs, prior_np, explicit_signs_np,
        )

    # The final solver-facing sign matrix, stored only when the solver receives it: the
    # UniLasso and cooperative solvers take no sign constraint.
    if signs_np is not None and _mode_spec(model.model_type).hard_constraints:
        model.derived_signs_ = pd.DataFrame(
            signs_np, index=y.columns, columns=x.columns,
        )

    if model.apply_ols_prior:
        # Only remaining hard-constraint conflicts lose their prior.
        prior_np = _zero_incompatible_priors(prior_np, signs_np, nonneg=model.nonneg)
        model.effective_beta_prior_ = pd.DataFrame(
            prior_np.copy(), index=y.columns, columns=x.columns,
        )

    # ── Expert bounds ──
    lower_bounds_np = upper_bounds_np = None
    if model.expert_prior_bound_n_std is not None:
        lower, upper, diagnostics = expert_prior_bounds(
            model, x, y, eff_span, model.ols_r2_, model.ols_beta_prior_,
            model.effective_beta_prior_,
        )
        model.prior_lower_bounds_ = lower
        model.prior_upper_bounds_ = upper
        model.prior_bound_diagnostics_ = diagnostics
        model.effective_prior_hac_lags_ = model.expert_prior_hac_lags
        lower_bounds_np, upper_bounds_np = lower.to_numpy(), upper.to_numpy()

    # ── Adaptive penalty weights (Zou 2006; opt-in) ──
    penalty_weights_np = None
    row_weights_np = None
    col_weights_np = None
    want_adaptive = model.auto_sign_constraints and model.auto_sign_adaptive_weights
    if want_adaptive and auto_slopes_np is not None:
        penalty_weights_np, row_weights_np, col_weights_np = adaptive_penalty_weights(
            model, auto_signs_np, auto_slopes_np, asset_clusters, is_lasso_mode,
        )
    if penalty_weights_np is not None:
        model.sign_penalty_weights_ = pd.DataFrame(
            penalty_weights_np, index=y.columns, columns=x.columns)
    model.sign_block_weights_ = None if col_weights_np is None else col_weights_np.copy()

    return _PreparedFit(
        asset_clusters=asset_clusters, linkage=linkage, cutoff=cutoff,
        is_lasso_mode=is_lasso_mode, signs_np=signs_np, prior_np=prior_np,
        penalty_weights_np=penalty_weights_np,
        row_weights_np=row_weights_np, col_weights_np=col_weights_np,
        lower_bounds_np=lower_bounds_np, upper_bounds_np=upper_bounds_np,
    )
