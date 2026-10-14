"""Preparation of signs, centres, bounds and penalty geometry for one fit.

Clustering, the sign matrix, the prior centres, the expert bounds and the adaptive penalty
weights do not depend on ``reg_lambda`` under the standard-error rule. A regularisation
path reuses them across the grid. Cluster-mean budgets instead use an unfloored first
pass at the current penalty, so each grid point repeats preparation. The returned
solver inputs are a :class:`_PreparedFit` in both cases.

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
from factorlasso.priors._budget_modes import leave_one_out_budgets, resolve_budgets
from factorlasso.cluster._hierarchical import (
    apply_cluster_correlation_transform, compute_clusters_from_corr_matrix,
)
from factorlasso.cluster._response import _ClusterGeometry, prepared_response_dependence
from factorlasso.linear_model._restrictions import (
    SIGN_DIAGNOSTIC_ATTRIBUTES, adaptive_penalty_weights, automatic_signs, combine_signs,
    expert_prior_bounds, ols_prior, overlay_explicit_prior, override_detected_signs,
)
from factorlasso.linear_model._settings import (
    expert_bounds_enabled, validate_excluded_factors, validate_fit_modes,
    validate_sign_settings,
)
from factorlasso.linear_model._types import _mode_spec
from factorlasso.utils._panel import get_x_y_np
from factorlasso.linear_model._exclusions import exclusion_mask, overlay_exclusions

#: Prior and bound diagnostics, reset at the start of every preparation.
PRIOR_DIAGNOSTICS = (
    'ols_betas_', 'ols_r2_', 'ols_beta_prior_', 'effective_beta_prior_', 'ols_prior_span_',
    'prior_lower_bounds_', 'prior_upper_bounds_', 'prior_bound_diagnostics_',
    'effective_prior_hac_lags_',
    'effective_factor_for_prior_', 'named_set_diagnostics_', 'prior_bound_contributions_',
    'budget_first_pass_coef_', 'resolved_budgets_', 'budget_sources_', 'budget_rule_status_',
    'excluded_factor_mask_',
)

#: Sign diagnostics, reset after the sign settings have been validated. ``derived_signs_``
#: is the solver-facing sign matrix; a fit that enforces no signs leaves it ``None``.
SIGN_DIAGNOSTICS = (
    'detected_signs_', 'sign_slopes_', 'sign_t_stats_', 'sign_effective_n_',
    'sign_valid_counts_', 'sign_penalty_weights_', 'sign_block_weights_', 'derived_signs_',
)

#: Every fitted attribute derived in preparation, including penalty-dependent cluster budgets.
PREPARATION_STATE = PRIOR_DIAGNOSTICS + ('effective_sign_span_',) + SIGN_DIAGNOSTICS


@dataclass(frozen=True)
class _PreparedFit:
    """Solver inputs for fixed geometry or one penalty-dependent cluster-budget fit."""
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
    budget_first_pass: Optional[Any] = None,
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
    model.excluded_factor_mask_ = exclusion_mask(model, x, y)

    asset_clusters, linkage, cutoff, is_lasso_mode = asset_partition(
        model, x, y, y_np, valid_mask, eff_span, eff_cluster_correlation_span,
        external_clusters, external_linkage, external_cutoff,
    )

    if model.expert_prior_budget_rule == 'cluster_mean':
        if asset_clusters is None:
            raise ValueError(f'cluster_mean requires a response partition, got {asset_clusters!r}')
        first = model.copy(kwargs=dict(
            expert_prior_budget_rule='n_std', expert_prior_bound_n_std=None,
            expert_prior_exposure_budget=None, apply_ols_prior=False,
            factor_for_prior=None, factors_beta_prior=None))
        external = (dict(external_clusters=asset_clusters, external_linkage=linkage,
                         external_cutoff=cutoff)
                    if _mode_spec(model.model_type).external_clusters else {})
        if budget_first_pass is None:
            first.fit(x=x, y=y, span=eff_span,
                      cluster_correlation_span=eff_cluster_correlation_span, **external)
        else:
            # Internal execution adapters may solve the identical clone with
            # explicit tolerances. Validate its specification and fitting panel.
            validate_first_pass(candidate=budget_first_pass, expected=first, x=x, y=y,
                                span=eff_span, clusters=asset_clusters)
            first = budget_first_pass
        model.budget_first_pass_coef_ = first.coef_.copy()
        automatic, model.budget_rule_status_ = leave_one_out_budgets(
            coefficients=first.coef_, clusters=first.clusters_)
        model.resolved_budgets_, model.budget_sources_ = resolve_budgets(
            responses=y.columns, factors=x.columns,
            declared=model.expert_prior_exposure_budget, automatic=automatic)

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
    explicit_signs_np = overlay_exclusions(explicit_signs_np, model.excluded_factor_mask_)
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
    if expert_bounds_enabled(model):
        lower, upper, diagnostics = expert_prior_bounds(
            model, x, y, eff_span, model.ols_r2_, model.ols_beta_prior_,
            model.effective_beta_prior_, signs_np=signs_np,
        )
        model.prior_lower_bounds_ = lower
        model.prior_upper_bounds_ = upper
        model.prior_bound_diagnostics_ = diagnostics
        model.effective_prior_hac_lags_ = model.expert_prior_hac_lags
        lower_bounds_np, upper_bounds_np = lower.to_numpy(), upper.to_numpy()
        if model.expert_prior_bound_centre == 'floor':
            prior_np = floor_centred_penalty(prior_np, lower_bounds_np, upper_bounds_np)

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


def validate_first_pass(*, candidate: Any, expected: Any, x: pd.DataFrame,
                        y: pd.DataFrame, span: Optional[float],
                        clusters: pd.Series) -> None:
    """Reject stale or differently configured internal first-pass results."""
    for name, value in expected.get_params().items():
        other = candidate.get_params().get(name)
        if isinstance(value, (pd.DataFrame, pd.Series)):
            equal = isinstance(other, type(value)) and value.equals(other)
        elif isinstance(value, np.ndarray):
            equal = isinstance(other, np.ndarray) and np.array_equal(value, other, equal_nan=True)
        else:
            equal = value == other
        if not equal:
            raise ValueError(f'budget_first_pass parameter {name!r} differs, got {other!r}')
    if (candidate.x_ is None or candidate.y_ is None or not candidate.x_.equals(x)
            or not candidate.y_.equals(y) or candidate.effective_span_ != span):
        raise ValueError('budget_first_pass has a different panel or span')
    if candidate.clusters_ is None:
        raise ValueError('budget_first_pass has no fitted clusters')
    eligible = candidate.clusters_.index
    if not candidate.clusters_.equals(clusters.reindex(eligible)):
        raise ValueError('budget_first_pass has different cluster assignments')


def floor_centred_penalty(prior_np: np.ndarray, lower_np: np.ndarray,
                          upper_np: np.ndarray) -> np.ndarray:
    """Return the penalty centres with every floored cell moved from its prior to its bound.

    With an elementwise absolute penalty, ``|b - floor|`` is linear on the feasible side of the
    floor, so the penalty is constant along the collinear direction of the floored loading and
    a competitor with a zero centre (Sepp 2026, Proposition 5(i)). A group norm keeps the
    floored cell in its block: the block geometry then shares the room above the floor among
    the block's loadings, and the result is not the separable one.
    """
    centre = prior_np.copy()
    centre[np.isfinite(lower_np)] = lower_np[np.isfinite(lower_np)]
    centre[np.isfinite(upper_np)] = upper_np[np.isfinite(upper_np)]
    return centre
