"""Validation of the estimator configuration and resolution of per-fit settings.

The checks run in the order the estimator has always applied them, with the same exception
types and messages: configuration checks on construction, the mode-dependent checks again at
each fit (parameters may have been changed by ``set_params``), and the input coercion of
``fit``. Nothing here changes a value the caller supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from factorlasso.beta_priors import _validate_prior_selection_type
from factorlasso.cluster_smoothing import ClusterSmootherType
from factorlasso.cluster_utils import (
    VALID_LINKAGE_METHODS, ClusterCorrelationTransform, DistanceTransform,
)
from factorlasso.dependence_utils import DependenceMeasure
from factorlasso.linear_model._solvers.common import _validate_loss_normalization
from factorlasso.linear_model._types import LassoModelType, _mode_spec
from factorlasso.prior_bounds import _validate_expert_bound_settings
from factorlasso.utils._ewm import _validate_span


def _selected_prior_factors(value) -> tuple:
    """Validate a scalar or ordered factor list without mutating estimator parameters."""
    if pd.api.types.is_scalar(value):
        return () if pd.isna(value) else (value,)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('factor_for_prior requires a scalar label or nonempty list/tuple')
    if any(not pd.api.types.is_scalar(label) or pd.isna(label) for label in value):
        raise ValueError('factor_for_prior selections require nonmissing scalar labels')
    if pd.Index(value).has_duplicates:
        raise ValueError('factor_for_prior selections must contain unique factors')
    return tuple(value)


def validate_loss_mode(model) -> None:
    """Reject invalid or unsupported objective conventions, including after mutation."""
    _validate_loss_normalization(model.loss_normalization)
    if (model.model_type == LassoModelType.UNILASSO
            and model.loss_normalization != 'sample'):
        raise ValueError('UNILASSO uses its existing unweighted two-stage loss; '
                         'loss_normalization must be sample')


def validate_sign_inputs_mode(model) -> None:
    """Reject hard sign inputs for the solvers that cannot enforce them."""
    if _mode_spec(model.model_type).hard_constraints:
        return
    if model.factors_beta_loading_signs is not None:
        raise ValueError(
            'factors_beta_loading_signs is not supported by the '
            f'{model.model_type.name} solver, which takes no sign constraint'
        )
    if model.nonneg:
        raise ValueError(
            f'nonneg=True is not supported by the {model.model_type.name} solver, '
            'which takes no sign constraint'
        )


def validate_ols_prior_mode(model) -> None:
    """Reject ambiguous flags and a mode whose solver has no beta prior."""
    _validate_expert_bound_settings(
        model.expert_prior_bound_n_std, model.expert_prior_hac_lags,
        model.expert_prior_hac_lags_freq_dict)
    if model.expert_prior_bound_n_std is not None:
        if not model.apply_ols_prior:
            raise ValueError('expert_prior_bound_n_std requires apply_ols_prior=True')
        if not _mode_spec(model.model_type).hard_constraints:
            raise ValueError('expert prior bounds require LASSO, group LASSO, HCGL or FCGL')
    _validate_prior_selection_type(model.prior_selection_type)
    if not isinstance(model.apply_ols_prior, (bool, np.bool_)):
        raise ValueError('apply_ols_prior must be a boolean')
    if model.factor_for_prior is not None:
        if not model.apply_ols_prior:
            raise ValueError('factor_for_prior requires apply_ols_prior=True')
        if not isinstance(model.factor_for_prior, (Mapping, pd.Series)):
            raise TypeError('factor_for_prior must be a mapping or pandas Series')
        if (isinstance(model.factor_for_prior, pd.Series)
                and not model.factor_for_prior.index.is_unique):
            raise ValueError('factor_for_prior response labels must be unique')
        for _, value in model.factor_for_prior.items():
            _selected_prior_factors(value)
    if model.model_type == LassoModelType.UNILASSO:
        if model.apply_ols_prior:
            raise ValueError('apply_ols_prior is not supported by the UNILASSO solver')
        # The UniLasso solver takes no beta prior; reject rather than ignore it silently.
        if model.factors_beta_prior is not None:
            raise ValueError(
                'factors_beta_prior is not supported by the UNILASSO solver, '
                'which takes no beta prior'
            )


def validate_configuration(model) -> None:
    """Construction-time checks of every configuration parameter."""
    validate_loss_mode(model)
    validate_ols_prior_mode(model)
    validate_sign_inputs_mode(model)
    if _mode_spec(model.model_type).grouping == "user" and model.group_data is None:
        raise ValueError(
            "group_data must be provided for model_type="
            f"{model.model_type.name}"
        )
    _validate_span(model.span)
    _validate_span(
        model.cluster_correlation_span, name="cluster_correlation_span"
    )
    if not (0.0 < model.cutoff_fraction <= 1.0):
        raise ValueError(
            f"cutoff_fraction must lie in (0, 1], "
            f"got {model.cutoff_fraction!r}"
        )
    if model.linkage_method not in VALID_LINKAGE_METHODS:
        raise ValueError(
            f"linkage_method must be one of {VALID_LINKAGE_METHODS}, "
            f"got {model.linkage_method!r}"
        )
    try:
        DistanceTransform(model.distance_transform)
    except ValueError:
        raise ValueError(
            f"distance_transform must be one of "
            f"{[t.value for t in DistanceTransform]}, "
            f"got {model.distance_transform!r}"
        ) from None
    try:
        ClusterCorrelationTransform(model.cluster_correlation_transform)
    except ValueError:
        raise ValueError(
            f"cluster_correlation_transform must be one of "
            f"{[item.value for item in ClusterCorrelationTransform]}, "
            f"got {model.cluster_correlation_transform!r}"
        ) from None
    try:
        DependenceMeasure(model.dependence_measure)
    except ValueError:
        raise ValueError(
            f"dependence_measure must be one of "
            f"{[m.value for m in DependenceMeasure]}, "
            f"got {model.dependence_measure!r}"
        ) from None
    if not 0.0 <= model.gerber_threshold <= 1.0:
        raise ValueError(
            f"gerber_threshold must lie in [0, 1], "
            f"got {model.gerber_threshold!r}"
        )
    if model.n_clusters is not None:
        if (not isinstance(model.n_clusters, (int, np.integer))
                or isinstance(model.n_clusters, bool)):
            raise ValueError(
                f"n_clusters must be an integer or None, "
                f"got {model.n_clusters!r}"
            )
        if model.n_clusters < 1:
            raise ValueError(
                f"n_clusters must be at least 1, got {model.n_clusters!r}"
            )
    try:
        smoother_type = ClusterSmootherType(model.cluster_smoother_type)
    except ValueError:
        raise ValueError(
            f"cluster_smoother_type must be one of "
            f"{list(ClusterSmootherType)}, got {model.cluster_smoother_type!r}"
        ) from None
    if model.smoother_delta < 0.0:
        raise ValueError(
            f"smoother_delta must be non-negative, got {model.smoother_delta!r}"
        )
    if not 0.0 <= model.smoother_lambda < 1.0:
        raise ValueError(
            f"smoother_lambda must lie in [0, 1), got {model.smoother_lambda!r}"
        )
    if smoother_type == ClusterSmootherType.HOLD and model.recluster_freq is None:
        raise ValueError(
            f"recluster_freq must be set for HOLD, got {model.recluster_freq!r}"
        )
    if smoother_type == ClusterSmootherType.NONE and model.recluster_freq is not None:
        raise ValueError(
            f"recluster_freq must be None when smoother is NONE, "
            f"got {model.recluster_freq!r}"
        )
    if model.group_penalty not in ("normalized", "yuan_lin"):
        raise ValueError(
            f"group_penalty must be 'normalized' or 'yuan_lin', "
            f"got {model.group_penalty!r}"
        )
    if not (0.0 <= model.l1_weight <= 1.0):
        raise ValueError(
            f"l1_weight must lie in [0, 1], got {model.l1_weight!r}"
        )


def validate_fit_modes(model) -> None:
    """Fit-time repeat of the mode checks; parameters may have changed since construction."""
    validate_loss_mode(model)
    validate_ols_prior_mode(model)
    validate_sign_inputs_mode(model)


def validate_sign_settings(model) -> None:
    """Fit-time checks of the automatic-sign span and variance settings."""
    _validate_span(model.auto_sign_ewma_span, name="auto_sign_ewma_span")
    if model.auto_sign_variance not in ('date', 'independent'):
        raise ValueError("auto_sign_variance must be 'date' or 'independent'")
    if model.auto_sign_use_fit_span and model.auto_sign_ewma_span is not None:
        raise ValueError("select either auto_sign_use_fit_span or auto_sign_ewma_span")


def validate_excluded_factors(model, x: pd.DataFrame) -> None:
    """``auto_sign_excluded_factors`` must name unique columns of ``x``."""
    excluded = model.auto_sign_excluded_factors
    if excluded is not None:
        if (isinstance(excluded, str)
                or not isinstance(excluded, (list, tuple))
                or any(not isinstance(name, str) for name in excluded)
                or len(set(excluded)) != len(excluded)):
            raise ValueError("auto_sign_excluded_factors must be unique factor names")
        missing = set(excluded) - set(x.columns)
        if missing:
            raise ValueError(
                f"auto_sign_excluded_factors not present in x: {sorted(missing)}"
            )


def validate_external_clusters(model_type, external_clusters, external_linkage,
                               external_cutoff) -> None:
    """External partitions are accepted only by the discovered-cluster group penalties."""
    if external_clusters is not None and not _mode_spec(model_type).external_clusters:
        raise ValueError(
            "external_clusters is supported only for HCGL and FCGL, "
            f"got model_type={model_type.name}"
        )
    if external_clusters is None and (
        external_linkage is not None or external_cutoff is not None
    ):
        raise ValueError("external linkage/cutoff metadata requires external_clusters")


def resolve_spans(model, span: Optional[float],
                  cluster_correlation_span: Optional[float]) -> Tuple[Optional[float],
                                                                      Optional[float]]:
    """Effective beta and clustering spans of one fit, validated.

    An explicit ``None`` check gives precedence to the per-call value: ``span or
    model.span`` would treat ``span=0`` as unset. A missing clustering span falls back to
    the model's clustering span and then to the effective beta span.
    """
    eff_span = model.span if span is None else span
    _validate_span(eff_span)
    configured_cluster_span = (
        model.cluster_correlation_span
        if cluster_correlation_span is None
        else cluster_correlation_span
    )
    eff_cluster_correlation_span = (
        eff_span if configured_cluster_span is None else configured_cluster_span
    )
    _validate_span(
        eff_cluster_correlation_span, name="cluster_correlation_span"
    )
    return eff_span, eff_cluster_correlation_span


def coerce_fit_inputs(
    x: Union[pd.DataFrame, pd.Series, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce Series/ndarray → DataFrame and validate shapes / index alignment.

    NumPy arrays are accepted for \\pkg{scikit-learn} interoperability
    (``Pipeline``, ``GridSearchCV``, ``cross_val_score`` pass ndarrays):
    a 1-D array becomes a single-column frame, a 2-D array gets generated
    ``x0, x1, ...`` / ``y0, y1, ...`` column names and a shared
    ``RangeIndex``. DataFrame inputs are unchanged, so existing callers
    and the named-index behaviour the rest of the pipeline relies on are
    unaffected.
    """
    if isinstance(x, np.ndarray):
        # A 1-D array of length T is one regressor observed T times —
        # mirror the y handling and the pd.Series-x convention. The
        # previous ``np.atleast_2d`` turned shape (T,) into a (1, T)
        # row (one observation, T features) and fit() then failed on
        # index alignment with a misleading error message.
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        x = pd.DataFrame(
            x, columns=[f"x{j}" for j in range(x.shape[1])]
        )
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = pd.DataFrame(
            y, columns=[f"y{k}" for k in range(y.shape[1])]
        )
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if isinstance(y, pd.Series):
        y = y.to_frame()
    if not isinstance(x, pd.DataFrame):
        raise TypeError(
            f"x must be pd.DataFrame, pd.Series, or np.ndarray, "
            f"got {type(x).__name__}"
        )
    if not isinstance(y, pd.DataFrame):
        raise TypeError(
            f"y must be pd.DataFrame, pd.Series, or np.ndarray, "
            f"got {type(y).__name__}"
        )
    if len(x) == 0:
        raise ValueError("Empty input: x and y must have at least one row")
    # ndarray inputs arrive with independent RangeIndexes of equal length;
    # align y onto x's index so the equality check below passes.
    if len(x) == len(y) and not x.index.equals(y.index):
        y = y.set_axis(x.index, axis=0)
    if not x.index.equals(y.index):
        raise ValueError(
            f"x and y must share the same index: "
            f"x has {len(x)} rows, y has {len(y)} rows"
        )
    return x, y
