"""
LASSO and Group LASSO factor model estimation using CVXPY.

Implements sparse multi-output regression with support for:

- **Standard L1 LASSO** — element-wise sparsity
- **Group LASSO** with predefined groups — structured sparsity
- **Hierarchical Clustering Group LASSO (HCGL)** — data-driven group
  discovery via Ward's method, then Group LASSO with adaptive penalties
- **Sign constraints** on regression coefficients
  (non-negative, non-positive, zero, free)
- **Prior-centered regularisation** — penalise ‖β − β₀‖ instead of ‖β‖
- **EWMA-weighted objectives** — exponential decay for non-stationary data
- **NaN-aware estimation** — validity masking preserves all usable data

Convention
----------
The factor model follows the paper convention (column vectors)::

    Y_t = α + β X_t + ε_t

where Y_t is ``(N × 1)``, X_t is ``(M × 1)``, β is ``(N × M)``,
and α is ``(N × 1)``.  *N* is the number of response variables
and *M* is the number of regressors (factors).

In Python, pandas DataFrames store observations as rows (T × N).
The code computes the equivalent row-major form ``Y = X β' + α``
internally, but stores β as ``coef_`` in the paper shape ``(N × M)``.

When ``demean=True`` (default), the intercept α is absorbed by subtracting
the (EWMA) rolling mean from both Y and X before estimation.  The fitted
intercept ``intercept_`` is recovered as the EWMA-weighted mean of residuals.

The API follows scikit-learn conventions: ``fit(X, y)`` estimates parameters,
``predict(X)`` returns fitted values, ``score(X, y)`` returns R².  Fitted
attributes carry a trailing underscore (``coef_``, ``intercept_``, etc.).

References
----------
Sepp A., Ossa I., Kastenholz M. (2026), "Robust Optimization of
Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
*Journal of Portfolio Management*, 52(4), 86–120.

Yuan, M., Lin, Y. (2006), "Model selection and estimation in regression
with grouped variables", *J. R. Statist. Soc. B*, 68(1), 49–67.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from factorlasso.cluster._smoothing import ClusterSmootherType
from factorlasso.cluster._hierarchical import (
    DEFAULT_CLUSTER_CORRELATION_TRANSFORM, DEFAULT_CUTOFF_FRACTION, DEFAULT_DISTANCE_TRANSFORM,
    DEFAULT_LINKAGE_METHOD, ClusterCorrelationTransform, DistanceTransform,
)
from factorlasso.cluster._dependence import (
    DEFAULT_DEPENDENCE_MEASURE, DEFAULT_GERBER_THRESHOLD, DependenceMeasure,
)
from factorlasso.linear_model import _inspection, _nowcast
from factorlasso.linear_model._dispatch import solve_prepared, solve_prepared_path
from factorlasso.linear_model._preparation import _PreparedFit, prepare_fit
from factorlasso.linear_model._settings import (
    coerce_fit_inputs, resolve_spans, validate_configuration, validate_external_clusters,
)
from factorlasso.linear_model._state import (
    fitted_state, install_fitted_state, owned, preparation_state,
)
from factorlasso.linear_model._types import (
    LassoEstimationResult, LassoModelType, LassoNowcastResult, _mode_spec,
)
from factorlasso.utils._panel import get_x_y_np


@dataclass
class LassoModel:
    """
    Configurable LASSO / Group LASSO / HCGL factor model estimator.

    Estimates the model ``Y_t = α + β X_t + ε_t`` with sparse β using
    L1 (LASSO) or Group L2/L1 (Group LASSO) regularisation via CVXPY.

    The API follows scikit-learn conventions:

    - ``fit(x, y)`` estimates parameters, returns ``self``
    - ``predict(x)`` returns Ŷ_t = α + β X_t (computed as ``X @ β' + α``)
    - ``score(x, y)`` returns mean R² across response variables
    - Fitted attributes use trailing underscore: ``coef_``, ``intercept_``

    Convention
    ----------
    β is ``(N × M)`` following the paper.  After ``fit()``:

    - ``coef_`` (also ``estimated_betas``): DataFrame (N × M)
    - ``intercept_``: Series (N,) — the α vector

    Sign constraints
    ~~~~~~~~~~~~~~~~
    ``factors_beta_loading_signs`` is ``(N × M)``::

        0  → constrained to zero
        1  → constrained non-negative
       -1  → constrained non-positive
       NaN → unconstrained (free)

    Enforced by ``LASSO``, ``GROUP_LASSO``, ``HIERARCHICAL_CLUSTER_GROUP_LASSO``
    and ``FACTOR_CLUSTER_GROUP_LASSO``; rejected with ``ValueError`` by
    ``UNILASSO`` and the cooperative modes, whose solvers take none.

    Prior-centered regularisation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``factors_beta_prior`` is ``(N × M)``.  The penalty becomes
    ``‖β − β₀‖`` instead of ``‖β‖``.

    Parameters
    ----------
    model_type : LassoModelType, default LASSO
        Selects the optimisation problem: ``LASSO`` (cellwise L1),
        ``UNILASSO`` (per-response two-stage univariate-guided regression,
        no grouping), ``GROUP_LASSO`` (external group partition),
        ``HIERARCHICAL_CLUSTER_GROUP_LASSO`` (HCGL row-grouped penalty on a
        discovered partition), ``FACTOR_CLUSTER_GROUP_LASSO`` (FCGL
        cluster-by-factor block penalty on the same discovered partition),
        ``COOPERATIVE_GROUP_LASSO`` (cooperative-LASSO on an external
        partition, soft within-block sign coherence), or
        ``COOPERATIVE_CLUSTER_GROUP_LASSO`` (cooperative-LASSO on the
        discovered partition).  HCGL and FCGL impose a hard pooled sign only
        when ``auto_sign_constraints=True``.  ``UNILASSO`` and the two
        cooperative modes take no sign constraint: ``factors_beta_loading_signs``
        and ``nonneg=True`` raise ``ValueError`` with them, derived signs are
        not enforced and ``derived_signs_`` stays ``None``; the cooperative
        penalty encourages sign coherence softly.
    loss_normalization : {"sample", "weight_sum"}, default "sample"
        ``sample`` preserves the historical loss divided by panel row count.
        ``weight_sum`` divides each response's weighted squared error by its
        own valid squared-weight mass, then sums over responses. Empty responses
        contribute zero loss. For balanced data with common mass S, the equivalent
        penalty is ``lambda_weight_sum = lambda_sample * T / S``. Unequal histories
        change relative response weights, so one conversion cannot preserve every
        old fit. This option does not change signs, targets or diagnostic SSE units.
        UniLasso retains its separate unweighted loss and rejects ``weight_sum``.
    reg_lambda : float, default 1e-5
        Penalty strength multiplying the L1 and group terms. Its scale depends
        on ``loss_normalization``, the observation frequency and the return
        units; ``LassoModelCV`` selects it on expanding time-series splits.
    span : float, optional
        EWMA span for observation weighting.  Must be ≥ 1 when provided.
        Float accepted — integer is the common case, but the recursion
        math does not require it.
    span_freq_dict : dict, optional
        Per-frequency override of ``span`` used by multi-frequency
        pipelines downstream (``optimalportfolios`` / ``rosaa``).  Keys
        are pandas freq codes (``'ME'``, ``'QE'``), values are spans at
        that frequency (float).  Carried through the model specification
        but not consumed by :meth:`fit`; the caller selects the right
        span when it slices per frequency.
    cluster_correlation_span : float, optional
        EWMA span used only to prepare the response panel and estimate the
        dependence matrix for cluster discovery. ``None`` (the default)
        uses the effective beta-estimation ``span`` and therefore preserves
        the historical coupled behaviour exactly.
    cluster_correlation_span_freq_dict : dict, optional
        Per-frequency clustering-correlation spans carried for downstream
        multi-frequency pipelines. Like ``span_freq_dict``, this mapping is
        resolved by the caller before :meth:`fit`.
    group_data : pd.Series, optional
        Group labels (required for ``GROUP_LASSO`` and
        ``COOPERATIVE_GROUP_LASSO``).
    cutoff_fraction : float, default 0.5
        Fraction of ``max(pdist)`` at which to cut the dendrogram when
        ``model_type`` is ``HIERARCHICAL_CLUSTER_GROUP_LASSO``,
        ``FACTOR_CLUSTER_GROUP_LASSO``, or
        ``COOPERATIVE_CLUSTER_GROUP_LASSO`` (all discover clusters the same
        way).  Ignored by ``LASSO``, ``UNILASSO``, ``GROUP_LASSO``, and
        ``COOPERATIVE_GROUP_LASSO``.
        See :func:`factorlasso.compute_clusters_from_corr_matrix`.
    linkage_method : str, default 'ward'
        Agglomerative linkage method for the cluster-discovery step, one of
        ``'single'``, ``'complete'``, ``'average'``, ``'weighted'``,
        ``'centroid'``, ``'median'``, or ``'ward'``.  Used only by the
        cluster-discovery modes and ignored otherwise.  The default
        ``'ward'`` reproduces the prior behaviour.
    distance_transform : DistanceTransform or str, default ONE_MINUS_RHO
        Correlation-to-distance transform for the cluster-discovery step:
        ``ONE_MINUS_RHO`` (``d = 1 - rho``), ``CHORD``
        (``d = sqrt(2 (1 - rho))``, the Euclidean chord under which Ward's
        variance criterion is exact), or ``ARCCOS`` (``d = arccos(rho)``).
        Used only by the cluster-discovery modes and ignored otherwise.
        The default reproduces the pre-0.9.0 behaviour exactly.
        ``cutoff_fraction`` is calibrated per transform and does not port
        across transforms; see
        :func:`factorlasso.compute_clusters_from_corr_matrix` for the
        granularity-preserving conversion when switching.
    cluster_correlation_transform : ClusterCorrelationTransform or str, default NONE
        Optional diagnostic/robustness transform of the signed dependence
        matrix before distance, linkage, and cluster discovery. ``NONE`` is
        the production default and an exact numerical bypass. ``REMOVE_PC1``
        removes the largest algebraic eigencomponent and restandardizes the
        residual matrix to unit diagonal. It does not residualize responses,
        fitted loadings, or the assembled covariance matrix. A changed cluster
        partition can nevertheless change fitted outputs indirectly when a
        cluster-based penalty or sign-pooling rule consumes that partition.
    dependence_measure : DependenceMeasure or str, default PEARSON
        Dependence measure used to build the clustering correlation
        matrix: ``'pearson'`` (the default and the pre-0.10.0 behaviour),
        ``'spearman'`` (Pearson correlation of ranks), or ``'gerber'``
        (Gerber et al. 2022 co-movement statistic).  Both alternatives
        are robust to outliers, which the linear correlation is not.
        Used only by the cluster-discovery modes and ignored otherwise.
        Every measure honours the effective clustering-correlation span,
        which defaults to the ``span`` weighting of the solver loss.
        CAUTION: ``cutoff_fraction`` does not port across measures — the
        Gerber statistic shrinks correlations toward zero by a
        data-dependent factor, and no closed-form remapping exists.  Set
        ``n_clusters`` instead whenever partitions are compared across
        measures.
    gerber_threshold : float, default 0.5
        Threshold ``c`` for ``DependenceMeasure.GERBER``, applied as
        ``c * sigma`` on each leg.  Observations below the threshold on
        both legs are treated as noise and discarded.  Ignored by the
        other measures.
    n_clusters : int, optional
        Target number of clusters for the cluster-discovery modes.  When
        set, the dendrogram is cut to at most ``n_clusters`` groups and
        ``cutoff_fraction`` is ignored.  None (default) uses the
        fractional-height cut, the pre-0.10.0 behaviour.  Prefer
        ``n_clusters`` when comparing partitions across distance
        transforms or dependence measures, since the fractional cut is
        calibrated against the scale of the distance matrix.
    cluster_smoother_type : ClusterSmootherType, default NONE
        Declarative causal temporal smoother used by rolling consumers.
        ``NONE`` leaves the current single-fit behaviour bit-identical;
        ``HOLD`` holds partitions between ``recluster_freq`` anchors;
        ``PARTITION_BONUS`` discounts distances for prior peers; and
        ``SIMILARITY_EWMA`` smooths the clustering similarity matrix.
    smoother_delta : float, default 0.05
        Non-negative prior-partition distance discount.
    smoother_lambda : float, default 0.7
        Prior-state weight for similarity EWMA, in ``[0, 1)``.
    recluster_freq : str, optional
        Optional pandas anchor frequency for rolling cluster updates.  It is
        required for ``HOLD`` and optional for ``PARTITION_BONUS`` and
        ``SIMILARITY_EWMA``.  When supplied to either smoother, its state and
        partition update only on anchor dates and the partition is held
        between anchors.  ``None`` updates on every estimation date.  It must
        remain ``None`` for ``NONE``.
    group_penalty : {"normalized", "yuan_lin"}, default "normalized"
        Per-group weighting for the group-LASSO penalty.  ``"normalized"``
        uses ``√(|g|/G)``, a heuristic cluster-size scaling that adjusts
        for the data-driven group count (not invariant to arbitrary
        partition refinements), and is the default —
        appropriate for HCGL where the number of groups is data-driven.
        ``"yuan_lin"`` uses the classical Yuan–Lin (2006) ``√|g|``.
        Ignored for ``model_type == LASSO``.  See
        :func:`solve_group_lasso_cvx_problem` for the full formula.
    l1_weight : float, default 0.0
        Sparse Group LASSO mixing parameter ``α ∈ [0, 1]``. Adds an
        elementwise L1 penalty ``α·λ·|β - β₀|`` on top of the standard
        group L2 penalty (which is scaled by ``(1 - α)``). Set ``α = 0``
        (default) for pure group LASSO — backward compatible with
        v0.3.1. Typical research values: ``α ∈ [0.05, 0.20]`` — preserve
        group structure as the primary mechanism while allowing
        additional within-group elementwise zeroing for assets whose
        loadings are noisy. Only consumed when ``model_type`` is
        ``GROUP_LASSO``, ``HIERARCHICAL_CLUSTER_GROUP_LASSO``, or
        ``FACTOR_CLUSTER_GROUP_LASSO``; ignored for pure
        ``LASSO`` since L1 is the only penalty already.
    nonneg : bool, default False
        If True, every loading is constrained to be non-negative. Rejected
        with ``ValueError`` by ``UNILASSO`` and the two cooperative modes,
        whose solvers take no sign constraint.
    factors_beta_loading_signs : pd.DataFrame, optional
        Hard sign matrix indexed by response and factor: ``1`` non-negative,
        ``-1`` non-positive, ``0`` fixed at zero, NaN free. Non-NaN entries
        take precedence over prior and automatically detected signs. Enforced
        by ``LASSO``, ``GROUP_LASSO``, ``HIERARCHICAL_CLUSTER_GROUP_LASSO`` and
        ``FACTOR_CLUSTER_GROUP_LASSO``; rejected with ``ValueError`` by
        ``UNILASSO`` and the cooperative modes.
    factors_beta_prior : pd.DataFrame, optional
        Explicit penalty centres, indexed by response and factor. With
        ``apply_ols_prior=True``, NaN defers to the computed prior and finite
        entries override it, including zero. A finite nonzero prior overrides
        a conflicting automatically detected sign or zero gate. Explicit hard
        signs still win; with OLS enabled their incompatible priors are zeroed.
        With the flag off, NaN retains its legacy zero meaning. Rejected with
        ``ValueError`` by ``UNILASSO``, whose solver takes no beta prior.
    apply_ols_prior : bool, default False
        Derive per-response weighted one-factor OLS priors on original inputs
        with an intercept and the effective LASSO squared-loss span. Use
        ``prior_selection_type`` to select the centres. After explicit prior
        overrides, nonzero prior signs take precedence over automatic signs;
        zero cells violating remaining explicit hard constraints. Do not
        reselect or redistribute blocked priors. Unsupported for UNILASSO.
    prior_selection_type : str, default 'highest_r2'
        The only supported selector, consumed when ``apply_ols_prior=True``.
        Select the factor with highest centred EWMA-weighted univariate
        R-squared for each response and assign its full OLS slope as the prior;
        other automatic prior cells are zero. R-squared uses weighted residual
        and centred total sums of squares with the effective squared-loss span.
        Ties use input factor-column order. Unsupported values raise even when
        automatic priors are disabled.
    factor_for_prior : mapping or pd.Series, optional
        Response-to-factor labels overriding the highest-R-squared selection.
        Requires ``apply_ols_prior=True``. A scalar selects its univariate slope;
        a nonempty ordered list/tuple selects slopes from one joint weighted OLS
        regression with intercept and complete finite rows. All other automatic
        centres in that row are zero. Omitted responses and missing values retain
        automatic selection.
        Superset response maps support cadence-group and rolling fits. Unknown
        factors raise; an unestimable selected slope gives a zero row without
        reselection. Finite explicit centres and sign filtering apply afterward.
    auto_sign_constraints : bool, default False
        If True, signs are derived inside ``fit()`` from the EWMA-demeaned,
        NaN-masked arrays returned by ``get_x_y_np`` (i.e. the same data the
        CVXPY solver consumes). Pooling strategy is dispatched by
        ``model_type``:

        * ``LASSO`` (or single-column y): per-y-column independent
          univariate sign derivation; rows of ``derived_signs_`` may differ.
        * ``GROUP_LASSO``: signs pooled within each ``group_data`` group;
          members share detected signs before per-cell prior/hard overrides.
        * ``HIERARCHICAL_CLUSTER_GROUP_LASSO``: signs pooled within each HCGL asset
          cluster (the same clustering the group solver uses).
        * ``FACTOR_CLUSTER_GROUP_LASSO``: signs pooled within each HCGL
          asset cluster, identically to ``HIERARCHICAL_CLUSTER_GROUP_LASSO``; the two
          modes share the sign derivation and differ only in the group
          norm of the penalty.

        A finite nonzero resolved prior (explicit, mapped OLS, or automatic
        OLS) overrides a conflicting detected sign, including an automatic
        zero gate. Zero or missing priors leave detection unchanged. This is
        per response/factor, so final signs may differ within a pooled cluster.
        Excluded factor columns remain exempt. Non-NaN entries of
        ``factors_beta_loading_signs`` take precedence over both prior and
        detected signs. Adaptive weights retain the original detected values.
    auto_sign_threshold_t : float, optional, default 0.75
        Noise-floor gate on the pooled univariate t-statistic: cells whose
        absolute statistic falls below it are pinned to zero, the others
        receive the slope's sign. ``None`` disables the gate. It is a
        screening rule, not a calibrated significance test; 0.75 corresponds
        to a two-sided p of about 0.45 under a normal reference. Used only
        with ``auto_sign_constraints=True``.
    auto_sign_ewma_span : float, optional
        EWMA span of the univariate slopes and scores behind derived signs.
        ``None`` (the default) weights dates equally. Mutually exclusive with
        ``auto_sign_use_fit_span=True``.
    auto_sign_use_fit_span : bool, default False
        If True, derive signs with the effective squared-loss span of the fit,
        including a span passed to ``fit``.
    auto_sign_variance : {"independent", "date"}, default "independent"
        Variance estimator of the gate's t-statistic. ``"independent"`` treats
        the responses of a pool as independent observations (the historical
        gate). ``"date"`` sums the scores by date, a sandwich variance under
        which duplicated or correlated responses do not inflate the evidence.
        Both are screening rules, not calibrated t tests.
    auto_sign_adaptive_weights : bool, default False
        If True, together with ``auto_sign_constraints=True``, each cell's L1
        penalty is weighted by ``1 / max(|b|, floor) ** gamma`` of its pooled
        univariate slope ``b`` (Zou, 2006); with ``l1_weight=0`` the weights
        enter the group norms by root-mean-square row aggregation (Wang and
        Leng, 2008). Cells pinned to zero by the gate stay at zero.
    auto_sign_adaptive_gamma : float, default 1.0
        Exponent ``gamma`` of the adaptive weights; 1 is the adaptive-LASSO
        default and larger values strengthen the reweighting.
    auto_sign_adaptive_floor : float, default 1e-3
        Floor applied to ``|b|`` before inversion, so near-zero slopes do not
        produce exploding weights.
    auto_sign_excluded_factors : list or tuple of str, optional
        Factor columns exempt from automatic signs and their t-stat zero gate.
        Explicit non-NaN ``factors_beta_loading_signs`` still apply; supply NaN
        there to allow either sign. Pooling, clustering and adaptive penalty
        weights are unchanged. Names must be unique and present in the fitted
        factor panel. None or an empty sequence preserves the existing fit.
    demean : bool, default True
        If True, each series is demeaned before estimation, with its EWMA
        mean when ``span`` is set and its sample mean otherwise;
        ``alpha_const_`` then holds the intercept consistent with the fitted
        loadings.
    solver : str, default 'CLARABEL'
        CVXPY solver of the primary solve.
    solver_fallbacks : sequence of str, optional
        Solver names tried in order only when the primary solver raises or
        returns a non-optimal status. ``None`` (the default) runs the primary
        solver once and lets its error propagate.
    warmup_period : int, optional, default 12
        Minimum number of valid observations of a response. A response with
        fewer receives zero loadings, NaN diagnostics and a warning, and is
        left out of the cluster assignment. It also sets the minimum sample,
        ``max(3, warmup_period)``, of the OLS prior regressions. ``None``
        disables the check.
    unilasso_loo : bool, default True
        ``UNILASSO`` only. If True, stage two uses leave-one-out
        (prevalidated) univariate fits, as in the published method; False
        uses in-sample univariate fits.
    unilasso_non_negative : bool, default True
        ``UNILASSO`` only. If True, the stage-two coefficients are
        non-negative, so each final loading keeps the sign of its univariate
        slope.

    expert_prior_bound_n_std : float or None, default None
        Optional individual sign-oriented magnitude floor from expert-selected or
        automatic highest-R-squared OLS targets: max(0, abs(target) - n_std * HAC_SE).
        Finite manual targets remain soft; hard signs take precedence. No cluster
        pooling enters the bound. Supported by LASSO, group LASSO, HCGL and FCGL.
    expert_prior_hac_lags : int, default 0
        Bartlett bandwidth on the original observation grid. Zero uses robust
        contemporaneous score products only. Missing rows retain their positions.
    expert_prior_hac_lags_freq_dict : dict or None, default None
        Optional consumer-resolved cadence map. Direct fits use the scalar lag;
        a multi-frequency consumer must select the appropriate scalar explicitly.

    Attributes (fitted, set by ``fit()``)
    --------------------------------------
    coef_ : pd.DataFrame, shape (N, M)
        Estimated factor loadings β.
    alpha_const_ : pd.Series, shape (N,)
        **Economic intercept α** — the constant term in the regression
        ``y = α + Xβ + ε`` paired consistently with the fitted β.
        Reconstructed from weighted means of ``y`` and ``X`` using the
        same weighting that produced β:

        * for ``span=None`` (uniform weights), this is the sample-mean
          reconstruction ``α = ȳ_sample − x̄_sample · β``, identical to
          the OLS intercept;
        * for ``span=integer`` (EWMA weights), this uses EWMA-weighted
          means with the same weights factorlasso applies in the loss
          function, so the ``(α, β)`` pair represents one coherent
          weighted-least-squares solution rather than two estimators
          under different weightings.

        This is the field to read when reporting "alpha after factor
        exposure".
    intercept_ : pd.Series, shape (N,)
        Raw solver output: the EWMA-weighted mean of residuals on the
        *demeaned* data, equal to ``estimation_result_.alpha``. Because
        the underlying solver fits a no-intercept model on centered data,
        this is a mechanical artefact of the fit, **not** the regression
        intercept in original units:

        * for ``span=None`` this is identically zero by the OLS
          first-order condition;
        * for ``span=integer`` it is a finite-sample EWMA-demean leftover.

        Preserved under this name for back-compatibility with code that
        read ``model.intercept_`` in pre-0.3.4 versions. New code should
        use ``alpha_const_`` for the economic intercept. Since v0.5.0 this
        diagnostic is computed in the nominal-span EWMA norm; v0.4.x and
        earlier used an effective span of ``≈ 2·span``.
    estimation_result_ : LassoEstimationResult
        Full diagnostics (alpha, ss_total, ss_res, r2).
    clusters_ : pd.Series or None
        Cluster labels (HCGL only).
    linkage_ : np.ndarray or None
        Scipy linkage matrix (HCGL only).
    cutoff_ : float or None
        Dendrogram cut distance (HCGL only).
    ols_betas_, ols_r2_ : pd.DataFrame or None
        Per-response univariate slopes and centred weighted R-squared before
        selection, populated only when ``apply_ols_prior=True``.
    ols_beta_prior_, effective_beta_prior_ : pd.DataFrame or None
        Selected OLS centres before overrides/constraints, and the actual
        solver centres after overrides and sign-conflict zeroing, respectively.
    ols_prior_span_ : float or None
        Effective squared-loss span used for OLS. None means uniform weighting
        when the flag is on; diagnostics are cleared when the flag is off.
    derived_signs_ : pd.DataFrame or None
        The final ``(N × M)`` sign matrix that was passed to the solver,
        in ``LassoModel.factors_beta_loading_signs`` convention
        (``+1`` non-negative, ``-1`` non-positive, ``0`` forced zero,
        ``NaN`` unconstrained). Populated whenever sign constraints were
        actually applied during the fit:

        * ``auto_sign_constraints=True`` only — pooled univariate signs
          from the EWMA-demeaned, NaN-masked arrays the solver consumes,
          identical across response rows.
        * ``factors_beta_loading_signs`` only — the user's matrix reindexed
          to the fit universe.
        * Both — auto-derived signs as the base layer, overlaid with the
          explicit per-cell values wherever ``factors_beta_loading_signs``
          is non-NaN (per-asset overrides for the asset-specific master
          constraints).

        Read this attribute to inspect, log, or render the constraints
        that actually shaped the fitted ``coef_``. ``None`` after a fit in
        ``UNILASSO`` or a cooperative mode, whose solvers take no sign
        constraint.
    fit_demeaned_ : bool
        Fit-time copy of the de-meaning decision. Unlike the mutable
        ``demean`` hyperparameter, this is immutable fitted provenance used
        to admit a model to :meth:`nowcast`.
    nowcast_residuals_ : pd.DataFrame, shape (T, N)
        Deep-copied original-unit residuals ``y - X @ beta`` captured after
        final warmup beta handling. The snapshot is independent of the
        caller-owned frames aliased by ``x_`` and ``y_``.
    nowcast_factors_complete_ : bool
        Fit-time flag recording whether every fitted factor observation was
        finite. Nowcast eligibility reads this snapshot rather than the
        caller-owned frame aliased by ``x_``.
    nowcast_final_response_complete_ : bool
        Fit-time flag recording whether the final fitted response row was
        fully observed. Nowcast eligibility reads this snapshot rather than
        the caller-owned frame aliased by ``y_``.

    prior_lower_bounds_, prior_upper_bounds_ : pandas.DataFrame or None
        Fitted individual coefficient bounds, response by factor; NaN is unbounded.
    prior_bound_diagnostics_ : pandas.DataFrame or None
        Selected-cell OLS/HAC statistics, floor and exclusion reasons.
    effective_prior_hac_lags_ : int or None
        Bandwidth used by the current fit, including a consumer's cadence override.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from factorlasso import LassoModel, LassoModelType
    >>> np.random.seed(42)
    >>> T, M, N = 200, 3, 5
    >>> X = pd.DataFrame(np.random.randn(T, M), columns=[f'f{i}' for i in range(M)])
    >>> beta_true = np.array([[1, 0, .5], [0, 1, 0], [.3, 0, 0],
    ...                       [0, .8, .2], [1, .5, 0]])
    >>> Y = pd.DataFrame(X.values @ beta_true.T + .1*np.random.randn(T, N),
    ...                   columns=[f'y{i}' for i in range(N)])
    >>> model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-4)
    >>> _ = model.fit(x=X, y=Y)
    >>> model.coef_.shape
    (5, 3)
    >>> y_hat = model.predict(X)
    >>> r2 = model.score(X, Y)
    """
    # ── Hyperparameters (constructor args) ────────────────────────────
    model_type: LassoModelType = LassoModelType.LASSO
    group_data: Optional[pd.Series] = None
    reg_lambda: float = 1e-5
    span: Optional[float] = None
    span_freq_dict: Optional[Dict[str, float]] = None
    cutoff_fraction: float = DEFAULT_CUTOFF_FRACTION
    linkage_method: str = DEFAULT_LINKAGE_METHOD
    distance_transform: Union[DistanceTransform, str] = DEFAULT_DISTANCE_TRANSFORM
    cluster_correlation_transform: Union[
        ClusterCorrelationTransform, str
    ] = DEFAULT_CLUSTER_CORRELATION_TRANSFORM
    dependence_measure: Union[DependenceMeasure, str] = DEFAULT_DEPENDENCE_MEASURE
    gerber_threshold: float = DEFAULT_GERBER_THRESHOLD
    n_clusters: Optional[int] = None
    cluster_smoother_type: ClusterSmootherType = ClusterSmootherType.NONE
    smoother_delta: float = 0.05
    smoother_lambda: float = 0.7
    recluster_freq: Optional[str] = None
    group_penalty: str = "normalized"
    l1_weight: float = 0.0
    demean: bool = True
    solver: str = 'CLARABEL'
    solver_fallbacks: Optional[Sequence[str]] = None
    warmup_period: Optional[int] = 12
    nonneg: bool = False
    factors_beta_loading_signs: Optional[pd.DataFrame] = None
    factors_beta_prior: Optional[pd.DataFrame] = None
    # Auto sign-constraint derivation (signs computed inside fit on the
    # solver-ready EWMA-demeaned, NaN-masked arrays). Pooling strategy is
    # determined by ``model_type``:
    #   * HIERARCHICAL_CLUSTER_GROUP_LASSO → pool within each asset cluster from HCGL
    #   * GROUP_LASSO          → pool within each ``group_data`` group
    #   * LASSO / single-col y → per-y-column independent derivation
    auto_sign_constraints: bool = False

    # Significance gate for auto-derived signs.  When set (>0), only
    # columns whose univariate ``|t|`` meets the threshold get a hard
    # sign constraint; columns failing the threshold are pinned to 0
    # (β forced to zero), excluding them from the regression.  This
    # enforces parsimony directly and is robust to the choice of
    # ``reg_lambda``.  Default 0.75 acts as a noise floor — it is
    # well below conventional significance levels but high enough
    # to filter columns whose univariate slope sign is dominated by
    # sampling noise (|t| < 0.75 ⇒ two-sided p > 0.45).  Pass
    # ``None`` to disable the gate and reproduce v0.3.6 behaviour.
    #
    # Typical alternative values: 0.5 (looser) to 1.0 (stricter).
    # Only effective when ``auto_sign_constraints=True``.
    auto_sign_threshold_t: Optional[float] = 0.75

    # ── Adaptive penalty weights (Zou 2006 adaptive LASSO) ──
    # When True and auto_sign_constraints=True, the L1 penalty becomes
    # weighted: each |β_kj| is scaled by 1 / max(|β̂_uni_kj|, floor)^gamma,
    # where β̂_uni is the pooled univariate slope (same quantity that
    # produces the sign matrix). Cells with strong univariate evidence
    # (large |β̂_uni|) get a lighter L1 penalty and can take larger
    # multivariate coefficients; cells with weak evidence get a heavier
    # penalty and are pushed harder toward the prior.
    #
    # Default ``False`` preserves v0.3.8 behaviour exactly. Independent of
    # the threshold gate: cells pinned to zero by the gate continue to be
    # forced to zero by the hard sign constraint, with the adaptive weight
    # acting only on the non-pinned cells.
    auto_sign_adaptive_weights: bool = False
    # Zou (2006) exponent γ on |β̂_uni|. γ=1 is the standard adaptive-Lasso
    # default. Larger values amplify the magnitude-aware reweighting.
    auto_sign_adaptive_gamma: float = 1.0
    # Stabiliser preventing weight explosion on near-zero slopes:
    # |β̂_uni| is clipped at this floor before inversion.
    auto_sign_adaptive_floor: float = 1e-3
    # ── UniLasso (model_type=UNILASSO) ──
    # loo: use leave-one-out (prevalidated) univariate fits in stage 2
    # (the published UniLasso); False uses in-sample univariate fits.
    unilasso_loo: bool = True
    # non_negative: theta >= 0 in stage 2, so the final coefficient
    # inherits the univariate sign (UniLasso's sign-preservation).
    unilasso_non_negative: bool = True
    # ── Fitted state (set by fit(), trailing underscore) ──────────────
    x_: Optional[pd.DataFrame] = None
    y_: Optional[pd.DataFrame] = None
    coef_: Optional[pd.DataFrame] = None
    intercept_: Optional[pd.Series] = None
    alpha_const_: Optional[pd.Series] = None
    estimation_result_: Optional[LassoEstimationResult] = None
    clusters_: Optional[pd.Series] = None
    linkage_: Optional[np.ndarray] = None
    cutoff_: Optional[float] = None
    valid_mask_: Optional[np.ndarray] = None
    effective_span_: Optional[float] = None
    effective_cluster_correlation_span_: Optional[float] = None
    derived_signs_: Optional[pd.DataFrame] = None
    fit_demeaned_: Optional[bool] = field(default=None, init=False)
    nowcast_residuals_: Optional[pd.DataFrame] = field(default=None, init=False)
    nowcast_factors_complete_: Optional[bool] = field(default=None, init=False)
    nowcast_final_response_complete_: Optional[bool] = field(default=None, init=False)
    # Optional independent clustering horizon. Appended after every historical
    # dataclass field so even positional callers retain their old mapping.
    cluster_correlation_span: Optional[float] = None
    cluster_correlation_span_freq_dict: Optional[Dict[str, float]] = None
    # Appended to preserve historical positional constructor arguments.
    auto_sign_excluded_factors: Optional[Sequence[str]] = None
    # Appended; every historical positional constructor argument is preserved.
    apply_ols_prior: bool = False
    prior_selection_type: str = 'highest_r2'
    ols_betas_: Optional[pd.DataFrame] = field(default=None, init=False)
    ols_r2_: Optional[pd.DataFrame] = field(default=None, init=False)
    ols_beta_prior_: Optional[pd.DataFrame] = field(default=None, init=False)
    effective_beta_prior_: Optional[pd.DataFrame] = field(default=None, init=False)
    ols_prior_span_: Optional[float] = field(default=None, init=False)
    # Appended to preserve every historical positional constructor argument.
    factor_for_prior: Optional[Union[Mapping, pd.Series]] = None
    # Appended: preserve positional compatibility. None retains equal weights.
    auto_sign_ewma_span: Optional[float] = None
    auto_sign_use_fit_span: bool = False
    auto_sign_variance: str = 'independent'
    effective_sign_span_: Optional[float] = field(default=None, init=False)
    detected_signs_: Optional[pd.DataFrame] = field(default=None, init=False)
    sign_slopes_: Optional[pd.DataFrame] = field(default=None, init=False)
    sign_t_stats_: Optional[pd.DataFrame] = field(default=None, init=False)
    sign_effective_n_: Optional[pd.DataFrame] = field(default=None, init=False)
    sign_valid_counts_: Optional[pd.DataFrame] = field(default=None, init=False)
    sign_penalty_weights_: Optional[pd.DataFrame] = field(default=None, init=False)
    sign_block_weights_: Optional[np.ndarray] = field(default=None, init=False)
    # Appended: retain historical constructor positions and objective default.
    loss_normalization: str = "sample"
    loss_weight_mass_: Optional[pd.Series] = field(default=None, init=False)
    loss_denominator_: Optional[pd.Series] = field(default=None, init=False)
    n_loss_rows_: Optional[int] = field(default=None, init=False)

    # Optional individual floors for explicit OLS selectors, never automatic winners.
    expert_prior_bound_n_std: Optional[float] = None
    expert_prior_hac_lags: int = 0
    expert_prior_hac_lags_freq_dict: Optional[Dict[str, int]] = None
    prior_lower_bounds_: Optional[pd.DataFrame] = field(default=None, init=False)
    prior_upper_bounds_: Optional[pd.DataFrame] = field(default=None, init=False)
    prior_bound_diagnostics_: Optional[pd.DataFrame] = field(default=None, init=False)
    effective_prior_hac_lags_: Optional[int] = field(default=None, init=False)

    def __post_init__(self):
        """Validate the configuration; see :mod:`factorlasso.linear_model._settings`."""
        validate_configuration(self)

    # ── Backward-compatible property aliases ─────────────────────────

    @property
    def estimated_betas(self) -> Optional[pd.DataFrame]:
        """Alias for ``coef_`` (backward compatibility)."""
        return self.coef_

    @estimated_betas.setter
    def estimated_betas(self, value):
        self.coef_ = value

    @property
    def clusters(self) -> Optional[pd.Series]:
        """Alias for ``clusters_`` (backward compatibility)."""
        return self.clusters_

    @clusters.setter
    def clusters(self, value):
        self.clusters_ = value

    @property
    def linkage(self) -> Optional[np.ndarray]:
        """Alias for ``linkage_`` (backward compatibility)."""
        return self.linkage_

    @linkage.setter
    def linkage(self, value):
        self.linkage_ = value

    @property
    def cutoff(self) -> Optional[float]:
        """Alias for ``cutoff_`` (backward compatibility)."""
        return self.cutoff_

    @cutoff.setter
    def cutoff(self, value):
        self.cutoff_ = value

    @property
    def x(self) -> Optional[pd.DataFrame]:
        """Alias for ``x_`` (backward compatibility)."""
        return self.x_

    @x.setter
    def x(self, value):
        self.x_ = value

    @property
    def y(self) -> Optional[pd.DataFrame]:
        """Alias for ``y_`` (backward compatibility)."""
        return self.y_

    @y.setter
    def y(self, value):
        self.y_ = value

    # ── scikit-learn compatibility ───────────────────────────────────

    @classmethod
    def _constructor_param_names(cls) -> List[str]:
        """Names of constructor (non-fitted) dataclass fields."""
        return [f.name for f in fields(cls) if not f.name.endswith("_")]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return constructor hyperparameters as a dict (sklearn-compatible).

        Parameters
        ----------
        deep : bool, default True
            Present for sklearn API parity; LassoModel has no nested
            estimators so this argument has no effect.

        Returns
        -------
        dict
            Mapping ``{param_name: value}`` for every constructor argument.
            Fitted attributes (trailing underscore) are excluded.
        """
        del deep  # unused, kept for API parity
        return {name: getattr(self, name) for name in self._constructor_param_names()}

    def set_params(self, **params: Any) -> "LassoModel":
        """
        Set constructor hyperparameters in place (sklearn-compatible).

        Returns ``self`` for method chaining.

        Raises
        ------
        ValueError
            If any key is not a valid constructor parameter.
        """
        valid = set(self._constructor_param_names())
        invalid = sorted(set(params) - valid)
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for LassoModel: {invalid}. "
                f"Valid parameters: {sorted(valid)}"
            )
        for name, value in params.items():
            setattr(self, name, value)
        return self

    # ── Core API ─────────────────────────────────────────────────────

    def copy(self, kwargs: Optional[Dict] = None) -> LassoModel:
        """Create a fresh, unfitted copy, optionally overriding parameters.

        Only constructor hyperparameters are carried over (the same set
        that :meth:`get_params` returns). Fitted state (``coef_``,
        ``estimation_result_``, ...) is **not** copied: the copy is a new
        estimator specification, not a snapshot of a fit. This matches
        \\pkg{scikit-learn}'s ``clone`` semantics.

        The previous implementation round-tripped the model through
        ``dataclasses.asdict``, which (a) carried stale fitted state into
        the copy, so a copy with a new ``reg_lambda`` still "looked
        fitted" with coefficients from the old one, and (b) recursively
        converted the nested :class:`LassoEstimationResult` dataclass
        into a plain ``dict``, corrupting the attribute's type.
        """
        params = self.get_params()
        if kwargs is not None:
            params.update(kwargs)
        return LassoModel(**params)

    def fit(
        self,
        x: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
        verbose: bool = False,
        span: Optional[float] = None,
        external_clusters: Optional[pd.Series] = None,
        external_linkage: Optional[np.ndarray] = None,
        external_cutoff: Optional[float] = None,
        cluster_correlation_span: Optional[float] = None,
    ) -> LassoModel:
        """
        Estimate model: Y_t = α + β X_t + ε_t.

        Parameters
        ----------
        x : pd.DataFrame or pd.Series, shape (T, M) or (T,)
            Regressor (factor) returns.  Series is converted to single-column DataFrame.
        y : pd.DataFrame or pd.Series, shape (T, N) or (T,)
            Response (asset) returns.  May contain NaNs.
            Series is converted to single-column DataFrame.
        verbose : bool, default False
            Print solver diagnostics.
        span : float, optional
            Per-call override of the model's ``span`` hyperparameter.
            ``None`` (the default) falls back to ``self.span`` without
            modification — previous versions used ``span or self.span``
            which would treat ``span=0`` as "unset".
        cluster_correlation_span : float, optional
            Per-call clustering-correlation EWMA span. ``None`` first falls
            back to ``self.cluster_correlation_span`` and, when that is also
            ``None``, to the effective beta ``span``. Thus callers that do
            not supply this argument retain the historical coupled span.
        external_clusters : pandas.Series, optional
            Asset-to-cluster partition for HCGL or FCGL. When provided,
            cluster discovery is skipped while the model type and penalty
            geometry remain unchanged.
        external_linkage : numpy.ndarray, optional
            Linkage metadata accompanying ``external_clusters``.
        external_cutoff : float, optional
            Dendrogram cutoff metadata accompanying ``external_clusters``.

        Returns
        -------
        self
            Updated with ``coef_`` (N × M) and ``intercept_`` (N,).
        """
        x, y = coerce_fit_inputs(x, y)
        validate_external_clusters(
            self.model_type, external_clusters, external_linkage, external_cutoff,
        )
        eff_span, eff_cluster_correlation_span = resolve_spans(
            self, span, cluster_correlation_span,
        )
        x_np, y_np, valid_mask = get_x_y_np(
            x=x, y=y, span=eff_span, demean=self.demean
        )
        prep = self._prepare_fit(
            x=x, y=y, x_np=x_np, y_np=y_np,
            valid_mask=valid_mask, eff_span=eff_span,
            eff_cluster_correlation_span=eff_cluster_correlation_span,
            external_clusters=external_clusters,
            external_linkage=external_linkage,
            external_cutoff=external_cutoff,
        )
        result = solve_prepared(self, prep, x_np, y_np, valid_mask, eff_span, verbose)
        self._finalize_fit(
            result=result, x=x, y=y, valid_mask=valid_mask, eff_span=eff_span,
            eff_cluster_correlation_span=eff_cluster_correlation_span,
            asset_clusters=prep.asset_clusters, linkage=prep.linkage, cutoff=prep.cutoff,
        )
        return self

    def _prepare_fit(
        self,
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
    ) -> "_PreparedFit":
        """derive the reg_lambda-independent solver inputs.

        Clustering, the sign matrix, the prior, and the adaptive penalty
        weights do not depend on ``reg_lambda``. Extracted verbatim from
        ``fit`` so a regularisation path derives them once and reuses them
        across the grid. Sets ``self.derived_signs_`` as the in-line code
        did.
        """
        return prepare_fit(
            self, x=x, y=y, x_np=x_np, y_np=y_np, valid_mask=valid_mask, eff_span=eff_span,
            eff_cluster_correlation_span=eff_cluster_correlation_span,
            external_clusters=external_clusters, external_linkage=external_linkage,
            external_cutoff=external_cutoff,
        )

    def _finalize_fit(
        self,
        result: LassoEstimationResult,
        x: pd.DataFrame,
        y: pd.DataFrame,
        valid_mask: np.ndarray,
        eff_span: Optional[float],
        eff_cluster_correlation_span: Optional[float],
        asset_clusters: Optional[pd.Series],
        linkage,
        cutoff,
    ) -> None:
        """store fitted state from a solver result.

        Warmup zeroing, ``coef_``, the economic intercept ``alpha_const_``,
        and cluster bookkeeping. Extracted verbatim from ``fit`` so the
        single fit and every point on a regularisation path share one
        post-processing path.
        """
        install_fitted_state(self, fitted_state(
            self, result=result, x=x, y=y, valid_mask=valid_mask, eff_span=eff_span,
            eff_cluster_correlation_span=eff_cluster_correlation_span,
            asset_clusters=asset_clusters, linkage=linkage, cutoff=cutoff,
        ))

    def fit_reg_lambda_path(
        self,
        x: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
        reg_lambdas: Sequence[float],
        verbose: bool = False,
        span: Optional[float] = None,
        cluster_correlation_span: Optional[float] = None,
    ) -> List["LassoModel"]:
        """Fit at each ``reg_lambda``, sharing one derivation.

        Returns one fitted model per value in ``reg_lambdas``, in the same
        order. Each returned model is equivalent to a fresh :meth:`fit` at
        that ``reg_lambda`` (same ``coef_``, ``alpha_const_``, diagnostics).

        For the group-LASSO family (GROUP_LASSO, HCGL, FCGL) the
        ``reg_lambda``-independent derivation (clustering, signs, adaptive
        weights) is computed once via :meth:`_prepare_fit` and the penalty
        path is solved with :func:`solve_group_lasso_path`, which reuses one
        canonical form across the grid. LASSO, the cooperative estimators,
        and UniLasso have no path solver, so each grid point is a full
        :meth:`fit`.

        Each returned model owns its fitted state: it holds the complete
        sign, prior, bound and adaptive-weight diagnostics of a fresh fit,
        as independent copies. Coefficients agree with a fresh fit up to
        solver tolerance, since the path is solved as one parametrised
        problem.

        Primitive behind ``LassoModelCV(use_lambda_path=True)``. For the
        group-LASSO family ``self`` is left partially updated (its
        preparation diagnostics, such as ``derived_signs_``, are set); use
        the returned models, not ``self``.

        Parameters
        ----------
        reg_lambdas : sequence of float
            Penalty weights, in any order. The returned list is aligned with
            this sequence.
        x, y, verbose, span, cluster_correlation_span
            As in :meth:`fit`.

        Returns
        -------
        list of LassoModel
            One fitted model per ``reg_lambda``.
        """
        lambdas = [float(lv) for lv in reg_lambdas]
        if len(lambdas) == 0:
            raise ValueError("reg_lambdas must be non-empty")

        if not _mode_spec(self.model_type).lambda_path:
            # No path solver for these modes; a full fit per grid point. The
            # derivation repeats, but the result is identical to fit().
            return self._fit_each(lambdas, x, y, verbose, span, cluster_correlation_span)

        x, y = coerce_fit_inputs(x, y)
        eff_span, eff_cluster_correlation_span = resolve_spans(
            self, span, cluster_correlation_span,
        )
        x_np, y_np, valid_mask = get_x_y_np(
            x=x, y=y, span=eff_span, demean=self.demean,
        )
        prep = self._prepare_fit(
            x=x, y=y, x_np=x_np, y_np=y_np,
            valid_mask=valid_mask, eff_span=eff_span,
            eff_cluster_correlation_span=eff_cluster_correlation_span,
        )

        if prep.is_lasso_mode:
            # Single asset (N=1): the group penalty is degenerate, so there is
            # no shared canonical form to exploit across the grid. Fall through
            # to a full fit per grid point, exactly as the non-path estimators
            # above. Each fit applies the same single-asset LASSO reduction.
            return self._fit_each(lambdas, x, y, verbose, span, cluster_correlation_span)

        results = solve_prepared_path(
            self, prep, x_np, y_np, valid_mask, eff_span, lambdas, verbose,
        )

        # Every returned model carries its own copy of the full preparation state that a fresh
        # fit at its reg_lambda would hold, then the solve-dependent state of its result.
        out: List["LassoModel"] = []
        for lam, result in zip(lambdas, results):
            params = self.get_params()
            params["reg_lambda"] = lam
            clone = LassoModel(**params)
            install_fitted_state(clone, preparation_state(self))
            clone._finalize_fit(
                result=result, x=x, y=y, valid_mask=owned(valid_mask),
                eff_span=eff_span,
                eff_cluster_correlation_span=eff_cluster_correlation_span,
                asset_clusters=owned(prep.asset_clusters),
                linkage=owned(prep.linkage), cutoff=prep.cutoff,
            )
            out.append(clone)
        return out

    def _fit_each(self, lambdas, x, y, verbose, span, cluster_correlation_span):
        """A fresh, full :meth:`fit` per ``reg_lambda`` (modes without a path solver)."""
        out: List["LassoModel"] = []
        for lam in lambdas:
            params = self.get_params()
            params["reg_lambda"] = lam
            out.append(LassoModel(**params).fit(
                x=x, y=y, verbose=verbose, span=span,
                cluster_correlation_span=cluster_correlation_span,
            ))
        return out

    def nowcast(
        self,
        x: pd.DataFrame,
        *,
        alpha_span: Optional[float] = None,
    ) -> LassoNowcastResult:
        """Nowcast future responses from realised factors and residual alpha.

        This analytic is available only for fits recorded with
        ``demean=True``. It keeps beta fixed, estimates statistical alpha as
        the terminal causal mean of the fit-time original-unit residuals
        ``y - X @ beta``, and returns ``X_target @ beta + stat_alpha``. It
        deliberately does not call :meth:`predict`: neither the economic
        ``alpha_const_`` nor the solver's de-meaned ``intercept_`` enters the
        nowcast.

        Parameters
        ----------
        x : pd.DataFrame, shape (K, M)
            Complete realised target factor returns. Columns must exactly
            equal the fitted factor columns in identity and order, and every
            target date must be strictly after the fit cutoff.
        alpha_span : float, optional
            EWMA span for the statistical residual alpha. ``None`` reuses
            the recorded effective beta span. If both are ``None``, alpha is
            the uniform mean of the residual history.

        Returns
        -------
        LassoNowcastResult
            Copied decomposition, fitted snapshots, and diagnostics.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted.
        ValueError
            If the fitted provenance or target data violate the causal
            nowcast contract.
        TypeError
            If ``x`` is not a pandas DataFrame with a DatetimeIndex.
        """
        return _nowcast.nowcast(self, x, alpha_span=alpha_span)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict response values in original units.

        Paper convention: Ŷ_t = α + β X_t, with α the **economic
        intercept** ``alpha_const_`` (the constant reconstructed from the
        same weighted means that produced β). Code (row-major):
        Ŷ = X @ β' + α.

        Versions before 0.5.1 added ``intercept_`` here — the EWMA-weighted
        residual mean on the *demeaned* data, which is identically zero for
        ``span=None`` and a finite-sample leftover otherwise. Predictions
        therefore omitted the asset means, and ``score()`` (used by
        :class:`LassoModelCV` and \\pkg{scikit-learn} model selection)
        understated R² for any response with a non-zero mean.

        When the model was fitted with ``demean=False``, the user asked for
        a through-origin fit and no constant is added: Ŷ = X @ β'.

        Parameters
        ----------
        x : pd.DataFrame, shape (T, M)
            Regressor data with columns matching ``fit()``.

        Returns
        -------
        pd.DataFrame, shape (T, N)
        """
        if self.coef_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # Accept ndarray for sklearn interop: map positionally to the fitted
        # factor columns (the order fit() saw).
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(np.atleast_2d(x), columns=list(self.coef_.columns))
        # Paper: Y_t = α + β X_t.  Row-major equivalent: Y = X @ β' + α
        y_hat = x[self.coef_.columns] @ self.coef_.T
        if self.demean and self.alpha_const_ is not None:
            y_hat = y_hat + self.alpha_const_.values
        return y_hat

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Mean R² across response variables.

        Parameters
        ----------
        x : pd.DataFrame, shape (T, M)
        y : pd.DataFrame, shape (T, N)

        Returns
        -------
        float
            Mean R² (higher is better).
        """
        y_hat = self.predict(x)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(
                np.atleast_2d(y), index=y_hat.index, columns=y_hat.columns
            )
        ss_res = ((y - y_hat) ** 2).sum(axis=0)
        ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        r2 = 1.0 - ss_res / ss_tot.replace(0, np.nan)
        return float(r2.mean())

    def __sklearn_tags__(self):
        """Estimator tags for \\pkg{scikit-learn} >= 1.6 interoperability.

        Declares the estimator as a multi-output regressor that tolerates
        NaN inputs, so that ``Pipeline``, ``GridSearchCV``, and
        ``cross_val_score`` accept it. Falls back gracefully on older
        \\pkg{scikit-learn} versions that do not call this hook.
        """
        try:
            from sklearn.utils import InputTags, Tags, TargetTags
        except Exception:  # pragma: no cover - older sklearn
            return None
        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True, multi_output=True),
            transformer_tags=None,
            classifier_tags=None,
            regressor_tags=None,
            input_tags=InputTags(allow_nan=True),
        )

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model.

        Reports the problem dimensions, the active model type, the number of
        discovered clusters (for the HCGL modes), the active-coefficient
        count and density, and the mean per-asset R-squared. Raises if the
        model has not been fitted.
        """
        return _inspection.summary(self)

    def plot_signs(self, ax=None):
        """Plot the derived sign matrix as a heatmap (requires matplotlib).

        Returns the matplotlib ``Axes``. Available only when
        ``auto_sign_constraints=True`` produced a ``derived_signs_`` matrix.
        """
        return _inspection.plot_signs(self, ax=ax)
