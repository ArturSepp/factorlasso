# factorlasso — API Compatibility Policy

`factorlasso` follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This document specifies the surface that is committed to be stable and the
process for changes to it.

## Stable surface (0.16 series)

The names in `factorlasso.__all__` are the authoritative public API for 0.16. Backward-incompatible
changes to their names, parameter signatures, default values, or documented return contracts will
not happen within the 0.16.x patch line. Modules that happen to be visible through `dir(factorlasso)`
but are absent from `__all__` are not additional public entry points.

### Core estimator

- `LassoModel`
  - Constructor parameters:
    `model_type`, `group_data`, `reg_lambda`, `span`, `span_freq_dict`,
    `cutoff_fraction`, `linkage_method`, `distance_transform`,
    `cluster_correlation_transform`, `dependence_measure`, `gerber_threshold`,
    `n_clusters`, `cluster_smoother_type`, `smoother_delta`, `smoother_lambda`,
    `recluster_freq`, `group_penalty`, `l1_weight`, `demean`, `solver`,
    `solver_fallbacks`, `warmup_period`, `nonneg`,
    `factors_beta_loading_signs`, `factors_beta_prior`,
    `auto_sign_constraints`, `auto_sign_threshold_t`,
    `auto_sign_adaptive_weights`, `auto_sign_adaptive_gamma`,
    `auto_sign_adaptive_floor`, `unilasso_loo`, and
    `unilasso_non_negative`.
  - Fitted attributes (trailing underscore):
    `coef_`, `intercept_`, `alpha_const_`, `estimation_result_`,
    `clusters_`, `linkage_`, `cutoff_`, `valid_mask_`,
    `effective_span_`, `derived_signs_`, `x_`, `y_`.
  - Methods: `fit`, `predict`, `score`, `get_params`, `set_params`, `copy`,
    `summary`, and `plot_signs`.

- `LassoModelCV`
  - Constructor: `lambdas`, `n_splits`, `base_model`, `refit`,
    `use_lambda_path`.
  - Fitted attributes: `best_lambda_`, `best_score_`, `cv_scores_`,
    `best_model_`.
  - Methods: `fit`, `predict`, `score`.

- `LassoModelDiagonalityCV`
  - Constructor: `lambdas`, `n_splits`, `base_model`, `refit`,
    `use_lambda_path`, `significance`, `zero_rtol`, and `min_periods`.
  - Fitted attributes: `best_lambda_`, `best_score_`, `passed_`,
    `threshold_`, `diagnostics_`, `fold_scores_`, `missing_factors_`, and
    `best_model_`.
  - Methods: `fit`, `predict`, `score`.

- `LassoModelType` enum: `LASSO`, `UNILASSO`, `GROUP_LASSO`,
  `HIERARCHICAL_CLUSTER_GROUP_LASSO`, `FACTOR_CLUSTER_GROUP_LASSO`,
  `COOPERATIVE_GROUP_LASSO`, and `COOPERATIVE_CLUSTER_GROUP_LASSO`.

- `LassoEstimationResult` dataclass: `estimated_beta`, `alpha`, `ss_total`,
  `ss_res`, and `r2`.

### Sign-constraint derivation

- `derive_sign_constraints(x, y, clusters=None, master_constraints=None,
  auto_sign_threshold_t=0.75, return_slopes=False)`
- `validate_cluster_signs(x, y, clusters, warn=True)`

### Clustering, dependence, and smoothing

- `DistanceTransform`, `ClusterCorrelationTransform`,
  `ClusterCorrelationTransformResult`, and `DependenceMeasure`.
- `compute_clusters_from_corr_matrix`, `compute_dependence_matrix`,
  `compute_gerber_matrix`, `apply_cluster_correlation_transform`, and
  `remove_first_principal_component`.
- `get_clusters_by_freq`, `get_cutoffs_by_freq`, `get_linkage_array`,
  and `get_linkages_by_freq`.
- `ClusterSmootherType`, `RollingClusterData`,
  `apply_partition_distance_bonus`, `compute_rolling_smoothed_clusters`, and
  `smooth_similarity_ewma`.
- `compute_co_association_panel`, `ClusterStabilityStatistics`, and
  `compute_cluster_stability_statistics`.
- `StabilityPoolingType` and `score_with_stability_pooled_clusters`.

### Factor covariance assembly

- `CurrentFactorCovarData` dataclass fields, including `derived_signs`.
- `RollingFactorCovarData` dataclass fields.
- `VarianceColumns` enum.

### Residual validation

- `ResidualDiagnostics`, `Sparsity`, `diagnose_residuals`,
  `effective_sparsity`, `marchenko_pastur_edge`,
  `missing_factor_components`, `null_threshold`, `raw_offdiagonal_mass`,
  `residual_correlation`, and `suggest_tolerance`.

### Offline cluster lineage

- `RiskClusterReport`, `TaxonomyConfig`, `analyze_cluster_lineage`, and
  `run_cluster_lineage_report`.

### EWMA, group-loading, and solver helpers

- `compute_ewm`, `compute_ewm_covar`, `compute_expanding_power`,
  `set_group_loadings`.
- `solve_lasso_cvx_problem`, `solve_group_lasso_cvx_problem`,
  `solve_group_lasso_path`, `solve_cooperative_group_lasso_cvx_problem`,
  `solve_unilasso_cvx_problem`, and `get_x_y_np`.

## Internal surface (not stable)

Anything absent from `factorlasso.__all__` is internal and may change without notice. In particular:

- All functions and classes with a leading underscore.
- Module-internal helpers (e.g. `_compute_sign_vector`,
  `_compute_sign_matrix_per_response`, `_adaptive_penalty_weights`,
  `_aggregate_to_row_weights`).
- Dataclass implementation details beyond their documented public fields.
- The CVXPY problem objects constructed inside `solve_*_cvx_problem`;
  callers depending on the internal structure of those objects (variables,
  parameters, constraints by index) are not protected.

## Deprecation policy

Any breaking change to the stable surface follows this process:

1. **Deprecation warning** added in a minor release using `DeprecationWarning`.
   The warning identifies the affected symbol, the replacement (if any), and
   the earliest release in which the old behaviour may be removed.
2. **At least one minor-version cycle** between deprecation warning and
   removal. For example, deprecation in 0.15 means the old surface remains
   throughout 0.16 and removal occurs no earlier than 0.17.
3. **Removal** in the next minor or major release after the deprecation
   cycle has elapsed.
4. **Changelog entry** in both the deprecation release and the removal
   release, under a `### Deprecated` or `### Removed` heading
   respectively.

## Numerical reproducibility

Within the 0.15.x patch line, fitted `coef_`, `derived_signs_`, and
`estimation_result_.r2` values for a given (data, parameters) tuple are
guaranteed to be bit-identical across patch releases on the same Python
and CVXPY version.

Across minor versions, numerical changes may occur only when the changelog
states the affected path and reason. A change to defaults or a documented
numerical contract follows the deprecation policy above; where practical, the
old solver path remains available for one minor cycle through an explicit
opt-in flag.

## Out of scope

The CVXPY solver dependency (`CLARABEL`, `ECOS`, `SCS`) is not pinned
beyond the `cvxpy>=1.3` requirement in `pyproject.toml`. Numerical
results may vary at the last few decimal places across CVXPY versions
or across underlying solver versions; this is treated as inherent to
the solver, not as a `factorlasso` regression.

## Version targets

- **0.15.x:** Bug fixes, documentation, and compatible additions only. No
  backward-incompatible public API or default changes.
- **Later 0.x minors:** Compatible additions are preferred. Any planned
  removal follows the deprecation cycle above.
- **1.0.0:** Requires an explicit maintainer stability review. Scientific-paper
  review and publication status are tracked separately from software versioning.

## Questions

If you depend on a specific behaviour and are unsure whether it is part
of the stable surface, open an issue at
<https://github.com/ArturSepp/factorlasso/issues> and the contract will
be clarified explicitly in the documentation.
