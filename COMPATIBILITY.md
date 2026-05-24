# factorlasso — API Compatibility Policy

`factorlasso` follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This document specifies the surface that is committed to be stable and the
process for changes to it.

## Stable surface (v0.4.0)

The following public symbols are part of the stable API. Backward-incompatible
changes to their names, parameter signatures, default values, or documented
return contracts will not happen within the v0.4.x line.

### Core estimator

- `LassoModel`
  - Constructor parameters:
    `model_type`, `group_data`, `reg_lambda`, `span`, `span_freq_dict`,
    `cutoff_fraction`, `group_penalty`, `l1_weight`, `demean`, `solver`,
    `warmup_period`, `nonneg`, `factors_beta_loading_signs`,
    `factors_beta_prior`, `auto_sign_constraints`, `auto_sign_threshold_t`,
    `auto_sign_adaptive_weights`, `auto_sign_adaptive_gamma`,
    `auto_sign_adaptive_floor`.
  - Fitted attributes (trailing underscore):
    `coef_`, `intercept_`, `alpha_const_`, `estimation_result_`,
    `clusters_`, `linkage_`, `cutoff_`, `valid_mask_`,
    `effective_span_`, `derived_signs_`, `x_`, `y_`.
  - Methods: `fit`, `predict`, `score`, `get_params`, `set_params`.

- `LassoModelCV`
  - Constructor: `lambdas`, `n_splits`, `base_model`, `refit`.
  - Fitted attributes: `best_lambda_`, `best_score_`, `cv_scores_`,
    `best_model_`.
  - Methods: `fit`, `predict`, `score`.

- `LassoModelType` enum: `LASSO`, `GROUP_LASSO`, `GROUP_LASSO_CLUSTERS`.

- `LassoEstimationResult` dataclass: `alpha`, `ss_total`, `ss_res`, `r2`.

### Sign-constraint derivation

- `derive_sign_constraints(x, y, clusters=None, master_constraints=None,
  auto_sign_threshold_t=0.75, return_slopes=False)`
- `validate_cluster_signs(signs, clusters)`

### Clustering utilities

- `compute_clusters_from_corr_matrix(corr_df, cutoff_fraction=0.5)`
- `get_clusters_by_freq`, `get_cutoffs_by_freq`, `get_linkage_array`,
  `get_linkages_by_freq`

### Factor covariance assembly

- `CurrentFactorCovarData` dataclass fields, including `derived_signs`.
- `RollingFactorCovarData` dataclass fields.
- `VarianceColumns` enum.

### EWMA and helpers

- `compute_ewm`, `compute_ewm_covar`, `compute_expanding_power`,
  `set_group_loadings`.

### Solver entry points (low-level)

- `solve_lasso_cvx_problem`, `solve_group_lasso_cvx_problem`,
  `get_x_y_np`.

## Internal surface (not stable)

Anything not listed above is internal and may change without notice. In
particular:

- All functions and classes with a leading underscore.
- Module-internal helpers (e.g. `_compute_sign_vector`,
  `_compute_sign_matrix_per_response`, `_adaptive_penalty_weights`,
  `_aggregate_to_row_weights`).
- The numerical layout of `LassoEstimationResult` beyond the four
  documented fields.
- The CVXPY problem objects constructed inside `solve_*_cvx_problem`;
  callers depending on the internal structure of those objects (variables,
  parameters, constraints by index) are not protected.

## Deprecation policy

Any breaking change to the stable surface follows this process:

1. **Deprecation warning** added in a v0.4.x minor release using
   `DeprecationWarning`. The warning identifies the affected symbol,
   the replacement (if any), and the release in which the old behaviour
   is removed.
2. **At least one minor-version cycle** between deprecation warning and
   removal. For example, deprecation in v0.4.5 means removal no earlier
   than v0.5.0.
3. **Removal** in the next minor or major release after the deprecation
   cycle has elapsed.
4. **Changelog entry** in both the deprecation release and the removal
   release, under a `### Deprecated` or `### Removed` heading
   respectively.

## Numerical reproducibility

Within the v0.4.x line, fitted `coef_`, `derived_signs_`, and
`estimation_result_.r2` values for a given (data, parameters) tuple are
guaranteed to be bit-identical across patch releases on the same Python
and CVXPY version.

Across minor versions (v0.4 → v0.5), numerical changes may occur if
they are documented and the CHANGELOG explains the reason. Such
changes are treated as breaking and follow the deprecation policy
above where the old solver path remains available for one minor cycle
via an explicit opt-in flag.

## Out of scope

The CVXPY solver dependency (`CLARABEL`, `ECOS`, `SCS`) is not pinned
beyond the `cvxpy>=1.3` requirement in `pyproject.toml`. Numerical
results may vary at the last few decimal places across CVXPY versions
or across underlying solver versions; this is treated as inherent to
the solver, not as a `factorlasso` regression.

## Version targets

- **v0.4.x:** Bug fixes, documentation, internal refactors. No public
  API changes.
- **v0.5.0:** First release that may introduce breaking changes to the
  surface listed above, with appropriate deprecation cycle.
- **v1.0.0:** Reserved for the production-final stamp following
  acceptance of the JSS methodology paper.

## Questions

If you depend on a specific behaviour and are unsure whether it is part
of the stable surface, open an issue at
<https://github.com/ArturSepp/factorlasso/issues> and the contract will
be clarified explicitly in the documentation.
