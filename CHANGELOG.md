# Changelog

All notable changes to `factorlasso` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [Unreleased]

## [0.9.0] — 2026-07-22

### Added
- `DistanceTransform` (exported): string enum selecting the
  correlation-to-distance transform for hierarchical clustering —
  `ONE_MINUS_RHO` (`d = 1 - rho`, default), `CHORD`
  (`d = sqrt(2(1 - rho))`, the Euclidean chord under which Ward's variance
  criterion is exact), and `ARCCOS` (`d = arccos(rho)`, the geodesic arc).
- `compute_clusters_from_corr_matrix` gains a `distance_transform` keyword
  (default `ONE_MINUS_RHO`), threaded like `linkage_method`. Plain strings
  (`'chord'`) are accepted. The default path is numerically identical to
  0.8.0: clipping `rho` to `[-1, 1]` before `1 - rho` equals the previous
  clip of `1 - rho` to `[0, 2]`.
- `LassoModel` gains a `distance_transform` constructor parameter, consumed
  by the cluster-discovery modes (`HIERARCHICAL_CLUSTER_GROUP_LASSO`,
  `FACTOR_CLUSTER_GROUP_LASSO`, `COOPERATIVE_CLUSTER_GROUP_LASSO`) and
  ignored otherwise. Flows through `get_params`/`set_params` and
  `LassoModelCV` automatically.
- `tests/test_distance_transform.py` — transform values, metric properties,
  exact 0.8.0 regression on the default path, monotone-invariance of
  rank-based linkages, `LassoModel` threading, and validation errors.

### Changed
- Documentation only: `compute_clusters_from_corr_matrix` documents that
  `cutoff_fraction` is calibrated per transform and does not port across
  transforms. Switching from `ONE_MINUS_RHO` at fraction `f` while
  preserving the implied pairwise merge threshold requires `sqrt(f)` under
  `CHORD` and `arccos(rho*)/arccos(rho_min)` with
  `rho* = 1 - f(1 - rho_min)` under `ARCCOS`. No behavioural change to any
  existing default.

## [0.8.0] — 2026-07-12

### Changed
- **Dependency floors raised.** Python `>=3.10` (was `>=3.9`), `numpy>=2.0`
  (was 1.22), `pandas>=2.2.0` (was 1.4), `scipy>=1.12.0` (was 1.9),
  `cvxpy>=1.3.0`. Python 3.9 is no longer supported; the 3.9 classifier is
  dropped. The dependency surface is unchanged in composition — `numpy`,
  `pandas`, `scipy`, `cvxpy`, `openpyxl` — only the floors move.

### Fixed
- Single-asset (N=1) estimation is completed across the clustering and
  group-based estimators. `compute_clusters_from_corr_matrix` short-circuits a
  1x1 correlation matrix, returning the lone asset in cluster 1, an empty
  `(0, 4)` linkage and a zero cutoff. SciPy's `squareform` / `linkage` are
  undefined for one observation — the condensed pairwise-distance vector is
  empty and `linkage` raises on an empty distance matrix — so the short-circuit
  keeps the function total: callers building group loadings receive a valid
  one-element `pd.Series` rather than an exception. This unblocks any fit whose
  frequency bucket holds exactly one asset (e.g. a mandate whose sole
  quarterly-rebalanced sleeve is a single hedge-fund proxy).
- `LassoModel.fit_reg_lambda_path` honours the N=1 reduction: in lasso mode the
  group path is bypassed and a full fit is run per grid point, instead of
  building a group loading from an empty cluster set.
- FCGL now names the assets it zeroes. When an asset has too few observations
  the warning lists the offending columns (first ten, then a `(+n more)`
  count) rather than reporting a bare count.

### Added
- `tests/test_single_asset_cluster.py` — covers the lone-cluster correlation
  case, a single-asset fit across every estimator, and the single-asset
  `reg_lambda` path across every estimator.
- JSS submission: Stage 0 usage example (`papers/jss_2026/paper/usage_example.py`)
  with the signs heatmap regenerated; `replicate.py` and the replication-zip
  builder updated to match.

## [0.7.2] — 2026-06-22

### Fixed
- Single-asset (N=1) estimation is handled across every clustering and
  group-based estimator. `compute_clusters_from_corr_matrix` short-circuits a
  1x1 correlation matrix (lone asset in cluster 1, empty linkage, zero cutoff).
  GROUP_LASSO, HCGL, and FCGL reduce to plain LASSO at N=1. The FCGL adaptive
  block-weight step and the `fit_reg_lambda_path` group path now honour that
  reduction instead of building a group loading from an empty cluster set. The
  cooperative estimators fit a single asset directly.

## [0.7.1] — 2026-06-22

### Fixed
- Cluster construction reached `squareform` through
  `scipy.cluster.hierarchy.distance`, an undocumented re-export that recent
  scipy builds no longer expose. `compute_clusters_from_corr_matrix` now
  imports `squareform` from `scipy.spatial.distance` directly. Every clustering
  estimator (HCGL, FCGL, Pearson clustering at `span=None`, auto-sign on
  clusters) now runs regardless of the installed scipy build. The change is
  behaviour-identical: both paths resolve to the same function and fitted
  coefficients are unchanged.

## [0.7.0] — 2026-06-20

### Added
- `solve_group_lasso_path(x, y, group_loadings, reg_lambdas, ...)`: solves
  the group-LASSO family over a grid of `reg_lambda` values, reusing a
  single canonical form. The penalty weight is passed to CVXPY as a
  `cvxpy.Parameter(nonneg=True)`, so the disciplined-parametrised programme
  is compiled once and the conic form is reused on each solve; the
  canonicalisation otherwise repeated per grid point is paid one time.
  Covers group LASSO, HCGL (`block_mode="row"`), FCGL
  (`block_mode="cluster_factor"`), the sparse-group L1 term, sign
  constraints, the prior, and adaptive reweighting. Results are identical
  to solver tolerance to calling `solve_group_lasso_cvx_problem` per grid
  point, and aligned with `reg_lambdas`. Intended for `reg_lambda`
  selection (CV/BIC), regularisation-path figures, rolling backtests with
  per-date selection, and threshold or cutoff sweeps; no benefit for a
  single `reg_lambda`. Measured speed-up at N=100, M=9 over a 15-point grid
  is about 1.5x, bounded by the canonicalisation share of each solve.

- `LassoModel.fit_reg_lambda_path(x, y, reg_lambdas, ...)`: fits at each
  grid point sharing one derivation. For the group-LASSO family
  (GROUP_LASSO, HCGL, FCGL) the `reg_lambda`-independent work (clustering,
  signs, adaptive weights) is computed once and the penalty path is solved
  with `solve_group_lasso_path`; LASSO, the cooperative estimators, and
  UniLasso fall back to a full fit per grid point. Each returned model is
  equivalent to a fresh `fit` at that `reg_lambda` (group family matches to
  solver tolerance, ~1e-9 to 1e-12 observed; fallback matches exactly).

- `LassoModelCV.use_lambda_path` (bool, default `False`): when `True` and
  the model is a group-LASSO-family estimator, each fold derives once and
  sweeps the grid via `fit_reg_lambda_path` instead of re-fitting per
  `reg_lambda`. **Default `False` keeps the cross-validation path
  byte-identical to before**; enabling it is numerically identical up to
  solver tolerance (`cv_scores_` agree to ~1e-10, identical `best_lambda_`
  observed) and faster for the group family. The flag is a no-op for
  non-group estimators (the per-lambda loop runs in both cases).

- `solver_fallbacks` parameter on `LassoModel` and on the underlying solver
  functions. When set to an ordered list of solver names (for example
  `["ECOS", "SCS"]`), a fit that fails with the primary solver is retried with
  each fallback in turn, and a `SolverError` is raised only if the primary and
  every fallback fail. The default is `None`, which preserves the previous
  behaviour exactly: a single solve whose outcome stands, verified byte for
  byte against the regression oracle (worst absolute difference 0 across 23
  scenarios). Intended for production callers who prefer graceful degradation
  over a hard failure on one solver.

### Changed
- Internal: the group-LASSO problem assembly is factored into a shared
  `_build_group_lasso_problem` helper used by both
  `solve_group_lasso_cvx_problem` and `solve_group_lasso_path`. The
  single-solve path is behaviour-preserving (existing test suite unchanged).
- Internal: `LassoModel.fit` is factored into `_prepare_fit` (the
  `reg_lambda`-independent derivation) and `_finalize_fit` (warmup zeroing,
  `coef_`, the economic intercept, cluster bookkeeping), with the solver
  dispatch left inline. Both blocks are relocated verbatim; `fit` output is
  byte-for-byte unchanged across a 23-scenario regression oracle spanning
  every estimator and option combination (worst absolute difference 0).

- The warmup zeroing step now emits a `UserWarning` reporting how many assets
  had fewer than `warmup_period` valid observations and were zeroed, rather
  than zeroing them silently. Numerical output is unchanged.


## [0.6.0] — 2026-06-19

### Added
- `linkage_method` parameter on `LassoModel` and on
  `compute_clusters_from_corr_matrix`, validated against the SciPy linkage
  set (`single`, `complete`, `average`, `weighted`, `centroid`, `median`,
  `ward`). The cluster-discovery step is now fully configurable, and the
  default `ward` reproduces the prior behaviour.

- `LassoModelType.UNILASSO`: a per-response univariate-guided estimator
  following Chatterjee, Hastie, and Tibshirani (2025). Stage one fits each
  factor univariately; stage two combines the univariate fits with non-negative
  weights, so each loading keeps its univariate sign. Two parameters control it:
  `unilasso_loo` (leave-one-out stage-1 fits, default `True`) and
  `unilasso_non_negative` (theta >= 0 in stage 2, default `True`). The mode uses
  no grouping and ignores `group_data` and `cutoff_fraction`.
- `LassoModelType.COOPERATIVE_GROUP_LASSO` and
  `LassoModelType.COOPERATIVE_CLUSTER_GROUP_LASSO`: the cooperative LASSO of
  Chiquet, Grandvalet, and Charbonnier (2012). The penalty splits each
  coefficient into positive and negative parts and penalises the two parts as
  separate groups, which encourages a soft within-group sign coherence the data
  can overrule. The first mode reads an external `group_data` partition; the
  second discovers clusters the same way as HCGL and FCGL via `cutoff_fraction`.
  Neither mode gates and neither sets a hard sign constraint.
- Public solvers `solve_unilasso_cvx_problem` and
  `solve_cooperative_group_lasso_cvx_problem`, exported from the package root.

### Changed (breaking)

- Renumbered the integer values of `LassoModelType` to insert `UNILASSO = 2`.
  The new ordering is `LASSO = 1`, `UNILASSO = 2`, `GROUP_LASSO = 3`,
  `HIERARCHICAL_CLUSTER_GROUP_LASSO = 4`, `FACTOR_CLUSTER_GROUP_LASSO = 5`,
  `COOPERATIVE_GROUP_LASSO = 6`, and `COOPERATIVE_CLUSTER_GROUP_LASSO = 7`. The
  values of `GROUP_LASSO`, `HIERARCHICAL_CLUSTER_GROUP_LASSO`, and
  `FACTOR_CLUSTER_GROUP_LASSO` therefore changed. Code that selects a mode by
  name is unaffected. Code or serialised configuration that selects a mode by
  integer value must update. No change to behaviour or to any numerical output.

### Notes

- Sparse Group LASSO remains a configuration of `GROUP_LASSO` through
  `l1_weight > 0` (Simon et al. 2013), not a separate enum member.

## [0.5.5] — 2026-06-15

### Changed (breaking)

- Renamed the `LassoModelType.CLUSTER_FACTOR_GROUP_LASSO` enum member to
  `LassoModelType.FACTOR_CLUSTER_GROUP_LASSO`, for consistency with
  `HIERARCHICAL_CLUSTER_GROUP_LASSO` (both now end in `_CLUSTER_GROUP_LASSO`)
  and so the member initials match the FCGL ("Factor-Clustering Group LASSO")
  acronym used in the paper and documentation. The integer value is unchanged
  (`4`), so code or serialised configuration that selects the mode by value
  continues to work. Code that references the member by name must update to
  `FACTOR_CLUSTER_GROUP_LASSO`; the old name no longer exists and raises
  `AttributeError`. No change to behaviour or to any numerical output — only
  the member name changed.

## [0.5.4] — 2026-06-14

### Changed (breaking)

- Renamed the `LassoModelType.GROUP_LASSO_CLUSTERS` enum member to
  `LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO`, matching the HCGL
  ("Hierarchical Clustering Group LASSO") name used throughout the
  accompanying paper. The integer value is unchanged (`3`), so any code
  or serialised configuration that selects the mode by value continues to
  work. Code that references the member by name must update to
  `HIERARCHICAL_CLUSTER_GROUP_LASSO`; the old name no longer exists and
  raises `AttributeError`. No change to the estimator's behaviour or to
  any numerical output — only the member name changed. Downstream callers
  (`optimalportfolios`, `rosaa`) and any user scripts must rename the
  reference.

## [0.5.3] — 2026-06-14

### Changed (non-functional)

- Documentation correction in `cluster_utils.py`: the Ward-linkage step is
  now described as a stable correlation-clustering heuristic on the
  correlation dissimilarity `1 - rho`, not as exact Ward variance
  minimisation in Euclidean space (which would use the chord distance
  `sqrt(2(1 - rho))`). No change to the clustering computation or to any
  numerical output — the dissimilarity fed to Ward is unchanged.
- Docstring wording for `group_penalty="normalized"` corrected: the
  `sqrt(|g|/G)` weight is a heuristic cluster-size scaling, not invariant to
  arbitrary partition refinements.

### Fixed (lint, no behaviour change)

- Resolved `ruff` E501 (line length) on the FCGL enum comment and an internal
  comment, and I001 (import-order) on the adaptive-weights import block in
  `lasso_estimator.py`. Restores a green CI lint stage. No code logic changed.

## [0.5.2] — 2026-06-14

### Added (non-breaking)

- New `LassoModelType.CLUSTER_FACTOR_GROUP_LASSO` model type. It reuses
  the HCGL cluster discovery of `GROUP_LASSO_CLUSTERS` but takes the
  group penalty over each cluster×factor block rather than over each
  asset row. The cluster is the group of the L2 norm, so whole
  cluster×factor blocks enter or leave the model together. Unlike the
  row-grouped mode this couples assets within a cluster, so the problem
  is **not** block-separable across assets. Adaptive weights for this
  mode aggregate the per-cell weights to per-(cluster, factor) blocks
  via the new `_aggregate_to_block_weights` helper. `GROUP_LASSO_CLUSTERS`
  is unchanged — its row-grouped penalty path is byte-for-byte identical,
  and the existing solver/group/cluster tests pass.
- `solve_group_lasso_cvx_problem` gains a `block_mode` parameter
  (`"row"` default, `"cluster_factor"` opt-in) and an optional
  `col_weights` argument of shape `(G, M)` for per-block adaptive
  reweighting.



## [0.5.1] — 2026-06-12

Numerical and API correctness release. Three of the six changes alter
numbers; each was checkpointed explicitly before shipping.

### Fixed (numerical — checkpointed)

- **`LassoModel.predict` now adds the economic intercept `alpha_const_`**
  instead of the demeaned-residual diagnostic `intercept_`, restoring the
  documented `Ŷ = α + βX` contract for `demean=True` fits. Previously
  predictions omitted the response means, and `score()` — hence
  `LassoModelCV` and scikit-learn `GridSearchCV`/`cross_val_score` —
  understated R² for any response with a non-zero mean. With
  `demean=False` the fit is through-origin and no constant is added
  (previously the residual-mean diagnostic was added). `intercept_`
  itself is unchanged.
- **`get_x_y_np` demeans on NaN-preserved arrays** (zero-fill moved after
  the demean step). Previously `fillna(0.0)` ran first, so an asset with
  valid fraction `f` was demeaned by the diluted mean `f·μ` instead of
  `μ`, injecting a constant offset `(1 − f)·μ` into the solver response
  on its valid window — a second-order bias on β and a first-order
  deflation of the residual-variance and R² diagnostics. `np.nanmean`
  (span=None) and the NaN-aware `compute_ewm` recursion (EWMA path) now
  compute the means over valid observations only. No effect on fully
  observed panels (bit-identical).
- **`derive_sign_constraints` cluster mode masks the slope numerator by
  `x_agg_valid`**, matching the denominator. Previously, rows where some
  cluster members were missing entered the numerator with a biased
  zero-filled aggregate while being excluded from the denominator,
  inflating the cluster-pooled slope (observed ~40% on a
  half-missing-member panel) and breaking the closed-form SSR identity
  behind the t-gate. Only the public regressor-cluster mode with
  NaN-bearing `x` was affected; the `LassoModel.fit` auto-sign path
  (asset-cluster pooling) never hits this branch.

### Fixed (API)

- 1-D `np.ndarray` `x` passed to `fit` is now interpreted as one
  regressor of length T (mirroring 1-D `y` and `pd.Series` handling).
  Previously `np.atleast_2d` produced a `(1, T)` row and `fit` failed
  with a misleading index-alignment error.
- `LassoModel.copy()` builds a fresh, unfitted estimator from
  `get_params()` (scikit-learn `clone` semantics). Previously it
  round-tripped through `dataclasses.asdict`, carrying stale fitted
  state into the copy and corrupting the nested `LassoEstimationResult`
  into a plain `dict`.

### Changed (JSS simulation harness)

- The calibrated benchmark (paper §6.5) and the empirical ETF
  application (§7) now run the **production configuration** of the
  MATF-CMA deployment, set a priori rather than tuned:
  `cutoff_fraction=0.40`, gate `auto_sign_threshold_t=1.0`, adaptive
  reweighting with `auto_sign_adaptive_floor=0.5`, plus the public
  sub-asset-class sign overlay in the empirical section. The synthetic
  ablation (§6.1–6.4) stays at package defaults (the "package
  profile"). The production EWMA span is excluded throughout the
  article: the simulation DGP is stationary (time-decayed weights are
  pure efficiency loss there), and the empirical section runs the same
  uniform-weight convention for consistency. This replaces the former
  DGP-grid-searched `cutoff_fraction=0.50` in §6.5.
- `metrics.sign_agreement_rate` thresholds `beta_hat` at the activity
  tolerance before taking signs. Interior-point coefficients at
  ±O(1e-10) carried platform-dependent noise signs that moved the metric
  by O(1e-3) between otherwise-identical runs (observed: 0.915 Windows
  vs 0.913 Linux on one estimator). Near-zero estimates on true-support
  cells now deterministically count as misses. Headline-ablation sign
  rates move down by 0.011–0.069; all other metrics bit-identical.
- Stale `test_matf_calibrated_raises_not_implemented` replaced with a
  smoke test of the implemented calibrated-DGP path.
- Committed `simulations/results/` artifacts replaced with the full
  5-seed grid (2,850/2,850 cells ok) so the shipped files back the
  published headline table directly; previously a 1-seed quick run was
  committed.

### Added

- Regression tests: cluster-mode NaN-in-x slope and gate, valid-window
  demeaning, economic-intercept prediction, through-origin
  `demean=False` prediction, 1-D `x` handling, `copy()` semantics
  (`tests/test_cluster_nan_x_and_api_fixes.py`).

### Fixed (numerical — checkpointed)

- **With `span=None`, HCGL clustering now computes the sample Pearson
  correlation** (pairwise-complete over valid observations), matching
  both the documented contract of the JSS paper §2.3 and the uniform
  loss weighting. Previously the call routed through
  `compute_ewm_covar`, whose `ewm_lambda = 0.94` default (effective
  span ≈ 32 observations, the RiskMetrics daily convention) silently
  applied — so a uniform-weight fit clustered on a trailing-window
  correlation (adjusted Rand index ≈ 0.40 against the Pearson
  partition on a T=120 panel). With a span set, the EWMA(span)
  correlation is used as before. Production always passes a span and
  is unaffected. `compute_ewm_covar` itself is unchanged (qis parity).



## [Unreleased] — Roadmap

- JSS submission (Q3 2026): cluster-pooled sign-derivation paper.
- Sphinx documentation under `docs/`, deployed to ReadTheDocs.
- Benchmark suite under `benchmarks/` comparing against `sparsegl`,
  `asgl`, `celer`, `adelie` on synthetic data.
- Simulation study harness under `simulations/` for support-recovery,
  sign-agreement, and prediction-MSE comparisons.
- Public reproduction notebook on the Fama-French 5-factor + S&P 500
  universe.

### Simulation harness (in-repo, not part of published wheel)

- Solver fallback chain `CLARABEL → ECOS → SCS` in
  `simulations.estimators._factorlasso_fit` handles rare pathological
  problem instances on the JSS 2026 grid (e.g. orthogonal factor
  covariance + cluster-pooled signs at very small `reg_lambda`).
- Result of solver choice surfaced as `solver_used` column in
  `results_long.parquet` for audit transparency.
- `scs` and `ecos` added to the `simulations` optional-dependency
  group so the fallback chain has alternates available. Without
  these, only CLARABEL is tried and cells with CLARABEL-specific
  numerical issues fail.


## [0.5.0] — 2026-06-06

### Changed — breaking, numerical (in-sample diagnostics only)

- In-sample diagnostics (`estimation_result_.alpha`, `ss_res`, `ss_total`,
  `r2`, and the `intercept_` alias / `summary()` mean R²) are now computed
  in the **nominal-span EWMA norm** (per-observation weight `lambda^k`,
  matching the solver's error norm). Previously these statistics reused the
  sqrt-decay solver weights *linearly*, which placed them at an effective
  span of ~`2*span` (e.g., a span-36 fit reported a ~72-span alpha).
  Solver loss, estimated betas (`coef_`), the economic intercept
  (`alpha_const_`), CV model selection (`LassoModel.score` /
  `LassoCV`), and production CMA alphas (`estimate_alpha`) are unchanged.
  `span=None` fits are unaffected (uniform weights are idempotent under
  squaring). The numerical delta is documented by the frozen fixture in
  `tests/test_diagnostic_norm.py::test_snapshot_delta_v050`.

### Added

- `tests/test_diagnostic_norm.py`: error-norm identity (solver loss is the
  nominal-span EWMA norm), diagnostic-norm equivalence against the
  `pandas.ewm(span, adjust=True)` convention, Kish-ESS check
  (ESS = span post-fix; the sqrt-linear reuse gives `(1+sqrt(lambda))/(1-sqrt(lambda)) ≈ 2*span`,
  kept as a regression guard), and the v0.4.x → v0.5.0 snapshot delta.
- Design rule documented in `_compute_solver_weights` /
  `_compute_solver_diagnostics`: solver weights are sqrt-decay row scalings
  and may only enter quadratic forms; any linear-EWMA statistic must use
  `weights**2`.

### Fixed — sign-gate SSR over-count under NaN factor columns

- The closed-form residual sum-of-squares used by the noise-floor t-gate
  (`_compute_sign_vector` and `_compute_sign_matrix_per_response` in
  `sign_constraints.py`) summed the response variation over a single global
  (or per-response) `Σ y²` that was not restricted to the rows where each
  factor is observed, while the slope denominator `D_j` and the degrees of
  freedom were already masked by `valid_x`. For a factor column carrying NaN
  (e.g. a later-inception factor series), the three terms in
  `SSR_j = ‖Y‖²_F − β_j² D_j` ranged over different row sets, over-counting
  `SSR_j`, inflating `σ²`, shrinking `|t_j|`, and over-conservatively gating
  that factor's sign constraint to zero. The fix masks the response
  sum-of-squares per factor by `valid_x`, matching `D_j` and `df_j`.
- Impact is confined to panels with NaN in factor columns. When every factor
  is fully observed the computation takes a fast path that is bit-identical to
  the prior code, so no published result, table, figure, or the §3.6 usage
  example changes. The bug never crashed, never flipped a derived sign, and
  never affected the fitted slopes (only the gate decision).
- Regression coverage in `tests/test_nan_valid_row_invariance.py`: per-response
  and cluster-pooled gates under NaN-in-X are checked against an honest
  drop-NaN reference and against the pre-fix global-`Σ y²` gate, so the tests
  fail on the old behaviour. Full suite 260 passing.

## [0.4.3] — 2026-06-02

**Documentation, citation, and JSS-manuscript consistency pass (no
functional code changes).**

### Fixed

- Adaptive-penalty-weight attribution corrected throughout (docstring of
  `sign_constraints._adaptive_penalty_weights`, the parameter-group
  comment in `lasso_estimator.py`, and `README.md`). The magnitude-aware
  penalty weight follows Zou (2006); it is distinct from the
  univariate-guided *sign* constraint of Richland et al. (2025) eq. (3.3),
  which is applied separately via the sign matrix. Earlier comments
  conflated the weight formula with eq. (3.3).
- `README.md` removed a dangling reference to a
  `factorlasso_sign_constraints_note.tex` technical note that is not part
  of the distribution; the derivation it pointed to lives in the JSS
  article appendix.
- `README.md` citation block and `CITATION.cff` version bumped to 0.4.3.
- JSS manuscript (`papers/jss_2026/paper/`): the §5.3 core-grid
  sparse-low-SNR figures in the prose (β-MSE 0.49 vs. 0.95, a 48 %
  reduction) and the matching summary sentence corrected to agree with
  the regenerated Figure 2 and the simulation parquet; the reproduction
  appendix's idiosyncratic-regime coherence gain corrected to 0.16;
  §5.2 coherence range stated as "approximately 0.70 (0.695 to 0.708)";
  a garbled sentence in the introduction reworded; package-version
  strings in Table 1's caption and the Computational Details section
  bumped to 0.4.3.
- JSS bibliography (`refs.bib`): the MATF-CMA self-citation changed from
  `@Article ... note = {Forthcoming}` to `@Unpublished ... "Working
  paper, under review at the Journal of Portfolio Management"`; the
  `factorlasso` package self-citation version note bumped to 0.4.3.
- `papers/jss_2026/simulations/study.yaml` header comment: stale
  `python -m simulations.run` invocation paths corrected to
  `python -m papers.jss_2026.simulations.run`.

### Changed

- Simulation reference grid (`papers/jss_2026/simulations/results/`)
  regenerated under 0.4.3 and merged from five single-seed runs
  (seeds 42–46). The merged grid is numerically identical to the 0.4.1
  reference (max absolute deviation 0 across every metric and cell),
  confirming the 0.4.1→0.4.3 changes are documentation-only; the
  `manifest.json` now records `factorlasso_version = 0.4.3`. The §5
  figures and Table 2 regenerate byte-identical from the refreshed
  parquet.

### Internal

- `tests/` and `papers/` brought to a clean `ruff` pass (import order,
  unused imports, f-strings without placeholders, over-length lines, and
  two semicolon-joined statements in the application plotting script).
  The package source (`factorlasso/`) was already clean. Test count and
  behaviour unchanged (252 tests, 243 pass / 9 skip).


## [0.4.2] — 2026-05-31

**Documentation and packaging metadata fixes (no functional code changes).**

### Fixed

- Citation block in `README.md` updated to `version = {0.4.2}` (was stale
  at 0.4.0).
- Test-count claim in `README.md` corrected to 252 shipped tests (was 201).
- Backward-compatibility note clarified: the "reproduces v0.3.8 fits
  bit-for-bit" claim now states this holds on fully observed panels only,
  cross-referencing the 0.4.1 valid-observation correctness fix for
  `NaN`-bearing panels.
- `Documentation` project URL (`pyproject.toml`) and `url` (`CITATION.cff`)
  redirected from the not-yet-live `factorlasso.readthedocs.io` to the
  GitHub repository.

### Added

- `README.md` quickstart now documents the scikit-learn interoperability
  (NumPy-array inputs, `Pipeline` / `GridSearchCV` / `cross_val_score`
  composition) and the `summary()` / `plot_signs()` helpers introduced in
  0.4.1.


## [0.4.1] — 2026-05-31

**Correctness fix (sign-derivation under heterogeneous inception dates).**

### Fixed

- The cluster-pooled (`_compute_sign_vector`) and per-response
  (`_compute_sign_matrix_per_response`) sign-derivation paths now compute
  the univariate slope denominator, the residual sum of squares, and the
  degrees of freedom over genuine (row, response) observations only,
  rather than over zero-filled rows with nominal sample length `T`. On
  panels with leading-`NaN` prefixes (assets with later inception dates),
  the previous code biased the pooled slope and inflated the gate
  `t`-statistic (the zero-filled rows deflated the residual variance
  because the dof charged the nominal `T`). The corrected slope now equals
  the honest drop-`NaN` univariate OLS slope to machine precision, and the
  gate `t`-statistic uses the valid-observation count in its dof.
- On fully observed panels (no `NaN`) the corrected code reproduces the
  previous nominal-`T` formula exactly, so all simulation-study results
  (which use complete synthetic panels) are unchanged. The S&P 500
  application is materially unchanged at the default `auto_sign_threshold_t
  = 0.75`: the 9 reduced-coverage constituents clear or fail the gate on
  the same side, so the derived sign matrix, the BIC tournament, and every
  printed Table 5 number are identical.

### Added

- `tests/test_nan_valid_row_invariance.py` pins the corrected behaviour:
  valid-row slope equality, gated-sign equality against the drop-`NaN`
  reference, and full-data backward compatibility with the nominal-`T`
  formula.
- scikit-learn interoperability: `LassoModel.fit`, `predict`, and `score`
  now accept NumPy arrays in addition to pandas objects, and the
  estimator declares `__sklearn_tags__` (sklearn >= 1.6), so it composes
  with `sklearn.pipeline.Pipeline`, `GridSearchCV`, and `cross_val_score`
  without wrapper code. Covered by `tests/test_sklearn_interop.py`.
- `LassoModel.summary()` returns a human-readable fit summary (dimensions,
  active-coefficient count, discovered cluster count, mean R^2);
  `LassoModel.plot_signs()` renders the derived sign matrix as a heatmap.
- JSS reproduction materials under `papers/jss_2026/`: single
  `replicate.py` entry point (quick/full modes with a captured
  session-info log), the S&P 500 application, and an empirical comparison
  script (`applications/compare_external.py`) benchmarking against
  scikit-learn, `asgl`, and `skglm`.

### Changed

- **License changed from MIT to GPL-3.0-or-later** (`LICENSE`,
  `pyproject.toml`, `CITATION.cff`, README). Note: the full GPL-3.0
  licence text must be present in `LICENSE` before distribution; the
  current file carries the standard GPL-3 header and notice.
- Application `run_application.py` lambda grid corrected to 8 points
  (10^-8 ... 10^-1); BIC-best selection breaks ties toward the smallest
  lambda; boxplot tick labels set via `set_xticklabels` for matplotlib
  cross-version compatibility.


## [0.4.0] — 2026-05-24

**Stabilisation release.** Promotes the v0.3.5 – v0.3.10 cluster-pooled
sign-derivation, closed-form SSR noise-floor gate, adaptive penalty
reweighting, and HCGL integration features to a stable API. No new
functional code, no breaking changes from v0.3.11; this release is a
deliberate API freeze before the JSS paper draft starts citing specific
function signatures.

### Stable API surface

The following surface is now committed to be backward-compatible until
v0.5.0 (see `COMPATIBILITY.md`):

- `LassoModel` constructor parameters, in particular the full
  `auto_sign_*` family (`auto_sign_constraints`, `auto_sign_threshold_t`,
  `auto_sign_adaptive_weights`, `auto_sign_adaptive_gamma`,
  `auto_sign_adaptive_floor`), `cutoff_fraction`, `l1_weight`,
  `group_penalty`, `factors_beta_loading_signs`, `factors_beta_prior`.
- `LassoModel` fitted attributes with trailing underscore, in particular
  `derived_signs_`, `clusters_`, `linkage_`, `cutoff_`, `alpha_const_`.
- `derive_sign_constraints` public signature
  (`x`, `y`, `clusters`, `master_constraints`, `auto_sign_threshold_t`,
  `return_slopes`).
- `validate_cluster_signs` public signature.
- `compute_clusters_from_corr_matrix` public signature.
- `CurrentFactorCovarData` / `RollingFactorCovarData` dataclass fields,
  including `derived_signs`.
- `LassoModelCV` constructor and fitted attributes.

Internal helpers (leading underscore in `sign_constraints.py`,
`lasso_estimator.py`, `ewm_utils.py`, etc.) remain unconstrained.

### Added

- `COMPATIBILITY.md` documenting the v0.4.x API stability commitment,
  the deprecation policy, and the upgrade path to v0.5.0.
- `simulations/` in-repo methodology study harness (not part of the
  published wheel): DGP, unified estimator interface, metrics module,
  orchestration CLI, JSS 2026 study YAML, 54 unit and smoke tests.
- New `simulations` optional dependency group in `pyproject.toml`
  (`pyyaml`, `joblib`, `pyarrow`); install with
  `pip install -e ".[simulations]"`.
- README section on the cluster-pooled sign-derivation mechanism with
  a minimal end-to-end example, surfacing the headline differentiator
  for first-time visitors.
- Mika Kastenholz added to `CITATION.cff` as co-author, reflecting the
  forthcoming JSS methodology paper. Preferred citation now points to
  the forthcoming MATF-CMA paper.
- Additional discoverability keywords in `pyproject.toml`
  (`sparse-group-lasso`, `adaptive-lasso`, `hcgl`,
  `cluster-pooled-sign-derivation`, `multi-response-regression`).

### Changed

- Package description updated to surface the cluster-pooled
  sign-derivation mechanism, HCGL integration, and adaptive penalty
  reweighting as the headline features. No code-level change.

### Migration

- **None.** Every v0.3.11 API contract holds bit-identically in v0.4.0.
  Confirmed by the full test suite (240 tests) plus a smoke test
  against the production rosaa pipeline config.

### Versioning policy from here

- v0.4.x: bug fixes, documentation, internal refactors. No public API
  changes.
- v0.5.0: any breaking API change to the stabilised surface. Will ship
  with at least one minor-version deprecation cycle of the affected
  symbols.
- v1.0.0: reserved for post-JSS-acceptance production-final stamp.


## [0.3.10] — 2026-05-23

### Fixed

- **`CurrentFactorCovarData` now carries a `derived_signs` field.**
  Closes a downstream-pipeline gap that surfaced from `rosaa`'s
  `unified_pipeline` write-out step: the pipeline tries to persist a
  `derived_signs` Excel sheet from `CurrentFactorCovarData.derived_signs`,
  but no such field existed in the dataclass. The auto-sign mechanism
  populated `LassoModel.derived_signs_` (since v0.3.5) but never
  propagated it through to the covariance snapshot consumed by
  downstream code. The field is `Optional[pd.DataFrame] = None`,
  defaults preserve backward compatibility, and the value flows through
  `filter_on_tickers` and `save`/`load` automatically.

- **`auto_sign_adaptive_weights=True` now has impact in the pure
  group-LASSO production config** (`l1_weight=0`). In v0.3.9 the adaptive
  reweighting was plumbed only through the L1 penalty term, which is
  zero-weighted in the production `GROUP_LASSO_CLUSTERS` configuration
  used by MAC and CMA pipelines. The adaptive flag therefore had no
  effect on the actual production estimator.

  This release routes the same per-cell adaptive weights through the
  group L2 norms via a per-asset row aggregation, following
  Wang & Leng (2008)'s adaptive group lasso. The penalty becomes

      λ · Σ_g w_g · Σ_k∈g  W_k · ||β_k - β⁰_k||_2

  where `W_k = sqrt(mean_{j: s_kj ≠ 0} W_kj²)` is the root-mean-square
  of the per-cell weights `W_kj = 1 / max(|β̂_uni_kj|, floor)^γ` over
  the non-pinned factors in asset row `k`. Gated cells (`s_kj = 0`)
  are excluded from the aggregation; their hard sign constraint
  already forces `β_kj = 0` independently.

### Added

- New `derived_signs: Optional[pd.DataFrame] = None` field on
  `CurrentFactorCovarData`. Stores the `(N × M)` solver-facing sign
  matrix actually consumed by the LASSO problem (auto-derived layer
  optionally overlaid with practitioner-set explicit signs).
  Round-trips through `filter_on_tickers` (subset/rename in lock-step
  with `y_betas`) and `save`/`load` (new `derived_signs` Excel sheet,
  optional — files predating this version load with the field set to
  `None`).

- New helper `factorlasso.sign_constraints._aggregate_to_row_weights`
  performing the RMS row aggregation. Returns a length-N vector of
  per-asset weights; assets with all cells pinned fall back to
  `W_k = 1` (no-op).
- New optional `row_weights: np.ndarray (N,)` parameter on
  `solve_group_lasso_cvx_problem`. When supplied, each per-asset L2
  norm `||β_k - β⁰_k||_2` in the group penalty is multiplied by
  `row_weights[k]` before being summed within its group. Plumbed
  automatically by `LassoModel.fit` when `auto_sign_adaptive_weights=True`.

### Rationale for row aggregation by root-mean-square

The L2 norm in the group penalty pairs naturally with an L2-aggregation
of the per-cell weights. RMS preserves Cauchy-Schwarz scaling: a row of
uniformly small slopes gets a uniformly large weight, a row dominated by
one strong slope gets a moderate weight. For an asset with
`|β̂_uni_kj| = 1` across all factors, the aggregation returns
`W_k = 1` exactly, preserving the existing per-cluster scaling
`√(|g|/G)` without any multiplicative drift — backward-compatible
with v0.3.9 default behaviour. Alternatives considered (mean, max,
sum) were rejected: mean would be too soft, max would be overly
aggressive on rows with one weak factor, sum would scale with row
length.

### Tests

- 4 new tests in `tests/test_auto_sign_adaptive_weights.py` covering:
  RMS aggregation correctness (uniform weights → 1.0, pinned cells
  excluded, all-pinned fallback to 1.0), multi-row vectorisation,
  production-config impact regression (the v0.3.9 no-op bug must not
  resurface), and backward-compatible default behaviour.

### Documentation

- README: extended the adaptive-weights subsection with a paragraph on
  the row-aggregation path and its impact in pure-group-LASSO configs.
- LaTeX note `factorlasso_sign_constraints_note.tex`: extended the
  adaptive-reweighting paragraph to cover both the L1 and group-L2
  formulations, with Wang & Leng (2008) cited as the adaptive group
  lasso antecedent.


## [0.3.9] — 2026-05-23

### Added

- **`auto_sign_adaptive_weights: bool = False`** field on `LassoModel`.
  When set to `True` alongside `auto_sign_constraints=True`, the L1
  penalty becomes elementwise reweighted by the inverse univariate-slope
  magnitude:

      λ · |β_kj| / max(|β̂_uni_kj|, floor)^γ

  Following Zou (2006)'s adaptive Lasso and the formulation in Richland
  et al. (2025) eq. (3.3). Strong-evidence factors (large |β̂_uni|) get
  a lighter L1 penalty and can take larger multivariate coefficients;
  weak-evidence factors get a heavier penalty and are pushed harder
  toward the prior. The Zou (2006) oracle property carries over: at
  γ = 1 the penalty is magnitude-aware without being a thresholding
  operator.

- Two associated configuration fields:
  - `auto_sign_adaptive_gamma: float = 1.0` — Zou (2006) exponent.
    γ = 1 is the standard adaptive-Lasso default; larger values amplify
    the magnitude-aware reweighting.
  - `auto_sign_adaptive_floor: float = 1e-3` — stabiliser preventing
    weight explosion on near-zero slopes. |β̂_uni| is clipped at this
    floor before inversion.

- New helper `factorlasso.sign_constraints._adaptive_penalty_weights`
  converts (slopes, signs) → adaptive penalty matrix.

- New optional `penalty_weights: np.ndarray (N, M)` parameter on
  `solve_lasso_cvx_problem` and `solve_group_lasso_cvx_problem`. When
  supplied, the L1 penalty becomes `λ · Σ W_kj |β_kj - β⁰_kj|`. Plumbed
  automatically by `LassoModel.fit` when the adaptive flag is active.

### Internal

- `_compute_sign_matrix_per_response` gained a `return_slopes: bool`
  parameter to expose the underlying univariate slope matrix needed by
  the adaptive-weight derivation. Backward compatible: existing
  call-sites that don't request slopes get unchanged behaviour.

### Tests

- 8 new tests in `tests/test_auto_sign_adaptive_weights.py` covering:
  default-False behaviour, no-op when sign-constraints disabled,
  Zou (2006) oracle property (strong factors preserved, weak factors
  shrunk harder), γ-exponent amplification, floor stabiliser preventing
  numerical explosion, and direct correctness of the
  `_adaptive_penalty_weights` helper.

### Documentation

- README: new subsection under *Data-driven sign constraints* documenting
  the adaptive-weight option, parameter triple, and pointer to the
  technical note for the formal derivation.


## [0.3.8] — 2026-05-22

### Performance

- **~5–140× speedup of the `auto_sign_constraints=True` derivation block.**
  The univariate t-statistic computation in `_compute_sign_vector` previously
  ran an `M`-deep Python loop that allocated a `(T, q)` residual matrix per
  column. Replaced with a closed-form pooled-OLS SSR identity that is fully
  vectorisable in `j`:

      SSR_j  =  ||Y||²_F  −  q · β_j² · (x_j' x_j)

  using `x_j' y_sum = β_j · q · (x_j' x_j)` from the slope definition. No
  residuals are materialised. Numerical output is bit-identical to the prior
  implementation (max abs diff ≈ 1e-15).

- **Bulk per-y-column path** for LASSO mode in `LassoModel.fit`. The previous
  implementation called `_compute_sign_vector` once per response column in
  an `N`-deep Python loop. Replaced with a single matrix-product + closed-form
  block (`_compute_sign_matrix_per_response`) that returns the full `(N, M)`
  sign matrix in one call. End-to-end rolling-window benchmark on a 25-factor
  × 160-asset universe: prior 2.5× slowdown of `auto_sign_constraints=True`
  vs `False` reduced to ~1.15–1.22×; the residual is dominated by the CVXPY
  solver itself processing the additional sign constraints.

### Tests

- **Cluster-mode threshold coverage.** Added four tests pinning the
  cluster-aggregated threshold gate behaviour (strong cluster survives,
  weak cluster pinned uniformly, master constraint can pierce a single
  cluster member, HCGL within-cluster sign coherence). Closes the v0.3.7
  coverage gap.

### Documentation

- **README** gains a dedicated section *"Data-driven sign constraints with
  a noise-floor gate"* covering `auto_sign_constraints`,
  `auto_sign_threshold_t`, the model-type dispatch table, and the explicit
  overlay semantic.
- **COMPARISON.md** gains a *"Data-driven sign derivation with noise gate"*
  row in the feature comparison table.

### Internal

- New helper `_compute_sign_matrix_per_response(x_arr, y_arr, threshold)`
  returns the bulk `(N, M)` sign matrix in a single vectorised call. Used by
  `LassoModel.fit` in LASSO / single-response mode. Cluster mode continues to
  use `_compute_sign_vector` per cluster — each call is now O(M) closed-form,
  not O(M) Python loop.


## [0.3.7] — 2026-05-22

### Added

- **`auto_sign_threshold_t`** — new optional parameter on `LassoModel` and
  on the public `derive_sign_constraints` function (default `0.75`). When
  `auto_sign_constraints=True` and the threshold is active, columns whose
  univariate t-statistic falls below the threshold have their sign
  constraint pinned to `0`, forcing β = 0 in the multivariate fit. This
  acts as a noise floor on the data-derived sign machinery: it prevents
  the multivariate solver from inheriting a noise-driven hard constraint
  for factors whose univariate slope sign is dominated by sampling noise.

  The default `0.75` is intentionally well below conventional significance
  thresholds (|t| = 0.75 corresponds to two-sided p ≈ 0.45). It is not a
  hypothesis test — only the worst noise-driven signs are filtered.
  Empirical calibration on thin financial panels: across five PE-strategy
  panels (T = 49–71 quarters), cap-weighted aggregate Direct Alpha shifts
  by ≤ 100 basis points across the threshold range [None, 1.0].

  Pass `auto_sign_threshold_t=None` to disable the gate and reproduce
  the v0.3.6 behaviour of always enforcing the slope-sign.

### Changed

- **`LassoModel(auto_sign_constraints=True)` is now noise-filtered by
  default.** Previously every auto-derived sign was enforced regardless
  of the strength of univariate evidence; now the default threshold of
  `0.75` filters out columns with negligible signal. Code that depends
  on the old behaviour can opt out by passing
  `auto_sign_threshold_t=None` explicitly.

### Internal

- `_compute_sign_vector` accepts `auto_sign_threshold_t` and computes
  per-column univariate t-statistics using the same no-intercept formula
  the slopes themselves are computed against. Cluster mode uses the
  cluster-mean regressor and shares the t-statistic across cluster
  members. Multi-response y pools residuals across response columns.



## [0.3.5] — 2026-05-22

### Added

- **`factorlasso.sign_constraints`** — new module providing data-driven
  derivation of sign constraints for ``LassoModel.factors_beta_loading_signs``
  from pooled univariate slopes. Addresses the within-cluster sign-alternation
  problem in tightly-collinear factor sets, where vanilla LASSO can assign
  opposite signs to two regressors with > 0.95 correlation, producing
  artificial long/short profiles inside a single factor cluster.

- **`derive_sign_constraints(x, y, clusters=None, master_constraints=None)`** —
  public function returning an ``(N × M)`` DataFrame compatible with
  ``LassoModel.factors_beta_loading_signs``. Two operating modes via the
  ``clusters`` argument:

  * **column-level** (``clusters=None``): one sign per regressor from its own
    pooled univariate slope. Allows within-cluster sign disagreement if two
    correlated regressors have opposite marginal effect on ``y``.
  * **cluster-level** (``clusters=<array>``): one sign per cluster from the
    slope of the cluster-mean regressor vs ``y``, broadcast to every cluster
    member. Guarantees within-cluster sign coherence — the property that
    eliminates within-cluster alternations in the fitted ``coef_``.

  Pooled across multi-response ``y`` (``Σ_k y[:, k]``); the resulting sign
  vector is broadcast across response rows of the output DataFrame.
  ``master_constraints`` accepts ``{name_or_idx: sign}`` for explicit
  column-level overrides on top of the data-derived signs. No preprocessing
  is performed on ``x`` or ``y`` — caller owns any centering/standardization.

- **`validate_cluster_signs(x, y, clusters)`** — diagnostic helper that
  flags regressors whose column-level univariate sign disagrees with their
  cluster's aggregate sign. Emits a ``UserWarning`` with the offending
  factor names. Use this before committing to a cluster spec to detect
  groupings that mix economically-different regressors.

- **`LassoModel.auto_sign_constraints: bool = False`** — new hyperparameter.
  When ``True``, signs are derived inside ``fit()`` from the EWMA-demeaned,
  NaN-masked arrays returned by ``get_x_y_np``, i.e. the exact same data
  the CVXPY solver consumes. This avoids the inconsistency of deriving
  signs on raw inputs while fitting on demeaned ones, and ensures
  per-fold sign derivation under ``LassoModelCV`` (no cross-fold leakage).

  The pooling strategy is dispatched by ``model_type`` — the auto-derivation
  uses the same asset-side grouping the solver does, with no extra user
  configuration:

  * ``LASSO`` (or single-column y): each y-column gets an independent
    univariate sign derivation. Rows of ``derived_signs_`` are produced
    one-by-one and may differ across responses.
  * ``GROUP_LASSO``: signs are pooled within each ``group_data`` group;
    every member of a group shares the same row in ``derived_signs_``.
  * ``GROUP_LASSO_CLUSTERS``: signs are pooled within each HCGL asset
    cluster (the same ``compute_clusters_from_corr_matrix`` output the
    group solver already uses).

  No factor-side cluster argument is exposed — sign-pooling structure is
  fully determined by the model type and its existing grouping inputs.

- **`LassoModel.derived_signs_`** — new fitted attribute (``pd.DataFrame``
  or ``None``) holding the final ``(N × M)`` sign matrix passed to the
  solver. Populated whenever sign constraints reached the solver — under
  ``auto_sign_constraints=True``, under explicit ``factors_beta_loading_signs``,
  or both. Provides a single audit artifact for monitoring and dendrogram
  / heatmap rendering of the constraint surface.

- **`tests/test_sign_constraints.py`** — 25 integration tests against the
  real ``LassoModel`` and ``LassoModelCV``, covering external API shape,
  drop-in compatibility with ``factors_beta_loading_signs``, cluster-mode
  coherence, CV propagation, demean-data sensitivity, misspecified-cluster
  diagnostics, the explicit-overlay-on-auto composition, and edge cases
  (X/Y shape mismatch, missing cluster assignments, invalid master values).

### Behaviour

- **`_compute_sign_vector`** (and through it, both ``derive_sign_constraints``
  and the internal ``LassoModel`` auto-sign path) is now NaN-agnostic at
  the function boundary. NaN entries in ``x`` or ``y`` are zero-filled on
  entry — mathematically equivalent to dropping those observations for the
  no-intercept univariate slope ``(x · y) / (x · x)``, since a zeroed row
  contributes nothing to either inner product. Previously, a single NaN
  in ``y`` poisoned every slope (because ``y.sum(axis=1)`` propagated NaN),
  and the external ``derive_sign_constraints`` returned an all-NaN sign
  matrix silently. The internal LassoModel path was incidentally safe
  because ``get_x_y_np`` zero-fills upstream, but the function itself was
  load-bearing on that. Now the safety is at the function boundary, where
  it belongs.

- When **both** ``auto_sign_constraints=True`` and ``factors_beta_loading_signs``
  are supplied to ``LassoModel``, they **compose** rather than conflict:

  * Auto-derived signs form the base layer — one pooled sign per factor,
    broadcast across responses (same sign across all assets).
  * The explicit ``factors_beta_loading_signs`` matrix is overlaid on top
    per-cell: non-NaN entries win, NaN entries inherit the auto value.

  This is the recommended pattern for production ROSAA pipelines where
  most factor signs should be data-driven (cluster-coherent) but specific
  assets need asset-specific overrides (e.g. a bond fund forced to zero
  equity loading regardless of marginal correlation).

- The single-layer paths are preserved unchanged. Setting only
  ``factors_beta_loading_signs`` works identically to ``< 0.3.5``; setting
  only ``auto_sign_constraints=True`` produces a pure data-derived
  ``(N × M)`` sign matrix.

- ``LassoModelCV`` automatically benefits from ``auto_sign_constraints``:
  when the ``base_model`` carries the flag, each fold derives its own
  signs from its training subset, eliminating the silent cross-fold
  leakage that exists when signs are derived once externally on full
  training data.




### Added

- **`LassoModel.alpha_const_`** — new attribute holding the **economic
  intercept α** of the regression in the original ``y = α + Xβ + ε``
  representation, paired consistently with the fitted β under the same
  weighted-least-squares objective:

  * for ``span=None`` (uniform weights), this is the sample-mean
    reconstruction ``α = ȳ_sample − x̄_sample · β`` (= OLS intercept);
  * for ``span=integer`` (EWMA weights), this is the EWMA-weighted-mean
    reconstruction using the same weights factorlasso applies in the
    loss function.

  The result is an internally consistent ``(α, β)`` pair: the weighted
  mean of residuals ``y − α − X·β`` is identically zero by the first-
  order condition. This is what users typically mean by "alpha" when
  decomposing returns into ``α + factor exposure``.

- **`examples/alpha_const_vs_intercept.py`** — worked example
  demonstrating the difference across span choices on a synthetic
  single-asset factor model with known α, including an explicit FOC
  check that confirms internal consistency at machine precision.

### Changed (documentation only, no behavioural change)

- **`LassoModel.intercept_` is now documented as raw solver output**, not
  the regression intercept. Its value is unchanged from 0.3.3 — it is
  populated from ``LassoEstimationResult.alpha``, which is the
  EWMA-weighted residual mean on the demeaned data the solver receives.
  Because the underlying solver fits a no-intercept model on centered
  data, this is a mechanical artefact of the fit:

  * for ``span=None`` it is identically zero by the OLS first-order
    condition;
  * for ``span=integer`` it is a finite-sample EWMA-demean leftover.

  Code that read ``model.intercept_`` in 0.3.3 continues to receive the
  same value. New code should use ``alpha_const_`` for the economic
  intercept.

- `LassoEstimationResult.alpha` docstring now explicitly states that the
  field is not the regression intercept and refers users to
  ``LassoModel.alpha_const_`` for the economic quantity.

- `LassoModel` class docstring updated with a clear distinction between
  ``alpha_const_`` (economic intercept α, weighted-consistent with β)
  and ``intercept_`` (raw solver output / EWMA-demean diagnostic).



## [0.3.2] — 2026-04-25

### Added

- **`l1_weight` keyword on `solve_group_lasso_cvx_problem` and `LassoModel`.**
  Adds an elementwise L1 penalty term on top of the existing group L2
  penalty, parameterised by a mixing weight `α ∈ [0, 1]`:

  ```
  P(β) = (1 - α) · λ · Σ_g w_g · ||β_g - β₀||_{2,1}
       +      α  · λ ·       ||β   - β₀||_1
  ```

  This is the Simon–Friedman–Hastie–Tibshirani (2013) Sparse Group LASSO
  formulation. The L1 term enables within-group elementwise sparsity:
  inside an HCGL cluster that is "active" under the group term,
  individual response-factor cells with noisy loadings can be zeroed
  out without killing the whole group.

  **Backward compatibility**: `l1_weight=0.0` is the default and the L1
  term is gated behind an explicit `if l1_weight > 0.0` check, so
  existing callers see a zero-cost no-op — β is bit-identical to v0.3.1
  (verified: Max |Δβ| = 0 across the test suite). The L1 term shares the
  same prior `β₀` as the group term, so the two shrinkage mechanisms
  compose consistently. Sign constraints and the `group_penalty`
  convention apply to both terms.

  **Limits**: at `α = 0` the problem reduces exactly to v0.3.1 pure
  group LASSO. At `α = 1` the group term drops out and the problem
  reduces to plain LASSO (verified parity: Max |Δβ| < 1e-4 against
  `solve_lasso_cvx_problem` at the same λ, solver-precision noise).
  Typical research range is `α ∈ [0.05, 0.20]`: the group term remains
  the primary selection mechanism while the L1 term prunes within-group
  noise.

  **Ignored for `model_type == LASSO`**, since L1 is the only penalty
  already — passing `l1_weight` has no effect in that path.

  **Validated at construction**: `l1_weight` outside `[0, 1]`, or
  non-finite (NaN, inf) raises `ValueError` in `LassoModel.__post_init__`
  and at solver entry.

### Tests

- New file `tests/test_l1_weight.py` with 26 tests covering: α=0
  backward compatibility (bit-identical to pre-feature code path at
  solver and LassoModel levels), LASSO-mode no-op, input validation,
  α=1 equivalence to pure LASSO, `l1_weight × group_penalty`
  interaction (group scheme is irrelevant at α=1 as expected),
  sparsity-progression property on an adversarial noise panel (spurious
  β mass decreases monotonically with α, at least 2× reduction from α=0
  to α=0.8), signal preservation at moderate α, sign constraint
  preservation at all α, and prior-centering applied to the L1 term.

### Fixed

- **`CITATION.cff` version alignment.** Bumped from 0.3.0 → 0.3.2 to
  match `pyproject.toml`.



## [0.3.1] — 2026-04-24

### Fixed

- **Ghost-asset cluster labels in `LassoModel.fit`.** Assets with fewer than
  `warmup_period` valid observations had their `coef_` zeroed and per-asset
  diagnostics (`alpha`, `ss_total`, `ss_res`, `r2`) NaN-ed — but still received
  cluster labels in `clusters_`. Downstream consumers that count or analyse
  clusters (e.g., `n_clusters` diagnostics, cluster-based risk attribution,
  regime detection) saw placeholder singleton labels for pre-launch or
  short-history assets, inflating cluster counts in early history (observed:
  83 raw vs 31 real clusters on a 160-asset multi-asset universe at
  2002-12-31; mean 40 vs 25 clusters across a 23-year backtest).

  **Fix**: cluster labels for short-history assets are now dropped from
  `clusters_` using the same mask already applied to `coef_`/diagnostics.
  After the fix, `clusters_`, `coef_`, and per-asset diagnostics are mutually
  consistent — a short-history asset has no cluster label, zeroed betas,
  and NaN statistics.

  **Backward compatibility**: when `warmup_period=None` (no warmup check),
  behavior is unchanged — `clusters_` contains labels for every asset in
  `y.columns`, as before.

  **Affected consumers**: any code that reads `clusters_` from a fitted
  `LassoModel` or the `clusters` field on `CurrentFactorCovarData`. In
  practice this includes HCGL (`GROUP_LASSO_CLUSTERS`) and external-groups
  (`GROUP_LASSO`) paths; `LASSO` and `OLS` paths are unaffected because
  they don't produce `clusters_`.



---

## [0.3.0] — 2026-04-20

### Added

- **`group_penalty` keyword on `solve_group_lasso_cvx_problem` and `LassoModel`.**
  Selects between two weighting schemes for the per-group `L_{2,1}` penalty:
  - `"normalized"` (default, unchanged from v0.2.2): `w_g = √(|g|/G)`.
    Group-count-invariant — keeps the effective regularisation scale stable
    across problems where the number of groups `G` varies. This is the
    appropriate choice for HCGL, where `G` is data-driven.
  - `"yuan_lin"` (opt-in): `w_g = √|g|`. Classical Yuan–Lin (2006).

  The two conventions are related by a constant factor `√G`, so results at
  `group_penalty="yuan_lin"` with regularisation `λ` match results at
  `group_penalty="normalized"` with regularisation `λ·√G`. Existing tuned
  `reg_lambda` values continue to produce bit-identical β under the default.

- **`cutoff_fraction` hyperparameter** on `LassoModel` and
  `compute_clusters_from_corr_matrix`. Controls the fraction of `max(pdist)` at
  which the HCGL dendrogram is cut; default `0.5` preserves existing behaviour.
  Smaller values produce more, tighter clusters; larger values produce fewer,
  looser clusters. Exported as `DEFAULT_CUTOFF_FRACTION` from
  `factorlasso.cluster_utils`.
- **Span validity checks** (`_validate_span`) applied at all public entry points:
  `LassoModel.__post_init__`, `LassoModel.fit`, `get_x_y_np`, `compute_ewm`,
  `compute_ewm_covar`, `ewm_recursion`, and the two CVXPY solvers. A span below 1
  now raises `ValueError` with a clear message instead of silently producing
  invalid decay factors.
- **`ewm_lambda` validity check** (`_validate_ewm_lambda`): the decay factor
  must lie in `[0, 1)`. Catches garbage values before the recursion is entered.
- **Shape/index guard in `CurrentFactorCovarData.get_y_covar`**: raises
  `ValueError` when `y_betas` and `y_variances` disagree on row ordering, rather
  than silently assembling a wrong-but-shape-matching covariance matrix.
- **Python 3.14 support** declared in `pyproject.toml` classifiers and CI matrix.

### Changed

- **`LassoModelCV.fit` exception handling narrowed.** Previously a bare
  `except Exception: pass` silently turned every fold failure into a NaN score.
  Now only solver-domain errors are swallowed — the tuple is
  `(cvx.error.SolverError, cvx.error.DCPError, ValueError,
  np.linalg.LinAlgError)`. `KeyboardInterrupt`, `MemoryError`, and bugs from
  invalid kwargs now propagate so real issues surface during debugging. A
  `verbose=True` CV run additionally logs each fold failure to stderr.
  `LassoModelCV` was new in 0.2.x and is not known to be used externally; if
  you rely on the old behaviour, pin 0.2.2.

### Fixed

- **`LassoModel.fit` span precedence**: `span or self.span` mistakenly treated
  `span=0` as "unset" and fell back to `self.span`. Replaced with an explicit
  `None` check. Zero span is invalid anyway (now caught by `_validate_span`), but
  the idiom was fragile.
- **`compute_expanding_power` positivity guard**: explicit check that
  `power_lambda > 0` and is finite before `np.log`. Previously a corrupt upstream
  value would poison the weight vector with `-inf` / `NaN` silently. Also guards
  `n >= 1`.
- **`examples/finance_factor_model.py`**: removed a dead line that fit and
  discarded an entire `LassoModel` without using the result. The example now
  uses the post-refactor `coef_` / `clusters_` attribute names directly.
- **`CITATION.cff`**: paper title corrected to _"Capital Market Assumptions Using
  Multi-Asset Tradable Factors: The MATF-CMA Framework"_ (the previous title
  conflated the CMA paper with an earlier SAA scope). Version aligned with
  `pyproject.toml`.
- **`factorlasso/__init__.py`**: imports sorted alphabetically within the
  first-party group to satisfy ruff `I001`. Section-label comments moved from
  between imports into `__all__` where they don't interrupt the sort.

### Documentation

- **README rewritten** in MLOSS style: one-line pitch, install + quickstart,
  three distinguishing features with snippets, when to use / when not, link to
  examples, citation, development. Release-notes content previously in the
  README has moved to this file.
- Docstrings for `solve_group_lasso_cvx_problem`, `LassoModel`, `get_y_covar`,
  `compute_clusters_from_corr_matrix`, `compute_expanding_power`, and
  `LassoModelCV` expanded to document new parameters and validation behaviour.

### Backward compatibility

All public API is preserved. Default behaviour of `solve_group_lasso_cvx_problem`
and `LassoModel` is bit-identical to v0.2.2 (verified via β-level parity test at
Max |Δβ| = 0 on a 140-asset × 8-factor HCGL problem with 15 clusters). Existing
`reg_lambda` values do not need retuning. The old sklearn-style attribute names
(`estimated_betas`, `clusters`, `linkage`, `cutoff`, `x`, `y`) remain available
as property aliases for `coef_`, `clusters_`, `linkage_`, `cutoff_`, `x_`, `y_`.

---

## [0.2.0] — 2026-04-18

### Added

- New module `factorlasso.cluster_utils` consolidating all clustering utilities.
- Public API exports: `get_clusters_by_freq`, `get_linkages_by_freq`,
  `get_cutoffs_by_freq`.
- Public API export: `get_linkage_array` (previously internal).

### Changed

- `compute_clusters_from_corr_matrix` moved from `lasso_estimator.py` to
  `cluster_utils.py` (still importable from top-level `factorlasso`).
- `get_linkage_array` moved from `factor_covar.py` to `cluster_utils.py`.
- `LassoModel` with `model_type=GROUP_LASSO` now populates `.clusters_` from
  externally-supplied `group_data`, for API uniformity with HCGL mode.

### Fixed

- **`compute_clusters_from_corr_matrix` distance-vector bug**: now uses
  `squareform(1 - C)` — the correct conversion from a correlation matrix to
  scipy's condensed pairwise-distance vector — instead of the previous buggy
  `pdist(1 - corr)`, which treated rows of `(1 - corr)` as observations in
  N-dimensional space and computed Euclidean distances between those rows. The
  previous metric was geometrically meaningful but not semantically what Ward
  linkage on correlations requires.
- `.gitignore` glob `*.idea/` corrected to `.idea/`.

### Internal

- All 143 existing tests continue to pass.
- Wheel correctly includes `cluster_utils.py`.

---

## [0.1.x] — 2026-03 to 2026-04

Initial public releases. Core features:

- `LassoModel` (dataclass estimator) with `LassoModelType` {LASSO, GROUP_LASSO,
  GROUP_LASSO_CLUSTERS} selector.
- `LassoModelCV` for time-series cross-validation of `reg_lambda`.
- `CurrentFactorCovarData` / `RollingFactorCovarData` containers for
  Σ_y = β Σ_x β' + D assembly.
- `ewm_utils` module: pure-NumPy reimplementation of `qis` EWMA recursion and
  covariance, matching to machine epsilon with one deliberate deviation on
  zero-variance diagonals in correlation mode.
- Sign constraints (0 / ±1 / NaN matrix) and prior-centered regularisation on β.
- sklearn-compatible `fit` / `predict` / `score` / `get_params` / `set_params`.
