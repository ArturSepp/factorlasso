# Changelog

All notable changes to `factorlasso` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
