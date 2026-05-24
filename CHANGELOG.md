# Changelog

All notable changes to `factorlasso` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



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
