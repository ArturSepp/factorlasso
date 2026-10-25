# factorlasso

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-03-22](https://github.com/ArturSepp/factorlasso/commit/3ea3c317f0c6bf663453c594ada381a96839f08a)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

**`factorlasso` estimates sparse multi-output factor models with sign constraints, prior-centered
shrinkage, data-driven grouped penalties, and consistent factor covariance assembly.**

It provides LASSO, Hierarchical Clustering Group LASSO (HCGL), Factor-Clustering Group LASSO
(FCGL), sparse-group, UniLasso, and cooperative penalties through auditable CVXPY formulations.
Its standalone output is a fitted `LassoModel` with labelled loadings, intercepts and
diagnostics; [inspect a completed estimator](docs/completed_estimator.md), including units,
dates, failure states and explicit table export, with the
[canonical offline example](examples/docs/completed_estimator.py). The financial factor-model
pipeline belongs to the private `factormodels` package; OptimalPortfolios
optimises supplied alphas and covariance, and QIS supplies general risk analytics and reports.
None is required to complete a factorlasso fit.

**Install:** `pip install factorlasso` · **Import:** `factorlasso` · **Status:** Stable

[![PyPI](https://img.shields.io/pypi/v/factorlasso?style=flat-square)](https://pypi.org/project/factorlasso/)
[![Python](https://img.shields.io/pypi/pyversions/factorlasso?style=flat-square)](https://pypi.org/project/factorlasso/)
[![CI](https://github.com/ArturSepp/factorlasso/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/factorlasso/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/factorlasso/badge/?version=latest)](https://factorlasso.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ArturSepp/factorlasso.svg?style=flat-square)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/factorlasso)](https://pepy.tech/project/factorlasso)
[![Monthly](https://static.pepy.tech/badge/factorlasso/month)](https://pepy.tech/project/factorlasso)

**Paper:** Sepp, A. and Kastenholz, M. (2026), *factorlasso: Sparse Multi-Output Regression with
Cluster-Grouped Sign Constraints in Python*, submitted to the *Journal of Statistical Software*.
**Methodology:** Sepp, A. and Kastenholz, M. A. (2026), *Gated Cluster-Pooled Sign Constraints for
Multi-Output Sparse Regression*, a revised working manuscript prepared for *Computational Statistics*. See
[Statement of need](#statement-of-need) for what each paper contributes, [Citation](#citation)
for the BibTeX entries and the [paper index](papers/README.md) for public availability.

**Documentation:** [factorlasso.readthedocs.io](https://factorlasso.readthedocs.io/en/latest/) ·
[getting started](docs/getting-started.md) · [task guides](docs/task-guides.md) ·
[choice guide](docs/comparison.md)

---

## Why factorlasso

Factor models are central to quantitative finance, underpinning the commercial risk systems in
industry use (among others, MSCI Barra, Axioma, and Bloomberg's multi-asset-class models). These
systems decompose asset returns into a few common factors and an idiosyncratic residual
(Rosenberg and McKibben 1973; Connor 1995). Practitioners who build factor-risk and
capital-market-assumption systems must estimate the loadings of *N* asset returns on *M*
tradable factors over a sample of length *T*, and they need software that produces loadings
stable enough to feed a portfolio optimiser. In financial applications the estimation regime is
adverse on four counts. First, the sample is short by statistical standards, because most core
cross-asset indices and funds launched in the late 2000s and 2010s, giving 5 to 25 years of
data. Second, the factors are often strongly correlated, so the design carries severe
multicollinearity. Third, the cross-section is moderate (*N* ∈ [50, 200] at the
multi-asset-class level). Fourth, the true coefficient matrix β ∈ ℝ^(N×M) exhibits a cluster
structure that the practitioner does not know a priori, and traditional provider-based asset
classifications are a weak proxy for the interdependence of assets. Assets in the same economic
group (regional equity, investment-grade credit, hedge funds, and so on) typically share signs on
the dominant factors, and an estimator that exploits this structure can outperform one that
treats each asset independently.

`factorlasso` fits the sparse multi-output linear model

$$
Y = X\beta^\top + \varepsilon,
\qquad \beta \in \mathbb{R}^{N \times M}
$$

with L1 and L2 penalties along with sign constraints on factor loadings. It is written in pure
numpy/pandas/scipy/cvxpy — no numba, no custom coordinate descent. The solver is CVXPY (default
`CLARABEL`), so problem formulation is explicit and auditable. It is useful when:

- some coefficients **must be zero, non-negative, or non-positive**, by asset, by factor, or both;
- you have a **prior** β₀ and want to penalise `‖β − β₀‖`, not `‖β‖`;
- you want **structured sparsity** with groups supplied by you or discovered by hierarchical
  clustering of the response correlation matrix, in either grouping geometry (HCGL over each
  asset's loadings, FCGL over a cluster's assets on each factor);
- you want **group-level selection with within-group elementwise sparsity** (sparse group LASSO);
- you want the **sign coherence within a group to be soft** (cooperative LASSO) or a
  **per-response univariate-guided** estimator with no grouping (UniLasso).

### Key differentiators

**1. Per-element sign constraints.** An `(N × M)` matrix `factors_beta_loading_signs` with
entries in `{0, 1, -1, NaN}` encodes equality-to-zero, non-negative, non-positive or free, so one
fit carries structural knowledge across responses. Scikit-learn's `Lasso` supports one
`positive` flag for the whole matrix; this is the custom CVXPY problem, packaged. See
[sign constraints and priors](docs/sign_constraints_and_priors.md).

**2. Data-driven sign constraints with a noise-floor gate.** `auto_sign_constraints=True`
derives signs inside `fit()` from pooled univariate slopes on the same EWMA-demeaned arrays the
solver consumes, pooling per `model_type` (per response for LASSO, per group or discovered
cluster for the grouped modes). `auto_sign_threshold_t` (default `0.75`, not a significance test)
pins factors below the noise floor to zero; explicit sign entries override the derived matrix
per cell; `auto_sign_adaptive_weights=True` adds Zou (2006) adaptive L1 weights from the same
slopes. See [gated cluster-pooled signs](docs/gated_cluster_pooled_signs.md) and
[adaptive penalty weights](docs/adaptive_penalty_weights.md).

**3. Prior-centered regularisation.** `factors_beta_prior` penalises `‖β − β₀‖` instead of
`‖β‖`; the prior is a soft target controlled by `reg_lambda`. `apply_ols_prior=True` builds an
empirical prior from weighted one-factor OLS slopes (`prior_selection_type="highest_r2"`, with
`factor_for_prior` overrides), re-estimated inside every training window. See
[prior targets](docs/prior_targets.md), [prior inference](docs/prior_inference.md) and the
[offline example](examples/ols_prior.py).

**4. Hierarchical Clustering Group LASSO (HCGL).** EWMA correlation of the responses → Ward's
linkage → a cut by `cutoff_fraction` or `n_clusters` → a row-grouped penalty weighted by cluster
size. `DistanceTransform` (`ONE_MINUS_RHO`, `CHORD`, `ARCCOS`), `DependenceMeasure` (`PEARSON`,
`SPEARMAN`, `GERBER`) and the opt-in `ClusterCorrelationTransform.REMOVE_PC1` diagnostic
control the partition; all measures are signed because the partition feeds sign pooling. See
[group penalties](docs/group_penalties_hcgl_fcgl.md), [cluster discovery](docs/cluster_discovery.md)
and [common-mode removal](docs/common_mode_removal.md).

**5. Sparse Group LASSO.** `l1_weight` = α ∈ [0, 1] adds an elementwise L1 term to the group L2
penalty (Simon, Friedman, Hastie and Tibshirani 2013): group-then-prune. `α = 0` reduces exactly
to pure group LASSO and drops the term from the problem. See
[group penalties](docs/group_penalties_hcgl_fcgl.md).

**6. Factor-Clustering Group LASSO (FCGL).** The complementary geometry: the L2 norm runs over a
cluster's assets on each factor, so a cluster-by-factor block enters or leaves together. Sign
derivation and adaptive weights are shared with HCGL; FCGL is one coupled cone programme and
matches HCGL in wall-clock to within a couple of percent at N = 500. Neither mode dominates in
general. See [group penalties](docs/group_penalties_hcgl_fcgl.md).

**7. UniLasso.** Per-response two-stage estimation with no grouping: univariate fits, then a
non-negative combination that keeps each univariate sign (`unilasso_loo`,
`unilasso_non_negative`), following Chatterjee, Hastie and Tibshirani (2025). See
[UniLasso](docs/unilasso.md).

**8. Cooperative LASSO.** Soft within-group sign coherence (Chiquet, Grandvalet and Charbonnier
2012): positive and negative parts are penalised as separate groups, on supplied groups
(`COOPERATIVE_GROUP_LASSO`) or discovered clusters (`COOPERATIVE_CLUSTER_GROUP_LASSO`). No hard
sign is derived or imposed. See [cooperative LASSO](docs/cooperative_lasso.md).

**9. Residual validation.** `diagnose_residuals` tests the strict-factor-structure assertion
`Σ = B Σ_F B' + D` directly, with the sphericity statistic against its chi-square null and the
largest residual eigenvalue against the Marchenko–Pastur edge; `missing_factor_components` names
the series that would define an omitted factor. `LassoModelDiagonalityCV` selects `reg_lambda`
on that criterion; `effective_sparsity` and `suggest_tolerance` count loadings at a
scale-aware tolerance. See [residual diagnostics](docs/residual_diagnostics.md) and
[penalty selection](docs/penalty_selection.md).

**10. Empirical residual correlation.** `CurrentFactorCovarData.get_y_covar` and
`RollingFactorCovarData.get_y_covars` assemble `B F B' + w D`; with
`residual_type="empirical"`, `D = S[(1 − ρ)I + ρR]S` adds a prepared common-period EWMA residual
correlation `R` while keeping the stored residual variances on the diagonal. See
[empirical residual correlation](docs/empirical_residual_correlation.md) and
[factor covariance assembly](docs/factor_covariance_assembly.md).

### When to use it — and when not

**Use it when:**

- Multi-output LASSO with heterogeneous sign constraints across the coefficient matrix.
- You have a prior `β₀` that should shrink the fit instead of zero.
- You need discovered-group structured sparsity (HCGL or FCGL), or group-level selection with
  within-group elementwise sparsity (sparse group LASSO at small-to-moderate α).
- You need to test whether the residual covariance is actually diagonal, or to select the
  penalty on that criterion rather than on prediction error.
- You want an auditable CVXPY formulation where heterogeneous constraints matter more than
  specialized-solver throughput.

**Reach for something else when:**

- You need non-linear models, random effects, or GLM link functions.
- You need the financial factor-model pipeline (factor definitions, NAVs, mixed-cadence
  covariance, premia): that is `factormodels`. Portfolio construction is `optimalportfolios`.

The maintained [choice guide](https://factorlasso.readthedocs.io/en/latest/comparison.html)
compares FactorLasso with scikit-learn, skglm, and groupyr by workflow fit, constraint and
grouping geometry, multi-output behaviour, missing-data handling, covariance assembly, solver
trade-offs, interoperability, dependencies, and license; it is dated, cites primary sources, and
includes cases favoring every project. The repository pointer is [`COMPARISON.md`](COMPARISON.md).

## Installation

```bash
pip install factorlasso
```

Upgrade using

```bash
pip install --upgrade factorlasso
```

Requires Python ≥ 3.10, CVXPY ≥ 1.5.2, SciPy ≥ 1.15.0, and numpy / pandas / openpyxl. Matplotlib
is not a runtime dependency; `plot_signs()` and `RiskClusterReport.to_figures()` import it only
when called.

**Upgrading to 1.0:** the old flat module paths (such as `factorlasso.lasso_estimator` and
`factorlasso.beta_priors`) have been removed. Use the package root or the capability
subpackages. All 74 root exports retain their signatures and defaults. See the
[migration guide and 1.x compatibility policy](COMPATIBILITY.md), including the limit on loading
pre-0.24 pickles.

## Table of contents

1. [Why factorlasso](#why-factorlasso)
2. [Installation](#installation)
3. [Statement of need](#statement-of-need)
4. [Overview](#overview)
5. [Five-minute quickstart](#five-minute-quickstart)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Changelog](#changelog)
9. [Feedback & contributing](#feedback-contributing)
10. [Citation](#citation)
11. [Ecosystem](#ecosystem)
12. [License](#license)
13. [Disclaimer](#disclaimer)

## Statement of need

Standard sparse-regression tools quietly misattribute risk in the regime financial factor models
live in: short samples, strongly correlated factors, and a group or cluster structure in the
loadings. `factorlasso` targets that regime. On a multi-asset benchmark it matches scikit-learn,
skglm, and asgl on generic accuracy while recovering a collinear credit factor that every one of
them shrinks to zero; the [credit-attribution case study](docs/app_multi_asset_credit_attribution.md)
and its [offline synthetic example](examples/docs/app_multi_asset_credit_attribution.py)
show how sign constraints, prior centres and grouped penalties affect factor attribution.
`factorlasso` is the estimation engine behind a multi-asset capital market assumptions and
allocation framework: the loadings it produces feed the ROSAA framework for robust strategic and
tactical asset allocation and the MATF-CMA framework for capital market assumptions.

| Paper | Implemented analytics | Where |
|---|---|---|
| Sepp, A. and Kastenholz, M. (2026), *factorlasso: Sparse Multi-Output Regression with Cluster-Grouped Sign Constraints in Python*, submitted to the Journal of Statistical Software | The estimator: `LassoModel` with its model types, sign constraints, priors, cross-validation, residual diagnostics and factor covariance assembly; sklearn interoperability | [sparse factor model](docs/sparse_factor_model.md), [software design](docs/software_design.md), [research availability](docs/scientific-replication.md) (manuscript and replication workspace retained locally) |
| Sepp, A. and Kastenholz, M. A. (2026), *Gated Cluster-Pooled Sign Constraints for Multi-Output Sparse Regression*, revised working manuscript prepared for Computational Statistics | Cluster-pooled sign screening (`auto_sign_constraints`, `auto_sign_threshold_t`), matched fitted-estimator comparisons and limitations; the revised experiments do not validate adaptive weights or prior targets | [gated cluster-pooled signs](docs/gated_cluster_pooled_signs.md), [`papers/sign_pooling_2026/`](papers/sign_pooling_2026/) (approved manuscript, figures, replication code and tests) |
| Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios*, The Journal of Portfolio Management 52(4), 86–120 | The loadings that feed the ROSAA risk model, consumed through `factormodels` and `optimalportfolios` | [portfolio risk models](docs/app_portfolio_risk_models.md) |
| Sepp, A., Hansen, E. and Kastenholz, M. (2026), *Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors*, [SSRN 6785958](https://ssrn.com/abstract=6785958) | The loadings and factor covariance behind the MATF capital market assumptions, consumed through `factormodels` | [factor covariance assembly](docs/factor_covariance_assembly.md) |

## Overview

The package root re-exports the public API; the capability subpackages are `linear_model`
(the `LassoModel` estimator, its dispatch over model types, nowcasting, the production
bootstrap and test inversion), `priors` (sign derivation, OLS priors, expert maps, bounds and
budget modes), `cluster` (dependence measures, hierarchical clustering, rolling smoothing and
stability statistics), `model_selection` (cross-validation and the diagonality selector),
`diagnostics` (residual diagnostics and cluster lineage), `covariance` (factor covariance
containers, residual correlation and alpha uncertainty), `inference` (weighted and HAC
inference, resampling and intervals) and `utils` (EWM, HAC and panel helpers).
The [software design](docs/software_design.md) page draws the imports and the
[API reference](https://factorlasso.readthedocs.io/en/latest/api.html) lists every public object.

### Methods

| `LassoModelType` | Penalty | Selected by | Article | Example |
|---|---|---|---|---|
| `LASSO` | `λ ‖β − β₀‖₁` with per-element sign constraints | default | [sign constraints and priors](docs/sign_constraints_and_priors.md) | [`sign_constraints_and_priors.py`](examples/docs/sign_constraints_and_priors.py) |
| `GROUP_LASSO` | group L2 over supplied `group_data` | `group_data` | [group penalties](docs/group_penalties_hcgl_fcgl.md) | [`group_penalties_hcgl_fcgl.py`](examples/docs/group_penalties_hcgl_fcgl.py) |
| `HIERARCHICAL_CLUSTER_GROUP_LASSO` | `(1−α) λ Σ_g w_g ‖β_g − β₀‖_{2,1} + α λ ‖β − β₀‖₁` over each asset's loading row, clusters discovered from the response correlation | `cutoff_fraction` or `n_clusters`, `span`, `l1_weight` | [group penalties](docs/group_penalties_hcgl_fcgl.md), [cluster discovery](docs/cluster_discovery.md) | [`cluster_discovery.py`](examples/docs/cluster_discovery.py) |
| `FACTOR_CLUSTER_GROUP_LASSO` | `(1−α) λ Σ_g w_g Σ_j ‖β_{g,j} − β⁰_{g,j}‖₂ + α λ ‖β − β⁰‖₁` over a cluster's assets on each factor | as HCGL | [group penalties](docs/group_penalties_hcgl_fcgl.md) | [`group_penalties_hcgl_fcgl.py`](examples/docs/group_penalties_hcgl_fcgl.py) |
| `UNILASSO` | stage-1 univariate fits, stage-2 non-negative combination | `unilasso_loo`, `unilasso_non_negative` | [UniLasso](docs/unilasso.md) | [`unilasso.py`](examples/docs/unilasso.py) |
| `COOPERATIVE_GROUP_LASSO`, `COOPERATIVE_CLUSTER_GROUP_LASSO` | positive and negative parts penalised as separate groups | `group_data` or `cutoff_fraction` | [cooperative LASSO](docs/cooperative_lasso.md) | [`cooperative_lasso.py`](examples/docs/cooperative_lasso.py) |

Data-driven signs (`auto_sign_constraints`) and the empirical prior (`apply_ols_prior`) compose
with every prior-aware type; UniLasso rejects the prior flag and the cooperative types never
gate. Observation weighting is EWMA with the effective `span` (uniform when `None`); see
[EWMA weighting and ragged histories](docs/ewma_weighting_and_ragged_histories.md).

### Analytics at a glance

| Area | What the package computes |
|---|---|
| Estimation | `LassoModel.fit(x, y)` → `coef_` (N × M), `alpha_const_`, `intercept_`, `predict`, `score`, `summary()`, `plot_signs()`; NumPy or pandas inputs; `__sklearn_tags__` for `Pipeline`, `GridSearchCV` and `cross_val_score` |
| Sign derivation | `derived_signs_` from gated, cluster-pooled univariate slopes; explicit overrides per cell; adaptive L1 and adaptive group weights |
| Priors | explicit `factors_beta_prior`; empirical OLS prior with `ols_betas_`, `ols_r2_`, `ols_beta_prior_`, `effective_beta_prior_`; expert-input maps described in [expert inputs](docs/expert_inputs.md) |
| Cluster discovery | `clusters_`, `linkage_`, `compute_clusters_from_corr_matrix`, `compute_rolling_smoothed_clusters` with an eligibility panel, `remove_first_principal_component` diagnostics |
| Cluster lineage and stability | `analyze_cluster_lineage` → persistent derived tracks and `RiskClusterReport` (offline diagnostic, not a signal); `compute_cluster_stability_statistics`, `score_with_stability_pooled_clusters`, `compute_co_association_panel` — point-in-time co-association weights; see [cluster lineage](docs/cluster_lineage.md) and [stability and pooled scoring](docs/cluster_stability_and_pooled_scoring.md) |
| Penalty selection | `LassoModelCV` (time-series expanding-window splits) and `LassoModelDiagonalityCV` (residual diagonality criterion) on shared splits and grid |
| Residual diagnostics | `diagnose_residuals`, `missing_factor_components`, `effective_sparsity`, `suggest_tolerance` |
| Factor covariance | `CurrentFactorCovarData`, `RollingFactorCovarData` → `B F B' + w D` with orthogonal or empirical residuals; `estimate_residual_correlation` → `ResidualCorrelationData`; ticker filtering and Excel save/load |
| Nowcasting | `LassoModel.nowcast(target_factors, alpha_span)` → `LassoNowcastResult` with factor component, statistical alpha, residuals and diagnostics; fails closed on look-ahead; see [residual-alpha nowcasting](docs/residual_alpha_nowcasting.md) |
| Production inference | [production-estimator bootstrap](docs/production_bootstrap.md): full FCGL refits and joint coefficient draws with signed portfolio weights (95% percentile ranges labelled as stability ranges); [production test inversion](docs/production_test_inversion.md): candidate exposures tested through complete null refits |

## Five-minute quickstart

```python
import numpy as np
import pandas as pd
from factorlasso import LassoModel, LassoModelType

rng = np.random.default_rng(0)
T, M, N = 200, 4, 10
dates = pd.date_range("2010-01-31", periods=T, freq="ME")
X = pd.DataFrame(
    rng.standard_normal((T, M)),
    index=dates,
    columns=[f"f{i}" for i in range(M)],
)
Y = pd.DataFrame(
    rng.standard_normal((T, N)),
    index=dates,
    columns=[f"y{i}" for i in range(N)],
)

model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5).fit(x=X, y=Y)

print(model.coef_.shape)         # (N, M) estimated β
print(model.alpha_const_.shape)  # (N,) estimated α, the regression intercept
print(model.predict(X).shape)    # fitted response panel
```

```text
(10, 4)
(10,)
(200, 10)
```

The API mirrors scikit-learn: `fit(x, y)`, `predict(x)`, `score(x, y)`, `get_params()`,
`set_params()`. Fitted attributes carry a trailing underscore. `fit`/`predict`/`score` accept
NumPy arrays as well as pandas objects, and the estimator declares `__sklearn_tags__`, so it
composes directly with `sklearn.pipeline.Pipeline`, `GridSearchCV`, and `cross_val_score`. The
[quickstart article](docs/quickstart.md) continues from return panels to a factor covariance
with [its script](examples/docs/quickstart.py).

## Examples

### Highlighted examples

- [Completed estimator](docs/completed_estimator.md) — what a finished `LassoModel` carries, its
  units, dates, failure states and table export:
  [`examples/docs/completed_estimator.py`](examples/docs/completed_estimator.py).
- [Credit-attribution case study](docs/app_multi_asset_credit_attribution.md) — sign constraints,
  prior centres and grouped penalties on a collinear credit factor:
  [`examples/docs/app_multi_asset_credit_attribution.py`](examples/docs/app_multi_asset_credit_attribution.py).
- [Residual-alpha nowcasting](docs/residual_alpha_nowcasting.md) — a fitted, de-meaned model
  extrapolates a missing response return once the realised factor return is known:
  [`examples/docs/residual_alpha_nowcasting.py`](examples/docs/residual_alpha_nowcasting.py).
- [Cluster lineage](docs/cluster_lineage.md) and
  [stability-pooled scoring](docs/cluster_stability_and_pooled_scoring.md) — persistent cluster
  tracks and point-in-time co-association weights from a rolling partition sequence:
  [`examples/docs/cluster_lineage.py`](examples/docs/cluster_lineage.py),
  [`examples/docs/cluster_stability_and_pooled_scoring.py`](examples/docs/cluster_stability_and_pooled_scoring.py).

### Runnable examples

Self-contained scripts at the top of [`examples/`](examples/):

- [`alpha_const_vs_intercept.py`](examples/alpha_const_vs_intercept.py) — the economic intercept
  and the CVXPY solver intercept conventions, explicitly.
- [`genomics_factor_model.py`](examples/genomics_factor_model.py) — QTL-style multi-response
  LASSO: genotype matrix → expression panel, with sign constraints derived from biological priors.
- [`finance_factor_model.py`](examples/finance_factor_model.py) — multi-asset factor decomposition
  with sign constraints and HCGL clustering.
- [`cv_lambda_selection.py`](examples/cv_lambda_selection.py) — time-series cross-validated
  `reg_lambda` selection via `LassoModelCV` with expanding-window splits.
- [`ols_prior.py`](examples/ols_prior.py) — weighted OLS prior checked against an independent
  least-squares calculation, with a prohibited-exposure (PE) exclusion.

Every documentation article has a canonical script of the same name under
[`examples/docs/`](examples/docs/); the [analytics gallery](docs/analytics_gallery.md) collects
their figures.

### From method to implementation

| Method | Implementation | Article · script |
|---|---|---|
| Gated cluster-pooled sign derivation and screening threshold | `auto_sign_constraints`, `auto_sign_threshold_t`, `derived_signs_` | [article](docs/gated_cluster_pooled_signs.md) · [`gated_cluster_pooled_signs.py`](examples/docs/gated_cluster_pooled_signs.py) |
| Adaptive L1 and adaptive group weights (Zou 2006; Wang and Leng 2008) | `auto_sign_adaptive_weights`, `auto_sign_adaptive_gamma`, `auto_sign_adaptive_floor` | [article](docs/adaptive_penalty_weights.md) · [`adaptive_penalty_weights.py`](examples/docs/adaptive_penalty_weights.py) |
| HCGL and FCGL group penalties, sparse-group mixing | `HIERARCHICAL_CLUSTER_GROUP_LASSO`, `FACTOR_CLUSTER_GROUP_LASSO`, `l1_weight` | [article](docs/group_penalties_hcgl_fcgl.md) · [`group_penalties_hcgl_fcgl.py`](examples/docs/group_penalties_hcgl_fcgl.py) |
| Distance transforms, dependence measures, dominant common-mode removal | `DistanceTransform`, `DependenceMeasure`, `ClusterCorrelationTransform` | [cluster discovery](docs/cluster_discovery.md), [common-mode removal](docs/common_mode_removal.md) · [`common_mode_removal.py`](examples/docs/common_mode_removal.py) |
| Strict-factor-structure test and diagonality-based penalty selection | `diagnose_residuals`, `LassoModelDiagonalityCV`, `effective_sparsity` | [residual diagnostics](docs/residual_diagnostics.md), [penalty selection](docs/penalty_selection.md) · [`residual_diagnostics.py`](examples/docs/residual_diagnostics.py) |
| Factor covariance with orthogonal or empirical residuals | `CurrentFactorCovarData`, `RollingFactorCovarData`, `estimate_residual_correlation` | [factor covariance assembly](docs/factor_covariance_assembly.md), [empirical residual correlation](docs/empirical_residual_correlation.md) · [`factor_covariance_assembly.py`](examples/docs/factor_covariance_assembly.py) |
| Production bootstrap and test inversion | FCGL refits with joint coefficient draws; null refits of candidate exposures | [production bootstrap](docs/production_bootstrap.md), [production test inversion](docs/production_test_inversion.md) · [`production_bootstrap.py`](examples/docs/production_bootstrap.py), [`production_test_inversion.py`](examples/docs/production_test_inversion.py) |

## Testing

```bash
uv sync --locked --group test
uv run --no-sync pytest
uv run --locked --only-group lint ruff check src/factorlasso tests
```

The suite covers estimator contracts, independent EWMA references, covariance assembly, cluster
diagnostics, and numerical parity with scikit-learn and skglm on their shared LASSO surface.
Matplotlib-dependent tests skip when it is not installed, so a core install stays green.

## Changelog

Release history is maintained in [CHANGELOG.md](CHANGELOG.md).

## Feedback & contributing <a name="feedback-contributing"></a>

- **Bug:** use the [bug-report form](https://github.com/ArturSepp/factorlasso/issues/new?template=bug_report.yml) with the version, Python/platform, a minimal reproducer, and expected versus actual output.
- **Feature:** use the [feature-request form](https://github.com/ArturSepp/factorlasso/issues/new?template=feature_request.yml) and describe the estimation goal, current workaround, and smallest useful API. In particular: which grouping, constraint, or covariance diagnostic is missing?
- **Question or methodology:** search or open an [issue](https://github.com/ArturSepp/factorlasso/issues) and identify the estimator, paper section, or convention involved.
- **Contribution:** follow [CONTRIBUTING.md](CONTRIBUTING.md) and look for [`good first issue`](https://github.com/ArturSepp/factorlasso/labels/good%20first%20issue) or [`help wanted`](https://github.com/ArturSepp/factorlasso/labels/help%20wanted) work.

## Citation

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

If you use `factorlasso` in academic work, please cite the software, the software paper
(submitted to the *Journal of Statistical Software*), the methodology paper for the
cluster-pooled sign derivation and screening gate (revised working manuscript prepared for
*Computational Statistics*), and the framework papers in which it was developed:

```bibtex
@software{factorlasso,
  author  = {Sepp, Artur and Kastenholz, Mika},
  title   = {factorlasso: Sparse Multi-Output Factor-Model Estimation in
             {Python}},
  year    = {2026},
  version = {1.0.6},
  url     = {https://github.com/ArturSepp/factorlasso},
}

@article{SeppKastenholz2026factorlasso,
  author  = {Sepp, Artur and Kastenholz, Mika},
  title   = {{factorlasso}: Sparse Multi-Output Regression with
             Cluster-Grouped Sign Constraints in {Python}},
  journal = {Journal of Statistical Software},
  year    = {2026},
  note    = {Submitted.}
}

@misc{SeppKastenholz2026sign,
  author  = {Sepp, Artur and Kastenholz, Mika A.},
  title   = {Gated Cluster-Pooled Sign Constraints for Multi-Output
             Sparse Regression},
  year    = {2026},
  note    = {Revised working manuscript; prepared for Computational Statistics.}
}

@article{SeppOssaKastenholz2026,
  author  = {Sepp, Artur and Ossa, Ivan and Kastenholz, Mika},
  title   = {Robust Optimization of Strategic and Tactical Asset Allocation
             for Multi-Asset Portfolios},
  journal = {The Journal of Portfolio Management},
  year    = {2026},
  volume  = {52},
  number  = {4},
  pages   = {86--120},
}

@misc{SeppHansenKastenholz2026MATF,
  author       = {Sepp, Artur and Hansen, Emilie and Kastenholz, Mika},
  title        = {Capital Market Assumptions and Strategic Asset Allocation Using
                  Multi-Asset Tradable Factors},
  year         = {2026},
  howpublished = {Working paper, SSRN 6785958},
  url          = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958}
}
```

## Ecosystem

This package is part of an open-source Python stack for quantitative finance. The
[ArturSepp profile](https://github.com/ArturSepp) is the canonical catalogue:

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, backtesting, and factsheet reporting |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and rolling backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) *(this package)* | Sparse factor-model estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data in pandas DataFrames |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain data and queries |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Numba-vectorised BSM and Bachelier pricing |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic-volatility pricing and calibration |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Closed-form trend-following analytics |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Multi-factor PME for private assets |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Goal-based allocation under regime-switching jump-diffusions |

`factorlasso` has no runtime dependency on another package in the stack. It is consumed by
`factormodels` (private) and, through its optional `factors`
extra, by `privateassets`.

## License

MIT — see [`LICENSE`](LICENSE) and the retained attributions in [`NOTICE`](NOTICE).

`1.0.4.post1` is the first verified MIT release after the GPL release series. It changes
licensing and release documentation; the 1.0.4 estimator API, numerical implementation and
defaults are unchanged. Earlier GPL releases and their grants remain available under their
original terms. This licence covers the package source and applicable documentation, examples
and synthetic fixtures. Dependencies, vendor data and SDKs, and manuscripts retain their own
terms; the package licence does not relicense them.

## Disclaimer

`factorlasso` is distributed FREE & WITHOUT ANY WARRANTY under the MIT License. See
[LICENSE](LICENSE) in the release for details. Use the routes in
[Feedback & contributing](#feedback-contributing) for bugs, feature requests and
methodology questions.
