---
myst:
  html_meta:
    description: >-
      Software design of factorlasso: the module layers from EWMA kernels to covariance
      containers, the data flow of a fit and of a rolling estimation, the scikit-learn boundary
      and the dependency surface.
---

# Software design

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

factorlasso is one estimator surrounded by the pieces it needs: numerical kernels below it, the
derivation of clusters, signs and priors beside it, selection and diagnostics around it, and
covariance containers after it. This page maps the modules to those roles, follows the data
through a fit, and states where the package ends.

## Module layers

| Subpackage | Internal modules | Responsibility |
|---|---|---|
| `factorlasso.utils` | `_ewm`, `_panel`, `_hac` | EWMA means and covariances, group loadings, and the preparation of the panels for the solvers: alignment, validity masks and de-meaning. |
| `factorlasso.inference` | `_wls`, `_geometry`, `_sandwich`, `_validation`, `_gaussian`, `_ar1` | Weighted regression statistics, coefficient and mean uncertainty, fixed linear geometry and Gaussian calibration. |
| `factorlasso.cluster` | `_dependence`, `_hierarchical`, `_smoothing`, `_stability`, `_standardization`, `_response` | Pearson, Spearman and Gerber dependence; shared response-panel preparation; distance, linkage and cut; causal rolling partitions; stability statistics and pooled scoring. |
| `factorlasso.priors` | `_signs`, `_ols`, `_bounds`, `_expert_map`, `_inference`, `_risk` | Derived sign constraints and adaptive penalty weights, OLS prior centres and expert bounds, the mapping of expert priors to factors, and prior inference and risk. |
| `factorlasso.linear_model` | `_estimator`, `_settings`, `_preparation`, `_restrictions`, `_dispatch`, `_state`, `_nowcast`, `_inspection`, `_types`, `_solvers` | `LassoModel`: validation, the penalty-independent preparation, the CVXPY programme of every mode, the fitted state and the nowcast. |
| `factorlasso.covariance` | `_factor_covar`, `_residual_correlation` | Dated factor-model snapshots, covariance assembly, and the prepared residual correlation. |
| `factorlasso.diagnostics` | `_residuals`, `_lineage` | Residual tests and effective sparsity, and the offline lineage of the risk clusters in a rolling covariance history. |
| `factorlasso.model_selection` | `_cv`, `_diagonality` | Expanding-window selection by held-out $R^2$ or residual diagonality. |

Every public name is exported from the package root and from the one subpackage that owns it;
the [API reference](api.rst) groups them by the article that documents them. Modules with a
leading underscore are internal. Version 1.0 removes the earlier flat modules such as
`factorlasso.lasso_estimator` and `factorlasso.cluster_utils`. Use the root or public
subpackage imports listed in the
[migration and compatibility policy](https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md).

The import graph runs one way. `utils` imports no other subpackage; `cluster` and
`inference` import `utils`; `priors` imports `utils` and `inference`; `covariance` may
import `utils`, `cluster` and `inference`; `linear_model` imports `utils`,
`cluster` and `priors`; `diagnostics` imports `utils` and `covariance`, because the lineage reads
covariance snapshots; and `model_selection` imports `utils`, `linear_model` and `diagnostics`.
The rolling clustering refers to `LassoModel` only for type checking, which keeps the graph
acyclic. A test of the package enforces these edges.

```mermaid
flowchart TB
    utils["utils: EWMA, group loadings, panel preparation"]
    cluster["cluster: dependence, partitions, smoothing, stability"]
    priors["priors: signs, prior centres, bounds, inference"]
    linear["linear_model: LassoModel and the solvers"]
    covariance["covariance: snapshots and assembly"]
    diagnostics["diagnostics: residual tests, offline lineage"]
    selection["model_selection: LassoModelCV, LassoModelDiagonalityCV"]
    utils --> cluster
    inference["inference: weighted statistics and Gaussian calibration"]
    utils --> inference
    inference --> priors
    utils --> priors
    cluster --> linear
    priors --> linear
    cluster --> covariance
    covariance --> diagnostics
    linear --> selection
    diagnostics --> selection
```

The shared utilities support clustering and inference. Prior applications use inference
while retaining floor and sign policy; alpha reporting retains its recursive weights
and reuses the existing calendar kernel. Existing prior-facing types keep their
module paths and delegate numerical work to the inference capability.

## A fit, step by step

`LassoModel.fit(x, y)` runs the same sequence for every mode; each step is skipped when the mode
or the settings do not need it.

1. **Prepare the panels.** Align dates, record the valid mask of every response, de-mean with
   uniform or EWMA weights, and zero-fill the masked cells for the solver
   ([EWMA weighting](ewma_weighting_and_ragged_histories.md)).
2. **Discover or accept clusters.** Build the dependence matrix of the responses, optionally
   remove the dominant mode, and cut the Ward tree; or take `group_data` or `external_clusters`
   ([cluster discovery](cluster_discovery.md)).
3. **Derive signs, centres and weights.** Pool univariate slopes over the clusters and gate them;
   compute OLS prior centres; resolve their precedence with explicit signs and priors; compute
   adaptive weights ([conventions](conventions.md)).
4. **Solve.** Build the CVXPY programme of the mode and solve it with the configured solver and
   fallbacks ([sparse factor model](sparse_factor_model.md)).
5. **Record.** Store the loadings, the intercepts, the fit diagnostics, the clusters and the
   solver-facing signs and centres, and the residual panel for nowcasts.

The fit restores every fitted attribute to its previous value if an exception escapes. This
includes diagnostics written during preparation. A non-raising solve with no solution still
warns and publishes its result. The [conventions](conventions.md) distinguish those cases
and explain labelled versus positional input alignment.

The penalty-independent part of steps 1 to 3 is shared across a grid by
`LassoModel.fit_reg_lambda_path`, which the selectors use for the group-LASSO family
([penalty selection](penalty_selection.md)). It returns independent fitted models and restores
the template's fitted state on both success and error.

## A rolling estimation

The package estimates one date at a time; the rolling loop belongs to the caller. A rolling
consumer computes causal partitions with `compute_rolling_smoothed_clusters`, fits a `LassoModel`
per date with `external_clusters`, and stores each date's loadings, factor covariance and residual
variances in a `CurrentFactorCovarData`, collected in a `RollingFactorCovarData`. OptimalPortfolios
does this in its
[factor covariance estimators](https://optimalportfolios.readthedocs.io/en/latest/covariance_estimators.html).
Stability statistics and pooled scoring read the partitions; the offline lineage reads the
rolling container and is the only component that looks at the whole panel at once
([offline cluster lineage](cluster_lineage.md)).

## The scikit-learn boundary

`LassoModel` follows the scikit-learn estimator protocol, `fit`, `predict`, `score`, `get_params`
and `set_params`, without inheriting from a scikit-learn class and without importing scikit-learn
at module level; a guarded hook imports it only when an installed scikit-learn asks for estimator
tags. The package's own selectors split time series by expanding windows, which generic shuffled
cross-validation does not ([scikit-learn interoperability](interoperability.md)).

## Dependency surface

The runtime dependencies are NumPy, pandas, SciPy, CVXPY and openpyxl. The default solver,
CLARABEL, is installed with CVXPY; other CVXPY solvers can be named in `solver` and
`solver_fallbacks`. factorlasso is a leaf of the author's package stack: it imports none of the
other packages, and OptimalPortfolios depends on it rather than the reverse. The linter bans
imports of the other stack packages outright and bans a module-level scikit-learn import.

## See also

- [Conventions and glossary](conventions.md): the notation, the loss and the precedence rules.
- [API reference](api.rst): every public name and the `LassoModel` parameter map.
- [Examples and recipes](task-guides.md): every runnable script.
- [Choosing a sparse-regression workflow](comparison.md).
