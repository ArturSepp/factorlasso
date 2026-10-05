# factorlasso — API Compatibility Policy

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Software: [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](CITATION.cff).

Version 1.0.0 establishes the capability subpackages as the supported layout
and retires the pre-1.0 compatibility facades. It preserves all 74 root exports,
their signatures and defaults, and the estimation behaviour of 0.25.0.
factorlasso follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version 1.0.2 adds `factorlasso.inference` for general weighted regression
and mean inference. Existing prior-facing entry points retain their defaults, result
fields and subpackage ownership. `PriorHacGeometry` and `Ar1PriorInterval` retain their
original class module paths; generic types have separate names and ownership.

## Stable surface in 1.x

`factorlasso.__all__` is the authoritative public API. Every name also appears in
exactly one capability subpackage's `__all__`, resolving to the same object.
Both import forms are supported throughout 1.x:

```python
from factorlasso import LassoModel, LassoModelCV
from factorlasso.linear_model import LassoModel
from factorlasso.model_selection import LassoModelCV
```

The [API reference](https://factorlasso.readthedocs.io/en/latest/api.html) documents
the public names, parameters and return contracts. The reviewed fixture
[`tests/data/api_contract.json`](tests/data/api_contract.json) pins root and
subpackage exports, callable signatures and defaults, enum members, dataclass
fields, public methods and class module paths. Compatible additions update that
fixture explicitly. Undocumented names visible through `dir(factorlasso)` are
not additional public entry points.

Within 1.x, existing public names, parameter signatures, defaults, documented
return contracts and fitted attributes remain compatible. `LassoModel` stores
constructor parameters unmodified, supports `get_params` and `set_params`,
returns `self` from `fit`, and names fitted attributes with a trailing underscore.
Compatible features may be added in minor releases. Bug fixes that correct
numerical results are identified in the changelog with their affected path.

## Upgrade from the flat layout

The 19 modules below were compatibility facades in 0.24 and 0.25. They and
`factorlasso._compat` are removed in 1.0.0, including their root module aliases
and `patch_points` helper. This is an explicit breaking change approved for the
1.0 transition; the former promise of continued facade availability was not
satisfied by a warning cycle. Install 0.25.0 if those paths are still required,
or migrate before upgrading.

Use the package root for any name in `factorlasso.__all__`, or its public home:

| Removed module (`factorlasso.` prefix) | Public home (`factorlasso.` prefix) |
|---|---|
| `beta_priors` | `priors` |
| `cluster_lineage` | `diagnostics` |
| `cluster_smoothing` | `cluster` |
| `cluster_standardization` | `cluster` |
| `cluster_statistics` | `cluster` |
| `cluster_utils` | `cluster` |
| `cv` | `model_selection` |
| `dependence_utils` | `cluster` |
| `diagonality` | `model_selection` |
| `ewm_utils` | `utils` |
| `expert_prior_map` | `priors` |
| `factor_covar` | `covariance` |
| `lasso_estimator` | `linear_model` |
| `prior_bounds` | `priors` |
| `prior_inference` | `priors` |
| `prior_risk` | `priors` |
| `residual_covar` | `covariance` |
| `residual_diagnostics` | `diagnostics` |
| `sign_constraints` | `priors` |

This table maps capabilities, not every incidental import in an old module.
For example, `get_x_y_np` belongs to `utils`, while `LassoModelCV` belongs to
`model_selection`. The [software design](docs/software_design.md) lists the
implementation owners. Private helpers and constants formerly re-exported by
facades remain private. Research instrumentation must patch a helper where
the implementation looks it up; an imported name may have several such
namespaces. There is no supported public patch-point API.

## Persistence

Pickles created with 0.23 or earlier can contain the removed flat module paths
and cannot be loaded directly in 1.0. Keep their original environment to load
them and export labelled numerical data or refit in 1.0. Do not rewrite module
strings in pickle bytes. Existing 0.24/0.25 class module paths remain unchanged
in 1.0, but arbitrary cross-version pickle compatibility is not guaranteed:
loading also requires compatible fitted fields and dependency versions.

## Fitted state and input alignment

`LassoModel.fit` replaces fitted attributes only when it completes: if an
exception escapes, every fitted attribute keeps its previous value, while
parameters set with `set_params` stay as set. A solve that fails without raising
warns and stores NaN coefficients. `fit_reg_lambda_path` leaves the fitted
attributes of its model unchanged. `x` and `y` must carry the same index labels;
an input with the default index `0..n-1` (including NumPy arrays) adopts the
other input's labels.

`LassoModel.nowcast(x, *, alpha_span=None)` is unit-preserving. It requires a fit
that recorded `demean=True`, exact finite target factor columns, and strictly
future sorted unique dates. Statistical alpha is the terminal adjust-false EWMA
of the fit-time original-unit residual snapshot, with leading missing values
removed and interior missing values holding the prior state. `alpha_span=None`
reuses `effective_span_`; a uniform fit uses the simple residual mean. Prediction
is `x @ coef_.T + stat_alpha` and includes neither `alpha_const_` nor `intercept_`.
Diagnostics copy `alpha_const_` and the nominal-span de-meaned solver `ss_total`,
`ss_res` and `r2` without clipping, and include Kish effective sample size from
normalized squared solver row weights.

## Internal surface

Leading-underscore modules and helpers, names outside the public `__all__`
lists, and the internal structure of CVXPY problems are not stable APIs.
Package code imports the private owner directly and follows the enforced
subpackage layering. Downstream callers should use the public root or
capability subpackage whenever possible.

## Future deprecations

Breaking changes to the stable 1.x surface require a new major version:

1. A minor release adds a `DeprecationWarning` identifying the replacement and
   earliest removal version, with a changelog entry.
2. The old surface remains available for at least one subsequent minor release.
3. Removal occurs in a major release, with explicit migration instructions and
   a `Removed` changelog entry.

The 1.0 removal above is the documented pre-1.0 transition, not a precedent for
removing supported 1.x imports without this process.

## Numerical reproducibility

The 1.0 transition changes import paths, not solver formulations, seeds or
estimator defaults. Reproducibility requires the full environment: Python,
NumPy/SciPy, CVXPY, the solver and numerical libraries, as well as data and
parameters. No bit-identical guarantee is made across different environments.
Supported solvers and dependencies are not pinned by the library; numerical
results can differ at the last few decimal places across their versions.
Numerical bug fixes and intended changes are recorded in the changelog and
validated against appropriate reference results.

Report uncertain public contracts in the
[issue tracker](https://github.com/ArturSepp/factorlasso/issues). Software
stability and scientific-paper review or publication status are tracked separately.
