---
myst:
  html_meta:
    description: >-
      The completed factorlasso estimator: inputs, fitted fields, units and dates,
      numerical validity, mutable state, explicit table export and downstream use.
---

# A completed estimator

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

The standalone output of factorlasso is a fitted `LassoModel`: labelled loadings,
intercepts and diagnostics from supplied factor and response panels. It is complete
when `fit` returns and the caller has checked the required responses for usable fitted
values. A portfolio, a factor-definition catalogue, an expected-return forecast or a
rendered report is not required to finish this estimation task.

## Inputs and completion

Supply a factor panel `x` with dates by factors, a response panel `y` with dates by
assets, and explicit estimator settings. The two labelled panels must have matching
date indices; see [conventions](conventions.md) for positional inputs, missing histories,
penalties, signs and priors. The caller selects the training cutoff before calling
`fit`. The estimator does not fetch data or infer point-in-time availability.

The [canonical offline script](https://github.com/ArturSepp/factorlasso/blob/main/examples/docs/completed_estimator.py)
uses 72 monthly synthetic observations, two factors and three assets. All inputs are
simple returns in decimal units. It selects uniform observation weights, de-meaning,
the CLARABEL solver and a zero LASSO penalty, so an augmented NumPy least-squares
system supplies an independent numerical reference. The zero penalty is a transparent
reference case, not a recommended penalty for a production universe.

```python
    model = LassoModel(
        model_type=LassoModelType.LASSO, reg_lambda=0.0,
        span=None, demean=True, solver="CLARABEL",
    )
    assert model.fit(x=x, y=y) is model
    result = model.estimation_result_
    assert isinstance(result, LassoEstimationResult)
```

Save the linked script and run `python completed_estimator.py` after installing
`factorlasso>=1.0.4`. It prints the fitted summary, cutoff, labelled loadings and
diagnostics; it needs no sibling package, checkout import, network or data file and
writes no files. It checks coefficients and intercepts against independent least
squares, predictions against that reference design, and residual variance and
R-squared against direct residual calculations.

## Inspectable fields, dates and units

| Object or field | Meaning and contract |
|---|---|
| `LassoModel` returned by `fit` | The same mutable estimator instance, now containing fitted state. |
| `coef_` | Asset-by-factor DataFrame in the supplied label order. A loading has response units per factor unit; decimal return on decimal return is dimensionless. |
| `alpha_const_` | Asset-labelled intercept in the original response units per input observation. This is the intercept used by `predict` for this de-meaned fit. |
| `intercept_` | Raw residual-mean diagnostic on the solver's de-meaned data, distinct from `alpha_const_`. |
| `estimation_result_` | `LassoEstimationResult` containing arrays `estimated_beta`, `alpha`, `ss_total`, `ss_res`, `r2`, in fitted asset/factor order. `ss_total` and `ss_res` use squared response units; `r2` is dimensionless. |
| `x_`, `y_`, `valid_mask_` | Fitted input panels and response validity mask. The example's copied caller metadata records training start, cutoff and availability explicitly. |
| `effective_span_` | Actual beta-estimation span; `None` means uniform weights in this example. |
| `clusters_`, `linkage_`, `derived_signs_`, prior diagnostics | Additional fitted information when the chosen mode/settings produce it; absence is legitimate when a feature is unused. |

The estimator does not attach a currency, a return convention, an annualisation factor
or an availability timestamp to `LassoModel`. Retain those as caller metadata alongside
the fitted result. Dates in `x_`/`y_` record supplied observations, not a provider's
release dates. The example records monthly decimal returns and no annualisation;
its residual variances are monthly squared returns. Covariance assembly, when requested
separately, also uses the units supplied by its caller.

## Numerical validity and failure states

There is no public `fit_status_` or solver-status enum on `LassoModel`. Successful
return alone does not certify every response. Check finite coefficients, intercepts
and the diagnostics relevant to the requested output, together with valid-observation
counts and the chosen warmup policy. Inspect warnings. A non-raising failed solve can
publish NaN coefficients; a warmup-excluded response can have zero loadings and NaN
diagnostics. This page's complete, finite panel avoids either case and checks all
responses explicitly. A zero coefficient on its own does not distinguish exclusion
from estimated sparsity.

If an exception escapes `fit`, its previous fitted state is restored. Parameters
already changed through `set_params` retain their new values. Consequently a failed
refit must not be reported as a newly completed fit merely because an older `coef_`
still exists. See the [fitted-state compatibility contract](https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md#fitted-state-and-input-alignment).

## Mutability and explicit export

Fitting replaces mutable attributes on the estimator. Its `x_` and `y_` panels can
retain aliases to caller inputs, and its numerical containers are not immutable
snapshots. Copy the fields needed by a downstream consumer before editing them or
reusing the estimator. The canonical script copies loadings, intercepts and diagnostics,
exports the loading table to an in-memory CSV string, reads it back with pandas and
checks that labels and values survive.

```python
    loadings_csv = tables["loadings"].to_csv()
    restored = pd.read_csv(StringIO(loadings_csv), index_col=0)
    pd.testing.assert_frame_equal(restored, tables["loadings"], atol=1e-14, rtol=1e-14)
```

Writing that string or a pandas table to a caller-selected file is an explicit later
action. A numerical-table export is not a full estimator checkpoint: retain settings,
input identity, dates, units and software/dependency versions if a fit must be audited
or reproduced. The package has no versioned estimator bundle format. Arbitrary
cross-version pickle compatibility is not guaranteed; follow the
[persistence policy](https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md#persistence).

## Downstream use is a separate step

The completed estimator can predict responses, support residual diagnostics or
provide loadings for the existing [covariance containers](factor_covariance_assembly.md).
These are optional operations on an already completed fit.

[factormodels](https://github.com/ArturSepp/tradablefactors) owns the financial
pipeline: factor definitions and histories, point-in-time estimation orchestration
using factorlasso, fitted factor-model covariance and model-specific inference.
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios) consumes supplied
alphas and covariance for optimisation. [QIS](https://github.com/ArturSepp/QuantInvestStrats)
provides general risk analytics and reporting. No sibling package is needed to run
this example or inspect the estimator, and factorlasso imports none of them.

## See also

- [Sparse factor model](sparse_factor_model.md): objective, solver and diagnostics.
- [Software design](software_design.md): estimation layers and rolling orchestration.
- [scikit-learn interoperability](interoperability.md): estimator protocol and selectors.
- [API reference](api.rst): public names and configuration ownership.
