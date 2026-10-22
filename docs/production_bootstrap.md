---
myst:
  html_meta:
    description: >-
      Joint dependent-wild bootstrap refits of production FCGL loadings, preserving
      ragged histories and testing the limits of nominal confidence coverage.
---

# Production factor-model bootstrap

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

A production-estimator bootstrap refits the same sparse estimator on perturbed
response histories. It measures sensitivity of the fitted loadings to sampling
noise while including data-dependent selection steps.

## Overview

The factor-cluster group LASSO (FCGL) estimator discovers response clusters, selects
automatic ordinary-least-squares prior centres, derives signs and penalty weights,
and solves a constrained, prior-centred convex problem. A confidence interval from
an independent weighted-least-squares fit describes a different estimator.
Full FCGL refits keep the production estimate as the reported point.

Full refitting is necessary to capture selection instability; it is not sufficient
to establish confidence coverage. In particular, shrinkage and thresholds make
ordinary bootstrap inference nonregular near zero and prior centres [2, 3].

## Inputs, notation, and assumptions

| Input | Meaning | Convention |
|---|---|---|
| X | Observed factor panel | Fixed design; caller supplies return or point-change units |
| Y | Asset responses | Original dates, frequency and missing mask retained |
| B | Production coefficients | Assets by factors |
| w | Current asset exposures | Signed; net-equity weights may sum above one |
| h | Bartlett bandwidth | Calendar distance at which multiplier correlation is zero |
| R | Bootstrap replications | Default 2,000; fixed seed |

Each call operates at one native cadence. Set the model's scalar `span`, penalty
and any cadence-specific clustering span or prior-HAC lag before calling. Missing
opening observations are retained. Partial factor rows, inadequate response
support and misaligned dates fail explicitly. Observed factors and histories are
conditioned on, rather than resampled. There is no fetching or fabrication of data.

## Methodology

First fit the original model and form its economic fitted mean and residuals:

$$
\widehat Y_t = \widehat\alpha + X_t\widehat B^{\top},\qquad
e_{ti}=Y_{ti}-\widehat Y_{ti}.
$$

Centre residuals using the same squared EWMA observation weights as the production
quadratic loss. Missing residual cells remain missing. Generate a common Gaussian
multiplier sequence for each replication with unit variance and covariance

$$
K_{ts}=\max(0,1-\lvert c_t-c_s\rvert/h),\qquad
Y^*_{ti}=\widehat Y_{ti}+\widetilde e_{ti}\xi_t.
$$

The same multiplier at a date applies to every asset. This retains observed
cross-asset residual dependence and permits time dependence through the Bartlett
kernel [1]. Residuals have no degrees-of-freedom rescaling; their finite-sample
shrinkage is one reason coverage must be checked.

Every draw starts from a deep copy of the original configuration, not the fitted
partition or selected prior. Clusters, automatic prior selection and centres,
automatic signs, adaptive weights and any data-dependent expert bounds are
recomputed. Explicit expert signs/targets, penalty settings and supplied external
group labels remain fixed. The current-fit bootstrap does not replay rolling
cluster smoothing, cross-validation or an upstream factor-selection procedure.

Aggregate joint asset draws before taking portfolio quantiles:

$$
b^*_f=\sum_i w_i B^*_{if}.
$$

`percentile` returns the 2.5th and 97.5th percentiles by default. These are central
95% **bootstrap stability ranges**, not automatically validated confidence intervals.
`basic` reflects those quantiles about twice the production estimate, giving a
candidate estimation-error interval. Neither construction repairs nonregular
selection automatically; basic endpoints may lie outside sign constraints.
Do not shift the production dot to the midpoint of either interval.

## Worked example

The offline example uses two synthetic factors and assets. One response has a
later start. Signed exposure aggregation allows a leveraged portfolio. Thirty
draws keep this teaching example quick; use the default or more for reporting.

```python
bootstrap = bootstrap_factor_betas(model, x, y, draws=30, seed=92, bandwidth=3.)
weights = pd.Series({'Asset A': .8, 'Asset B': .4})
table = bootstrap.portfolio_intervals(weights, confidence=.95)
```

The example independently aggregates the raw draws in pandas and checks their
quantiles against the returned portfolio table. It also verifies retained response
counts and that the supplied model remains unfitted and unmodified.

## Implementation in factorlasso

`bootstrap_factor_betas` returns `FactorBootstrapResult`, which contains original
coefficients, joint draws, response support and per-draw diagnostics. Its
`intervals` and `portfolio_intervals` methods report estimates, endpoints,
bootstrap bias and near-zero frequency. Coverage status starts as
`coverage_not_validated`; the package never silently certifies a confidence claim.

Run [the canonical example](../examples/docs/production_bootstrap.py):

```console
python examples/docs/production_bootstrap.py
```

The reproducible synthetic audit in
[benchmarks/bootstrap_coverage.py](../benchmarks/bootstrap_coverage.py) tests known
portfolio loadings under strong signals, weak/zero coefficients and a short,
ragged, correlated 13-factor design. It reports per-factor empirical coverage for
percentile and basic intervals, with binomial Wilson Monte Carlo bounds across
independent panels. It does not count correlated coefficients as independent
trials or change thresholds after seeing the results.

```console
python benchmarks/bootstrap_coverage.py --experiments 100 --draws 199 --workers 4 --output PATH
```

Choose a new output directory outside the checkout. More simulations improve the
precision of coverage measurement; they cannot establish universal validity.
Report both the simulation design and failed coverage cells. Report-range labels
must stay “stability” unless a defensible inference procedure is validated for the
intended target and setting.

## Interpretation and limitations

- Conditional inference excludes uncertainty in factor construction, synthetic
  product assumptions, chosen factor universe, current weights and future regimes.
- Full-model data-dependent steps are recomputed, but a fixed penalty does not
  include uncertainty from selecting that penalty elsewhere.
- Short and collinear histories can yield poor coverage or unstable decomposition.
  A coefficient constrained to zero by specification has no statistical interval;
  an automatically gated zero still has selection uncertainty.
- Percentile intervals can exclude the original estimate and need not be symmetric.
  Zero-width sampled intervals do not prove a population loading is known exactly.
- Bounds are pointwise, not simultaneous, posterior credible or future-P&L bands.
- For mixed cadences, do not independently bootstrap sleeves and then combine
  percentiles. A caller may supply coordinated date-labelled multipliers, but owns
  their cross-cadence construction and validation.
- Any solver failure aborts the run. Dropping difficult draws would distort the
  distribution. Inspect convergence across increasing draw counts and sensitivity
  to the dependence bandwidth before interpreting narrow endpoints.

## See also

- [Group penalties and FCGL](group_penalties_hcgl_fcgl.md)
- [Weighted regression inference](weighted_inference.md)
- [EWMA weighting and ragged histories](ewma_weighting_and_ragged_histories.md)

## References

1. Shao, X. (2010). The Dependent Wild Bootstrap. *Journal of the American Statistical
   Association*, 105, 218–235. [DOI](https://doi.org/10.1198/jasa.2009.tm08744).
   The smooth-statistic theory does not itself validate FCGL selection inference.
2. Chatterjee, A., and Lahiri, S. N. (2011). Bootstrapping Lasso Estimators.
   *Journal of the American Statistical Association*, 106, 608–625.
   [DOI](https://doi.org/10.1198/jasa.2011.tm10159).
3. Zhou, Q., and Min, S. (2017). Uncertainty Quantification Under Group Sparsity.
   *Biometrika*, 104, 613–632. [Paper](https://arxiv.org/abs/1507.01296).
   Its modified parametric bootstrap assumptions do not automatically extend to
   this data-clustered, constrained, prior-centred, dependent and ragged pipeline.
