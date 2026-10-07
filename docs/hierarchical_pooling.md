---
myst:
  html_meta:
    description: >-
      Partial pooling of correlated mean estimates, with uncertain group centres
      and dispersions, explicit priors, joint posterior draws and numerical checks.
---

# Hierarchical pooling of correlated means

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Partial pooling uses a shared distribution to stabilize related mean estimates while
retaining individual differences. This implementation accepts a full measurement-error
covariance and integrates uncertainty in both group centres and group dispersions.

## Overview

An observed cross-sectional spread combines latent heterogeneity and estimation noise.
A hierarchical model separates those components under explicit distributional assumptions.
The population dispersion parameter and the spread of posterior point estimates answer
different questions: shrinkage compresses the latter. This distinction follows the
discussion of pooling in [Gelman and Pardoe (2006)](#references).

The likelihood may be supplied by [EWMA alpha uncertainty](alpha_uncertainty.md), or by
another estimator with a declared joint error covariance. The generic method does not
fit returns, alter EWMA spans, select factors or annualize its inputs.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
| --- | --- | --- |
| $y$ | Vector of estimated means | Caller-declared units, fixed ordered asset labels |
| $V$ | Full measurement-error covariance | Squared mean units, PSD, same ordered axes |
| $Z$ | Group-membership indicator matrix | One group per asset; first-appearance group order |
| $\mu_g$ | Latent group centre | Mean units |
| $\tau_g$ | Latent between-asset standard deviation | Mean units, strictly positive continuous prior |
| $m_0,s_\mu,s_\tau$ | Prior locations and scales | Explicit caller choices, in mean units |

The Gaussian measurement-error model is conditional on the supplied covariance. A fitted
HAC covariance is a plug-in input whose own uncertainty is not inferred here. Conditional
independence of latent random effects is a separate assumption from measurement-error
independence; the latter is not imposed.

## Methodology

The hierarchical model is

$$
y\mid a\sim N(a,V),\qquad
a\mid\mu,\tau\sim N(Z\mu,T),\qquad T_{ii}=\tau_{g(i)}^2.
$$

Independent group priors are $\mu_g\sim N(m_{0g},s_{\mu g}^2)$ and
$\tau_g\sim\operatorname{HalfNormal}(s_{\tau g})$. Let $P=\operatorname{diag}(s_\mu^2)$.
Integrating the group centres gives the marginal likelihood

$$
y\mid\tau\sim N(Zm_0,V+T+ZPZ^\top).
$$

Tensor Gauss-Legendre quadrature integrates group dispersions. By default a coarse
pilot concentrates the numerical nodes using a mixture with 20 percent half-normal prior
mass and 80 percent fitted lognormal mass. The prior/proposal density ratio is included
explicitly. This changes the numerical integration measure, not the statistical prior.
The proposal retains full positive support. The nonadaptive option uses the prior CDF.

Conditional simulation draws group centres, latent means and measurement noise from the
prior, then conditions those joint Gaussian draws on the observed vector. Sampling the
quadrature nodes with their posterior weights propagates dispersion uncertainty as well.
The returned covariance therefore includes uncertainty in both group-level parameters.

For a fixed quadratic functional $a^\top M a$, posterior mean $m$ and covariance $C$ obey

$$
E[a^\top M a\mid y]=m^\top M m+\operatorname{tr}(MC).
$$

Using only the functional at the posterior mean omits the uncertainty term. Nonlinear
functionals can instead be evaluated on the coupled posterior draws in their application.

## Worked example

The [canonical synthetic example](../examples/docs/hierarchical_pooling.py) uses four
annual mean estimates and correlated errors. It compares the posterior dispersion mean
and evidence with an independent adaptive integral of a multivariate normal density.
It also checks that zero measurement error fixes every latent mean at its observation.

```python
    posterior = pool_gaussian_means(
        means, covariance, groups, mean_prior_scale=.05,
        dispersion_prior_scale=.05, quadrature_order=128, draws=4096, seed=37)
```

The example's 0.05 prior scales correspond to five annual percentage points when the
inputs are decimal annual returns. They are illustrative choices, not package defaults
or a universal calibration for financial means.

## Implementation in factorlasso

`pool_gaussian_means` returns `HierarchicalMeanPosterior`. Both are exported from the
package root and `factorlasso.inference`. Required prior scales make modelling choices
explicit. Inputs with missing means, mismatched labels, invalid covariances or invalid
prior scales fail rather than being aligned, repaired or dropped silently.

The result includes individual posterior summaries, the joint covariance, coupled latent
mean/group-centre/group-scale draws, group summaries, quadrature nodes and diagnostics.
Credible endpoints are equal-tail posterior quantiles. Group-dispersion endpoints interpolate
marginal quadrature CDF midpoints; other endpoints use conditional simulation. Repeat at
higher quadrature order
and with independent draws to check integration and Monte Carlo precision separately.
The effective grid-node count measures weight concentration; it is not an MCMC ESS.

Run `examples/docs/hierarchical_pooling.py` with the prescribed external interpreter.
Only the existing NumPy, pandas and SciPy runtime dependencies are used. No factor-model
or production alpha default changes when importing or calling the pooling method.

## Interpretation and limitations

These intervals are posterior credible intervals under the stated hierarchy. They are
distinct from the [analytical confidence bounds](weighted_inference.md). Neither estimated
measurement covariance nor factor/model-selection uncertainty is automatically included.

Small groups may be prior-sensitive; compare plausible centre and dispersion scales.
The continuous half-normal prior has no atom at zero, so a positive lower dispersion
endpoint does not establish a test against identical latent means. A diffuse prior is
not assumption-free, and normal random effects may be unsuitable for multimodal groups.
Pooling categories should reflect meaningful similarity, not a desired numerical result.

Tensor cost grows exponentially with group count; `max_nodes` guards resources. The
method is intended for a small number of groups. Quadrature and finite draws are numerical
approximations, with no automatic convergence certificate. A coarse grid can severely
understate interval width; inspect refinement results before using the reported quantiles.
Near-singular plug-in covariance treats some contrasts as very precise and can strongly
affect inference. Correlated errors can move group centres away from raw class averages.

## See also

- [Residual alpha uncertainty](alpha_uncertainty.md)
- [Weighted regression and mean inference](weighted_inference.md)
- [Cluster stability and pooled scoring](cluster_stability_and_pooled_scoring.md)

## References

- Gelman, A. and Pardoe, I. (2006). Bayesian Measures of Explained Variance and Pooling
  in Multilevel (Hierarchical) Models. *Technometrics*, 48(2), 241–251.
  [DOI](https://doi.org/10.1198/004017005000000517),
  [author manuscript](https://sites.stat.columbia.edu/gelman/research/published/rsquared.pdf).
  The correlated-input likelihood, half-normal prior and integration algorithm here are
  implementation choices; no empirical coverage guarantee is inherited from that paper.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
