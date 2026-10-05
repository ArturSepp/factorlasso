---
myst:
  html_meta:
    description: >-
      Conditional uncertainty of recursive EWMA residual means with joint calendar HAC
      covariance, exact initialization weights and explicit inference assumptions.
---

# Conditional uncertainty of residual alpha

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Residual alpha is an exponentially weighted mean of fitted factor-model residuals.
Its estimation covariance describes uncertainty in that mean, rather than the covariance
of future portfolio returns. The routines here preserve the production EWMA recursion.

## Overview

The joint covariance combines exact recursive influence weights with a shared calendar
Bartlett kernel. It retains dependence across assets and dates without changing a
portfolio risk model's residual covariance. Inference is conditional on fitted loadings.

## Inputs, notation, and assumptions

The residual panel has ordered, unique rows and labelled asset columns. Missing values
retain their dates. Every asset has its own native-observation EWMA span. The caller
supplies strictly increasing calendar coordinates and a bandwidth in those units.
For monthly and quarterly series, month ordinals preserve the original timing.

Residual units are declared by their producer. Annualised residuals use scale one;
native-period residuals need an explicit mean annualisation multiplier. Standard errors
scale by that multiplier and covariances by the product of the two asset multipliers.

The mean/dependence interpretation assumes a stationary constant residual mean with
the fitted model held fixed. It does not assume that LASSO loadings are known in truth.

## Methodology

Apply the existing `compute_ewm` recursion to impulse vectors to recover exact weights
$q_{ti}$. This preserves initialization and FFILL behaviour without a second EWMA rule.
Write $m_i=\sum_t q_{ti}$ and $\widehat a_i=\sum_t q_{ti}e_{ti}$.
For leading missing data, initialization at zero can give $m_i<1$. The target is then
$m_i\mu_i$ under a constant residual mean $\mu_i$. The code does not normalize away
that finite-history initialization.

Centered innovations use $e_{ti}-\widehat a_i/m_i$. Form scores with weights $q_{ti}$,
the declared unit multiplier and the conditional mean correction $\sqrt{n_i/(n_i-1)}$.
Let S contain those scores, with zero contribution on missing rows. Then

$$
V_a=S^\top K S,\qquad K_{ts}=\max(0,1-|c_t-c_s|/b).
$$

This shares the Bartlett construction with the prior HAC routines. The score contains
the weight, so its quadratic product contains both weight factors. Effective sample
size is $m_i^2/\sum_t q_{ti}^2$ and is not substituted into the correction.

A common six-month bandwidth gives five monthly lags and one quarterly lag. Applying
unrelated bandwidths to separate matrix entries does not have the same PSD guarantee.
The implementation rejects insufficient overlapping history instead of declaring
unobserved alpha errors independent. Small overlap remains visible in the diagnostics.

For a fixed PSD metric M, the estimated quadratic is $\widehat a^\top M\widehat a$;
its noise contribution is $\operatorname{tr}(M V_a)$ under unbiasedness for the target.
The signed correction may be negative. Conservative Gaussian-region bounds apply the
triangle inequality with a displacement radius bounded using the largest eigenvalue
and rank of $V_a^{1/2} M V_a^{1/2}$. They permit zero under the null. Their known-covariance
Gaussian coverage does not become exact when an estimated HAC covariance is inserted.

The optional `calibrate_alpha_uncertainty` adapter retains the observed alpha weights and
constructs their exact `LinearHacGeometry`. Known-shape Gaussian and bounded AR1 calibration
then use the shared [inference layer](weighted_inference.md). For weights $q$, mass $m$,
unit multiplier $s$ and constant residual mean $\mu$, the linear target is $sm\mu$.
The adapter does not replace the recursive estimate by a normalized regression intercept.
It checks the saved residual fingerprint and reproduces the original estimate and standard
error before applying model-based calibration.

Known-shape covariance matrices describe the observation errors on each asset's observed
support. They must be declared rather than treated as known after estimation. AR1 calibration
requires regular native support; an internally gapped series receives an unsupported status
and unavailable bounds. A complete quarterly series uses a quarterly AR coefficient even
when its HAC kernel is expressed in months. These marginal results do not calibrate projected
alpha, a selected portfolio or a joint universe comparison automatically.

## Worked example

The [offline example](../examples/docs/alpha_uncertainty.py) uses synthetic residuals,
checks exact alpha replay and verifies the simulated joint covariance independently.

```python
result = estimate_alpha_uncertainty(residuals, 60, calendar=calendar, bandwidth=6.)
np.testing.assert_allclose(result.estimates, compute_ewm(residuals, span=60).iloc[-1])
assert result.diagnostics.loc['b', 'weight_mass'] < 1
```

## Implementation in factorlasso

- `estimate_alpha_uncertainty` returns `AlphaUncertainty` with estimates, covariance,
  exact weights, individual diagnostics, pairwise overlap and method metadata.
- `sample_gaussian_estimates` provides reproducible joint sensitivity scenarios.
- `calibrate_alpha_uncertainty` supplies explicit normal, known-shape Gaussian or bounded
  AR1 pointwise intervals, with inference status and calibration audit.
- `gaussian_quadratic_summary` reports observed/noise/corrected quadratics and
  conservative Gaussian-region bounds. Its legacy default remains spectral;
  `method='weighted_chi2'` delegates to the tighter generic quadratic calibration.

Run `python examples/docs/alpha_uncertainty.py` in the prescribed environment. No
optional dependencies or changes to `CurrentFactorCovarData.estimate_alpha` are needed.

## Interpretation and limitations

Default marginal intervals use a normal multiplier and are approximate. The finite-sample
coverage of these HAC intervals is not guaranteed, especially with strong dependence
or short effective histories. A fixed span does not acquire unlimited precision from
an expanding archive. The existing Gaussian/AR prior calibration has its own assumptions
and is applied only through an explicit calibration request on supported observations.
Neither a corrected multiplier nor a bootstrap interval establishes that the estimated
joint covariance or its quadratic trace noise contribution is unbiased. Assess covariance
bias and interval coverage separately. The residual-bootstrap interfaces in the generic
inference layer are experimental approximations and require application-specific validation.

Beta estimation, sparse selection, omitted factors, changing residual means and future
returns require additional analysis. Rank/sign frequencies from Gaussian scenarios are
sensitivity statistics, not posterior probabilities. A sample with zero estimated
variance is flagged; it does not establish that the unknown population variance is zero.

## See also

- [Classical and HAC standard errors](weighted_inference.md#standard-deviation-standard-error-and-confidence-intervals)
- [EWMA weighting and ragged histories](ewma_weighting_and_ragged_histories.md)
- [Prior inference](prior_inference.md)
- [Residual alpha nowcasting](residual_alpha_nowcasting.md)

## References

- Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica*, 55(3), 703–708. [DOI](https://doi.org/10.2307/1913610).
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
