---
myst:
  html_meta:
    description: >-
      Optional Gaussian calibration of weighted OLS and HAC prior uncertainty,
      continuous AR parameter coverage, and conditional two-factor radius diagnostics.
---

# Prior uncertainty and conditional floor risk

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

A prior interval describes uncertainty in a named regression coefficient. A prior floor
restricts a fitted coefficient. These serve different purposes: a coverage-calibrated
interval need not minimize coefficient risk. The optional functions here expose that
distinction without changing any estimator defaults or automatically imposing bounds.

## Overview

Weighted ordinary least squares (OLS) and a heteroskedasticity and autocorrelation
consistent (HAC) variance can be written as a linear and a quadratic form. Their ratio
admits finite-Gaussian calibration when the covariance shape is known. For unknown
stationary AR(1) dependence, a conservative full-domain calculation or an adaptive
residual-direction confidence set accounts for nuisance-parameter uncertainty.

A separate two-factor diagnostic evaluates the limiting coefficient risk of a
zero-centred nonnegative Lasso with a named-factor floor. Its conditional optimal
radius is an analytical tuning reference, not a general FCGL setting.

## Inputs, notation, and assumptions

| Input | Meaning | Convention |
|---|---|---|
| `design` | Fixed full-rank matrix D | T rows, P columns, with intercept if intended |
| `weights` | Nonnegative observation weights | Same loss weights as the prior regression |
| `responses` | Complete response vectors | One row per response in a batch |
| `hac_lags` | Bartlett bandwidth | Original regular observation grid |
| `covariance_shape` | Known Gaussian covariance up to scale | Symmetric positive definite |
| `phi_max` | Bound on stationary AR dependence | Strictly between zero and one |
| `alpha`, `delta` | Interval and nuisance-set error budgets | Delta is used only in adaptive mode |

The inference model is $y=D\gamma+\sigma\varepsilon$, with fixed D and Gaussian errors.
The mean must belong to the declared design space. Factor selection and the design
must be fixed independently of response noise. Coverage is pointwise for one named
coefficient, not simultaneous across factors or valid after highest-R-squared selection.

The example uses dimensionless synthetic observations; no return annualisation is
performed. If rows represent returns, use one explicitly chosen frequency and return
convention throughout. An EWMA span is a decay parameter, not a hard lookback window.
Missing responses are rejected; dropping gaps before an AR call would change its time
model. No warmup is added implicitly: the caller supplies the fitting window and needs
more positive-weight observations than design columns. Rolling calls must use past data.

## Methodology

### The actual weighted HAC statistic

Let $H=(D^{\top}WD)^{-1}D^{\top}W$, let $h^{\top}$ be its selected coefficient row,
and let $R=I-DH$. With Bartlett kernel K and the existing finite-sample correction,

$$
\widehat b=h^{\top}y,\qquad
\widehat v=y^{\top}Qy,\qquad
Q=\frac{T_{obs}}{T_{obs}-P}R^{\top}\operatorname{diag}(h)K\operatorname{diag}(h)R.
$$

Both weight factors are present in Q. Effective sample size does not replace the
observed-row count in this correction. A weighted SVD constructs H; the library checks
rank and computes the variance after removing the fitted mean for numerical stability.

For known covariance shape $V=CC^{\top}$, the unknown-scale pivot is
$\lvert a^{\top}\xi\rvert/\sqrt{\xi^{\top}B\xi}$, where $a=C^{\top}h$,
$B=C^{\top}QC$ and $\xi$ is standard Gaussian. Numerator and denominator are generally
dependent. A one-positive-eigenvalue Gaussian quadratic-form integral calibrates its
tail; no estimated Student degrees of freedom are substituted. Gaussian quadratic-form
probabilities are classical (Imhof, 1961); this implementation uses a specialized
nonoscillatory integral rather than the general Imhof inversion algorithm.

### Unknown AR dependence

With covariance entries $V_{ij}=\phi^{\lvert i-j\rvert}$ and
$\lvert\phi\rvert\le\phi_{max}$, divide the atanh parameter into equal cells of
half-width h. Throughout a cell the covariance is bounded above and below by its
centre covariance multiplied by $\exp(2h)$ and $\exp(-2h)$. The Gaussian directional
density ratio is then at most $\exp(2Th)$.

Full-domain mode calibrates each cell centre at tail $\alpha\exp(-2Th)$ and takes
the largest critical value. It does not spend an error budget on an unused shape set.
Adaptive mode removes the unknown mean and scale with residual contrasts. Their angular
central Gaussian densities (Tyler, 1987) define a fixed-mixture likelihood-ratio confidence
set with error at most delta. Retaining whole cells that could meet the density threshold
covers continuous off-grid values. Calibrating the retained cells at
$(\alpha-\delta)\exp(-2Th)$ and maximizing their critical values gives the desired
union-bound guarantee, following the nuisance-confidence-set principle of Berger and
Boos (1994). The mixture is fixed before observing the response; a fitted likelihood
maximum cannot replace it in this expectation argument.

The adaptive mode is not uniformly narrower. Selecting the narrower realized interval
from the two modes has no guarantee from these calculations.

### Conditional floor risk and information

The diagnostic objective is half the standardized quadratic loss plus
$\lambda(b_1+b_2)$, with nonnegative coefficients. Truth is $(\theta,\eta)$,
$0\le\eta\le d$, and a fixed prior offset satisfies $\lvert e\rvert\le\tau$.
The floor is $(z_1-\rho d+e-r)_+$. First take $\rho$ to one, then let the primary
signal-to-noise ratio grow. A joint limit also requires
$\theta\sqrt{1-\rho}/s_B\to0$, where $s_B$ is the orthogonal score noise scale.

For independent marginal and orthogonal Gaussian score components, the limiting
worst risk balances fixed positive and negative bias endpoints. When $d=0$, the
optimal radius is zero for $\tau\le\lambda$, and otherwise equals
$\lambda(\tau-\lambda)/(4\tau-\lambda)$. The implementation also handles positive d.
Correlated score noise can be supplied to the risk diagnostic but invalidates that
particular optimal-radius formula. Neither rule is a universal SE multiplier.

The Gaussian information helper profiles an unrestricted intercept using the known
full noise covariance. It returns the efficient sole-driver variance, the conditional
secondary variance and their coupling. Its risk-rate benchmark has a lower constant
1/32 and an upper constant 2 over the stated nonnegative dominance class. Comparing
EWMA coefficient risk with it requires matching coefficient units and checking the
actual weighting efficiency. A fixed short span need not be efficient for a constant
coefficient observed over expanding history; this is a different problem from tracking
a drifting endpoint.

## Worked example

The [canonical offline example](../examples/docs/prior_inference.py) creates a fixed
synthetic design and verifies its SE against `compute_expert_prior_statistics`.
It also checks the known-shape and continuous-AR constructions and verifies the
radius calculation by independent endpoint algebra. No portfolio or return forecast
coverage is asserted.

```python
radius = two_factor_minimax_radius(secondary_bound=0, bias_bound=8, penalty=4)
np.testing.assert_allclose(radius, 4/7)
```

The result is a coefficient-scale radius for the example's limiting experiment.
It is not an instruction to set `expert_prior_bound_n_std` to that number.

## Implementation in factorlasso

- `compute_prior_hac_geometry` returns `PriorHacGeometry`, including linear and quadratic
  forms and a `statistics` method returning coefficient and HAC SE arrays.
- `gaussian_prior_critical_value` returns the known-shape two-sided multiplier.
- `compute_ar1_prior_interval` returns `Ar1PriorInterval`. Full-domain mode is the default;
  `adaptive=True` uses the residual-direction confidence cover. Audit fields include
  retained cells, parameter edges, cell critical values and the allocated tail probability.
- `two_factor_minimax_radius` and `two_factor_limit_risk` expose the conditional limiting
  radius and worst coefficient risk over the fixed exposure and prior-error budgets.
- `gaussian_dominance_information` returns `GaussianDominanceInformation`; `risk_rate`
  supplies the known-covariance oracle benchmark for a supplied secondary-exposure bound.

The functions require only core dependencies. Run `python examples/docs/prior_inference.py`
with the repository's prescribed interpreter, or copy the standalone script after installing
the package. The example uses linear algebra and deterministic quadrature, not an optimization
solver. Existing `LassoModel` parameters and production floor calculations remain unchanged.

## Interpretation and limitations

Coverage statements require the fixed correct Gaussian mean/covariance model. Plugging in
a fitted covariance shape, omitting factors, selecting the reported factor using the same
responses, or interpreting a weighted coefficient as a current endpoint requires additional
analysis. Automatic-selector floor construction still uses the existing HAC tuning rule;
these functions do not establish confidence coverage for that selection procedure.

Calibration uses floating-point eigenvalues, quadrature and root finding. It is exact in
model distribution, up to reported numerical tolerances, not a formal interval-arithmetic
certificate. Very coarse grids that would demand numerically unstable tail probabilities
are rejected. Runtime increases with sample size and cell count; batched responses reuse
the same cell calibration. The limiting radius is not a finite-sample, arbitrary-signal,
random-prior-error or prior-centred FCGL optimum.

## See also

- [Prior targets](prior_targets.md)
- [EWMA weighting and ragged histories](ewma_weighting_and_ragged_histories.md)
- [Structured group penalties](group_penalties_hcgl_fcgl.md)

## References

- Imhof, J. P. (1961). Computing the distribution of quadratic forms in normal variables.
  *Biometrika*, 48(3–4), 419–426. [DOI](https://doi.org/10.1093/biomet/48.3-4.419).
- Tyler, D. E. (1987). Statistical analysis for the angular central Gaussian distribution
  on the sphere. *Biometrika*, 74(3), 579–589. [DOI](https://doi.org/10.1093/biomet/74.3.579).
- Berger, R. L. and Boos, D. D. (1994). P values maximized over a confidence set for the
  nuisance parameter. *Journal of the American Statistical Association*, 89(427), 1012–1016.
  [DOI](https://doi.org/10.1080/01621459.1994.10476836).
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
