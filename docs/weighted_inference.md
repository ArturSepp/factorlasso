---
myst:
  html_meta:
    description: >-
      Classical, heteroskedasticity-robust and HAC standard errors and confidence
      intervals, with weighted regression, EWMA means and factorlasso calibration APIs.
---

# Weighted regression and mean inference

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Weighted regression inference estimates coefficients and the uncertainty caused by observing
a finite response history. The reusable inference layer combines weighted least squares (WLS)
with heteroskedasticity and autocorrelation consistent (HAC) covariance. Optional Gaussian
calibration chooses a confidence-interval multiplier for an explicitly declared mean and
covariance model. Prior floors and residual-alpha reporting use these calculations through
their own application interfaces.

## Overview

Classical and HAC confidence intervals both combine a parameter estimate, its standard
error and a critical value. HAC changes the estimated covariance to allow unequal noise
variances and serial dependence. Holding the observations, weights and regression design
fixed, this changes uncertainty without changing the fitted coefficients.

| Covariance calculation | Variance assumption | Dependence assumption |
|---|---|---|
| Classical ordinary least squares | One common conditional error variance | Errors are uncorrelated across observations |
| Heteroskedasticity-consistent (HC) | Conditional variances may differ | No cross-observation score covariance is included |
| Heteroskedasticity and autocorrelation consistent (HAC) | Conditional variances may differ | Includes weighted lagged score covariances up to a chosen bandwidth |

These describe covariance estimators, not complete coverage guarantees. Correct specification
of the estimated mean, identification, suitable moments and appropriate dependence conditions
remain necessary. White (1980) develops heteroskedasticity-consistent inference; Newey and
West (1987) add lagged covariance terms with a positive semidefinite construction.

Use the statistics interface for regression coefficients and standard errors, including a
weighted mean with no explanatory variables. It retains gaps on the original observation grid
and computes lag products without a dense observation covariance matrix. Request full
coefficient covariance when contrasts or joint uncertainty are needed.

Use the geometry interface when a fixed linear target and quadratic HAC variance are needed
for calibration. It builds dense matrices and therefore has a larger memory cost. A known
Gaussian covariance shape permits finite-model calibration up to numerical tolerances; a
bounded stationary autoregressive model of order one (AR1) permits conservative calibration
over a continuous range of correlations. These guarantees concern a fixed mean model and
target, independently of response noise.

## Inputs, notation, and assumptions

Let $D$ be a design with $T$ rows and $P$ columns, $y$ one response, and
$W=\operatorname{diag}(w_1,\ldots,w_T)$ nonnegative loss weights. Let $n$ count usable
positive-weight observations and $L$ be the Bartlett bandwidth in grid periods.
The mean model is $E[y]=D\theta$, and a fixed vector $c$ identifies the target $c^{\top}\theta$.

| Interface | Design and response conventions |
|---|---|
| Statistics | Supply explanatory variables without an intercept; `fit_intercept=True` adds it. An array with zero columns requests a weighted mean. Nonfinite rows receive zero weight. |
| Geometry | Supply the complete finite design, including an intercept if intended. Choose exactly one coefficient index or nonzero contrast. No rows are dropped. |
| Calibration | Supply complete finite responses. The covariance shape must be known up to scale, or belong to the declared regular-grid stationary AR1 family. |

An exponentially weighted moving average (EWMA) span $s$ corresponds to weights
$w_t=(1-2/(s+1))^{T-t}$. The caller supplies the weights; a common positive rescaling cancels.
The statistics helper requires at least three usable observations, the declared `min_periods`,
positive residual degrees of freedom, and full weighted rank. It reports unidentified fits
with NaN values and a status. Geometry rejects an unidentified design.

All examples use dimensionless synthetic observations without annualisation. Choose and state
one return frequency when applying them to returns. Bandwidth is measured in original grid
positions, not elapsed time inferred from labels. Rolling calls must use only available past
observations. Explicit irregular calendar distances belong to the alpha application's kernel;
dropping dates before an AR1 call would change its statistical model.

## Methodology

### Standard deviation, standard error and confidence intervals

For a complete equally weighted sample of $T$ observations, let $\bar y$ be the sample
mean and let $\widehat\sigma_y$ be the sample standard deviation:

$$
\widehat\sigma_y^2=\frac{1}{T-1}\sum_{t=1}^{T}(y_t-\bar y)^2,
\qquad
\widehat{\mathrm{SE}}_{\mathrm{classic}}(\bar y)=\frac{\widehat\sigma_y}{\sqrt T}.
$$

The standard deviation describes dispersion of individual observations. The standard error
describes uncertainty in an estimated parameter, here the mean. Multiplying the raw standard
deviation by a normal quantile does not give a confidence interval for that mean. A future
observation requires a separate prediction interval, which also includes future observation noise.

At confidence level $1-\alpha$, the classical mean interval is

$$
\bar y\pm t_{1-\alpha/2,T-1}\frac{\widehat\sigma_y}{\sqrt T}.
$$

Here $t_{p,\nu}$ denotes a Student quantile with $\nu$ degrees of freedom. This interval is
exact for independent, identically distributed Gaussian observations. A normal multiplier
$z_{1-\alpha/2}$ gives a large-sample approximation; $z_{0.975}\approx1.96$ for a two-sided
95% interval. These distributional statements require their assumptions, rather than following
from a standard-error formula alone. See the [NIST mean-interval description](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).

### HAC for a sample mean

Let $u_t=y_t-\bar y$ and define the sample lag covariance with a common divisor $T$:

$$
\widehat\gamma_\ell=\frac1T\sum_{t=\ell+1}^{T}u_tu_{t-\ell},
\qquad
k_\ell=1-\frac{\ell}{L+1}.
$$

Specializing factorlasso's Bartlett sandwich to this complete, equally weighted mean gives

$$
\widehat{\mathrm{SE}}_{\mathrm{HAC}}^2(\bar y)=
\frac{T}{T-1}\frac1T
\left[\widehat\gamma_0+
2\sum_{\ell=1}^{\min(L,T-1)}k_\ell\widehat\gamma_\ell\right].
$$

The leading $T/(T-1)$ is the package's finite-sample correction. At $L=0$, this reduces
exactly to $\widehat\sigma_y^2/T$. That equality is specific to the complete, equally
weighted mean: in a general regression, HC and classical standard errors need not agree.

Positive lag covariances increase the estimated uncertainty relative to the zero-lag
calculation; negative ones can decrease it. Consequently HAC is not always wider or more
conservative. Reordering a sample preserves its mean and standard deviation but changes its
lag products, so chronological order is part of the statistical input.

For a stationary AR1 process with coefficient $\phi$ and marginal variance $\sigma_y^2$,
the exact variance identity is

$$
\operatorname{Var}(\bar y)=\frac{\sigma_y^2}{T}
\left[1+2\sum_{\ell=1}^{T-1}\left(1-\frac{\ell}{T}\right)\phi^\ell\right].
$$

For a long sample, summing the geometric series gives

$$
\operatorname{Var}(\bar y)\approx\frac{\sigma_y^2}{T}\frac{1+\phi}{1-\phi},
\qquad
\frac{\mathrm{SE}_{\mathrm{dependent}}}{\mathrm{SE}_{\mathrm{iid}}}
\approx\sqrt{\frac{1+\phi}{1-\phi}}.
$$

At $\phi=0.5$, the long-sample standard error is about 1.73 times the independent-observation
calculation using the same marginal variance. This is an analytical comparison of true
variances, not a promise that a particular estimated HAC bandwidth recovers that ratio.
The canonical example verifies the finite-sample identity against an explicit AR1 covariance
matrix. This inflation concerns the mean; regression-coefficient uncertainty depends on
regressors as well as residuals.

### Classical regression covariance and weighted HAC

The coefficient estimate and residuals are

$$
\widehat\theta=(D^{\top}WD)^{-1}D^{\top}Wy,\qquad
\widehat u=y-D\widehat\theta.
$$

For unweighted ordinary least squares on a complete design ($n=T$), the classical covariance is

$$
\widehat V_{\mathrm{classic}}=\widehat\sigma_u^2(D^{\top}D)^{-1},
\qquad
\widehat\sigma_u^2=\frac{\widehat u^{\top}\widehat u}{n-P}.
$$

It uses one common residual variance. Conditional on a fixed full-rank design and independent
Gaussian errors of that common variance, a coefficient interval uses the Student multiplier
with $n-P$ degrees of freedom. Heteroskedasticity-robust covariance instead retains the
individual squared residuals in the score products; HAC additionally retains their lag products.

For scores $g_t=w_td_t\widehat u_t$, let $G$ have rows $g_t^{\top}$,
$A=D^{\top}WD$, and $K_{ts}=\max(0,1-\lvert t-s\rvert/(L+1))$. The covariance is

$$
\widehat V=\frac{n}{n-P}A^{-1}G^{\top}KGA^{-1}.
$$

This is the weighted-score Bartlett construction based on Newey and West (1987).
The weights enter zero-lag outer products twice. Missing rows have zero scores but retain their
lag positions. The correction uses the observed count; the diagnostic
$n_{\mathrm{eff}}=(\sum_t w_t)^2/\sum_t w_t^2$ describes weight concentration and is not a
replacement for residual degrees of freedom or an adjustment for serial correlation.

At `hac_lags=0`, $K=I$ and the implementation retains the individual weighted squared
residuals. For unweighted OLS this is the HC1 covariance, namely White's HC0 sandwich
multiplied by $n/(n-P)$. It does not revert to the classical common-variance formula.
For positive lags, the extra terms involve $g_tg_{t-\ell}^{\top}$ and its transpose.
Thus the relevant serial dependence for regression inference is in weighted regressor-residual
products, rather than the residual series alone.

### EWMA weights and effective sample size

EWMA loss weights express recency. They do not assert that older observations have larger
error variance. For model errors $\varepsilon=y-D\theta$, the familiar model-based WLS
covariance $\sigma^2 A^{-1}$ presumes
$\operatorname{Cov}(\varepsilon\mid D)=\sigma^2W^{-1}$ on positive-weight support. With
arbitrary fixed loss weights and independent homoskedastic errors, the covariance is instead

$$
\operatorname{Cov}(\widehat\theta\mid D)=
\sigma^2 A^{-1}D^{\top}W^2DA^{-1}.
$$

This follows by taking the covariance of the linear WLS estimate. The distinction between
precision weights and recency weights explains why both weight factors appear in the
factorlasso sandwich. The [statsmodels WLS documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html)
states the inverse-variance convention for model-based WLS.

For a normalized weighted mean, write $q_t=w_t/\sum_s w_s$. Under independent errors of
common variance $\sigma_y^2$, weight concentration alone gives

$$
\operatorname{Var}\left(\sum_t q_ty_t\right)
=\sigma_y^2\sum_t q_t^2=\frac{\sigma_y^2}{n_{\mathrm{eff}}}.
$$

Serial dependence adds the cross-date terms
$2\sum_{t>s}q_tq_s\operatorname{Cov}(y_t,y_s)$. Heteroskedasticity also prevents replacing
all marginal variances by one common value without further assumptions. Dividing an estimated
standard deviation by $\sqrt{n_{\mathrm{eff}}}$ therefore does not supply a general HAC
standard error. Once factorlasso has returned `standard_error`, it must not be divided by
another sample-size factor.

### Covariance geometry and Gaussian calibration

For $H=(D^{\top}WD)^{-1}D^{\top}W$, $h^{\top}=c^{\top}H$ and $R=I-DH$, geometry gives

$$
\widehat b=h^{\top}y,\qquad \widehat v=y^{\top}Qy,\qquad
Q=\frac{n}{n-P}R^{\top}\operatorname{diag}(h)K\operatorname{diag}(h)R.
$$

The generic geometry checks $h^{\top}D=c^{\top}$ and $QD=0$, along with dimensions,
rank, symmetry, positive semidefiniteness and residual-map consistency. It owns read-only
copies of its arrays. These algebraic checks cannot establish exogeneity or model correctness.

Under $y=D\theta+\sigma C\xi$, with $\xi\sim N(0,I)$ and known $V=CC^{\top}$, define
$a=C^{\top}h$ and $B=C^{\top}QC$. Calibration solves

$$
\Pr\left(\frac{\lvert a^{\top}\xi\rvert}
{\sqrt{\xi^{\top}B\xi}}>k_{\alpha}\right)=\alpha.
$$

The resulting interval is $\widehat b\pm k_{\alpha}\sqrt{\widehat v}$.
The unknown scale cancels, and numerator and denominator dependence is retained. Gaussian
quadratic-form probabilities are classical (Imhof, 1961); this implementation uses a specialized
single-positive-eigenvalue integral rather than general Imhof inversion.

For $V_{ts}=\phi^{\lvert t-s\rvert}$ and $\lvert\phi\rvert\le\phi_{\max}$, divide
$\operatorname{atanh}(\phi)$ into $J$ cells with half-width
$\eta=\operatorname{atanh}(\phi_{\max})/J$. Each centre is calibrated at

$$
\alpha_{\mathrm{cal}}=(\alpha-\delta)\exp(-2T\eta).
$$

Full-domain mode sets $\delta=0$ and maximizes the critical value over all cells. Adaptive mode
spends $\delta$ on a residual-direction confidence cover and maximizes over retained cells.
The cover uses angular central Gaussian densities (Tyler, 1987), a fixed mixture and a Markov
bound. Continuous-cell allowances retain off-grid parameters. The error allocation follows
the nuisance-confidence-set principle of Berger and Boos (1994); the mixture and cell bounds
are implementation choices. Adaptive mode need not give a narrower interval, and choosing
the narrower realized mode has no guarantee from this calculation.

### Fixed weighted means and joint residual resampling

The separate weighted-mean builder retains the supplied estimator weights $q$ and their
mass $m=\mathbf{1}^{\top}q$. With unit scale $s$, its linear form is $h=sq$, its residual
map is $R=I-\mathbf{1}q^{\top}/m$, and its constant-mean target is $sm\mu$. Its calendar
HAC variance form uses the observed-row correction $T/(T-1)$. Zero weights retain observed
rows in that count; missing observations should be excluded from the scalar geometry while
preserving their calendar distances. The alpha application supplies its exact recursion weights.

For a panel, the residual dependent-wild bootstrap multiplies all observed assets at each date
by a common Gaussian multiplier, with multiplier covariance given by the calendar Bartlett
kernel. It retains each missing mask and re-estimates weighted centring and HAC standard
errors in every replicate. Shao (2010) provides the dependent-wild-bootstrap framework;
fixed-span initialization and ragged multivariate inference here require additional validation.
The method is explicitly reported as an unvalidated bootstrap approximation.

Studentized intervals use the quantile of absolute replicate estimation error divided by its
re-estimated standard error. A simultaneous option uses the maximum over the supplied fixed
target family. Normal intervals instead use pointwise Gaussian or Bonferroni multipliers.
Bootstrap draws do not account for factor selection or model refitting in this interface.

### Quadratic error norm bounds

For a fixed positive semidefinite metric $M$ and Gaussian estimation error with known
covariance $V$, the squared error norm has the distribution $\sum_j\lambda_j Z_j^2$, where
$\lambda_j$ are the eigenvalues of $V^{1/2}MV^{1/2}$ and the $Z_j$ are independent standard
normals. Let $r$ be the square root of its desired quantile and let
$g=\sqrt{\widehat a^{\top}M\widehat a}$. A triangle inequality gives the conservative interval

$$
\left[\max(0,g-r)^2,(g+r)^2\right].
$$

The central quadratic quantile uses characteristic-function inversion, with separate sine
and cosine Fourier integrals and explicit convergence checks. Equal eigenvalues use a scaled
chi-square reference. This is distinct from the specialized pivot integral used above.
The legacy spectral option replaces the weighted sum by its maximum-eigenvalue envelope.
An experimental bootstrap option obtains the error-norm quantile from centred residual
bootstrap errors. None of these procedures makes plug-in covariance exact or validates
post-selection coverage. The reported noise correction remains signed and its reliability
depends separately on covariance estimation.

### Scalar quadratic and positive-part bounds

The scalar interfaces target a functional of a Gaussian mean directly. They require a
fixed metric and known estimation-error covariance for their coverage interpretation.
They use the full covariance spectrum, including cross-coordinate dependence.

For $Y\sim N(\theta,S)$, let $Q=\lVert Y\rVert^2$, $k=\lVert\theta\rVert^2$,
and let $L$ and $\lambda_j$ denote the largest and individual eigenvalues of $S$.
The exact Gaussian moment-generating function (MGF) implies the direction-uniform envelope

$$
\log E[\exp(tQ)]\leq c(t)+\frac{tk}{1-2tL},\qquad
c(t)=-\frac12\sum_j\log(1-2t\lambda_j),\qquad t<\frac{1}{2L}.
$$

The bound holds for either sign of $t$: $t/(1-2t\lambda)$ is increasing in
$\lambda$ on this domain. With tail probability $\delta=(1-\mathrm{confidence})/2$
and observed value $q$, define

$$
h(t,q)=\frac{\log\delta+tq-c(t)}{t}(1-2tL).
$$

Chernoff's inequality gives a lower endpoint $\max(0,\sup_{0<t<1/(2L)}h(t,q))$
and an upper endpoint $\max(0,\inf_{t<0}h(t,q))$. The implementation searches
a bounded reparameterization of $t$. Any admissible search candidate gives a conservative
endpoint; a missed optimum can widen it. A zero covariance gives the observed value exactly.
`quadratic_scalar_confidence_summary` applies this construction to
$\widehat a^{\top}M\widehat a$ using $Y=M^{1/2}\widehat a$ and
$S=M^{1/2}VM^{1/2}$. Its signed noise correction is
$\widehat a^{\top}M\widehat a-\operatorname{tr}(MV)$.

`positive_part_confidence_summary` instead targets
$k_+=\sum_i w_i\max(a_i,0)^2$ for fixed nonnegative metric weights $w_i$.
Transform coordinates by $\sqrt{w_i}$ and use the resulting $S$ and $L$.
The inequality $\lVert(\theta+e)_+\rVert\leq\lVert\theta_++e\rVert$ permits
the same upper-sampling-tail bound and hence the same lower confidence endpoint.
For the other endpoint, convexity of $g(\theta)=\lVert\theta_+\rVert$ gives a
supporting direction whose Gaussian error variance is at most $L$. Therefore

$$
U_+=\left(\sqrt{q_+}+z_{1-\delta}\sqrt L\right)^2.
$$

This uses the true supporting direction only in the proof; it does not freeze an estimated
positive set. The zero target is covered by nonnegativity. A fixed trace subtraction is
not valid across changing positive sets, so its noise correction is unavailable.

These envelope inversions and their positive-part combination are implementation derivations.
[Hsu, Kakade and Zhang (2012)](https://doi.org/10.1214/ECP.v17-2079) provide background on
MGF bounds for quadratic forms; their broader subgaussian setting is not asserted here.
Zero-signal quadratics have a vanishing first derivative, making ordinary first-order
Wald and bootstrap arguments unsuitable without further justification; see
[Chen and Fang (2019)](https://doi.org/10.1016/j.jeconom.2019.01.011).
These bounds remain conservative and need not be narrower than the error-norm bounds.
Taking the narrowest endpoints across different interval methods is not generally justified.

### Joint coefficient and residual-variance regions

`joint_wls_gaussian_region` provides a conservative region for every coefficient and
residual variance in a prespecified panel regression. It fits the full weighted design,
including an intercept. It does not use a selected LASSO support as though it were fixed.
The marginal noise law for asset $i$ must be Gaussian with covariance $\sigma_i^2 R$,
where $R$ is a known temporal shape with unit diagonal. Cross-asset dependence is unrestricted.
Fixed missing masks subset $R$ on the original calendar; they do not compress serial distances.

Write $H_i$ for the WLS coefficient map, $P_i=I-D_iH_i$ for its residual map and
$Q_i=P_i^{\top}W_iP_i/\operatorname{tr}(W_i)$. The normalized residual statistic is

$$
v_i=\frac{y_i^{\top}Q_i y_i}{\operatorname{tr}(Q_iR_i)}.
$$

Its expectation is $\sigma_i^2$. The distribution of $v_i/\sigma_i^2$ is a weighted
sum of independent chi-square variables with one degree of freedom. The weights are
the eigenvalues of $R_i^{1/2}Q_iR_i^{1/2}/\operatorname{tr}(Q_iR_i)$.
The existing quadratic-quantile implementation supplies the two tail quantiles.

For total failure probability $\alpha$, allocate $\alpha/(2N)$ to each two-sided
variance interval and $\alpha/(2NP)$ to each of the $NP$ coefficient-error events,
where $P=M+1$. The variance endpoints are $v_i/q_{1-\alpha/(4N)}$ and
$v_i/q_{\alpha/(4N)}$. If $U_i$ denotes the variance upper endpoint, coefficient $j$
has the conservative interval

$$
\widehat\theta_{ij}\pm z_{1-\alpha/(4NP)}
\sqrt{U_i h_{ij}^{\top}R_i h_{ij}}.
$$

On the intersection of the variance and Gaussian-error events, every endpoint encloses
its target. Bonferroni's inequality gives simultaneous coverage at least $1-\alpha$;
independence between coefficient and variance estimates is unnecessary. This error-budget
split and upper-scale construction are implementation choices. They can produce wide bounds.
The guarantee is conditional on the fixed correct Gaussian mean and covariance shape,
subject to numerical quadrature tolerances; substituting an estimated shape is insufficient.

WLS normalizes its loss weights. To target the production initialized EWMA residual alpha,
the caller must scale the fitted intercept by the original EWMA mass and the return
annualization factor using `coefficient_scale`. Loadings can retain unit scale. Residual
variance outputs remain per observation and require a separate stated annualization.
Batch input reuses geometry for panels with identical missing masks. Confidence applies
to each panel's parameter family, not simultaneously to the simulation batch.

## Worked example

### Compare the same mean with classical and HAC uncertainty

The [canonical offline example](../examples/docs/weighted_inference.py) uses eight synthetic,
dimensionless observations on a regular grid. An empty regressor matrix requests an intercept-only
fit. Both covariance choices give the same mean, 0.500, and use the same observations.

```python
observations = .5 + np.array([-2., -2., -1., -1., 1., 1., 2., 2.])
n = len(observations)
no_factors = np.empty((n, 0))
classic_se = observations.std(ddof=1)/np.sqrt(n)
hc = compute_wls_hac_statistics(no_factors, observations, hac_lags=0,
                                return_covariance=True)
hac = compute_wls_hac_statistics(no_factors, observations, hac_lags=2,
                                 return_covariance=True)
hc_interval = linear_confidence_intervals(hc.coefficients, hc.covariance)
hac_interval = linear_confidence_intervals(hac.coefficients, hac.covariance)
```

The interval helper defaults to pointwise normal calibration at 95% confidence. Supplying
`return_covariance=True` makes the coefficient covariance available to it. The full script
includes the imports and checks the HAC variance against an explicit sum over every date pair.

| Calculation | Standard error | Critical value | Interval endpoints |
|---|---:|---:|---|
| Classical mean SE with normal approximation; identical to HC here | 0.598 | 1.96 | -0.671 to 1.67 |
| The same classical mean SE with Student calibration, 7 degrees of freedom | 0.598 | 2.36 | -0.913 to 1.91 |
| Bartlett HAC with two lags and normal approximation | 0.859 | 1.96 | -1.18 to 2.18 |

The first and third rows isolate the covariance-estimation difference; the first and second
isolate the critical-value difference. The Student row is calculated explicitly with
`scipy.stats.t`, not by the normal interval helper. These short synthetic data demonstrate
arithmetic, not repeated-sampling coverage. The example also reorders the observations:
the sample standard deviation stays unchanged, while the HAC SE decreases below the classical
SE, demonstrating that HAC need not widen an interval.

### Weighted regression and model-based calibration

The [canonical offline example](../examples/docs/weighted_inference.py) creates a fixed
synthetic design and checks its coefficients and full covariance against a normal-equation
solve and explicit sums of calendar score products. It also checks a coefficient contrast
and an intercept-only mean against their independent algebraic references.

```python
geometry = compute_wls_hac_geometry(design, weights, hac_lags=2, coefficient=1)
estimate, standard_error = geometry.statistics(response)
critical = gaussian_critical_value(geometry, np.eye(24), alpha=.05)
interval = (estimate-critical*standard_error, estimate+critical*standard_error)
```

The identity covariance is a declared model in this synthetic example. Substituting a fitted
covariance matrix would require separate treatment of its estimation uncertainty.

### Joint region for an ordinary mean and variance

The same offline script checks the joint-region result for one response and no factors
against the independent sample-variance chi-square formula. Here the variance statistic
reduces to the unbiased sample variance. This is an arithmetic reference for the error
allocation; it does not establish robustness to non-Gaussian returns.

```python
joint = joint_wls_gaussian_region(
    np.empty((n, 0)), observations[:, None], np.ones((n, 1)),
    covariance_shape=np.eye(n))
```

### Scalar targets with a correlated covariance

The same synthetic example retains correlation and checks the quadratic target, its trace
correction and the positive-part upper endpoint against direct algebra.

```python
mean = np.array([.3, -.2, .5])
covariance = np.array([[.1, .08, -.02], [.08, .2, .03], [-.02, .03, .08]])
weights = np.array([2., .5, 1.])
quadratic = quadratic_scalar_confidence_summary(mean, covariance, np.diag(weights))
positive = positive_part_confidence_summary(mean, covariance, weights)
```

Here covariance is declared known. This arithmetic example does not establish empirical
coverage after substituting an estimated HAC covariance or estimating the metric.

## Implementation in factorlasso

### Choosing the covariance and the interval

Covariance estimation and critical-value calibration are separate choices in the current API.
The standard errors are square roots of coefficient-covariance diagonal entries; the helper
does not estimate a different set of coefficients when the HAC lag count changes.

| Requested calculation | Implemented interface and behavior |
|---|---|
| Classical OLS covariance or an ordinary Student interval | Reference formulas above; there is no `classical` covariance switch in the WLS HAC helpers or `student` method in the linear interval helper. The teaching example calculates the Student comparison with SciPy. |
| Heteroskedasticity-robust WLS uncertainty | `compute_wls_hac_statistics(..., hac_lags=0)`; includes the observed-count correction and keeps original row positions. |
| WLS uncertainty with serial dependence | The same function with `hac_lags=L` for a positive integer. Bartlett weights are fixed; callers choose the lag count. |
| Approximate normal intervals from an estimated covariance | `linear_confidence_intervals(..., method='normal')`; its covariance input may be HC or HAC. The default is pointwise, with an explicit Bonferroni option for simultaneous intervals. |
| Gaussian calibration for a declared covariance shape | `gaussian_critical_value` with fixed geometry; the unknown scalar noise level cancels. An estimated covariance shape does not inherit this guarantee. |
| Gaussian calibration over bounded stationary AR1 dependence | `compute_ar1_interval`; complete regular-grid support, declared AR bound, and full-domain or adaptive cell calibration. |
| Experimental studentized bootstrap intervals | `bootstrap_weighted_means` with `linear_confidence_intervals(..., method='bootstrap_t')`; the returned status explicitly identifies an unvalidated approximation. |

The WLS helpers default to `hac_lags=0`, so serial-dependence adjustment is opt-in.
They have no automatic bandwidth selector. Calendar-mean and alpha interfaces instead accept
`bandwidth`, the distance at which the Bartlett kernel becomes zero. On a regular grid of unit
spacing, `bandwidth=L+1` matches `hac_lags=L`. A six-month bandwidth therefore includes five
monthly lag positions; this is a different parameter convention from a lag count.

### Public interfaces and application adapters

- `WlsHacStatistics` and `compute_wls_hac_statistics` provide coefficients, standard errors,
  optional full covariance, observation counts, effective sample size, gaps and fit status.
- `LinearHacGeometry` and `compute_wls_hac_geometry` represent a fixed coefficient or contrast
  and its quadratic variance. `statistics` accepts one response or a batch of response rows.
- `gaussian_critical_value` returns the two-sided multiplier for known covariance shape.
- `joint_wls_gaussian_region` combines a full-design WLS inference anchor with simultaneous
  coefficient and residual-variance bounds under a known Gaussian temporal shape.
- `Ar1Interval` and `compute_ar1_interval` return intervals and a continuous-cell audit,
  including retained cells, edges, centre critical values and the allocated tail probability.
- `compute_weighted_mean_hac_geometry` represents a fixed weighted mean without normalizing
  its estimator weights or changing its initialization target.
- `bootstrap_weighted_means` produces joint residual-bootstrap errors and re-estimated SEs.
  With `return_residuals=True`, it also exposes a `(draws, T, N)` panel in the
  original input units, with the original missing cells and common multipliers.
  This supports downstream factor refits. Refitting and optimisation are owned
  by the caller; the output remains an unvalidated bootstrap approximation.
  With `return_covariances=True`, `covariance_draws` contains each replicate's full
  joint HAC estimate, including off-diagonal dependence. Its diagonal equals the
  squared reported SEs. These draws describe a bootstrap estimator distribution;
  they are not draws from a Bayesian covariance posterior.
- `weighted_mean_hac_expectation` computes exact joint covariance factors and expected
  centered HAC factors under a declared separable temporal covariance and fixed observation
  masks. It retains initialization masses and native observation counts. Elementwise ratios
  diagnose covariance attenuation; applying those ratios to a sample covariance is not
  guaranteed to preserve positive semidefiniteness or deliver calibrated intervals.
- `calibrate_covariance_moments` applies a PSD-preserving congruence to one covariance
  or a batch. Given a fixed declared expectation $M$ and target $V$, it uses principal
  symmetric roots to form $T = V^{1/2}M^{-1/2}$ and returns $TST^{\top}$.
  The identity $TMT^{\top}=V$ corrects the first moment when $M=E[S]$ is correct.
  It requires numerically positive-definite $M$, applies no ridge or pseudoinverse,
  and does not remove covariance estimation noise. Fitted moments require separate
  validation; moment matching alone does not calibrate confidence or credible intervals.
- `linear_confidence_intervals` provides explicit normal or experimental studentized intervals,
  with pointwise or simultaneous scope and unavailable-result handling.
- `gaussian_quadratic_quantile` computes central positive Gaussian quadratic quantiles.
- `quadratic_confidence_summary` reports observed/noise/corrected quadratics and error-norm
  bounds, keeping the statistical method and its limitations in the returned metadata.
- `quadratic_scalar_confidence_summary` and `positive_part_confidence_summary` provide
  opt-in scalar Gaussian bounds with explicit plug-in limitations. They accept one mean
  vector or a batch of row vectors sharing the same covariance and fixed metric. Confidence
  is pointwise for each functional and row, not simultaneous across the batch.

These names are exported from `factorlasso` and `factorlasso.inference`. The existing
prior-facing types and functions retain their signatures, result fields, module paths and
ownership in `factorlasso.priors`; they delegate to the shared numerical implementations.
The production statistics path uses lagged influence products rather than dense geometry.
The generic layer depends only on mathematical utilities and the existing core dependencies.

Run the example with the repository's prescribed external interpreter, or after installing
the package. The implementation uses a scaled singular-value decomposition; the statistics
helper centers the data when fitting an intercept to protect large offsets.

The [prior application](prior_inference.md) retains `compute_expert_prior_statistics`,
`compute_prior_hac_geometry`, `gaussian_prior_critical_value` and `compute_ar1_prior_interval`
as adapters. Their uncertainty concerns the selected small regression. The prior-floor policy
and a calibrated confidence interval are separate operations.

The [alpha application](alpha_uncertainty.md) provides `estimate_alpha_uncertainty` for the
exact recursive EWMA estimate, joint calendar HAC covariance and approximate pointwise normal
intervals. `calibrate_alpha_uncertainty` optionally selects `normal`, `known_shape` or `ar1`
calibration while retaining the saved estimator weights. Irregular native support can make
AR1 bounds unavailable. These adapters do not silently renormalize initialization mass or
refit the factor model as part of uncertainty estimation.

## Interpretation and limitations

Bandwidth controls the dependence included in the variance estimate. Too few lags can miss
important dependence; including more lags can add estimation noise. Choosing whichever
bandwidth gives the narrowest interval has no coverage guarantee. Classical HAC consistency
is an asymptotic result under moment and weak-dependence conditions with appropriate bandwidth
behavior; a finite fixed-span EWMA history does not automatically meet a large-sample limit.
The package's $n/(n-P)$ correction is a convention, not a finite-sample coverage theorem.

Ordinary HAC standard errors are not automatically exact confidence intervals. Gaussian
calibration requires the fixed correct mean and covariance assumptions and uses floating-point
eigenvalues, quadrature and root finding. Its numerical tolerances are not an interval-arithmetic
proof. The coverage is pointwise, with no automatic adjustment for factor selection, multiple
reported coefficients, omitted means, penalized bias or a drifting endpoint.

The scalar bounds also require known Gaussian estimation covariance. Plug-in HAC input is
accepted with explicit unvalidated status; a simulation pilot cannot establish coverage for
every fitted model. A metric chosen from the same data introduces additional uncertainty.
Dividing a functional and its endpoints by the number of coordinates rescales the result
without improving its relative precision.

Recursive EWMA residual alpha can have initialization weight mass below one. Its target is
then that mass times the constant residual mean. Reusing the kernel does not justify silently
normalizing alpha into the WLS mean. The alpha interface retains its original recurrence,
joint calendar covariance and approximate interpretation conditional on the fitted factor model.

Other estimators may reuse influence aggregation, but must supply their own derivatives,
normalization and corrections. GLS requires its own geometry; ridge bias and Lasso selection
need additional inference methods. No inference guarantee follows from moving an estimator
behind a common interface.

## See also

- [Prior uncertainty and conditional floor risk](prior_inference.md)
- [Residual alpha uncertainty](alpha_uncertainty.md)
- [EWMA weighting and ragged histories](ewma_weighting_and_ragged_histories.md)
- [Software design](software_design.md)

## References

- Hsu, D., Kakade, S. M. and Zhang, T. (2012). A tail inequality for quadratic forms of
  subgaussian random vectors. *Electronic Communications in Probability*, 17, paper 52.
  [DOI](https://doi.org/10.1214/ECP.v17-2079).
- Chen, Q. and Fang, Z. (2019). Inference on functionals under first order degeneracy.
  *Journal of Econometrics*, 210(2), 459–481.
  [DOI](https://doi.org/10.1016/j.jeconom.2019.01.011).
- NIST/SEMATECH. *e-Handbook of Statistical Methods*, Section 7.4.7.3,
  [Bonferroni's method](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm).
- NIST/SEMATECH. *e-Handbook of Statistical Methods*, Section 1.3.5.2,
  [Confidence limits for the mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct
  test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
  [DOI](https://doi.org/10.2307/1912934).
- Shao, X. (2010). The dependent wild bootstrap. *Journal of the American Statistical
  Association*, 105(489), 218–235. [DOI](https://doi.org/10.1198/jasa.2009.tm08744).
- Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity
  and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
  [DOI](https://doi.org/10.2307/1913610).
- Imhof, J. P. (1961). Computing the distribution of quadratic forms in normal variables.
  *Biometrika*, 48(3–4), 419–426. [DOI](https://doi.org/10.1093/biomet/48.3-4.419).
- Tyler, D. E. (1987). Statistical analysis for the angular central Gaussian distribution
  on the sphere. *Biometrika*, 74(3), 579–589. [DOI](https://doi.org/10.1093/biomet/74.3.579).
- Berger, R. L. and Boos, D. D. (1994). P values maximized over a confidence set for the
  nuisance parameter. *Journal of the American Statistical Association*, 89(427), 1012–1016.
  [DOI](https://doi.org/10.1080/01621459.1994.10476836).
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
- statsmodels developers. [Weighted least squares: inverse-variance weight convention](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html).
