---
myst:
  html_meta:
    description: >-
      Conditional Monte Carlo test inversion using full production FCGL refits,
      independent coverage calibration, signed portfolio contrasts and explicit limits.
---

# Production factor-model test inversion

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-22](https://github.com/ArturSepp/factorlasso/commit/8141e2e3577704ae539495d6a304aae0667d98be)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Test inversion constructs a confidence set by retaining candidate underlying loadings
that a test cannot reject. The test statistic is the full production estimate, including
selection and constraints. Reported production loadings are unchanged.

## Overview

The factor-cluster group LASSO (FCGL) estimator is biased and nonregular near selection
boundaries. Ordinary bootstrap percentiles can therefore have poor coverage even when
every selection step is refitted. Test inversion instead simulates each candidate null
and compares the observed FCGL statistic with its null distribution.

Conditional Gaussian simulation eliminates nuisance regression coefficients and a common
variance scale. A supplied, independently fixed covariance shape gives a finite-sample
Monte Carlo test [1]. An estimated shape requires separate simulation calibration and
independent validation. Its coverage claim is restricted to the audited design family.

## Inputs, notation, and assumptions

Stack observed response cells, in date then asset order, into $y$. The design $D$ has
an intercept and factor slopes for each asset, omitting only externally fixed zero
coefficients. The model is $y=D\gamma+\varepsilon$, with covariance
$\sigma^2 C$. A scalar contrast $b=c^{\top}\gamma$ can be an asset loading or a
signed portfolio loading. Portfolio weights are supplied by the caller without normalisation.

Factors and responses retain their native frequency, original missing dates and existing
return conventions. There is no annualisation. The FCGL configuration, including EWMA
span, solver, clustering, priors, signs and penalties, is copied unchanged into each refit.
The conditional design must have full column rank and residual degrees of freedom.

The default working shape estimates per-asset variances with residual degrees-of-freedom
correction. Each cross-asset correlation uses residuals projected on the pair's common
dates and union of allowed factors; correlating separately fitted, unequal histories
can attenuate shared noise. Every pair needs at least three common residual degrees of
freedom. Correlations use an adaptive shrinkage toward independence, capped at 50%, and
a 0.001 eigenvalue floor. Strong, precisely shared residual variation receives little
shrinkage. This is an implementation-specific working regularisation rule.

Given those marginal scales and correlations, a common AR(1) coefficient is profiled
over $[0,0.85]$ using a restricted-likelihood criterion [2]. This feasible profile is
not joint estimation of all covariance parameters. These are working assumptions, not
established properties of financial returns; the entire shape is re-estimated in every
outer calibration and validation panel.

## Methodology

Whiten the data and design using $C$. Write $z=C^{-1/2}y$, $A=C^{-1/2}D$ and
$P_A$ for the orthogonal projection onto the design. Define

$$
s^2=c^{\top}(A^{\top}A)^{-1}c, \qquad
a=A(A^{\top}A)^{-1}c/s, \qquad
m=P_Az-a(a^{\top}z).
$$

Under the null $c^{\top}\gamma=b$, condition on $m$ and the null residual length
$R_b=\lVert z-m-ab/s\rVert$. Simulate

$$
z^*=m+ab/s+R_b u,
$$

where $u$ is uniform on the unit sphere orthogonal to the null nuisance design.
Unwhiten, restore the missing-data pattern, and rerun the complete FCGL estimator.
The statistic is $T_b=\lvert c^{\top}\widehat\gamma_{FCGL}-b\rvert$ and its
Monte Carlo probability is $(1+\sum_{j=1}^{B}1_{\lbrace T_{b,j}^*\geq T_b\rbrace})/(B+1)$.
Ties count against rejection. Failed fits stop inference; they are never discarded.

For estimated covariance, `calibrate_inversion_tests` selects the largest attainable
rejection threshold achieving a predeclared conservative training coverage target in
every design. It then freezes that threshold and evaluates disjoint validation seeds.
Every design's one-sided Clopper-Pearson lower coverage bound, Bonferroni adjusted across
designs, must clear the requested coverage. Passing one convenient design is insufficient.
Validation failures do not cause automatic threshold retuning.

Numerical inversion expands both tails, searches a grid, refines the outer acceptance
boundaries and returns their hull. Open tails remain infinite. Disconnected acceptance
regions, endpoint resolution and all candidate tests are exported. This finite search
is an approximation, not a proof that no remote acceptance region exists.

## Worked example

The [offline script](../examples/docs/production_test_inversion.py) uses 48 synthetic monthly
observations, two factors, two responses, an eight-month late start for B, EWMA span 36
and CLARABEL. The Gaussian covariance shape is known by construction. It checks that
the production point is unchanged and accepted by its own null test.

```python
engine = FactorTestInversion(model, x, y,
    covariance=np.eye(int(y.notna().sum().sum())), draws=39, seed=7)
point = float(baseline.Equity @ np.array([.7, .3]))
test = engine.test(contrast, point)
assert test['pvalue'] == 1.
pd.testing.assert_frame_equal(engine.estimate, baseline)
assert model.coef_ is None
```

This small example demonstrates the API and algebra, not 95% empirical coverage.
The reproducible outer experiment is `benchmarks/test_inversion_coverage.py`.
Use independent calibration and validation seed ranges and retain both trial tables.

## Implementation in factorlasso

- `FactorTestInversion` owns the conditional geometry, complete null refits and search.
- `FactorTestInversionResult` contains unchanged production points, interval bounds,
  every candidate test and numerical diagnostics.
- `calibrate_inversion_tests` selects a threshold and audits independent coverage.
- `TestInversionCalibration` records the threshold, validation table and pass/fail gate.

`test` can stop early only after acceptance is certain for the predeclared fixed number
of draws. It returns lower and upper probability bounds; omitted draws cannot reverse
that acceptance. No early rejection is allowed. `intervals` requires matching draw counts
and a passing calibration for estimated covariance. The caller remains responsible for
matching the audited estimator, design, noise family and contrasts to the application.

Null refits can reuse factor-only competitor geometry keyed by the exact factor data,
observation mask, selected factor set and weighting settings. Every response statistic,
selection and bound is still recalculated. This private inference cache is absent from
ordinary production fits and is checked against uncached refits with exact equality.

## Interpretation and limitations

The interval estimates an underlying linear exposure using the production estimator as
its statistic. It is not an interval for a different least-squares point estimate, a
bootstrap stability range, or a prediction interval for future portfolio losses.

Coverage is pointwise, not simultaneous across all displayed factors. Fixed-design
Gaussian exactness requires a truly known covariance shape. Plug-in covariance and
calibration over a finite simulation family do not provide a distribution-free or
uniform guarantee. Validate heavy tails, heteroskedasticity, dependence, short histories,
collinearity, selection boundaries and relevant contrasts before reporting a coverage claim.

The conditional projection internally uses unrestricted nuisance regression geometry;
it does not replace the production point. External restrictions can legitimately intersect
the resulting confidence set, but data-selected zero loadings cannot be treated as
known zero parameters. Wide or unbounded intervals are valid outcomes and must remain visible.
Synthetic product assumptions and future structural changes are outside the model.

Coverage validation must be accompanied by a power and interval-width review.
A small rejection threshold can pass a coverage gate while producing little useful
information about the parameter. In particular, absolute distance from the candidate
can have low power when the estimator is shrunk toward a different value. Frequent
ties or selected zero loadings can also prevent rejection of distant candidates.
Neither a passing coverage audit nor complete FCGL refits establish informative bounds.
Inspect candidate-test curves, open tails, collinearity and native observation counts
before using intervals for hedge sizing. Do not shorten intervals for presentation
or promote bootstrap stability ranges to confidence intervals. Reporting can retain
wide confidence sets in a separate diagnostic table while showing unchanged exposure
estimates on their own chart scale.

## See also

- [Production factor-model bootstrap](production_bootstrap.md)
- [Weighted regression inference](weighted_inference.md)
- [Group penalties](group_penalties_hcgl_fcgl.md)

## References

1. Dufour, J.-M. (2006). Monte Carlo tests with nuisance parameters: A general approach
   to finite-sample inference and nonstandard asymptotics. *Journal of Econometrics*,
   133, 443-477. [doi:10.1016/j.jeconom.2005.06.007](https://doi.org/10.1016/j.jeconom.2005.06.007).
   The package uses conditional Gaussian elimination and application-specific empirical
   calibration; it does not inherit general validity for estimated covariance shapes.
2. Patterson, H. D. and Thompson, R. (1971). Recovery of inter-block information when
   block sizes are unequal. *Biometrika*, 58(3), 545-554.
   [doi:10.1093/biomet/58.3.545](https://doi.org/10.1093/biomet/58.3.545).
   The package uses the restricted-likelihood principle to profile a working AR(1)
   parameter with plug-in scales and correlations; it does not inherit exact coverage
   for that estimated covariance shape.
