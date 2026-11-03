---
myst:
  html_meta:
    description: >-
      Fixed-group Gaussian theory for conditional pooled signs, with exact weighted
      regression geometry, false exclusions, sign clipping and independently checked risk.
---

# Conditional pooled signs: fixed-group Gaussian theory

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Conditional pooled signs use joint factor-regression coefficients averaged across fixed
response groups. This article derives their Gaussian sampling geometry and the risk of a
scalar estimator that clips a raw estimate to a learned sign or excludes it. These are exact
reference calculations under known covariance parameters. They do not give the risk of the
complete learned-group, correlated-factor HCGL or FCGL estimator.

The four analytics names on this page are opt-in capabilities. They do not change
`LassoModel` or production defaults.

## Overview

Pooling can reduce noise in a direction estimate, but a univariate regression need not target
the direction of a conditional factor loading. For a cluster-average true loading vector
$\overline\beta_c$ and factor covariance $\Sigma_x$, the population marginal slope is

$$
b^{\mathrm{marg}}_{c,j}=\overline\beta_{c,j}+
\sum_{\ell\ne j}\frac{\Sigma_{x,j\ell}}{\Sigma_{x,jj}}\overline\beta_{c,\ell}.
$$

The joint regression targets $\overline\beta_c$ when the factor model is correctly specified,
the design is identified and the errors are exogenous. A precise cluster average can still
have the wrong sign for a heterogeneous member. Sign restrictions are established forms of
regularization; their usefulness depends on assumptions about directions and design
geometry, as in [Meinshausen (2013)](https://doi.org/10.1214/13-EJS818). The analytics here do
not transfer that paper's high-dimensional guarantees to a data-dependent gate.

## Inputs, notation, and assumptions

| Input or symbol | Meaning | Convention |
|---|---|---|
| $X$ | Fixed factor design | $T$ dates by $M$ factors, finite and complete |
| $\beta$ | True response loadings | $N$ responses by $M$ factors |
| $\Omega$ | Residual covariance across responses | Known, symmetric positive semidefinite |
| $\Gamma$ | Residual covariance across dates | Known, symmetric positive semidefinite; identity by default |
| $A$ | Fixed pooling weights | Nonnegative group-by-response rows summing to one |
| $W$ | Regression date weights | Diagonal matrix of supplied weights |
| $Z,G$ | Raw individual estimate and pooled gate | Jointly Gaussian scalars |
| $\theta,\gamma$ | Their respective means | The individual truth can differ from the pooled mean |
| $s_Z,s_G,r$ | Standard deviations and correlation | Known distribution parameters |
| $\tau$ | Gate threshold | Nonnegative; positive infinity makes every gate weak |

The geometry assumes $Y=\boldsymbol 1\alpha^\top+X\beta^\top+E$ and separable Gaussian
errors with $\operatorname{Cov}(E_{t,i},E_{u,k})=\Gamma_{tu}\Omega_{ik}$. The full weighted
design must have column rank. If `fit_intercept=False`, the model has no intercept. Supplying
date dependence is not the same as estimating or calibrating it. Preprocessing, weights and
groups are fixed independently of this Gaussian error draw.

No frequency conversion or annualization is performed. Coefficients and their standard
deviations use the supplied response/factor units; scalar mean squared error uses squared
coefficient units. A factor-premium contribution risk uses a premium in matching units and
return horizon. The example is synthetic and dimensionless, with no rolling history, warmup,
missing-data rule or production preprocessing hidden in the calculation.

## Methodology

### Joint conditional-regression geometry

Let $D=[\boldsymbol 1,X]$ when an intercept is included, and $D=X$ otherwise. Let $L$ be the
slope rows of $(D^\top WD)^{-1}D^\top W$. Define

$$
V=L\Gamma L^\top,\qquad
\widehat\beta=(LY)^\top,\qquad
\widehat b=A\widehat\beta.
$$

The expectations and covariance follow by linearity:

$$
E[\widehat\beta]=\beta,\qquad E[\widehat b]=A\beta,\qquad
\operatorname{Cov}(\widehat\beta_{i,j},\widehat\beta_{k,\ell})
=\Omega_{ik}V_{j\ell}.
$$

For one individual response $i$ and group $c$ on factor $j$,

$$
s_Z^2=\Omega_{ii}V_{jj},\qquad
s_G^2=(A\Omega A^\top)_{cc}V_{jj},\qquad
\operatorname{Cov}(Z,G)=(\Omega A^\top)_{ic}V_{jj}.
$$

Thus the same-data dependence between the gate and fit is retained. With independent dates
and ordinary least squares, $V$ is the inverse centered factor Gram matrix. Factor
collinearity inflates uncertainty through this matrix. In an equal-weight cluster of size
$m$ with equal residual variance $\sigma^2$ and equicorrelation $\rho$, the pooled residual
variance is $\sigma^2[\rho+(1-\rho)/m]$. Averaging cannot remove a shared residual component.

### Exclusion, release and sign clipping

Write $h=\tau s_G$. The scalar exclusion estimator is defined by three cases:

| Gate event | Estimate |
|---|---|
| $G\ge h$ with a positive gate | $\max(Z,0)$ |
| $G\le-h$ with a negative gate | $\min(Z,0)$ |
| Weak gate | $0$ |

The release estimator uses $Z$ on the weak event and the same clipped values on active
events. At a deterministic zero gate, the gate is weak. Sign clipping and weak-gate
exclusion are distinct sources of zeros. A formula that merely retains $Z$ whenever the
gate passes omits the clipping effect.

For $s_G>0$, put $U=(G-\gamma)/s_G$, $a=-\tau-\gamma/s_G$ and
$b=\tau-\gamma/s_G$. Conditional on $U=u$,

$$
Z\mid U=u\sim N(m(u),v^2),\qquad
m(u)=\theta+r s_Zu,\qquad v=s_Z\sqrt{1-r^2}.
$$

For $v>0$, the positive clipped first and second moments are

$$
M_+(m,v)=m\Phi(m/v)+v\phi(m/v),\qquad
Q_+(m,v)=(m^2+v^2)\Phi(m/v)+mv\phi(m/v).
$$

The negative moments are $M_-(m,v)=m-M_+(m,v)$ and
$Q_-(m,v)=m^2+v^2-Q_+(m,v)$. Their continuous limits apply at $v=0$.
Integrating positive moments over $u\ge b$ and negative moments over $u\le a$ against
$\phi(u)$ gives the exclusion estimate's first and second moments. Release adds the
unclipped moments over $a<u<b$. Bias, variance and mean squared error then follow from
$E[\widehat\theta]$, $E[\widehat\theta^2]$ and $\theta$. The implementation evaluates
deterministic integrals of these analytic moments; it does not use Monte Carlo to obtain
the reported risk. Truncated-normal moments have a classical literature, including
[Tallis (1961)](https://doi.org/10.1111/j.2517-6161.1961.tb00408.x); the gate-specific
partition and identities here are directly derived special cases.

Let $p=\Phi(b)-\Phi(a)$ be the weak-gate probability. Exclusion adds false-exclusion cost
$\theta^2p$ when the individual effect is nonzero. The exact difference between exclusion
and release risks is

$$
R_{\mathrm{exclude}}-R_{\mathrm{release}}
=\theta^2p-s_Z^2\left[p+r^2\left(a\phi(a)-b\phi(b)\right)\right].
$$

The active-gate clipping terms cancel in this difference because both policies use them.
They remain part of each policy's total risk. There is no universal sign of the difference:
the individual effect, pooled signal and gate/fit dependence all matter.

For a nonzero individual coefficient, the wrong-gate probability is the probability of the
opposite active direction: $\Phi(a)$ for $\theta>0$, and $1-\Phi(b)$ for $\theta<0$,
even when its pooled mean has another sign. False exclusion is the weak probability $p$
under exclusion. For a true zero, false admission means an active gate, with probability
$1-p$; it is not the probability that the final clipped estimate is nonzero. For $v>0$,
the probability of clipping is

$$
P_{\mathrm{clip}}=
\int_b^{\infty}\Phi(-m(u)/v)\phi(u)\mathrm{d}u+
\int_{-\infty}^{a}\Phi(m(u)/v)\phi(u)\mathrm{d}u.
$$

These distinctions prevent gate diagnostics from being misreported as support-recovery
results. The API also evaluates degenerate cases directly, including $v=0$ and a
deterministic gate.

### Financial loss and the scope of coordinatewise clipping

For fixed premiums $\pi_j$, expected component-attribution squared error is the sum of
$\pi_j^2R_j$. This identity does not require independent coefficient errors. Total
expected-return squared error also includes their cross moments and cannot generally be
obtained by summing marginal risks.

Coordinatewise clipping equals the unpenalized sign/zero-constrained least-squares solution
when the intercept-residualized weighted slope Gram matrix is diagonal. With correlated
factors the projection uses the Gram metric, so the coordinates interact. For example,
with Gram matrix having diagonal one and off-diagonal $0.9$, raw fit $(-1,1)$ and two
nonnegative restrictions, coordinate clipping gives $(0,1)$ whereas the constrained
least-squares solution is $(0,0.1)$. Group penalties add further coupling. The Gaussian
geometry above remains valid for the unconstrained WLS estimates, but its scalar clipped
risks must not be presented as exact HCGL/FCGL risks in such designs.

A useful bridge to correlated designs is a projection bound. For positive-definite Gram
matrix $Q$ and nonempty closed convex constraint set $C$, let $P_C$ be its projection in
the $Q$ norm, and let $d_Q(\beta,C)$ be the distance from the truth to that set. Then

$$
\Vert P_C(z)-\beta\Vert_Q
\leq \Vert z-\beta\Vert_Q+d_Q(\beta,C).
$$

Insert $P_C(\beta)$, apply the triangle inequality, and use the nonexpansiveness of the
projection to obtain this bound. It holds pointwise for a data-dependent set when the same
realized $C$ is held fixed in both projections. If the truth is feasible, the distance is
zero and projection cannot increase this Gram-metric error. This statement does not imply
factor-premium attribution or expected-return loss dominance; those losses generally use
different metrics. It also does not provide the exact risk of a penalized HCGL/FCGL fit.

## Worked example

The [canonical offline script](../examples/docs/conditional_pooled_signs_theory.py) checks
the geometry against an independent least-squares operator, checks scalar identities in
closed form, and integrates general risks in the reverse conditioning order. It uses NumPy
least squares and SciPy deterministic quadrature; no convex solver is required.

The design has 60 synthetic dates, four equally pooled responses and two factors with
correlation $0.9$. All four true loading rows are $(1,-0.5)$. The conditional pooled mean is
$(1,-0.5)$, while the marginal slopes are $(0.55,0.4)$. The second factor's marginal sign
therefore differs from its conditional sign. Residual correlation is $0.25$; pooling reduces
each coefficient's variance to $0.438$ of the individual variance, not one quarter.

```python
    geometry = compute_conditional_gaussian_geometry(x, beta, omega, pooling)
```

Two scalar checks illustrate improvement and failure. When $Z=G$ is standard normal,
threshold-one exclusion has risk $0.801$ versus the raw fit's risk of one. An independent
uninformative sign applied to a raw estimate with true mean two and standard deviation one
instead raises risk to $2.50$, even at threshold zero where there is no weak-gate exclusion.

```python
    exclusion = gaussian_sign_gate_risk(0., 0., 1., 1., 1., threshold=1., action='exclude')
    release = gaussian_sign_gate_risk(0., 0., 1., 1., 1., threshold=1., action='release')
```

```python
    clipping = gaussian_sign_gate_risk(2., 0., 1., 1., 0., threshold=0., action='exclude')
```

## Implementation in factorlasso

- `compute_conditional_gaussian_geometry` constructs the fixed-design WLS map and its exact
  Gaussian covariance, without fitting observed responses or deriving a production gate.
- `ConditionalGaussianGeometry` retains read-only coefficient means, covariance ingredients,
  individual/gate standard deviations and same-data correlations. Asset/factor inputs and
  pooling membership must be ordered consistently; the API uses NumPy arrays.
- `gaussian_sign_gate_risk` evaluates the specified scalar exclusion or release estimator,
  allowing a pooled `gate_mean` different from the individual `beta`. It accepts correlations
  at either endpoint, deterministic variances and an infinite threshold; invalid distribution
  parameters are rejected rather than repaired with a ridge or a marginal-regression fallback.
- `GaussianSignGateRisk` reports mean, second moment, bias, variance, MSE, raw-fit baseline
  MSE, positive/negative/weak-event MSE components, gate/clipping probabilities and a
  numerical integration-error estimate.

`probability_wrong_sign` describes the gate's direction relative to a nonzero individual
truth. It is zero by convention for a null truth. `probability_false_exclusion` is zero for
release and for a null truth. `probability_false_admission` is zero for nonzero truths.
`probability_clipped` records an active sign restriction setting a conflicting raw estimate
to zero; it excludes zeros caused solely by a weak gate.

## Interpretation and limitations

- Fixed pooling does not establish sign coherence. Heterogeneous groups may supply precise
  evidence against a member's true direction; the separate `beta` and `gate_mean` expose this.
- The normal calculation uses known distribution standard deviations, not estimated HC1 or
  HAC standard errors. At a centered null gate, threshold one activates a direction with
  probability $2\Phi(-1)$, approximately 31.7%. It is a regularizer, not a 5% test.
- The full scalar law retains gate/fit dependence, including use of the same observations.
  It does not account for groups or tuning parameters selected from those observations.
- A fixed EW span and capped history do not provide a growing-information asymptotic
  regime. Sign or support consistency requires additional conditions and is not asserted.
- A deterministic temporal transformation must also transform the residual covariance.
  Choosing `date_covariance=None` after recursive demeaning does not automatically represent
  the original transformed errors. Empirical HC1 calibration needs separate study.
- The functions neither assert universal risk reduction nor guarantee better realized
  returns. They provide a reproducible reference for identifying false-exclusion and
  sign-clipping mechanisms before analyzing a complete constrained estimator.

## See also

- [Gated cluster-pooled sign derivation](gated_cluster_pooled_signs.md): the existing marginal
  gate and its operational conventions.
- [Weighted regression and mean inference](weighted_inference.md): estimated HC/HAC statistics
  and their distinct inferential scope.
- [Group penalties](group_penalties_hcgl_fcgl.md): the coupled HCGL and FCGL objectives.
- [Cooperative LASSO](cooperative_lasso.md): a different sign-coherent regularizer.

## References

- Meinshausen, N. (2013). Sign-constrained least squares estimation for high-dimensional
  regression. *Electronic Journal of Statistics* 7, 1607–1631.
  [DOI 10.1214/13-EJS818](https://doi.org/10.1214/13-EJS818).
- Tallis, G. M. (1961). The Moment Generating Function of the Truncated Multi-Normal
  Distribution. *Journal of the Royal Statistical Society: Series B (Methodological)*
  23(1), 223–229. [DOI 10.1111/j.2517-6161.1961.tb00408.x](https://doi.org/10.1111/j.2517-6161.1961.tb00408.x).
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
