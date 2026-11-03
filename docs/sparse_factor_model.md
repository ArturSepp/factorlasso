---
myst:
  html_meta:
    description: >-
      The sparse multi-output factor model in factorlasso: the weighted squared loss with an L1
      penalty on an N by M loading matrix, its optimality conditions, the two intercepts, the units
      of the penalty, zero-centred and pooled-target ridge fits, and examples verified against
      independent least-squares and coordinate-descent calculations.
---

# Sparse multi-output factor model

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-22](https://github.com/ArturSepp/factorlasso/commit/fe2063860f701ac5a71951161bf391a1285d503d)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

The sparse factor model regresses $N$ response series on $M$ candidate factors at once and
penalises the absolute size of every loading, so that loadings the data do not support are
estimated as zero. It is the base estimator of the package: sign constraints, priors, group
penalties and the covariance decomposition all act on the loading matrix defined here.
The same estimator interface also provides a dense ridge baseline with a squared-L2 penalty.
The separate `fit_pooled_loading_shrinkage` function shrinks joint loading estimates toward
supplied response groups' mean WLS loadings, with asset-specific unpenalized intercepts.

## Overview

A time-series factor model explains each response by a small number of common factors,

$$
Y_t = \alpha + \beta X_t + \varepsilon_t .
$$

With a broad candidate factor set most entries of the $N \times M$ matrix $\beta$ are zero: a
government bond fund does not load on commodities. Ordinary least squares does not know that. It
returns a small non-zero estimate for every cell, and those estimates are noise that enters every
later use of $\beta$: risk attribution, hedging, and the covariance matrix
$\beta \Sigma_x \beta^{\top} + D$.

The LASSO of Tibshirani (1996) adds the penalty $\lambda \lVert \beta \rVert_1$ to the squared
loss. The penalty has a kink at zero, so the minimiser sets a loading exactly to zero unless the
correlation of its factor with the current residual exceeds a threshold proportional to
$\lambda$. The estimator trades a shrinkage bias on the loadings it keeps for the removal of the
loadings it should not keep. The worked example measures that trade: on 60 months of data the
loading error falls from 0.074 for least squares to 0.045 at the best penalty.

The article answers four questions.

1. Which optimisation problem does `LassoModel` solve, and in which units is `reg_lambda`?
2. How can a fitted $\hat\beta$ be verified to be the minimiser?
3. Which of the two fitted intercepts is the regression intercept?
4. What does the penalty cost and buy along a grid of penalties?

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| `x`, $X$ | Factor panel | $T \times M$; rows are periods, columns are factors; same return units as `y` |
| `y`, $Y$ | Response panel | $T \times N$ on the same index as `x`; missing values allowed |
| $\beta$, `coef_` | Loading matrix | $N \times M$, responses by factors; dimensionless when `x` and `y` share units |
| $\alpha$, `alpha_const_` | Regression intercept | Length $N$, in the units of `y` per period |
| $\lambda$, `reg_lambda` | Penalty strength | Units of `y` squared; default $10^{-5}$, calibrated for decimal returns |
| $W$, `span` | Observation weights | `span=None` gives unit weights; an EWMA span gives $w_t^2 = \lambda_w^{T-t}$ with $\lambda_w = 1 - 2/(\mathrm{span}+1)$ |
| `demean` | Centring of `x` and `y` before the solve | Default `True`: sample means for `span=None`, running EWMA means otherwise |
| $V$ | Validity mask | $V_{ti} = 1$ when $Y_{ti}$ is observed, else 0; multiplies $W$ |
| `warmup_period` | Minimum valid observations per response | Default 12; a response with fewer has its loadings set to zero and its diagnostics to NaN |
| `solver` | CVXPY solver | Default CLARABEL, an interior-point method |

The model assumes a linear relation with loadings constant over the estimation window, and
factor observations in every row used. It assumes nothing about the distribution of
$\varepsilon_t$ for estimation. Rows are weighted but not reordered, so the estimator has no
look-ahead beyond the window passed to `fit`.

## Methodology

### Objective

For centred panels $\tilde X$ and $\tilde Y$, `LassoModelType.LASSO` solves

$$
\hat\beta = \arg\min_{\beta}
\frac{1}{T} \lVert W \odot V \odot (\tilde X \beta^{\top} - \tilde Y) \rVert_F^2 +
\lambda \lVert \beta - \beta_0 \rVert_1 ,
$$

where $\odot$ is the element-wise product, $\lVert \cdot \rVert_F$ the Frobenius norm,
$\lVert \cdot \rVert_1$ the sum of absolute values over all $NM$ cells, and $\beta_0$ a prior
loading matrix that defaults to zero. Optional sign constraints restrict $\beta$ cell by cell.
Priors and signs are the subject of
[sign constraints and priors](sign_constraints_and_priors.md); this article takes
$\beta_0 = 0$ and no constraints.

Three conventions in this objective differ from other LASSO software and matter when a penalty
is carried over.

The loss is a mean, not a half mean. scikit-learn minimises
$\frac{1}{2T} \lVert y - Xw \rVert_2^2 + a \lVert w \rVert_1$, so a penalty $\lambda$ here
corresponds to $a = \lambda / 2$ there. On the worked example the two agree to $2 \times 10^{-5}$
in every loading with scikit-learn 1.9; the check is not part of the canonical script because
scikit-learn is not a dependency of the package.

The penalty is not scale-free. Multiplying `x` and `y` by $c$ multiplies the loss by $c^2$ and
leaves $\beta$ and the penalty term unchanged, so the same solution needs $c^2 \lambda$. A
penalty tuned on decimal returns must be multiplied by $10^4$ for returns in percent. The
worked example verifies this.

The divisor is the number of rows $T$ for every response. A response observed on a fraction $f$
of the rows has a loss smaller by about the factor $f$ against the same penalty, and is shrunk
more. The same holds for a short EWMA span, whose squared weights sum to about
$(\mathrm{span}+1)/2$ instead of $T$.

### Optional normalization by valid weight mass

The preceding equations and the worked example use the historical default
`loss_normalization="sample"`. Setting `loss_normalization="weight_sum"` instead
normalizes each response separately. Define $S_i = \sum_t W_{ti}^2 V_{ti}$ and let
$e_{ti}$ be its regression residual. The fitting loss becomes

$$
L(\beta) = \sum_{i:S_i>0} \frac{\sum_t W_{ti}^2 V_{ti} e_{ti}^2}{S_i}.
$$

An empty response contributes zero loss. The loss remains a sum over responses;
it is not additionally divided by their number. Missing observations add no weight,
and multiplying all observation weights for a response by a constant leaves its
loss unchanged. Invariance concerns the loss for fixed transformed inputs: changing
centring, target estimation, sign rules or cluster composition can independently
change a complete fit. Short histories still carry greater estimation uncertainty;
normalizing the loss does not supply additional information.

For a balanced panel with common weight mass $S$, equivalent penalties satisfy

$$
\lambda_{\mathrm{normalized}} = \lambda_{\mathrm{sample}} T/S.
$$

The [EWMA weighting and ragged histories](ewma_weighting_and_ragged_histories.md) article
measures the extra shrinkage of short histories under each convention and checks this
conversion and the closed-form shrinkage of a one-factor fit.

With unequal masses, the relative response weights change and a single lambda
conversion cannot preserve every old fit. Calibration must therefore accompany
activation. `loss_weight_mass_`, `loss_denominator_` and `n_loss_rows_` record the
fitted convention's inputs. Diagnostic residual sums retain their original EWMA
units. LASSO, ridge, group/cluster and cooperative models support this option and
cross-validation; group/cluster models also provide compiled regularization paths. UniLasso
retains its separate unweighted two-stage objective and rejects the option.

### Ridge baseline

`LassoModelType.RIDGE` replaces the L1 penalty by the
squared Frobenius norm (Hoerl and Kennard 1970):

$$
\widehat\beta = \arg\min_{\beta}
\sum_{i:S_i>0}\frac{\sum_t W_{ti}^2 V_{ti}
(\widetilde X_t\beta_i-\widetilde Y_{ti})^2}{d_i} +
\lambda\lVert\beta\rVert_F^2.
$$

Here $d_i=T$ for `loss_normalization="sample"` and $d_i=S_i$ for `"weight_sum"`.
The masks, observation weights and centring conventions are the same as for LASSO. This is
ordinary zero-centred, unconstrained ridge: signs, forced zeros, prior targets, adaptive
penalties and grouping options are rejected when enabled. These restrictions keep the baseline
distinct from the package's directional and structured penalties. The default model remains
LASSO. A group L2 norm is not a squared-L2 ridge penalty, and setting `l1_weight=0` in a group
model does not select ridge.

For positive finite `reg_lambda`, ridge continuously shrinks coefficients and generally retains
all factors. Zero penalty gives weighted minimum-norm least squares; positive infinity gives
exactly zero coefficients. No sparsity threshold or LASSO `lambda_max` is applied to ridge.
The intercept is reconstructed after fitting and is not penalised. With a positive EW span,
the package's running-mean transformation is preserved; it is not replaced by joint estimation
of a single weighted intercept. No automatic factor standardisation is added: any training-only
scaling belongs to the caller's study protocol.

On identical transformed inputs, scikit-learn's unnormalised ridge objective matches when its
penalty is `alpha = d_i * reg_lambda` and its sample weights are $W_{ti}^2 V_{ti}$.
For ragged responses under weight-mass normalisation, $d_i$ can differ by response. Scikit-learn
is only an optional external reference; it is not a factorlasso runtime dependency.

The equal-weight formula below is a mathematical characterization, not a prescription to form
an explicit matrix inverse:

$$
\widehat\beta^\top =
(\widetilde X^\top\widetilde X/T+\lambda I)^{-1}
\widetilde X^\top\widetilde Y/T,\qquad \lambda>0.
$$

Ridge provides a comparison for generic coefficient shrinkage under correlated factors. It does
not impose shared directions or establish that its attribution is more accurate. Its penalty
must be selected independently on training data; copying a LASSO penalty is not a fair tuning
procedure merely because the configuration parameter has the same name.

### Squared shrinkage toward pooled joint loadings

`fit_pooled_loading_shrinkage` is an explicit
complete-panel fit. Let $B_0$ be the individual joint weighted least-squares loading matrix and
let $G_i$ be the average of its rows in response $i$'s supplied group. With strictly positive
date weights $w_t$ normalized to mean one, it fits

$$
\widehat B=\arg\min_B
\frac{1}{T}\sum_t w_t\lVert\widetilde Y_t-B\widetilde X_t\rVert_2^2
+\lambda\lVert B-G\rVert_F^2,\qquad
\widehat\alpha=\bar y_w-\widehat B\bar x_w.
$$

Here both panels are centered using their fixed training-window weighted means. The intercept
is asset-specific and unpenalized; it is not pooled with other assets. All factors enter the
joint WLS target. The automatic selected-factor OLS prior supplies a different target, and
passing a target to an L1/group penalty does not turn that penalty into squared shrinkage.
Existing `LassoModelType.RIDGE` remains zero-centred and rejects prior/sign inputs.

The fit reuses the ridge solver on the centered response minus its pooled-target prediction,
then adds the target back. Zero penalty returns individual joint WLS exactly; positive infinity
returns the pooled target exactly with reconstructed intercepts. The full weighted-centered
design must have rank $M$ even at a positive or infinite penalty, because the WLS target must be
identified. Thus $T\ge M+1$ is allowed, including the boundary where an HC1 gate has no residual
degrees of freedom. Missing observations are rejected. No sign cone or factor standardization
is implicit, and changing coordinates can change pooling and fixed-penalty shrinkage.

The result exposes the target, individual WLS, normalized weights, means, objective and numerical
diagnostics. At finite penalty the independently recomputed scaled stationarity residual must
not exceed $10^{-7}$; an invalid numerical result raises. This threshold is not a coefficient
error bound for ill-conditioned designs. At infinity the finite-penalty objective and
stationarity are `None`, while the distance from the target is exactly zero. Default solver and
optional fallback choices follow the existing ridge solver. No new statistical distribution or
forecast-dominance theorem is implied by this target-centred adaptation.

### Separability

With no group penalty the objective is a sum over responses,

$$
\sum_{i=1}^{N} \left[ \frac{1}{T} \lVert w_i \odot (\tilde X \beta_i - \tilde y_i) \rVert_2^2 +
\lambda \lVert \beta_i \rVert_1 \right],
$$

where $\beta_i$ is row $i$ of $\beta$ as a column vector and $w_i$ the weight column of response
$i$. The joint fit therefore equals $N$ single-response fits. The matrix form exists because the
[group penalties](group_penalties_hcgl_fcgl.md) couple the rows.

### Optimality conditions

The objective is convex. For unit weights, $\hat\beta$ is a minimiser if and only if the
gradient of the loss,

$$
G = \frac{2}{T} (\tilde X \hat\beta^{\top} - \tilde Y)^{\top} \tilde X \in \mathbb{R}^{N \times M},
$$

satisfies, for every cell,

$$
G_{ij} = -\lambda \operatorname{sign}(\hat\beta_{ij}) \quad \text{if } \hat\beta_{ij} \neq 0,
\qquad
\lvert G_{ij} \rvert \le \lambda \quad \text{if } \hat\beta_{ij} = 0 .
$$

These Karush-Kuhn-Tucker conditions give two results used below. A loading is non-zero only if
the correlation of its factor with the residual reaches the threshold, which is how the penalty
selects. Every loading is zero when $\lambda$ reaches

$$
\lambda_{\max} = \max_{i,j} \left\lvert \frac{2}{T} (\tilde Y^{\top} \tilde X)_{ij} \right\rvert ,
$$

which bounds the useful range of a penalty grid from above.

### Centring and the two intercepts

The solver fits no intercept. `fit` centres both panels and solves for $\beta$ alone. It then
reports two different quantities.

`alpha_const_` is the regression intercept in the units of `y`, reconstructed with the weights of
the loss,

$$
\hat\alpha = \bar y_w - \hat\beta \bar x_w ,
$$

where $\bar y_w$ and $\bar x_w$ are sample means for `span=None` and EWMA-weighted means
otherwise. `predict` returns $\hat\alpha + X \hat\beta^{\top}$.

`intercept_` is the weighted mean of the residuals of the centred problem. For `span=None` it is
zero to machine precision by construction. With an EWMA span it is the small leftover of
one-sided running means. It is kept for backward compatibility and is not an estimate of
$\alpha$.

### Solver and numerical zeros

The problem is passed to CVXPY (Diamond and Boyd 2016) and solved by CLARABEL (Goulart and Chen
2024), an interior-point method. Interior-point iterates approach the boundary without reaching
it, so a loading that is zero at the optimum is returned as a small number. In the worked example
those numbers are below $2 \times 10^{-5}$, while the smallest kept loading is
$3.5 \times 10^{-3}$. A count of kept loadings therefore needs a tolerance.
`effective_sparsity` applies one and `suggest_tolerance` locates the gap; both are described in
[residual diagnostics](residual_diagnostics.md).

## Worked example

The example uses synthetic data with a fixed seed. Six monthly return series load on three of
eight candidate factors: ten of the 48 cells of the generating matrix are non-zero. Factor
returns have a volatility of 0.04 per month and idiosyncratic returns 0.02 per month, in decimal
units. The generating intercepts lie between -0.001 and 0.003 per month. The first 60 of 120
months are used for estimation and the last 60 only for held-out $R^2$.

The model is the default estimator with its arguments written out:

```python
def fit_lasso(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float = REG_LAMBDA) -> fl.LassoModel:
    """Fit the cell-wise L1 model with uniform observation weights and sample-mean centring."""
    model = fl.LassoModel(
        model_type=fl.LassoModelType.LASSO,
        reg_lambda=reg_lambda,
        span=None,                         # uniform weights; an EWMA span would discount old rows
        demean=True,                       # centre x and y, so no intercept enters the programme
    )
    return model.fit(x=x, y=y)
```

### The fit is the minimiser of the stated objective

At `reg_lambda` $= 10^{-3.5} \approx 3.2 \times 10^{-4}$ the fit keeps 13 of 48 loadings: all
ten generating loadings and three false ones. The reference is a coordinate-descent solver
written in NumPy for the same objective. Each update is the closed-form soft-threshold
$\beta_j \leftarrow S(\rho_j, \lambda/2) / z_j$ with $\rho_j$ the mean product of factor $j$ and
the partial residual, $z_j$ the mean square of factor $j$, and
$S(\rho, \tau) = \operatorname{sign}(\rho) \max(\lvert \rho \rvert - \tau, 0)$:

```python
def lasso_coordinate_descent(
    x: np.ndarray,
    y: np.ndarray,
    reg_lambda: float,
    n_sweeps: int = 2000,
) -> np.ndarray:
    """Minimise ``(1/T) ||x b - y||^2 + reg_lambda ||b||_1`` for one centred response."""
    n_obs, n_factors = x.shape
    b = np.zeros(n_factors)
    scale = np.sum(x ** 2, axis=0) / n_obs
    for _ in range(n_sweeps):
        previous = b.copy()
        for j in range(n_factors):
            partial_residual = y - x @ b + x[:, j] * b[j]
            rho = x[:, j] @ partial_residual / n_obs
            b[j] = np.sign(rho) * max(abs(rho) - 0.5 * reg_lambda, 0.0) / scale[j]
        if np.max(np.abs(b - previous)) < 1e-14:
            break
    return b
```

| Check | Result |
|---|---|
| Largest difference between `coef_` and the coordinate-descent solution | $1.9 \times 10^{-5}$ |
| Difference of the two objective values | below $10^{-9}$ |
| Support of `coef_` above $10^{-3}$ against the exact support of the reference | identical, 13 cells |
| Gradient on kept cells against $-\lambda \operatorname{sign}(\hat\beta_{ij})$ | equal within $10^{-6}$ |
| Largest absolute gradient on zeroed cells | $3.08 \times 10^{-4}$, below $\lambda = 3.16 \times 10^{-4}$ |
| Joint fit against six single-response fits | equal within $3 \times 10^{-4}$ |
| `reg_lambda` $= 10^{-12}$ against `numpy.linalg.lstsq` with an intercept | equal within $10^{-6}$ |
| `reg_lambda` $= 1.01 \lambda_{\max}$, with $\lambda_{\max} = 3.9 \times 10^{-3}$ | every loading below $10^{-3}$ |
| Percent returns with $10^{4} \times$ `reg_lambda` against decimal returns | equal within $10^{-4}$ |
| `alpha_const_` against $\bar y - \hat\beta \bar x$ | equal |
| Largest absolute `intercept_` | $4 \times 10^{-18}$ |

The optimality conditions are checked directly on the fitted matrix:

```python
gradient = 2.0 / N_TRAIN * (x_centred @ beta.T - y_centred).T @ x_centred      # (N, M)
kept = np.abs(beta) > ZERO_TOLERANCE
assert np.allclose(gradient[kept], -REG_LAMBDA * np.sign(beta[kept]), atol=1e-6)
assert np.all(np.abs(gradient[~kept]) <= REG_LAMBDA * (1.0 + 1e-6))
```

The agreement with the reference is $10^{-5}$ in the loadings and $10^{-9}$ in the objective.
The objective is flat near its minimum, so an interior-point solution that is optimal to solver
tolerance can differ from the exact minimiser in the fifth decimal of a loading. Comparisons of
loadings across solvers, or between a joint and a single-response fit, should allow for that.

### What the penalty costs and buys

[![Two panels. Left: the eight loadings of one response along the penalty grid; equity and credit rise from zero toward their true values 0.8 and 0.4 as the penalty falls, and six irrelevant factors leave zero only at small penalties. Right: loading error of all 48 loadings along the grid, with a minimum of 0.045 below the least-squares level of 0.074.](images/sparse_factor_model_path.png)](images/sparse_factor_model_path.png)

**Figure 1.** Synthetic teaching exhibit. The penalty falls to the right in both panels. Left:
the eight estimated loadings of `asset_2`, whose generating loadings are 0.8 on equity and 0.4 on
credit (blue circles, with the true values dashed) and zero on six other factors (orange
squares). Right: root mean squared error of all 48 estimated loadings against the generating
matrix, estimated on 60 months; the dashed line is ordinary least squares on the same rows.
Select the image for the full-resolution view.

| `reg_lambda` | Kept | False | Loading RMSE | $R^2$ in sample | $R^2$ held out |
|---|---|---|---|---|---|
| $10^{-2}$ | 0 | 0 | 0.301 | 0.000 | -0.116 |
| $10^{-3}$ | 10 | 0 | 0.143 | 0.565 | 0.456 |
| $5.6 \times 10^{-4}$ | 11 | 1 | 0.083 | 0.691 | 0.572 |
| $3.2 \times 10^{-4}$ | 13 | 3 | 0.053 | 0.736 | 0.605 |
| $1.8 \times 10^{-4}$ | 20 | 10 | 0.045 | 0.756 | 0.611 |
| $10^{-4}$ | 29 | 19 | 0.049 | 0.767 | 0.606 |
| $10^{-5}$ | 48 | 38 | 0.071 | 0.776 | 0.581 |
| $10^{-6}$ | 48 | 38 | 0.074 | 0.776 | 0.576 |

The path shows the two regimes of the estimator. At $10^{-3}$ the fit keeps exactly the ten
generating loadings, but it shrinks them: the equity loading of `asset_2` is 0.55 against a true
0.8, and the loading error of 0.143 is twice that of least squares. As the penalty falls the
kept loadings approach their least-squares values and false loadings enter. The loading error
is smallest, 0.045, at $1.8 \times 10^{-4}$, where the held-out $R^2$ also peaks. That model
keeps ten false loadings: the penalty that predicts best is smaller than the penalty that
recovers the support. At $10^{-6}$ the fit is least squares, with a loading error of 0.074 and
all 38 irrelevant cells non-zero.

In-sample $R^2$ rises monotonically as the penalty falls and cannot be used to choose it.

### Independent ridge reference

The canonical example also fits ridge to the same training panel and compares its coefficients
with an augmented least-squares problem. The reference appends $\sqrt{T\lambda}I$ to the centred
factor matrix and zero rows to the responses, then calls `numpy.linalg.lstsq`. It also checks
the ordinary least-squares and infinite-penalty endpoints, without changing the LASSO figure
or its reported numerical results.

```python
def fit_ridge(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float = REG_LAMBDA) -> fl.LassoModel:
    """Fit the zero-centred ridge baseline on the same uniformly weighted training panel."""
    return fl.LassoModel(
        model_type=fl.LassoModelType.RIDGE,
        reg_lambda=reg_lambda,
        span=None,
        demean=True,
        loss_normalization="sample",
    ).fit(x=x, y=y)
```

### Pooled fit and an explicitly supplied expected-return forecast

The same synthetic training panel is also fitted with three supplied response groups and
unequal positive date weights:

```python
groups = pd.Series([0, 0, 1, 1, 2, 2], index=y.columns)
weights = pd.Series(np.linspace(0.5, 1.5, len(x)), index=x.index)
fitted = fl.fit_pooled_loading_shrinkage(
    x, y, groups, REG_LAMBDA, weights=weights,
)
```

The canonical script constructs the joint WLS target independently and augments the weighted
least-squares system with the square-root penalty times the identity and target. It checks the
returned loadings, intercepts and infinite endpoint against that reference. These are synthetic
identities; the groups and penalty are illustrative, not empirical model-selection results.

For an expected-return forecast, premiums and admitted alpha are separate inputs. Here all
inputs are one-month decimal excess returns, the premiums are supplied for illustration, and
the policy admits zero residual alpha:

```python
# Supplied one-month excess-return premiums; admit no residual alpha to the forecast.
premium = pd.Series([0.003, 0.001, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0], index=x.columns)
admitted_alpha = pd.Series(0.0, index=y.columns)
assert premium.index.equals(fitted.coefficients.columns)
assert admitted_alpha.index.equals(fitted.coefficients.index)
forecast = fitted.coefficients.dot(premium) + admitted_alpha
```

The script verifies this forecast against the independent coefficient reference. Replacing the
supplied premium by observed factor returns would instead be factor-return reconstruction.
Neither this fit nor `LassoModel.predict` estimates a premium history or decides which regression
alpha an investment mandate should admit. Such choices require an explicit information cutoff,
horizon and unit convention.

### Select within the available history, refit and forecast

The final part of the same offline script is a complete, bounded forecasting recipe. It uses
the first 60 synthetic months available at an explicit decision date. Rows after that date do
not enter preparation, selection or the final fit. Every input return and supplied premium is
a one-month decimal excess return. The supplied alpha has the same horizon; no cash return or
annualization is added.

Within the 60-month history, training prefixes of 30, 40 and 50 observations each precede a
ten-observation validation block. Each prefix has its own calendar-month EW weights with span
24, weighted means and scales, response correlations, CHORD/Ward tree, group cut, conditional
gate and joint-WLS target. The cut is 0.6 times the final Ward merge height; it is not the core
clustering default's fraction of maximum pairwise distance. The example keeps gate and penalty
groups equal explicitly. The gate is inspectable, but it does not impose restrictions on the
pooled-shrinkage comparator.

Pooled shrinkage is fitted on standardized inputs at the illustrative penalties 0, 0.1, 1 and
positive infinity. For each candidate, validation loss is the mean squared raw reconstruction
error divided response by response by that prefix's squared training scale. All validation
blocks must have finite losses. A failed numerical fit invalidates the candidate, and the
largest penalty within $10^{-12}$ of the smallest average loss is selected:

```python
def select_forecast_penalty(losses: pd.DataFrame) -> float:
    """Require every blocked score; choose the largest penalty within 1e-12 of the minimum."""
    complete = np.isfinite(losses.to_numpy()).all(axis=1)
    if not np.any(complete):
        raise RuntimeError("no candidate has a finite score in every validation block")
    average = losses.loc[complete].mean(axis=1)
    return float(max(average.index[average <= average.min() + 1e-12]))
```

This criterion tunes factor-return reconstruction, not expected-return error. The small
standardized grid is a teaching choice, not a replacement for any frozen paper grid.
`LassoModelCV` uses different folds, scoring and tie/failure rules; supplying a full-sample gate
or target to it would not rebuild those objects inside each fold.

After selection, the recipe rebuilds every training-derived quantity on all 60 eligible months
and refits the selected penalty. If $s_x$ and $s_y$ are the saved training scales, analysis-unit
loadings are converted as $\hat B_{ij}=s_{y,i}\hat B^*_{ij}/s_{x,j}$, and the raw regression
intercept is $\hat\alpha_i=\bar y_i-\hat B_i\bar x$. The forecast admits the separately supplied
alpha rather than automatically using this regression intercept:

```python
decision_date = x.index[N_TRAIN - 1]
premium = pd.Series([0.003, 0.001, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0], index=x.columns)
admitted_alpha = pd.Series(0.0, index=y.columns)  # distinct from the regression intercept
result = forecast_recipe(x, y, decision_date, premium, admitted_alpha,
                         premium_asof=decision_date, alpha_asof=decision_date)
```

`forecast_recipe` is an example-local function, not a package export. Its result exposes the
validation-loss table, selected penalty, original-unit loadings and regression intercept,
final groups, gate statistics, pooled target and expected-return forecast. Premium/alpha labels
must match the factor/response order exactly; duplicate, reordered, missing or nonfinite inputs
are rejected. Their declared information dates cannot follow the decision date. The caller
still owns the truth of those availability declarations and the common-horizon convention.

For comparison, the same final training window is fitted with conditional HCGL at a fixed
illustrative penalty of 0.1, using the explicit gate restrictions with automatic signs disabled.
The inputs are square-root preweighted, and centering inside `LassoModel` is disabled. The raw
loadings are recovered before multiplying by the supplied premium. This is a separate fixed
penalty composition, not a selected paper HCGL procedure. Default solver output does not
guarantee the paper's stricter stationarity acceptance; its separate auditor/refinement remains
necessary. The recipe does not estimate a premium history, execute a portfolio or compare
empirical investment performance.

The script checks the selected pooled fit against independently augmented least squares.
Additional tests independently reconstruct every fold's weighted regression, Ward groups,
validation scores and final forecast, and verify the behavior under a failed candidate,
near ties, altered validation outcomes and data after the decision date. An injected
full-history preparation error fails those independent score checks.

## Implementation in factorlasso

Implementation and canonical example verified with factorlasso 1.0.0 on 2026-10-03.
The ridge mode and pooled-loading function are opt-in additions; existing LASSO defaults
remain unchanged.

All names below are exported from the top-level package and documented in the
[API reference](api.rst).

| Public name | Role |
|---|---|
| `LassoModel` | The estimator. `fit(x, y)` returns `self`; `predict(x)` returns $\hat\alpha + X\hat\beta^{\top}$; `score(x, y)` returns the mean $R^2$ over responses; `get_params` and `set_params` follow scikit-learn; `summary()` prints a one-screen description of a fitted model. |
| `LassoModelType` | Selects the penalty. `LASSO` is the cell-wise L1 penalty; `RIDGE` is the zero-centred squared-L2 baseline; the other members are the group, cluster, cooperative and univariate-guided variants. |
| `LassoEstimationResult` | Solver output stored as `estimation_result_`: `estimated_beta`, `alpha` (the residual mean that becomes `intercept_`), and the weighted `ss_total`, `ss_res` and `r2` per response. |
| `solve_lasso_cvx_problem` | The CVXPY programme for NumPy inputs. `LassoModel` calls it after centring; it accepts `valid_mask`, `span`, `factors_beta_loading_signs`, `factors_beta_prior` and `penalty_weights`. |
| `get_x_y_np` | Converts identically indexed pandas panels to the arrays the solver receives: centred, zero-filled where missing, with the validity mask. With EWMA demeaning the first row is dropped. |
| `fit_pooled_loading_shrinkage` | Standalone complete-panel weighted fit toward supplied response groups' mean joint WLS loadings, with squared target distance and an unpenalized intercept. |
| `PooledLoadingShrinkageResult` | Owned `coefficients`, `pooled_target` and `individual_wls` frames; `intercept`, `target_intercept`, `x_mean`, `y_mean`, copied groups/date weights, penalty/rank and numerical diagnostics. Its pandas contents remain mutable. |

The standalone fit accepts only real finite DataFrames `x` and `y`, including a one-column
frame for one response. Their unique nonmissing row indexes must match exactly in order;
column labels must also be unique/nonmissing. `groups` is a response-indexed Series of
nonmissing strings or finite nonboolean real IDs, and optional `weights` is a strictly positive
finite Series on the same row index. The helper performs no alignment, row dropping, grouping
or standardization. Build every training-derived group, scale and target inside its own fitting
prefix; a static matrix supplied to existing CV is not a per-fold callback. The fit does not
change `LassoModelCV` scoring, tie rules or failure handling.

The constructor parameters explained in this article are `model_type` (default
`LassoModelType.LASSO`, which selects the penalty), `reg_lambda`, `demean`, `solver` and
`solver_fallbacks`. The [API reference](api.rst) maps every other `LassoModel` parameter to
the article that explains it.

Fitted attributes carry a trailing underscore: `coef_`, `alpha_const_`, `intercept_`,
`estimation_result_`, and the references `x_` and `y_` to the fitted panels. The low-level path
reproduces the estimator:

<!-- fragment -->
```python
import factorlasso as fl

x_np, y_np, valid_mask = fl.get_x_y_np(x=x, y=y, span=None)
result = fl.solve_lasso_cvx_problem(x=x_np, y=y_np, valid_mask=valid_mask, reg_lambda=3e-4)
result.estimated_beta          # equals LassoModel(reg_lambda=3e-4).fit(x=x, y=y).coef_
```

This fragment assumes factor and response panels `x` and `y`. The runnable version is the
canonical script
[examples/docs/sparse_factor_model.py](../examples/docs/sparse_factor_model.py),
which needs only the core dependencies, runs offline, and asserts every number in the tables
above:

```console
python examples/docs/sparse_factor_model.py
```

`fit` requires equal row counts and rejects mismatched labelled indexes. An unlabelled input
adopts the other input's index when lengths agree; see [alignment conventions](conventions.md).
If an exception escapes, all fitted attributes revert to their values before the call. A
LASSO low-level solve with fewer than five rows, or one that ends without a solution and without
raising, returns NaN loadings with a warning. Warmup handling may subsequently zero short-history
responses. Solver exceptions propagate unless `solver_fallbacks` names solvers to try in turn;
exhausting that chain raises a solver error.

The numbers in this article were produced with factorlasso 0.20.0.dev2, CVXPY 1.9 and CLARABEL on
Python 3.12. Figure 1 is regenerated from the same script by the documentation analytics runner
described in the [documentation standard](documentation_standard.md); its
[provenance record](images/analytics_manifest.json) holds the configuration, the source identity
and the hash of the image.

## Interpretation and limitations

- **Shrinkage bias.** Kept loadings are biased toward zero, by $\lambda/2$ divided by the mean
  square of the factor for an orthogonal design. When the support matters more than the
  magnitudes, refit the kept cells without a penalty; the
  [group penalties](group_penalties_hcgl_fcgl.md) article shows how with a sign matrix.
- **Ridge is dense.** Ridge shrinks magnitudes without selecting factor support or enforcing
  economic signs. A smaller coefficient change need not imply better premium attribution or
  portfolio decisions; compare those outcomes separately from fitted-return error.
- **Pooled shrinkage depends on the target.** A heterogeneous group can introduce bias, and
  changing analysis units changes the target and penalty geometry. A target-centred fit is not
  evidence that group members share signs. Neither improved reconstruction nor a small
  stationarity residual establishes more accurate expected-return forecasts.
- **Selection is not guaranteed.** The LASSO recovers the true support only under conditions on
  the factor correlations (Zhao and Yu 2006). With correlated factors it can keep the wrong one
  of a pair. The worked example uses uncorrelated factors;
  [sign constraints and priors](sign_constraints_and_priors.md) treats an 85% correlated pair.
- **Prediction and support need different penalties.** The penalty with the best held-out $R^2$
  keeps false loadings. `LassoModelCV` selects by held-out $R^2$ and `LassoModelDiagonalityCV`
  by the diagonality of held-out residuals; both are described in
  [penalty selection](penalty_selection.md).
- **`reg_lambda` is in squared return units.** The default of $10^{-5}$ suits decimal returns of
  monthly to quarterly volatility. Rescale it by $c^2$ when the data are rescaled by $c$, and
  expect a different value for daily data.
- **The default penalises unequal histories unequally.** Under `loss_normalization="sample"`,
  the loss of every response is divided by the same $T$, so a short history or a short EWMA span
  raises the effective penalty. The optional `"weight_sum"` convention normalises each
  response's valid weight mass and changes the penalty scale.
- **No exact zeros.** Count loadings with `effective_sparsity`, not with `!= 0`. `summary()`
  reports the bare count.
- **No standard errors after LASSO selection.** The fitted penalised coefficients have no
  post-selection inference here. The [prior targets](prior_targets.md) and
  [prior inference](prior_inference.md) tools report uncertainty for selected OLS statistics
  and specified Gaussian prior models; they do not supply LASSO confidence intervals.

## See also

- [Quickstart](quickstart.md) for the estimator inside the full workflow.
- [Sign constraints and priors](sign_constraints_and_priors.md) for $\mathcal{C}$ and $\beta_0$.
- [Group penalties: HCGL and FCGL](group_penalties_hcgl_fcgl.md) for penalties that couple
  responses.
- [Residual diagnostics](residual_diagnostics.md) for counting kept loadings and testing what
  the factors leave behind.
- [Factor covariance assembly](factor_covariance_assembly.md) for the use of $\hat\beta$ in
  $\Sigma_y$.
- [API reference](api.rst) for signatures.

## References

- Hoerl, A. E., and Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal
  Problems. *Technometrics*, 12(1), 55–67.
  [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634).
- Diamond, S., and Boyd, S. (2016). CVXPY: a Python-embedded modeling language for convex
  optimization. *Journal of Machine Learning Research* 17(83), 1-5.
- Goulart, P. J., and Chen, Y. (2024). Clarabel: an interior-point solver for convex conic
  programs. arXiv:2405.12762.
- Sepp, A., and Kastenholz, M. A. (2026). factorlasso: Sparse Multi-Output Regression with
  Cluster-Grouped Sign Constraints in Python. Submitted to the *Journal of Statistical
  Software*. Manuscript retained locally; not distributed in this checkout.
- Sepp, A., Ossa, I., and Kastenholz, M. (2026). Robust Optimization of Strategic and Tactical
  Asset Allocation for Multi-Asset Portfolios. *The Journal of Portfolio Management* 52(4),
  86-120.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal
  Statistical Society: Series B* 58(1), 267-288. DOI 10.1111/j.2517-6161.1996.tb02080.x.
- Zhao, P., and Yu, B. (2006). On model selection consistency of Lasso. *Journal of Machine
  Learning Research* 7, 2541-2563.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
