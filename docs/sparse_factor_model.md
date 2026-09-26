---
myst:
  html_meta:
    description: >-
      The sparse multi-output factor model in factorlasso: the weighted squared loss with an L1
      penalty on an N by M loading matrix, its optimality conditions, the two intercepts, the units
      of the penalty, and a worked example verified against coordinate descent and least squares.
---

# Sparse multi-output factor model

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-22](https://github.com/ArturSepp/factorlasso/commit/fe2063860f701ac5a71951161bf391a1285d503d)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

The sparse factor model regresses $N$ response series on $M$ candidate factors at once and
penalises the absolute size of every loading, so that loadings the data do not support are
estimated as zero. It is the base estimator of the package: sign constraints, priors, group
penalties and the covariance decomposition all act on the loading matrix defined here.

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
units. LASSO, group/cluster and cooperative models support this option, including
regularization paths and cross-validation. UniLasso retains its separate unweighted
two-stage objective and rejects the option.

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

## Implementation in factorlasso

All names below are exported from the top-level package and documented in the
[API reference](api.rst).

| Public name | Role |
|---|---|
| `LassoModel` | The estimator. `fit(x, y)` returns `self`; `predict(x)` returns $\hat\alpha + X\hat\beta^{\top}$; `score(x, y)` returns the mean $R^2$ over responses; `get_params` and `set_params` follow scikit-learn; `summary()` prints a one-screen description of a fitted model. |
| `LassoModelType` | Selects the penalty. `LASSO` is the cell-wise L1 penalty of this article; the other members are the group, cluster, cooperative and univariate-guided variants. |
| `LassoEstimationResult` | Solver output stored as `estimation_result_`: `estimated_beta`, `alpha` (the residual mean that becomes `intercept_`), and the weighted `ss_total`, `ss_res` and `r2` per response. |
| `solve_lasso_cvx_problem` | The CVXPY programme for NumPy inputs. `LassoModel` calls it after centring; it accepts `valid_mask`, `span`, `factors_beta_loading_signs`, `factors_beta_prior` and `penalty_weights`. |
| `get_x_y_np` | Converts the panels to the arrays the solver receives: centred, zero-filled where missing, with the validity mask. With an EWMA span the first row is dropped. |

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

`fit` raises `ValueError` when `x` and `y` do not share an index. A solve with fewer than five
rows, or one that ends without a solution, returns NaN loadings with a warning instead of an
exception. An error raised by the solver propagates unless `solver_fallbacks` names solvers to
try in turn.

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
- **Unequal histories are penalised unequally.** The loss of every response is divided by the
  same $T$, so a short history or a short EWMA span raises the effective penalty.
- **No exact zeros.** Count loadings with `effective_sparsity`, not with `!= 0`. `summary()`
  reports the bare count.
- **No standard errors.** The package reports point estimates. Inference after selection is a
  separate problem that the package does not address.

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

- Diamond, S., and Boyd, S. (2016). CVXPY: a Python-embedded modeling language for convex
  optimization. *Journal of Machine Learning Research* 17(83), 1-5.
- Goulart, P. J., and Chen, Y. (2024). Clarabel: an interior-point solver for convex conic
  programs. arXiv:2405.12762.
- Sepp, A., and Kastenholz, M. A. (2026). factorlasso: Sparse Multi-Output Regression with
  Cluster-Grouped Sign Constraints in Python. Submitted to the *Journal of Statistical
  Software*. [Manuscript](../papers/jss_2026/paper/article.pdf).
- Sepp, A., Ossa, I., and Kastenholz, M. (2026). Robust Optimization of Strategic and Tactical
  Asset Allocation for Multi-Asset Portfolios. *The Journal of Portfolio Management* 52(4),
  86-120.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal
  Statistical Society: Series B* 58(1), 267-288. DOI 10.1111/j.2517-6161.1996.tb02080.x.
- Zhao, P., and Yu, B. (2006). On model selection consistency of Lasso. *Journal of Machine
  Learning Research* 7, 2541-2563.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
