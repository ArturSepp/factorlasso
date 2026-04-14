# 🚀 **Sparse Factor Model Estimation: factorlasso**

> `factorlasso` package implements sign-constrained LASSO, prior-centered regularisation, and hierarchical group LASSO (HCGL) for sparse multi-output factor model estimation with integrated factor covariance assembly

---

| 📊 Metric | 🔢 Value |
| --- | --- |
| PyPI Version | [![PyPI](https://img.shields.io/pypi/v/factorlasso?style=flat-square)](https://pypi.org/project/factorlasso/) |
| Python Versions | [![Python](https://img.shields.io/pypi/pyversions/factorlasso?style=flat-square)](https://pypi.org/project/factorlasso/) |
| License | [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) |

### 📈 Package Statistics

| 📊 Metric | 🔢 Value |
| --- | --- |
| Total Downloads | [![Total](https://pepy.tech/badge/factorlasso)](https://pepy.tech/project/factorlasso) |
| CI Status | [![CI](https://github.com/ArturSepp/factorlasso/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSepp/factorlasso/actions) |
| Coverage | [![Coverage](https://img.shields.io/badge/coverage-%3E90%25-brightgreen)](https://github.com/ArturSepp/factorlasso) |
| GitHub Stars | [![GitHub stars](https://img.shields.io/github/stars/ArturSepp/factorlasso?style=flat-square&logo=github)](https://github.com/ArturSepp/factorlasso/stargazers) |
| GitHub Forks | [![GitHub forks](https://img.shields.io/github/forks/ArturSepp/factorlasso?style=flat-square&logo=github)](https://github.com/ArturSepp/factorlasso/network/members) |

## **The Problem**

In many applications — portfolio construction, genomics, macro-econometrics — you need to estimate a factor model

$$Y_t = \alpha + \beta X_t + \varepsilon_t$$

where $Y_t \in \mathbb{R}^{N}$ are response variables (asset returns, gene expressions), $X_t \in \mathbb{R}^{M}$ are factors, $\beta \in \mathbb{R}^{N \times M}$ are sparse factor loadings, and $\alpha \in \mathbb{R}^{N}$ is the intercept.

In practice, you face several challenges that standard LASSO packages don't handle:

1. **Domain knowledge constrains coefficient signs** — equity assets should have non-negative equity beta; government bonds should not load on commodity factors. Standard LASSO ignores this.
2. **You have prior estimates** and want to shrink toward them, not toward zero — the penalty should be $\|\beta - \beta_0\|$ not $\|\beta\|$.
3. **Variables have different history lengths** — some assets start trading later than others. Dropping rows with any NaN discards valid data for all other variables.
4. **You need a consistent covariance matrix** — the factor covariance $\Sigma_y = \beta \Sigma_x \beta^\top + D$ must use the *same* $\beta$ from estimation, not a separate estimate.
5. **Data is non-stationary** — recent observations should carry more weight (EWMA weighting).

`factorlasso` solves all five in a single `fit()` call. The implementation follows scikit-learn conventions (`fit` / `predict` / `score` / `coef_` / `intercept_`).

The methodology is based on the Hierarchical Clustering Group LASSO (HCGL) framework introduced in:

> Sepp A., Ossa I., Kastenholz M. (2026), "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios", *The Journal of Portfolio Management*, 52(4), 86–120. [Paper link](https://eprints.pm-research.com/17511/143431/index.html)

and the Capital Market Assumptions framework in the companion paper:

> Sepp A., Hansen E., Kastenholz M. (2026), "Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors", *Under revision at the Journal of Portfolio Management*.

## **Installation**

Install using
```bash
pip install factorlasso
```

Upgrade using
```bash
pip install --upgrade factorlasso
```

Clone using
```bash
git clone https://github.com/ArturSepp/factorlasso.git
```

Core dependencies:
`numpy`, `pandas`, `scipy`, `cvxpy`, `openpyxl`

## Table of Contents

1. [Quick Start](#quick-start)
2. [Convention: Paper vs Code](#convention-paper-vs-code)
3. [Sign Constraints](#sign-constraints)
4. [Prior-Centered Regularisation](#prior-centred-regularisation)
5. [Hierarchical Clustering Group LASSO (HCGL)](#hierarchical-clustering-group-lasso-hcgl)
6. [NaN-Aware Estimation](#nan-aware-estimation)
7. [Factor Covariance Assembly](#factor-covariance-assembly)
8. [API Summary](#api-summary)
9. [Estimation Methods](#estimation-methods)
10. [Applications](#applications)
11. [Related Packages](#related-packages)
12. [References](#references)
13. [Citation](#citation)

## **Quick Start**

```python
import numpy as np, pandas as pd
from factorlasso import LassoModel, LassoModelType

# Simulate Y_t = β X_t + noise  (code uses row-major: Y = X @ β' + noise)
np.random.seed(42)
T, M, N = 200, 3, 5
X = pd.DataFrame(np.random.randn(T, M), columns=['f0', 'f1', 'f2'])
beta_true = np.array([[1, 0, .5], [0, 1, 0], [.3, 0, 0], [0, .8, .2], [1, .5, 0]])
Y = pd.DataFrame(X.values @ beta_true.T + .1*np.random.randn(T, N),
                  columns=[f'y{i}' for i in range(N)])

# Fit sparse factor model
model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-4)
model.fit(x=X, y=Y)
print(model.coef_.round(2))       # β (N × M)
print(model.intercept_.round(4))  # α (N,)

# Predict and score (scikit-learn compatible)
y_hat = model.predict(X)  # Ŷ_t = α + β X_t  (code: X @ β' + α)
r2 = model.score(X, Y)    # mean R² across response variables
```

## **Convention: Paper vs Code**

The factor model in the paper uses **column vectors**:

$$Y_t = \alpha + \beta\, X_t + \varepsilon_t, \qquad \beta \in \mathbb{R}^{N \times M}$$

where $Y_t \in \mathbb{R}^{N \times 1}$ and $X_t \in \mathbb{R}^{M \times 1}$.

In Python, **pandas DataFrames store observations as rows**. The code works with the row-major equivalent:

| Symbol | Paper (column-vector) | Code (row-major, pandas) |
|--------|----------------------|--------------------------|
| $Y$ | $(N \times T)$ | `y`: DataFrame $(T \times N)$ |
| $X$ | $(M \times T)$ | `x`: DataFrame $(T \times M)$ |
| $\beta$ | $(N \times M)$ | `coef_`: DataFrame $(N \times M)$ — **same as paper** |
| $\alpha$ | $(N \times 1)$ | `intercept_`: Series $(N,)$ |

The coefficient matrix `coef_` is stored in the **paper convention** $(N \times M)$.
The prediction `Y = X @ β' + α` in code is the row-major form of the paper's `Y_t = α + β X_t`.

## **Sign Constraints**

Enforce domain knowledge on coefficient signs using a constraint matrix where
`1` = non-negative, `-1` = non-positive, `0` = constrained to zero, `NaN` = free:

```python
signs = pd.DataFrame([[1, np.nan, 1], [np.nan, 1, 0], [1, 0, np.nan],
                       [np.nan, 1, 1], [1, 1, np.nan]],
                      index=Y.columns, columns=X.columns)

model = LassoModel(reg_lambda=1e-4, factors_beta_loading_signs=signs)
model.fit(x=X, y=Y)
# All constrained coefficients satisfy their sign requirements by construction
```

## **Prior-Centred Regularisation**

Shrink toward a non-zero prior instead of zero. When you have prior estimates
$\beta_0$ (e.g., from a previous estimation period or theoretical model),
the penalty becomes $\|\beta - \beta_0\|$ instead of $\|\beta\|$:

```python
beta_prior = pd.DataFrame(beta_true, index=Y.columns, columns=X.columns)
model = LassoModel(reg_lambda=1e-2, factors_beta_prior=beta_prior)
model.fit(x=X, y=Y)  # shrinks toward beta_prior instead of zero
```

## **Hierarchical Clustering Group LASSO (HCGL)**

Automatically discover group structure among response variables via
hierarchical clustering on their correlation matrix (Ward's method),
then apply Group LASSO with group-adaptive penalties:

```python
model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-5, span=52,
)
model.fit(x=X, y=Y)
print(model.clusters)  # auto-discovered groups
```

## **Cross-Validated Regularisation**

Picking `reg_lambda` by hand is fine for exploration, but for serious
work you want it chosen by data. `LassoModelCV` sweeps a log-spaced
grid using **expanding-window time-series splits** — random K-fold leaks
future information into training folds and is the wrong default for
returns data, so the package gives you the right CV out of the box:

```python
import numpy as np
from factorlasso import LassoModel, LassoModelCV

cv = LassoModelCV(
    lambdas=np.logspace(-6, -1, 15),  # default: 20-point grid on [1e-6, 1e-1]
    n_splits=5,
    base_model=LassoModel(span=52),    # inherits all hyperparameters except reg_lambda
).fit(x=X, y=Y)

print(cv.best_lambda_, cv.best_score_)
print(cv.cv_scores_.mean(axis=1))      # diagnostic: stability across folds

# After fit, predict/score delegate to a model refit on the full dataset
y_hat = cv.predict(X_new)
oos_r2 = cv.score(X_test, Y_test)
```

Per-fold solver failures leave a `NaN` in `cv_scores_` and the sweep
continues; you only get a `RuntimeError` if every fold for every lambda
fails. See `examples/cv_lambda_selection.py` for an end-to-end
illustration on a synthetic asset-factor panel, including an
out-of-sample comparison against fixed-lambda baselines.

## **NaN-Aware Estimation**

Variables with different history lengths are handled naturally.
Instead of dropping any row containing a NaN (which discards valid observations
for all other variables), `factorlasso` applies a binary validity mask that
zeros out the contribution of missing observations per variable while
preserving all available data:

```python
Y_with_gaps = Y.copy()
Y_with_gaps.iloc[:50, 3] = np.nan   # variable y3 starts 50 periods later
Y_with_gaps.iloc[:100, 4] = np.nan  # variable y4 starts 100 periods later

model = LassoModel(reg_lambda=1e-4)
model.fit(x=X, y=Y_with_gaps)
# All 5 variables estimated using their full available history
# No data discarded for y0, y1, y2 despite gaps in y3, y4
```

## **Factor Covariance Assembly**

After estimation, assemble the consistent factor covariance decomposition
$\Sigma_y = \beta \Sigma_x \beta^\top + D$ where $\beta$ is the *same*
matrix from the LASSO estimation — guaranteed consistency:

```python
from factorlasso import CurrentFactorCovarData, VarianceColumns

sigma_y = CurrentFactorCovarData(
    x_covar=factor_covariance,   # Σ_x (M × M)
    y_betas=model.coef_,          # β (N × M) from estimation
    y_variances=diagnostics_df,   # residual variances D
).get_y_covar()
# sigma_y is (N × N) positive semi-definite by construction
```

## **API Summary**

The API follows scikit-learn conventions: `fit` / `predict` / `score`.

| Method | Description |
|--------|-------------|
| `model.fit(x, y)` | Estimate α, β — returns `self` |
| `model.predict(x)` | Return Ŷ_t = α + β X_t (row-major: `X @ β' + α`) |
| `model.score(x, y)` | Return mean R² |
| `model.get_params()` / `set_params(**)` | scikit-learn compatibility (works with `GridSearchCV`, `Pipeline`) |
| `LassoModelCV(...).fit(x, y)` | Time-series CV for `reg_lambda` (expanding-window splits) |

| Fitted attribute | Shape | Description |
|-----------------|-------|-------------|
| `coef_` | (N, M) | Factor loadings β |
| `intercept_` | (N,) | Intercept α |
| `estimated_betas` | (N, M) | Alias for `coef_` (backward compat) |
| `clusters_` | (N,) | HCGL cluster labels |
| `estimation_result_` | — | Full diagnostics (r2, ss_res, ss_total) |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | `LassoModelType` | `LASSO` | Estimation method |
| `reg_lambda` | `float` | `1e-5` | Regularisation strength |
| `span` | `int` | `None` | EWMA span for observation weighting |
| `factors_beta_loading_signs` | `DataFrame` | `None` | Sign constraint matrix (N × M) |
| `factors_beta_prior` | `DataFrame` | `None` | Prior β₀ matrix (N × M) |
| `group_data` | `Series` | `None` | Group labels (required for `GROUP_LASSO`) |
| `demean` | `bool` | `True` | Subtract (rolling) mean before estimation |
| `solver` | `str` | `'CLARABEL'` | CVXPY solver name |
| `warmup_period` | `int` | `12` | Min observations before including a variable |

## **Estimation Methods**

| Method | `LassoModelType` | Penalty |
|--------|-------------------|---------|
| LASSO | `LASSO` | $\lambda\|\beta - \beta_0\|_1$ |
| Group LASSO | `GROUP_LASSO` | $\sum_g \lambda\sqrt{|g|/G}\|\beta_{g,:} - \beta_{0,g,:}\|_2$ |
| HCGL | `GROUP_LASSO_CLUSTERS` | Same as Group LASSO with auto-clustering |

All methods support sign constraints, prior-centered shrinkage, EWMA weighting, and NaN-aware estimation.

## **Applications**

The methodology is domain-agnostic. Examples are provided for:

1. `examples/finance_factor_model.py` — Multi-asset factor models with sign-constrained betas and consistent covariance estimation
2. `examples/genomics_factor_model.py` — Gene expression driven by pathway activity factors with biological sign priors

The same estimation problem (sparse factor loadings with sign priors and consistent covariance) appears in macro-econometrics, signal processing, and multi-task learning.

### Illustration: multi-asset factor model with HCGL

```python
from factorlasso import LassoModel, LassoModelType

model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-5,
    span=52,                                 # 1-year EWMA half-life (weekly data)
    factors_beta_loading_signs=sign_matrix,   # domain-knowledge constraints
    factors_beta_prior=prior_betas,           # shrink toward prior, not zero
)
model.fit(x=factor_returns, y=asset_returns)

# Inspect results
print(model.coef_)           # sparse factor loadings (N × M)
print(model.intercept_)      # intercept α (N,)
print(model.clusters_)       # auto-discovered asset groups
print(model.score(factor_returns, asset_returns))  # mean R²
```

## **Related Packages**

| Package | Key Difference |
|---------|----------------|
| [scikit-learn](https://scikit-learn.org/) `Lasso` | No sign constraints, no multi-output Group LASSO |
| [skglm](https://contrib.scikit-learn.org/skglm/) | No sign constraints, no prior-centered shrinkage |
| [abess](https://abess.readthedocs.io/) | Best-subset selection (L0), not L1/Group L2 |
| [group-lasso](https://pypi.org/project/group-lasso/) | No sign constraints, no EWMA, no prior-centered |

`factorlasso` is the only package that combines sign-constrained penalised regression, prior-centered shrinkage, HCGL clustering, NaN-aware estimation, and integrated factor covariance assembly.

## **References**

1. Sepp A., Ossa I., Kastenholz M. (2026), "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios", *The Journal of Portfolio Management*, 52(4), 86–120. [Paper link](https://eprints.pm-research.com/17511/143431/index.html)

2. Sepp A., Hansen E., Kastenholz M. (2026), "Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors", *Under revision at the Journal of Portfolio Management*.

## **Citation**

If you use `factorlasso` in your research, please cite the software and the underlying papers:

```bibtex
@software{sepp2026factorlasso,
  author = {Sepp, Artur},
  title = {factorlasso: Sparse Factor Model Estimation with Constrained LASSO in Python},
  year = {2026},
  url = {https://github.com/ArturSepp/factorlasso}
}

@article{seppossa2026,
  author = {Sepp, Artur and Ossa, Ivan and Kastenholz, Mika},
  title = {Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios},
  journal = {The Journal of Portfolio Management},
  volume = {52},
  number = {4},
  pages = {86--120},
  year = {2026}
}

@article{sepphansen2026,
  author = {Sepp, Artur and Hansen, Emilie and Kastenholz, Mika},
  title = {Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors},
  journal = {Under revision at the Journal of Portfolio Management},
  year = {2026}
}
```

## **Disclaimer**

`factorlasso` package is distributed FREE & WITHOUT ANY WARRANTY under the MIT License.

See [LICENSE](LICENSE) for details.

Please report any bugs or suggestions by opening an [issue](https://github.com/ArturSepp/factorlasso/issues).
