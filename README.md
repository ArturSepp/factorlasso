# **factorlasso**: Constrained Multi-Output Sparse Regression with Joint Covariance Assembly

[![PyPI](https://img.shields.io/pypi/v/factorlasso?style=flat-square)](https://pypi.org/project/factorlasso/)
[![Python](https://img.shields.io/pypi/pyversions/factorlasso?style=flat-square)](https://pypi.org/project/factorlasso/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/factorlasso)](https://pepy.tech/project/factorlasso)
[![CI](https://github.com/ArturSepp/factorlasso/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSepp/factorlasso/actions)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)](https://github.com/ArturSepp/factorlasso)
[![GitHub stars](https://img.shields.io/github/stars/ArturSepp/factorlasso?style=flat-square&logo=github)](https://github.com/ArturSepp/factorlasso/stargazers)

`factorlasso` is a scikit-learn-compatible Python package for **multi-output penalized regression under structured constraints**, plus consistent joint covariance assembly. It fills a specific gap in the sparse regression ecosystem: estimating sparse coefficient matrices when you need *any combination* of (i) element-wise sign constraints, (ii) shrinkage toward an informative prior, (iii) data-driven group discovery via hierarchical clustering, (iv) exponentially-weighted observations for non-stationary data, and (v) joint estimation of the implied response covariance matrix using the *same* coefficient estimates.

No other package in the Python ecosystem combines these five features. See [COMPARISON.md](COMPARISON.md) for an axis-by-axis comparison with `scikit-learn`, `skglm`, `groupyr`, `group-lasso`, and `asgl`.

---

## **The problem this solves**

Estimate coefficients $\beta \in \mathbb{R}^{N \times M}$ in the multi-output linear model

$$Y_t = \alpha + \beta X_t + \varepsilon_t, \qquad t = 1, \dots, T$$

where $Y_t \in \mathbb{R}^{N}$ are responses, $X_t \in \mathbb{R}^{M}$ are regressors, and $\beta$ is sparse. `factorlasso` handles five practical requirements that standard penalized-regression packages do not address jointly:

1. **Element-wise sign constraints** — the analyst knows *a priori* that some coefficients must be non-negative, non-positive, or exactly zero. Standard LASSO ignores this.
2. **Prior-centered shrinkage** — when a prior estimate $\beta_0$ is available, the natural penalty is $\|\beta - \beta_0\|$ rather than $\|\beta\|$. This is a convex-optimization analogue of Bayesian regression with an informative Laplace prior.
3. **Data-driven group discovery (HCGL)** — Group LASSO requires groups as input. `factorlasso` can discover them from the correlation structure of $Y$ via Ward's hierarchical clustering, then fit group-sparse $\beta$ in the same call.
4. **Ragged histories** — rows of $Y$ where some responses are missing are preserved via per-response validity masks rather than listwise deletion.
5. **Joint covariance consistency** — the implied response covariance $\Sigma_Y = \beta \Sigma_X \beta^\top + D$ is assembled using the *same* $\beta$ from estimation, eliminating the Factor Alignment Problem ([Ceria & Stubbs, 2013](https://www.pm-research.com/content/iijpormgmt/39/4/22)) that arises when $\beta$ for $\mu$ is estimated separately from $\beta$ for $\Sigma$.

All five compose: any subset can be active in a single `.fit(x, y)` call, and the package is a drop-in scikit-learn estimator.

---

## **Quick start**

```bash
pip install factorlasso
```

```python
import numpy as np, pandas as pd
from factorlasso import LassoModel, LassoModelType

rng = np.random.default_rng(0)
T, M, N = 200, 3, 5
X = pd.DataFrame(rng.standard_normal((T, M)), columns=[f'f{i}' for i in range(M)])
beta_true = np.array([[1, 0, .5], [0, 1, 0], [.3, 0, 0], [0, .8, .2], [1, .5, 0]])
Y = pd.DataFrame(X.values @ beta_true.T + .1*rng.standard_normal((T, N)),
                  columns=[f'y{i}' for i in range(N)])

model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-4)
model.fit(x=X, y=Y)

print(model.coef_.round(2))       # β (N × M)
print(model.intercept_.round(4))  # α (N,)

# sklearn-compatible prediction and scoring
y_hat = model.predict(X)          # Ŷ_t = α + β X_t  (row-major: X @ β' + α)
r2    = model.score(X, Y)         # mean R² across responses
```

---

## **When should I use this instead of scikit-learn / skglm / groupyr?**

The short answer: when your problem requires *constraints or structure* that the fast specialized solvers can't express, and you are willing to trade raw speed for flexibility. `factorlasso` uses CVXPY as its backend — slower than `skglm`'s Anderson-accelerated coordinate descent on problems both packages can solve, but expressive enough to handle prior-centered penalties, element-wise sign constraints, validity masks, and group-adaptive penalties *simultaneously*. See [COMPARISON.md](COMPARISON.md) for the full feature matrix.

**Reach for `factorlasso` when you have:**
- Mixed sign priors on individual coefficients (not a blanket `positive=True`)
- An informative prior $\beta_0 \ne 0$ you want to shrink toward
- Responses with different observation-window lengths
- A need for the covariance matrix $\Sigma_Y$ consistent with the estimated $\beta$
- Unknown group structure that should be discovered from the data

**Reach for `skglm` / `celer` / `group-lasso` when you have:**
- High-dimensional data (millions of features) and need raw speed
- Standard penalties (L1, L2, Group L2, SLOPE, MCP) with no sign or prior constraints
- No need for joint covariance output

---

## **Applications**

The methodology is domain-agnostic — the same estimation problem appears across fields:

| Field | Response variables $Y$ | Regressors $X$ | Why sign constraints matter |
|-------|------------------------|---------------|-----------------------------|
| **Multi-asset portfolio construction** | asset returns | factor returns | equities $\ge 0$ on equity factor, gov bonds not loading on commodities |
| **Genomics / pathway analysis** | gene expression | pathway activities | known up/down-regulation priors |
| **Macro-econometrics** | country/sector outputs | common macro factors | sign priors from economic theory |
| **Neuroimaging** | voxel BOLD signals | task/stimulus regressors | excitation vs. inhibition priors |
| **Signal processing** | sensor channels | latent sources | physical non-negativity (spectra, abundances) |

See `examples/finance_factor_model.py`, `examples/genomics_factor_model.py`, and `examples/cv_lambda_selection.py` for worked examples.

The design and empirical validation is described in:

> Sepp, Ossa, Kastenholz (2026), "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios", *Journal of Portfolio Management*, 52(4), 86–120. [link](https://eprints.pm-research.com/17511/143431/index.html)
>
> Sepp, Hansen, Kastenholz (2026), "Capital Market Assumptions Using Multi-Asset Tradable Factors: The MATF-CMA Framework", under revision at *Journal of Portfolio Management*.

---

## **Feature tour**

### Sign constraints

Element-wise constraints via a matrix where `1 = non-negative`, `-1 = non-positive`, `0 = fixed zero`, `NaN = free`:

```python
signs = pd.DataFrame([[1, np.nan, 1], [np.nan, 1, 0], [1, 0, np.nan],
                       [np.nan, 1, 1], [1, 1, np.nan]],
                      index=Y.columns, columns=X.columns)

model = LassoModel(reg_lambda=1e-4, factors_beta_loading_signs=signs).fit(x=X, y=Y)
# Every constrained coefficient satisfies its sign requirement by construction.
```

### Prior-centered regularization

Shrink toward a non-zero prior $\beta_0$ instead of zero:

```python
beta_prior = pd.DataFrame(beta_true, index=Y.columns, columns=X.columns)
model = LassoModel(reg_lambda=1e-2, factors_beta_prior=beta_prior).fit(x=X, y=Y)
# Penalty becomes ‖β − β₀‖₁ instead of ‖β‖₁.
```

### Hierarchical Clustering Group LASSO (HCGL)

When groups are unknown *a priori*, discover them from the correlation structure of $Y$ using Ward's method, then apply Group LASSO with group-adaptive penalties:

```python
model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-5, span=52,
    cutoff_fraction=0.5,   # dendrogram cut point, tunable
).fit(x=X, y=Y)
print(model.clusters_)     # auto-discovered group labels
```

Compared to Group LASSO with user-supplied groups, HCGL lets you fit a group-sparse model when the grouping is itself a nuisance parameter — common in unsupervised exploratory settings.

### Cross-validated regularization

Time-series expanding-window CV (random K-fold leaks future information and is the wrong default for temporal data):

```python
from factorlasso import LassoModelCV

cv = LassoModelCV(
    lambdas=np.logspace(-6, -1, 15),
    n_splits=5,
    base_model=LassoModel(span=52),  # inherits all hyperparameters except reg_lambda
).fit(x=X, y=Y)

print(cv.best_lambda_, cv.best_score_)
print(cv.cv_scores_.mean(axis=1))    # per-lambda mean fold R²
```

### Ragged histories

Per-response validity masks — no listwise deletion:

```python
Y_with_gaps = Y.copy()
Y_with_gaps.iloc[:50, 3]  = np.nan   # y3 starts 50 periods later
Y_with_gaps.iloc[:100, 4] = np.nan   # y4 starts 100 periods later

model = LassoModel(reg_lambda=1e-4).fit(x=X, y=Y_with_gaps)
# All 5 responses estimated on their full available history.
# Valid observations of y0, y1, y2 are not discarded because y3, y4 had gaps.
```

### Joint covariance assembly

The implied response covariance uses the *same* $\beta$ from estimation:

```python
from factorlasso import CurrentFactorCovarData

snapshot = CurrentFactorCovarData(
    x_covar=factor_covariance,    # Σ_X (M × M)
    y_betas=model.coef_,           # β (N × M) — same matrix used in prediction
    y_variances=diagnostics_df,    # residual variances D
)
sigma_y = snapshot.get_y_covar()   # (N × N), positive semi-definite by construction
```

This eliminates the Factor Alignment Problem: the $\beta$ that maps $X \to Y$ is the same $\beta$ that maps $\Sigma_X \to \Sigma_Y$, guaranteeing the expected-value and second-moment representations are mutually consistent.

---

## **Convention: math vs. code**

The model in math uses **column vectors**, $Y_t = \alpha + \beta X_t + \varepsilon_t$ with $\beta \in \mathbb{R}^{N \times M}$. In Python, `pandas` DataFrames store observations as rows, so the code works with the row-major equivalent:

| Symbol | Math (column-vector) | Code (row-major, pandas) |
|--------|----------------------|--------------------------|
| $Y$ | $(N \times T)$ | `y`: DataFrame $(T \times N)$ |
| $X$ | $(M \times T)$ | `x`: DataFrame $(T \times M)$ |
| $\beta$ | $(N \times M)$ | `coef_`: DataFrame $(N \times M)$ — **same as math** |
| $\alpha$ | $(N \times 1)$ | `intercept_`: Series $(N,)$ |

The coefficient matrix `coef_` is stored in the **mathematical convention** $(N \times M)$. The prediction `Y = X @ β' + α` in code is the row-major form of $Y_t = \alpha + \beta X_t$.

---

## **API summary**

Follows scikit-learn conventions — compatible with `GridSearchCV`, `Pipeline`, etc.

| Method | Description |
|--------|-------------|
| `model.fit(x, y)` | Estimate α, β — returns `self` |
| `model.predict(x)` | Ŷ_t = α + β X_t |
| `model.score(x, y)` | Mean R² across responses |
| `model.get_params()` / `set_params(**)` | sklearn parameter protocol |
| `LassoModelCV(...).fit(x, y)` | Time-series CV for `reg_lambda` |

| Fitted attribute | Shape | Description |
|------------------|-------|-------------|
| `coef_` | (N, M) | β |
| `intercept_` | (N,) | α |
| `clusters_` | (N,) | HCGL cluster labels |
| `estimation_result_` | — | Full diagnostics (r2, ss_res, ss_total) |

| Constructor parameter | Default | Purpose |
|-----------------------|---------|---------|
| `model_type` | `LASSO` | `LASSO` / `GROUP_LASSO` / `GROUP_LASSO_CLUSTERS` |
| `reg_lambda` | `1e-5` | Regularization strength |
| `span` | `None` | EWMA span for observation weighting |
| `factors_beta_loading_signs` | `None` | Sign constraint matrix (N × M) |
| `factors_beta_prior` | `None` | Prior β₀ matrix (N × M) |
| `group_data` | `None` | Group labels (required for `GROUP_LASSO`) |
| `cutoff_fraction` | `0.5` | HCGL dendrogram cut point |
| `demean` | `True` | Subtract (rolling) mean before estimation |
| `solver` | `'CLARABEL'` | CVXPY solver name |
| `warmup_period` | `12` | Minimum observations per response |

---

## **Estimation methods**

| Method | `LassoModelType` | Penalty |
|--------|-------------------|---------|
| LASSO | `LASSO` | $\lambda\|\beta - \beta_0\|_1$ |
| Group LASSO | `GROUP_LASSO` | $\sum_g \lambda\sqrt{|g|/G}\,\|\beta_{g,:} - \beta_{0,g,:}\|_2$ |
| HCGL | `GROUP_LASSO_CLUSTERS` | Same as Group LASSO, groups discovered from Y's correlation |

All methods support sign constraints, prior-centered shrinkage, EWMA weighting, and ragged-history estimation.

---

## **Installation**

```bash
pip install factorlasso                    # latest release
pip install --upgrade factorlasso          # upgrade
git clone https://github.com/ArturSepp/factorlasso.git  # dev clone
```

Dependencies: `numpy ≥ 1.22`, `pandas ≥ 1.4`, `scipy ≥ 1.9`, `cvxpy ≥ 1.3`, `openpyxl ≥ 3.1`.

---

## **Citation**

```bibtex
@software{sepp2026factorlasso,
  author = {Sepp, Artur},
  title  = {factorlasso: Constrained Multi-Output Sparse Regression with Joint Covariance Assembly},
  year   = {2026},
  url    = {https://github.com/ArturSepp/factorlasso},
}

@article{seppossa2026,
  author  = {Sepp, Artur and Ossa, Ivan and Kastenholz, Mika},
  title   = {Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios},
  journal = {The Journal of Portfolio Management},
  volume  = {52},
  number  = {4},
  pages   = {86--120},
  year    = {2026},
}

@unpublished{sepphansen2026,
  author = {Sepp, Artur and Hansen, Emilie and Kastenholz, Mika},
  title  = {Capital Market Assumptions Using Multi-Asset Tradable Factors: The {MATF-CMA} Framework},
  note   = {Under revision at the Journal of Portfolio Management},
  year   = {2026},
}
```

---

## **License**

MIT. See [LICENSE](LICENSE).

Bug reports and feature requests welcome via [issues](https://github.com/ArturSepp/factorlasso/issues).
