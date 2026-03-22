# factorlasso

**Sparse factor model estimation with sign-constrained LASSO, prior-centered regularisation, and hierarchical group LASSO (HCGL)**

[![CI](https://github.com/ArturSepp/factorlasso/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSepp/factorlasso/actions)
[![PyPI](https://img.shields.io/pypi/v/factorlasso?style=flat-square)](https://pypi.org/project/factorlasso/)
[![Python](https://img.shields.io/pypi/pyversions/factorlasso?style=flat-square)](https://pypi.org/project/factorlasso/)
[![Coverage](https://img.shields.io/badge/coverage-%3E90%25-brightgreen)](https://github.com/ArturSepp/factorlasso)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Overview

`factorlasso` estimates the sparse multi-output factor model

$$Y_t = \alpha + \beta X_t + \varepsilon_t$$

where $Y_t$ is $(N \times 1)$, $X_t$ is $(M \times 1)$, $\beta$ is $(N \times M)$, and $\alpha$ is $(N \times 1)$ intercept, under:

- **Sign constraints** on individual coefficients (non-negative, non-positive, zero, or free)
- **Prior-centered regularisation** — penalise $\|\beta - \beta_0\|$ instead of $\|\beta\|$, shrinking toward domain-specific priors
- **Group structure** — Group LASSO with user-defined groups or automatic hierarchical clustering (HCGL)
- **EWMA-weighted observations** — exponential decay for non-stationary data
- **NaN-aware estimation** — validity masking handles variables with different observation lengths

After estimation, `factorlasso` assembles the consistent factor covariance decomposition

$$\Sigma_y = \beta\,\Sigma_x\,\beta^\top + D$$

where $\Sigma_x$ is the factor covariance and $D$ is diagonal idiosyncratic variance.

**No existing Python package** combines sign-constrained penalised regression with prior-centered shrinkage and integrated factor covariance assembly.

## Convention: Paper vs Code

The factor model in the paper uses **column vectors**:

$$Y_t = \alpha + \beta\, X_t + \varepsilon_t, \qquad \beta \in \mathbb{R}^{N \times M}$$

where $Y_t \in \mathbb{R}^{N \times 1}$ and $X_t \in \mathbb{R}^{M \times 1}$. For $T$ observations stacked as columns: $Y = \alpha\,\mathbf{1}^\top + \beta\, X + E$ where $Y$ is $(N \times T)$ and $X$ is $(M \times T)$.

In Python, **pandas DataFrames store observations as rows** (the standard in ML). So the code works with transposed shapes:

| Symbol | Paper (column-vector) | Code (row-major, pandas) |
|--------|----------------------|--------------------------|
| $Y$ | $(N \times T)$ | `y`: DataFrame $(T \times N)$ |
| $X$ | $(M \times T)$ | `x`: DataFrame $(T \times M)$ |
| $\beta$ | $(N \times M)$ | `coef_`: DataFrame $(N \times M)$ — **same as paper** |
| $\alpha$ | $(N \times 1)$ | `intercept_`: Series $(N,)$ |

The coefficient matrix `coef_` is stored in the **paper convention** $(N \times M)$.  The prediction `Y = X @ β' + α` in code is equivalent to the paper's `Y_t = α + β X_t` — just the row-major form of the same linear algebra.

## Installation

```bash
pip install factorlasso
```

## Quick Start

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

model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-4)
model.fit(x=X, y=Y)
print(model.coef_.round(2))       # β (N × M)
print(model.intercept_.round(4))  # α (N,)
```

### Predict and Score (scikit-learn compatible)

```python
y_hat = model.predict(X)  # Ŷ_t = α + β X_t  (code: X @ β' + α)
r2 = model.score(X, Y)    # mean R² across response variables
```

### Sign Constraints

```python
# 1 = non-negative, -1 = non-positive, 0 = zero, NaN = free
signs = pd.DataFrame([[1, np.nan, 1], [np.nan, 1, 0], [1, 0, np.nan],
                       [np.nan, 1, 1], [1, 1, np.nan]],
                      index=Y.columns, columns=X.columns)

model = LassoModel(reg_lambda=1e-4, factors_beta_loading_signs=signs)
model.fit(x=X, y=Y)
```

### Prior-Centered Regularisation

```python
beta_prior = pd.DataFrame(beta_true, index=Y.columns, columns=X.columns)
model = LassoModel(reg_lambda=1e-2, factors_beta_prior=beta_prior)
model.fit(x=X, y=Y)  # shrinks toward beta_prior instead of zero
```

### Hierarchical Clustering Group LASSO (HCGL)

```python
model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-5, span=52,
)
model.fit(x=X, y=Y)
print(model.clusters)  # auto-discovered groups
```

### Factor Covariance Assembly

```python
from factorlasso import CurrentFactorCovarData, VarianceColumns
from factorlasso.ewm_utils import compute_ewm_covar

# Assemble Sigma_y = beta @ Sigma_x @ beta.T + D
sigma_y = CurrentFactorCovarData(
    x_covar=factor_covariance,
    y_betas=model.coef_,
    y_variances=diagnostics_df,
).get_y_covar()
```

## API Summary

The API follows scikit-learn conventions: `fit` / `predict` / `score`.

| Method | Description |
|--------|-------------|
| `model.fit(x, y)` | Estimate α, β — returns `self` |
| `model.predict(x)` | Return Ŷ_t = α + β X_t (row-major: `X @ β' + α`) |
| `model.score(x, y)` | Return mean R² |

| Fitted attribute | Shape | Description |
|-----------------|-------|-------------|
| `coef_` | (N, M) | Factor loadings β |
| `intercept_` | (N,) | Intercept α |
| `estimated_betas` | (N, M) | Alias for `coef_` (backward compat) |
| `clusters_` | (N,) | HCGL cluster labels |

## Estimation Methods

| Method | `LassoModelType` | Penalty |
|--------|-------------------|---------|
| LASSO | `LASSO` | $\lambda\|\beta - \beta_0\|_1$ |
| Group LASSO | `GROUP_LASSO` | $\sum_g \lambda\sqrt{|g|/G}\|\beta_{g,:} - \beta_{0,g,:}\|_2$ |
| HCGL | `GROUP_LASSO_CLUSTERS` | Same as Group LASSO with auto-clustering |

All methods support sign constraints, prior-centered shrinkage, EWMA weighting, and NaN-aware estimation.

## Applications

The methodology is domain-agnostic.  Examples are provided for:

- **Finance** — Multi-asset factor models with sign-constrained betas and consistent covariance estimation ([`examples/finance_factor_model.py`](examples/finance_factor_model.py))
- **Genomics** — Gene expression driven by pathway activity factors with biological sign priors ([`examples/genomics_factor_model.py`](examples/genomics_factor_model.py))

The same estimation problem (sparse factor loadings with sign priors and consistent covariance) appears in macro-econometrics, signal processing, and multi-task learning.

## Dependencies

Only standard scientific Python:

- `numpy ≥ 1.22`
- `pandas ≥ 1.4`
- `scipy ≥ 1.9`
- `cvxpy ≥ 1.3`

## Related Packages

| Package | Key Difference |
|---------|----------------|
| [scikit-learn](https://scikit-learn.org/) `Lasso` | No sign constraints, no multi-output Group LASSO |
| [skglm](https://contrib.scikit-learn.org/skglm/) | No sign constraints, no prior-centered shrinkage |
| [abess](https://abess.readthedocs.io/) | Best-subset selection (L0), not L1/Group L2 |
| [group-lasso](https://pypi.org/project/group-lasso/) | No sign constraints, no EWMA, no prior-centered |

`factorlasso` is the only package that combines sign-constrained penalised regression, prior-centered shrinkage, HCGL clustering, and integrated factor covariance assembly.

## References

Sepp A., Ossa I., Kastenholz M. (2026), "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios", *The Journal of Portfolio Management*, 52(4), 86–120. [Paper link](https://eprints.pm-research.com/17511/143431/index.html)

## Citation

```bibtex
@software{sepp2026factorlasso,
  author = {Sepp, Artur},
  title = {factorlasso: Sparse Factor Model Estimation with Constrained LASSO in Python},
  year = {2026},
  url = {https://github.com/ArturSepp/factorlasso}
}
```

## License

MIT — see [LICENSE](LICENSE).
