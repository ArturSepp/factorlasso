# Migration Guide: optimalportfolios → factorlasso

This guide explains how `factorlasso` was extracted from `optimalportfolios`
and what changes are needed in downstream code.

## What moved

| Module | Old location | New location |
|--------|-------------|-------------|
| `LassoModel` | `optimalportfolios.lasso.lasso_estimator` | `factorlasso.lasso_estimator` |
| `LassoModelType` | `optimalportfolios.lasso.lasso_estimator` | `factorlasso.lasso_estimator` |
| `LassoEstimationResult` | `optimalportfolios.lasso.lasso_estimator` | `factorlasso.lasso_estimator` |
| `solve_lasso_cvx_problem` | `optimalportfolios.lasso.lasso_estimator` | `factorlasso.lasso_estimator` |
| `solve_group_lasso_cvx_problem` | `optimalportfolios.lasso.lasso_estimator` | `factorlasso.lasso_estimator` |
| `CurrentFactorCovarData` | `optimalportfolios.covar_estimation.factor_covar_data` | `factorlasso.factor_covar` |
| `RollingFactorCovarData` | `optimalportfolios.covar_estimation.factor_covar_data` | `factorlasso.factor_covar` |
| `VarianceColumns` | `optimalportfolios.covar_estimation.factor_covar_data` | `factorlasso.factor_covar` |

## What stays in optimalportfolios

- `FactorCovarEstimator` — rolling estimation wrapper (uses `qis.TimePeriod`)
- `EwmaCovarEstimator` — EWMA covariance estimation
- `covar_reporting.py` — plotting and reporting
- All portfolio optimisation solvers
- `get_linear_factor_model()` method (uses `qis.LinearModel`)

## Backward compatibility

`optimalportfolios` v5+ re-exports all moved symbols from `factorlasso`.
Existing imports continue to work:

```python
# Still works (re-export from factorlasso):
from optimalportfolios import LassoModel, LassoModelType

# New recommended import:
from factorlasso import LassoModel, LassoModelType
```

## Changes needed in optimalportfolios

### 1. Add dependency
```toml
# pyproject.toml
dependencies = [
    "factorlasso>=0.1.0",
    ...
]
```

### 2. Replace optimalportfolios/lasso/__init__.py
```python
# Backward-compatible re-exports
from factorlasso import (
    LassoModelType, LassoModel, LassoEstimationResult,
    solve_lasso_cvx_problem, solve_group_lasso_cvx_problem,
)
```

### 3. Update factor_covar_estimator.py (1 line)
```python
# Old:
from optimalportfolios.lasso.lasso_estimator import LassoModel
# New:
from factorlasso import LassoModel
```

### 4. Update factor_covar_data.py imports
```python
# Old:
from optimalportfolios.covar_estimation.factor_covar_data import CurrentFactorCovarData
# New:
from factorlasso import CurrentFactorCovarData
```

### 5. Re-export in optimalportfolios/covar_estimation/__init__.py
```python
# Add re-exports for backward compatibility
from factorlasso import (
    CurrentFactorCovarData, RollingFactorCovarData, VarianceColumns,
)
```

## Key difference: qis dependency

`factorlasso` has **no dependency on qis**.  The 5 `qis` functions used by
the lasso estimator were reimplemented in `factorlasso.ewm_utils`:

| qis function | factorlasso replacement |
|-------------|----------------------|
| `qis.compute_expanding_power` | `factorlasso.ewm_utils.compute_expanding_power` |
| `qis.compute_ewm` | `factorlasso.ewm_utils.compute_ewm` |
| `qis.compute_ewm_covar` | `factorlasso.ewm_utils.compute_ewm_covar` |
| `qis.compute_ewm_covar_newey_west` | `factorlasso.ewm_utils.compute_ewm_covar_newey_west` |
| `qis.set_group_loadings` | `factorlasso.ewm_utils.set_group_loadings` |

The implementations are numerically equivalent.
