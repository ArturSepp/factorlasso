---
myst:
  html_meta:
    description: >-
      Use factorlasso with scikit-learn pipelines and model selection through estimator
      conventions, without adding scikit-learn to the factorlasso runtime dependencies.
---

# scikit-learn interoperability

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/factorlasso/commit/c89cf6d380358d4744592feeccb6073d989bc004)*

This page belongs to the documentation of [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

factorlasso follows the estimator conventions that scikit-learn relies on, while keeping
scikit-learn outside its runtime dependencies. A core installation imports NumPy, pandas, SciPy,
CVXPY and openpyxl; it neither installs nor imports scikit-learn.

## What is compatible

`LassoModel` provides `fit`, `predict`, `score`, `get_params` and `set_params`:

- constructor parameters are stored unmodified, and `fit` returns `self`;
- fitted state uses a trailing underscore;
- two-dimensional NumPy arrays are accepted and receive generated column names, while pandas
  inputs keep their labels;
- `score` returns the mean $R^2$ across response columns;
- the estimator tag hook advertises multi-output targets and missing-value support when a
  compatible scikit-learn version calls it.

The repository tests the estimator with cloning, `Pipeline`, `GridSearchCV` and
`cross_val_score`. Those tools may convert pandas objects to arrays, so use explicit DataFrames
when response and factor names are part of the downstream contract.

Install the two projects independently when composing them:

```console
python -m pip install factorlasso scikit-learn
```

For repository development, the `test` dependency group installs scikit-learn together with the
interoperability tests (`uv sync --locked --group test`). Its presence is for testing and
composition: package code performs no module-level scikit-learn import. The guarded
`__sklearn_tags__` hook imports `sklearn.utils` only when the installed scikit-learn calls that
method.

## Compatibility boundary

The authoritative surface is the set of names in `factorlasso.__all__`, rendered in the
[API reference](api.rst). The project promises scikit-learn estimator behaviour, not inheritance
from a scikit-learn base class and not a runtime dependency on its validation utilities.
Module-internal helpers and CVXPY problem internals are not stable APIs.

The package's own `LassoModelCV` and `LassoModelDiagonalityCV` always use expanding time-series
splits. A generic shuffled cross-validator changes the statistical question and can introduce
look-ahead, so choose the cross-validation object deliberately when using scikit-learn
orchestration.

The [compatibility policy](https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md)
defines the stable surface, the deprecation cycle and the numerical reproducibility boundary.
