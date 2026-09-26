---
myst:
  html_meta:
    description: >-
      Install factorlasso from PyPI and run a deterministic sparse multi-output regression
      offline with the core Python dependencies, then read what the fitted attributes mean.
---

# Installation and first fit

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/factorlasso/commit/01d87fd8542897d6157b8b3ffb250d6240ae5e9a)*

This page belongs to the documentation of [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

## Installation

factorlasso supports Python 3.10 and later. Install the released package from PyPI:

```console
python -m pip install factorlasso
```

The core installation includes NumPy, pandas, SciPy, CVXPY and openpyxl. It does not install
scikit-learn or Matplotlib. Interoperability with scikit-learn is achieved by following its
estimator conventions, and plotting is imported only by the methods that draw. The optional
`docs` extra installs the documentation toolchain, and the `simulations` extra the dependencies
of the JSS simulation harness in the source repository.

## A deterministic offline fit

The example below builds a small synthetic factor model with five responses and three factors,
fits it through the supported top-level API and prints structural output rather than
solver-sensitive decimals. It needs no network, data file, notebook, plotting backend or sibling
package. The code is an excerpt of
[`examples/docs/getting_started.py`](../examples/docs/getting_started.py), which also checks the
loadings against ordinary least squares.

```python
import numpy as np
import pandas as pd

import factorlasso as fl
```

```python
rng = np.random.default_rng(7)
x = pd.DataFrame(rng.normal(size=(120, 3)), columns=["growth", "rates", "inflation"])
beta = np.array(
    [
        [0.8, 0.0, -0.2],
        [0.0, 0.6, 0.1],
        [-0.4, 0.2, 0.0],
        [0.3, -0.5, 0.2],
        [0.1, 0.1, 0.7],
    ]
)
y = pd.DataFrame(
    x.to_numpy() @ beta.T + 0.05 * rng.normal(size=(len(x), len(beta))),
    columns=[f"asset_{i}" for i in range(len(beta))],
)

model = fl.LassoModel(reg_lambda=1e-4).fit(x=x, y=y)
prediction = model.predict(x)

print(model.coef_.shape)
print(prediction.shape)
print(bool(np.isfinite(prediction.to_numpy()).all()))
```

Expected output:

```text
(5, 3)
(120, 5)
True
```

## What the result means

- `coef_` is the $N \times M$ loading matrix, indexed by response and then by factor. Here it is
  $5 \times 3$, and at this small penalty it is close to ordinary least squares.
- `alpha_const_` holds the fitted regression intercept of each response, in the units of the
  responses. `intercept_` is a solver diagnostic, not the intercept; see the
  [sparse factor model](sparse_factor_model.md) article.
- `predict` returns a DataFrame with the response columns for pandas inputs.
- Fitted attributes end with an underscore, as in scikit-learn.

The [conventions page](conventions.md) defines the shapes, units and sign encoding used on every
page, and the [API reference](api.rst) lists the complete supported top-level surface.

## Next steps

Continue with the [quickstart](quickstart.md). It runs penalty selection, estimation with derived
signs, residual diagnostics and covariance assembly on one panel with known loadings. The
methodology articles, listed on the [documentation home](index.md), explain each step, and the
[examples and recipes](task-guides.md) route from a task to its article and script, with checked
recipes for constrained regression, cluster-aware estimation and covariance assembly.
