Installation and first success
==============================

.. meta::
   :description: Install factorlasso and run a deterministic sparse multi-output regression
      offline with core Python dependencies.

Installation
------------

FactorLasso supports Python 3.10 and later. Install the released package from PyPI:

.. code-block:: console

   python -m pip install factorlasso

The core installation includes NumPy, pandas, SciPy, CVXPY, and openpyxl. It does not install
scikit-learn or Matplotlib. scikit-learn interoperability is implemented through estimator
conventions, while plotting is imported only by the methods that need it.

Deterministic offline fit
-------------------------

This example creates a small synthetic factor model, fits it through the supported top-level API,
and checks structural output rather than solver-sensitive coefficient decimals. It needs no
network, proprietary data, notebook, plotting backend, or sibling package.

.. code-block:: python

   import numpy as np
   import pandas as pd

   import factorlasso as fl

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

Expected structural output:

.. code-block:: text

   (5, 3)
   (120, 5)
   True

What the result means
---------------------

``coef_`` is indexed by response and factor, ``intercept_`` stores the fitted economic intercept,
and ``predict`` preserves the response columns for pandas inputs. Fitted attributes use a trailing
underscore. See :doc:`api` for the complete supported top-level surface.

Next steps
----------

The repository `README <https://github.com/ArturSepp/factorlasso/blob/main/README.md>`_ documents
sign matrices, prior-centred penalties, group construction, cross-validation, residual diagnostics,
covariance assembly, rolling estimation, and cluster lineage. Focused task guides will be added
without changing the estimator's numerical contracts.
