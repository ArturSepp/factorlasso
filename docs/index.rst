factorlasso
===========

.. meta::
   :description: Sparse multi-output factor-model estimation with sign constraints,
      prior-centred shrinkage, data-driven grouped penalties, and consistent factor
      covariance assembly in Python.

``factorlasso`` is a Python library for sparse multi-output factor-model estimation with sign
constraints, prior-centred shrinkage, data-driven grouped penalties, and consistent factor
covariance assembly.

It is designed for three connected tasks:

* fit multi-output regressions with element-wise signs, informative priors, and ragged histories;
* discover or supply groups for HCGL, FCGL, sparse-group, and cooperative penalties; and
* select regularisation, diagnose residual factor structure, and assemble factor covariance.

The package is used in quantitative-finance workflows, but its regression API also accepts general
numeric pandas or NumPy inputs. It is a leaf package in the ArturSepp open-source stack and does not
depend on the portfolio-construction or reporting libraries that consume it.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   getting-started
   task-guides
   interoperability
   scientific-replication
   api

Choose the right path
---------------------

* Start with :doc:`getting-started` for a deterministic, offline fit using core dependencies.
* Follow :doc:`task-guides` for constrained regression, cluster-aware estimation, or model
  selection and covariance assembly.
* Check :doc:`interoperability` before composing the estimator with scikit-learn.
* Use :doc:`scientific-replication` only when reproducing a manuscript rather than evaluating the
  package.
* Use :doc:`api` for the supported top-level public surface.
* Read the `comparison guide <https://github.com/ArturSepp/factorlasso/blob/main/COMPARISON.md>`_
  for a qualified feature snapshot and solver trade-offs.
* Review the `compatibility policy
  <https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md>`_ before depending on a
  specific signature or numerical contract.
* Follow the `changelog <https://github.com/ArturSepp/factorlasso/blob/main/CHANGELOG.md>`_ for
  release history.

Project links
-------------

* `PyPI <https://pypi.org/project/factorlasso/>`_
* `Source repository <https://github.com/ArturSepp/factorlasso>`_
* `Issue tracker <https://github.com/ArturSepp/factorlasso/issues>`_
* `Citation metadata <https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff>`_
* `License <https://github.com/ArturSepp/factorlasso/blob/main/LICENSE>`_

FactorLasso is licensed under GPL-3.0-or-later.
