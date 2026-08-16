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
   api

Choose the right path
---------------------

* Start with :doc:`getting-started` for a deterministic, offline fit using core dependencies.
* Use :doc:`api` for the supported top-level public surface.
* Read the `comparison guide <https://github.com/ArturSepp/factorlasso/blob/main/COMPARISON.md>`_
  for a qualified feature snapshot and solver trade-offs.
* Review the `compatibility policy
  <https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md>`_ before depending on a
  specific signature or numerical contract.
* Follow the `changelog <https://github.com/ArturSepp/factorlasso/blob/main/CHANGELOG.md>`_ for
  release history.

Scientific replication is separate
----------------------------------

Installing and evaluating the library does not require running a paper replication. The
`JSS replication tree <https://github.com/ArturSepp/factorlasso/tree/main/papers/jss_2026>`_ and
`sign-pooling replication tree
<https://github.com/ArturSepp/factorlasso/tree/main/papers/sign_pooling_2026>`_ preserve their own
scientific inputs and commands. Their submitted-paper status is not a software stability claim.

Project links
-------------

* `PyPI <https://pypi.org/project/factorlasso/>`_
* `Source repository <https://github.com/ArturSepp/factorlasso>`_
* `Issue tracker <https://github.com/ArturSepp/factorlasso/issues>`_
* `Citation metadata <https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff>`_
* `License <https://github.com/ArturSepp/factorlasso/blob/main/LICENSE>`_

FactorLasso is licensed under GPL-3.0-or-later.
