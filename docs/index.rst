factorlasso
===========

.. meta::
   :description: Sparse multi-output factor-model estimation with sign constraints,
      prior-centered shrinkage, data-driven grouped penalties, and consistent factor
      covariance assembly in Python.

Sparse multi-output factor-model estimation with sign constraints, prior-centered shrinkage,
data-driven grouped penalties, and consistent factor covariance assembly.

It is designed for three connected tasks:

* fit multi-output regressions with element-wise signs, informative priors, and ragged histories;
* discover or supply groups for HCGL, FCGL, sparse-group, and cooperative penalties; and
* select regularisation, diagnose residual factor structure, and assemble factor covariance.

The package is used in quantitative-finance workflows, but its regression API also accepts general
numeric pandas or NumPy inputs. It is a leaf package in the ArturSepp open-source stack and does not
depend on the portfolio-construction or reporting libraries that consume it.

Start here
----------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - To
     - Read
   * - Install the package and run the smallest deterministic fit
     - :doc:`getting-started`
   * - Run the whole workflow once, with analytics and code, on a panel with known loadings
     - :doc:`quickstart`
   * - See every exhibit with its question, sample, script and producer
     - :doc:`analytics_gallery`
   * - Follow a recipe for constrained regression, cluster-aware estimation, model selection or
       covariance assembly
     - :doc:`task-guides`
   * - Compose the estimator with scikit-learn
     - :doc:`interoperability`
   * - Choose among FactorLasso, scikit-learn, skglm and groupyr by workflow fit and documented
       trade-offs
     - :doc:`comparison`
   * - Reproduce a manuscript rather than evaluate the package
     - :doc:`scientific-replication`
   * - Look up the supported top-level public surface
     - :doc:`api`
   * - Depend on a specific signature or numerical contract
     - `Compatibility policy
       <https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md>`_
   * - Follow the release history
     - `Changelog <https://github.com/ArturSepp/factorlasso/blob/main/CHANGELOG.md>`_

Workflow and methodology
------------------------

The package estimates the factor model and the covariance decomposition it implies,

.. math::

   Y_t = \alpha + \beta X_t + \varepsilon_t,
   \qquad
   \Sigma_y = \beta \Sigma_x \beta^{\top} + D,

for :math:`N` responses and :math:`M` factors, with :math:`\beta` an :math:`N \times M` loading
matrix. Each step of the workflow has a methodology article with the definition, the equations,
a worked example checked against an independent calculation, a reproducible exhibit, the public
entry points, limitations and references.

.. list-table::
   :header-rows: 1
   :widths: 6 34 30 30

   * - Step
     - Question
     - Entry point
     - Article
   * - 1
     - Which loadings does each response carry, and what does the penalty cost?
     - ``LassoModel``
     - :doc:`sparse_factor_model`
   * - 2
     - Which signs are admissible, and what should a strong penalty shrink to?
     - ``factors_beta_loading_signs``, ``factors_beta_prior``
     - :doc:`sign_constraints_and_priors`
   * - 3
     - Should evidence be pooled across similar responses?
     - ``LassoModelType``, ``solve_group_lasso_cvx_problem``
     - :doc:`group_penalties_hcgl_fcgl`
   * - 4
     - Is anything systematic left in the residuals?
     - ``diagnose_residuals``
     - :doc:`residual_diagnostics`
   * - 5
     - Which covariance matrix does the model imply?
     - ``CurrentFactorCovarData``
     - :doc:`factor_covariance_assembly`

The :doc:`quickstart` runs the five steps on one panel. Articles on observation weighting and
ragged histories, derived sign constraints, adaptive weights, the cooperative and
univariate-guided variants, penalty selection, cluster discovery and stability, cluster lineage,
nowcasting and the empirical residual correlation are planned; until they are written the
:doc:`task-guides` and the repository README cover those features.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   getting-started
   quickstart
   analytics_gallery
   task-guides
   interoperability
   comparison
   scientific-replication
   api

.. toctree::
   :maxdepth: 1
   :caption: Methodology: estimation

   sparse_factor_model
   sign_constraints_and_priors
   group_penalties_hcgl_fcgl

.. toctree::
   :maxdepth: 1
   :caption: Methodology: covariance and residuals

   factor_covariance_assembly
   residual_diagnostics

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   documentation_standard

Project links
-------------

* `PyPI <https://pypi.org/project/factorlasso/>`_
* `Source repository <https://github.com/ArturSepp/factorlasso>`_
* `Issue tracker <https://github.com/ArturSepp/factorlasso/issues>`_
* `Citation metadata <https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff>`_
* `License <https://github.com/ArturSepp/factorlasso/blob/main/LICENSE>`_

FactorLasso is licensed under GPL-3.0-or-later.
