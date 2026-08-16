Choosing a sparse-regression workflow
=====================================

.. meta::
   :description: Neutral, source-audited guidance for choosing between FactorLasso,
      scikit-learn, skglm, and groupyr.

This page helps choose a workflow; it does not rank packages. Each project is preferable in at
least one common setting, and no popularity signal is used. Runtime depends on the design matrix,
penalty, tolerance, and hardware, so the guide makes no cross-package speed claim.

Audit snapshot
--------------

The comparison was reconciled on **2026-08-16** against FactorLasso 0.15.0, the
`scikit-learn stable documentation <https://scikit-learn.org/stable/>`_ published on that date,
`skglm 0.5 <https://pypi.org/project/skglm/0.5/>`_, and
`groupyr 0.3.4 <https://pypi.org/project/groupyr/0.3.4/>`_. The released skglm and groupyr source
distributions were inspected where their hosted API documentation did not match the latest
release. In particular, skglm's main documentation identifies itself as development documentation,
and groupyr's hosted manual still identifies itself as 0.3.2. Those manuals are cited only where
the released source preserves the described contract.

Choose by task
--------------

* Choose **scikit-learn** for standard Lasso or Elastic Net inside its pipelines, preprocessing,
  model-selection, and metadata-routing ecosystem. Its ``Lasso`` supports one or multiple targets;
  ``MultiTaskLasso`` is the separate choice when all tasks should share selected features.
* Choose **skglm** when a standard or custom sparse generalized-linear objective fits its modular
  datafit, penalty, and working-set solver architecture. Its released estimators cover standard,
  weighted, grouped, and multi-task objectives as separate APIs.
* Choose **groupyr** when the central requirement is sparse group lasso over predefined,
  non-overlapping feature groups, including its dedicated regression or classification
  cross-validation workflow.
* Choose **FactorLasso** when responses are the grouped objects and the workflow must combine
  response-specific histories, element-wise coefficient constraints or priors, group discovery,
  time-weighted estimation, residual diagnostics, and factor-covariance assembly.

Decision matrix
---------------

The word *group* is not interchangeable across these projects. In skglm and groupyr, a group is a
partition of predictor columns in a single-response group penalty. In FactorLasso, HCGL and FCGL
group response rows and apply structure across their coefficient vectors.

.. list-table:: Workflow and contract comparison
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Question
     - FactorLasso 0.15.0
     - scikit-learn stable
     - skglm 0.5
     - groupyr 0.3.4
   * - Workflow fit
     - Sparse multi-output factor models plus residual and covariance analysis.
     - General-purpose supervised-learning workflows around standard linear estimators.
     - Modular sparse generalized-linear objectives and specialized solvers.
     - Sparse group lasso regression and classification with predefined feature groups.
   * - Constraint and prior expressiveness
     - Each coefficient can be non-negative, non-positive, fixed at zero, or free; penalties may
       shrink to a supplied coefficient prior.
     - ``Lasso(positive=True)`` applies blanket non-negativity for a single target. The surveyed
       Lasso APIs do not take an element-wise sign matrix or a nonzero shrinkage target.
     - ``Lasso`` and ``GroupLasso`` offer blanket positivity; weighted penalties and modular
       constraint penalties are available. The released estimator APIs do not accept
       FactorLasso's response-by-factor sign or prior matrices.
     - ``SGL`` mixes feature-level L1 and group L2 penalties. Its released constructor has no
       coefficient-sign or prior-centre parameter.
   * - Grouping geometry
     - Supplied or discovered groups of response rows; HCGL, FCGL, cooperative, group, and sparse
       group penalties are available.
     - ``MultiTaskLasso`` groups the coefficients for one predictor across targets, creating shared
       feature support; ``Lasso`` has no feature-group assignment parameter.
     - ``GroupLasso`` partitions predictor columns for one target. ``MultiTaskLasso`` is a separate
       estimator with shared feature support across tasks.
     - ``groups`` is a list of non-overlapping predictor-column index arrays; ``l1_ratio`` mixes
       within-feature and whole-group sparsity.
   * - Multiple outputs
     - One fit estimates the full response-by-factor matrix and allows response-specific masks,
       constraints, and priors.
     - ``Lasso`` accepts multiple targets independently; ``MultiTaskLasso`` couples their support
       with an L21 penalty.
     - ``MultiTaskLasso`` accepts a target matrix. Released ``GroupLasso`` is single-output, so the
       grouped and multi-task contracts are not combined in that estimator.
     - Released ``SGL`` and ``SGLCV`` validate a one-dimensional target.
   * - Observation weights and missing values
     - EWMA weights are part of the estimator contract; each response may have its own finite-row
       mask for ragged histories.
     - ``Lasso.fit`` accepts sample weights, but the surveyed Lasso estimators require finite input;
       missing values need preprocessing. ``MultiTaskLasso.fit`` has no sample-weight parameter.
     - ``WeightedQuadratic`` can express observation weights in the modular API. Released estimators
       use finite-array validation, so missing values need preprocessing.
     - The released ``SGL`` fit API has no sample-weight parameter and uses finite, single-output
       validation; missing values need preprocessing.
   * - Covariance assembly
     - ``FactorCovarianceResult`` assembles :math:`B\Sigma_F B^\mathsf{T}+D` from the fitted loadings
       and aligned factor/residual estimates.
     - The surveyed regression estimators return coefficients and predictions, not this factor
       covariance decomposition.
     - The surveyed regression estimators do not return this decomposition. skglm's separate
       ``GraphicalLasso`` estimates a sparse precision matrix, which is a different workflow.
     - The surveyed estimators return coefficients and predictions, not a factor covariance
       decomposition.
   * - Model selection
     - Time-series cross-validation can select by prediction or residual-diagonality criteria.
     - Dedicated CV estimators and generic model-selection tools include ``TimeSeriesSplit``.
     - Regularization paths and warm starts are exposed; generic scikit-learn model selection can
       wrap compatible estimators.
     - ``SGLCV`` and ``LogisticSGLCV`` support grid or sequential model-based optimization.
   * - Solver and runtime trade-off
     - CVXPY expresses heterogeneous constraints and priors, with modeling and solver overhead that
       should be measured for the intended problem.
     - Coordinate descent is specialized for the standard Lasso-family objectives it exposes.
     - Working sets, coordinate or block-coordinate solvers, and Numba target sparse objectives;
       custom compositions must satisfy the solver/datafit/penalty contracts.
     - Proximal-gradient optimization is delegated to ``copt``; warm starts accelerate its grid
       path, while Bayesian tuning does not use the same path order.
   * - API interoperability
     - Provides ``fit``, ``predict``, ``score``, and parameter methods following scikit-learn
       conventions without making scikit-learn a runtime dependency.
     - Defines the native estimator, pipeline, model-selection, and metadata-routing contracts.
     - Estimators follow scikit-learn conventions and depend on scikit-learn.
     - Estimators follow scikit-learn conventions and depend on scikit-learn.
   * - Core runtime dependencies
     - NumPy, pandas, SciPy, CVXPY, and openpyxl.
     - NumPy, SciPy, joblib, threadpoolctl, and Narwhals in the audited release metadata.
     - NumPy, SciPy, Numba, and scikit-learn.
     - NumPy, SciPy, scikit-learn, copt, scikit-optimize, and tqdm.
   * - License
     - GPL-3.0-or-later.
     - BSD 3-Clause.
     - BSD 3-Clause.
     - BSD 3-Clause.

Primary-source ledger
---------------------

The following sources support the competitor cells above. Absence claims are deliberately scoped
to the named released estimators, not to every extension a user could write.

**scikit-learn**

* The `Lasso API <https://scikit-learn.org/stable/modules/generated/
  sklearn.linear_model.Lasso.html>`_ documents its objective, coordinate-descent solver,
  ``positive`` flag, target shapes, and sample-weighted ``fit``.
* The `MultiTaskLasso API <https://scikit-learn.org/stable/modules/generated/
  sklearn.linear_model.MultiTaskLasso.html>`_ documents the L21 objective, target matrix, and
  shared-support coefficient shape.
* The official `NaN-support list <https://scikit-learn.org/stable/modules/impute.html#estimators-
  that-handle-nan-values>`_ does not include either Lasso estimator; the same guide documents
  preprocessing alternatives.
* The `project metadata <https://pypi.org/project/scikit-learn/>`_ supports the dependency and BSD
  3-Clause rows.

**skglm**

* The `0.5 release artifacts <https://pypi.org/project/skglm/0.5/>`_ define the audited
  ``Lasso``, ``WeightedLasso``, ``MultiTaskLasso``, ``GroupLasso``, ``WeightedQuadratic``, and
  ``GraphicalLasso`` contracts and the release dependencies.
* The official `API index <https://contrib.scikit-learn.org/skglm/api.html>`_ describes the modular
  datafits, penalties, and solvers, but labels itself development documentation. This guide uses it
  only for contracts also present in the 0.5 release artifacts.
* The `skglm paper <https://www.jmlr.org/papers/v26/24-0008.html>`_ describes its working-set and
  coordinate-descent architecture. No benchmark result from the paper is turned into a runtime
  claim here.
* The `0.5 project metadata <https://pypi.org/project/skglm/0.5/>`_ records its BSD 3-Clause license.

**groupyr**

* The `0.3.4 release artifacts <https://pypi.org/project/groupyr/0.3.4/>`_ define the audited
  ``SGL`` and ``SGLCV`` signatures, finite single-output validation, dependencies, and license.
* The official `groupyr guide <https://nrdg.github.io/groupyr/>`_ documents predefined feature
  groups, the sparse-group mixing parameter, scikit-learn workflow, grid paths, Bayesian tuning,
  and the ``copt`` solver. The hosted guide says 0.3.2; these cited contracts remain present in the
  inspected 0.3.4 release.

Boundary of the comparison
--------------------------

The comparison covers released public workflows, not every objective that could be assembled by
extending a package. It excludes download counts, repository stars, maintainer-activity labels,
maximum feature-count estimates, and unreplicated speed multipliers. The three alternative
projects are documentation and benchmark references only; FactorLasso does not add them as runtime
dependencies.

If a source or release changes a cell, please open a
`correction issue <https://github.com/ArturSepp/factorlasso/issues>`_ with the project version and a
primary-source link.
