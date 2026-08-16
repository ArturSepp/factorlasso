Task-oriented guides
====================

.. meta::
   :description: Fit constrained sparse factor models, use cluster-aware penalties, select
      regularisation, diagnose residual structure, and assemble factor covariance with
      factorlasso.

These three paths use only names in the supported top-level :doc:`api`. The examples are
deterministic, offline, and executed by the documentation test suite. They check structural and
algebraic contracts instead of solver-sensitive coefficient decimals.

Constrained and prior-centred multi-output regression
-----------------------------------------------------

Use :class:`~factorlasso.LassoModel` when several responses share the same factor panel but may
need different coefficient constraints or priors. Rows of both control matrices are responses;
columns are factors. In ``factors_beta_loading_signs`` the values mean:

* ``1``: coefficient constrained non-negative;
* ``-1``: coefficient constrained non-positive;
* ``0``: coefficient fixed at zero; and
* ``NaN``: coefficient unconstrained.

``factors_beta_prior`` has the same shape and centres the penalty on the supplied coefficient
matrix rather than on zero. A missing prior cell is treated as zero. When automatic and explicit
signs are both enabled, non-missing explicit cells override the derived layer and missing cells
inherit it. The actual solver-facing matrix is retained in ``derived_signs_``.

.. testcode:: constrained

   import numpy as np
   import pandas as pd

   import factorlasso as fl

   rng = np.random.default_rng(12)
   index = pd.date_range("2020-01-31", periods=80, freq="ME")
   x = pd.DataFrame(
       rng.normal(size=(80, 2)), index=index, columns=["growth", "rates"]
   )
   beta = np.array([[0.7, -0.2], [-0.4, 0.0], [0.1, 0.6]])
   y = pd.DataFrame(
       x.to_numpy() @ beta.T + 0.04 * rng.normal(size=(80, 3)),
       index=index,
       columns=["asset_a", "asset_b", "asset_c"],
   )
   y.loc[index[:8], "asset_c"] = np.nan

   signs = pd.DataFrame(
       [[1.0, np.nan], [-1.0, 0.0], [np.nan, 1.0]],
       index=y.columns,
       columns=x.columns,
   )
   prior = pd.DataFrame(0.0, index=y.columns, columns=x.columns)
   prior.loc["asset_a", "growth"] = 0.5

   model = fl.LassoModel(
       reg_lambda=1e-3,
       span=20,
       warmup_period=12,
       factors_beta_loading_signs=signs,
       factors_beta_prior=prior,
   ).fit(x=x, y=y)

   coefficient_constraints_hold = (
       model.coef_.loc["asset_a", "growth"] >= -1e-6
       and model.coef_.loc["asset_b", "growth"] <= 1e-6
       and abs(model.coef_.loc["asset_b", "rates"]) <= 1e-6
       and model.coef_.loc["asset_c", "rates"] >= -1e-6
   )
   print(model.coef_.shape)
   print(model.valid_mask_.shape)
   print(bool(coefficient_constraints_hold))
   print(model.derived_signs_.equals(signs))

.. testoutput:: constrained

   (3, 2)
   (79, 3)
   True
   True

Ragged histories are not imputed into the objective. Missing response cells, and every response on
a row where all factors are missing, receive zero weight through ``valid_mask_``. Demeaning happens
before the missing cells are zero-filled for the convex solver. With the default
``warmup_period=12``, a response with fewer than 12 valid observations is warned about, receives a
zero coefficient row, has ``NaN`` diagnostics, and is omitted from any fitted cluster assignment.

``span=None`` gives uniform observation weights. A finite span uses
``lambda = 1 - 2 / (span + 1)``; the quadratic loss receives square-root decay so that squaring the
row weights produces the requested nominal-span EWMA. ``alpha_const_`` is the economic intercept
paired with ``coef_`` under the same weighting. The backward-compatible ``intercept_`` is the
weighted residual mean on demeaned inputs and should not be reported as economic alpha.

Solver failure is explicit. With ``solver_fallbacks=None``, the selected solver is called once and
its exception propagates. A non-empty ordered fallback list retries after exceptions or non-optimal
statuses and raises ``cvxpy.error.SolverError`` if no solver reaches an optimal or
optimal-inaccurate status. Some low-level solve paths can instead warn and return an all-``NaN``
result when a backend returns no coefficient value, so production code should also check fitted
outputs for finiteness.

Cluster-aware HCGL, FCGL, and cooperative estimation
----------------------------------------------------

Choose the mode by the geometry you need:

* ``GROUP_LASSO`` takes a supplied ``group_data`` partition. HCGL discovers its asset partition
  from the response dependence matrix and applies the row-grouped penalty used by the package.
* ``FACTOR_CLUSTER_GROUP_LASSO`` discovers the same partition but groups each cluster-by-factor
  loading block, so the optimisation is not separable across responses.
* ``COOPERATIVE_GROUP_LASSO`` takes supplied groups;
  ``COOPERATIVE_CLUSTER_GROUP_LASSO`` discovers them. Cooperative penalties softly favour a
  shared sign within a cluster-by-factor block; they do not impose the hard pooled sign gate.
* ``external_clusters`` on ``fit`` lets HCGL or FCGL consume a point-in-time partition computed
  elsewhere while preserving their penalty geometry. It is not accepted by the other modes.

The following fit asks HCGL to discover at most two groups. ``clusters_`` is an estimated partition,
while ``linkage_`` and ``cutoff_`` retain the corresponding dendrogram metadata.

.. testcode:: groups

   import numpy as np
   import pandas as pd

   import factorlasso as fl

   rng = np.random.default_rng(23)
   x = pd.DataFrame(rng.normal(size=(90, 2)), columns=["market", "style"])
   common_1 = rng.normal(size=90)
   common_2 = rng.normal(size=90)
   y = pd.DataFrame(
       {
           "a": common_1 + 0.15 * rng.normal(size=90),
           "b": common_1 + 0.15 * rng.normal(size=90),
           "c": common_2 + 0.15 * rng.normal(size=90),
           "d": common_2 + 0.15 * rng.normal(size=90),
       }
   )
   hcgl = fl.LassoModel(
       model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
       n_clusters=2,
       reg_lambda=1e-3,
       warmup_period=12,
   ).fit(x=x, y=y)

   print(len(hcgl.clusters_))
   print(bool(hcgl.clusters_.nunique() <= 2))
   print(hcgl.linkage_.shape)

.. testoutput:: groups

   4
   True
   (3, 4)

The default dependence is Pearson and the default distance is ``1 - rho``. Under Ward linkage the
default is a stable heuristic, not exact Euclidean Ward variance minimisation; use
``DistanceTransform.CHORD`` for the exact chord geometry. A fractional dendrogram cut is calibrated
to its distance scale and does not port across distance transforms or dependence measures. Use
``n_clusters`` for like-for-like comparisons. Spearman and Gerber dependence are available as
robustness specifications.

``ClusterCorrelationTransform.REMOVE_PC1`` is also a robustness diagnostic. It removes the largest
algebraic eigencomponent from the clustering dependence matrix and restandardises that residual
matrix. It does not residualise response returns, loadings, or the assembled covariance; any fitted
change is indirect through the changed partition. Call
:func:`~factorlasso.remove_first_principal_component` directly when its eigenvalue and residual
diagnostics are needed.

Rolling partitions must use :func:`~factorlasso.compute_rolling_smoothed_clusters`, which truncates
the response panel at every estimation date and carries only prior smoother state. This executable
check changes future observations and independently verifies that earlier partitions do not move.

.. testcode:: rolling-clusters

   import numpy as np
   import pandas as pd

   import factorlasso as fl

   rng = np.random.default_rng(31)
   dates = pd.date_range("2024-01-01", periods=80, freq="D")
   base = rng.normal(size=(80, 2))
   y = pd.DataFrame(
       {
           "a": base[:, 0] + 0.1 * rng.normal(size=80),
           "b": base[:, 0] + 0.1 * rng.normal(size=80),
           "c": base[:, 1] + 0.1 * rng.normal(size=80),
           "d": base[:, 1] + 0.1 * rng.normal(size=80),
       },
       index=dates,
   )
   estimation_dates = [dates[39], dates[59]]
   cluster_spec = fl.LassoModel(
       model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
       n_clusters=2,
       warmup_period=12,
       cluster_smoother_type=fl.ClusterSmootherType.SIMILARITY_EWMA,
   )
   before = fl.compute_rolling_smoothed_clusters(y, estimation_dates, cluster_spec)
   changed_future = y.copy()
   changed_future.loc[dates[60]:, :] = 100.0
   after = fl.compute_rolling_smoothed_clusters(
       changed_future, estimation_dates, cluster_spec
   )

   unchanged = all(
       before.clusters[date].equals(after.clusters[date])
       for date in estimation_dates
   )
   print(len(before.clusters))
   print(bool(unchanged))

.. testoutput:: rolling-clusters

   2
   True

Cluster lineage answers a different question. :func:`~factorlasso.analyze_cluster_lineage` consumes
an already estimated :class:`~factorlasso.RollingFactorCovarData` and links raw groups across the
full panel for labelling, governance, and reporting. Its matching is joint across dates and its
taxonomy uses lifetime summaries. It is therefore a look-ahead diagnostic, not a point-in-time
cluster estimator, tradeable signal, or covariance estimator.

Model selection, residual diagonality, and factor covariance
-------------------------------------------------------------

:class:`~factorlasso.LassoModelCV` uses expanding-window splits and chooses ``reg_lambda`` by mean
held-out prediction R-squared across responses. That is appropriate when predictive fit is the
objective. A factor covariance model makes a different assertion: after the retained factors are
removed, the residual covariance is diagonal. Prediction score can prefer a dense model or leave a
material common residual component, so it can be the wrong selection criterion for this task.

:class:`~factorlasso.LassoModelDiagonalityCV` evaluates residual diagonality on held-out windows and
selects the sparsest penalty on the supplied grid that passes. If none passes, it selects the
minimum-statistic candidate and exposes the missing residual components in ``missing_factors_``.
For a one-off in-sample diagnostic, :func:`~factorlasso.diagnose_residuals` is available, but its
own documentation warns about in-sample optimism.

Both selectors are time-series procedures: they use expanding training windows followed by held-out
windows, never shuffled folds. CV solver failures are stored as ``NaN`` for the affected fold;
unexpected programming errors propagate.

Once factor covariance, loadings, and residual variances have been estimated under one stated
frequency and scaling convention, :class:`~factorlasso.CurrentFactorCovarData` assembles
``Sigma_y = beta Sigma_x beta' + D``. It does not infer or change annualisation. Do not mix daily
factor covariance with monthly residual variance, decimal returns with percentage returns, or
annualised and per-period inputs.

.. testcode:: covariance

   import numpy as np
   import pandas as pd

   import factorlasso as fl

   factor_covariance = pd.DataFrame(
       [[0.04, 0.01], [0.01, 0.09]],
       index=["growth", "rates"],
       columns=["growth", "rates"],
   )
   loadings = pd.DataFrame(
       [[1.0, 0.2], [0.4, -0.5]],
       index=["asset_a", "asset_b"],
       columns=factor_covariance.columns,
   )
   residual_variance = pd.Series([0.02, 0.03], index=loadings.index)
   diagnostics = pd.DataFrame(
       {fl.VarianceColumns.RESIDUAL_VARS.value: residual_variance}
   )
   snapshot = fl.CurrentFactorCovarData(
       x_covar=factor_covariance,
       y_betas=loadings,
       y_variances=diagnostics,
   )
   assembled = snapshot.get_y_covar()
   independent = (
       loadings.to_numpy()
       @ factor_covariance.to_numpy()
       @ loadings.to_numpy().T
       + np.diag(residual_variance.to_numpy())
   )

   print(assembled.shape)
   print(bool(np.allclose(assembled, independent)))
   print(bool(np.allclose(assembled, assembled.T)))

.. testoutput:: covariance

   (2, 2)
   True
   True

The container checks that the loading and residual-variance row indices agree before assembly.
``residual_var_weight`` scales only the diagonal residual term; it is a deliberate sensitivity
parameter, not an annualisation control. :class:`~factorlasso.RollingFactorCovarData` stores dated
snapshots and provides panel accessors without changing those conventions.
