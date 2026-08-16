scikit-learn interoperability
=============================

.. meta::
   :description: Use factorlasso with scikit-learn conventions without adding scikit-learn to the
      factorlasso runtime dependency set.

FactorLasso follows the estimator conventions needed by scikit-learn while keeping scikit-learn
outside its runtime dependency set. A core installation imports NumPy, pandas, SciPy, CVXPY, and
openpyxl; it does not import or install scikit-learn.

What is compatible
------------------

:class:`~factorlasso.LassoModel` provides ``fit``, ``predict``, ``score``, ``get_params``, and
``set_params``. Constructor parameters are retained unmodified, ``fit`` returns ``self``, and fitted
state uses trailing underscores. Two-dimensional NumPy arrays are accepted and receive generated
column names; pandas inputs preserve their labels. ``score`` returns mean R-squared across response
columns. The estimator tag hook advertises multi-output targets and missing-value support when a
compatible scikit-learn version calls it.

The repository tests the estimator with cloning, ``Pipeline``, ``GridSearchCV``, and
``cross_val_score``. Those tools may convert pandas objects to arrays, so use explicit DataFrames
when response and factor names are part of the downstream contract.

Install the two projects independently when composing them:

.. code-block:: console

   python -m pip install factorlasso scikit-learn

For repository development, ``python -m pip install -e ".[dev]"`` includes scikit-learn and the
interop tests. Its presence is for testing and composition; package code performs no module-level
scikit-learn import. The guarded ``__sklearn_tags__`` hook imports ``sklearn.utils`` only when the
installed scikit-learn calls that method.

Compatibility boundary
----------------------

The authoritative surface is the set of names in ``factorlasso.__all__``, rendered in :doc:`api`.
The project promises scikit-learn estimator behaviour, not inheritance from a scikit-learn base
class and not a runtime dependency on its validation utilities. Module-internal helpers and CVXPY
problem internals are not stable APIs.

FactorLasso's own :class:`~factorlasso.LassoModelCV` and
:class:`~factorlasso.LassoModelDiagonalityCV` always use expanding time-series splits. A generic
shuffled cross-validator changes the statistical question and can introduce look-ahead. Select the
cross-validation object deliberately when using scikit-learn orchestration.

See the `API compatibility policy
<https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md>`_ for the exact 0.15-series
surface, deprecation cycle, and numerical reproducibility boundary.
