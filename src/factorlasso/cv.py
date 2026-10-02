"""Historical location of LassoModelCV and its expanding-window splits.

Compatibility module of the 0.23.0 layout. Every name is re-exported from its canonical
owner; new code imports from :mod:`factorlasso` or :mod:`factorlasso.model_selection`. Assigning a
name here raises :class:`AttributeError` (see :mod:`factorlasso._compat`).
"""

# ruff: noqa: F401

from factorlasso._compat import guard_legacy_module
from factorlasso.linear_model._estimator import LassoModel
from factorlasso.linear_model._types import LassoModelType
from factorlasso.model_selection._cv import (
    DEFAULT_LAMBDA_GRID, expanding_window_splits, _FOLD_ERRORS, LassoModelCV,
)

guard_legacy_module(__name__)
