"""The private mode-capability table states the documented mode families.

Validation, preparation, dispatch and both selectors read this one table, so a wrong entry
would change several behaviours at once. The expected families are written out here from the
estimator documentation rather than derived from the table.
"""

from factorlasso import LassoModelType as T
from factorlasso.linear_model._types import (
    _MODE_SPECS, _MODES_WITHOUT_SIGN_CONSTRAINTS, _UNKNOWN_MODE, _mode_spec,
)


def _family(predicate):
    """Modes whose capability record satisfies ``predicate``, in enum order."""
    return [mode for mode in T if predicate(_mode_spec(mode))]


def test_every_mode_has_one_record():
    """No mode falls back to the unknown-mode record."""
    assert set(_MODE_SPECS) == set(T)
    assert _mode_spec("not a mode") is _UNKNOWN_MODE


def test_documented_families():
    """Grouping, solver, geometry, constraints, external partitions and paths."""
    assert _family(lambda s: s.grouping == "user") == [
        T.GROUP_LASSO, T.COOPERATIVE_GROUP_LASSO]
    assert _family(lambda s: s.grouping == "discovered") == [
        T.HIERARCHICAL_CLUSTER_GROUP_LASSO, T.FACTOR_CLUSTER_GROUP_LASSO,
        T.COOPERATIVE_CLUSTER_GROUP_LASSO]
    assert _family(lambda s: s.solver == "group") == [
        T.GROUP_LASSO, T.HIERARCHICAL_CLUSTER_GROUP_LASSO, T.FACTOR_CLUSTER_GROUP_LASSO]
    assert _family(lambda s: s.block_mode == "cluster_factor") == [
        T.FACTOR_CLUSTER_GROUP_LASSO]
    assert _family(lambda s: s.external_clusters) == [
        T.HIERARCHICAL_CLUSTER_GROUP_LASSO, T.FACTOR_CLUSTER_GROUP_LASSO]
    assert _family(lambda s: s.lambda_path) == [
        T.GROUP_LASSO, T.HIERARCHICAL_CLUSTER_GROUP_LASSO, T.FACTOR_CLUSTER_GROUP_LASSO]
    assert _family(lambda s: s.single_response_lasso) == _family(lambda s: s.lambda_path)
    assert list(_MODES_WITHOUT_SIGN_CONSTRAINTS) == [
        T.UNILASSO, T.COOPERATIVE_GROUP_LASSO, T.COOPERATIVE_CLUSTER_GROUP_LASSO]
