"""Shared response-clustering kernels of the fit and of the rolling partitions.

The fit and the rolling path share the prepared-panel dependence and the linkage/cut kernel
but prepare their observations differently: the fit masks a response observation when the
whole factor row is missing, the rolling path has no factors. Equal prepared inputs must give
equal outputs; the input difference must stay visible.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from factorlasso import (
    LassoModel, LassoModelType, compute_clusters_from_corr_matrix, get_x_y_np,
)
from factorlasso.cluster._hierarchical import _corr_to_distance
from factorlasso.cluster._response import (
    _ClusterGeometry, _ResponseDependenceSettings, linkage_and_cut, prepared_response_dependence,
)
from factorlasso.cluster._smoothing import (
    _cluster_distance_matrix, _correlation_input, _iter_correlation_inputs,
)


def _panel(missing_factor_rows=False):
    """Two blocks of three responses, three factors, ragged response histories."""
    rng = np.random.default_rng(20261002)
    index = pd.date_range("2010-01-31", periods=120, freq="ME")
    x = pd.DataFrame(rng.standard_normal((120, 3)), index=index, columns=["f0", "f1", "f2"])
    blocks = rng.standard_normal((120, 2))
    y = pd.DataFrame(blocks[:, [0, 0, 0, 1, 1, 1]] + 0.5 * rng.standard_normal((120, 6)),
                     index=index, columns=[f"y{j}" for j in range(6)])
    y.iloc[:20, 0] = np.nan
    if missing_factor_rows:
        x.iloc[::5, :] = np.nan
    return x, y


def _model(**overrides):
    """An HCGL configuration whose clustering settings the kernels read."""
    params = dict(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, n_clusters=2,
                  span=36, dependence_measure="spearman")
    params.update(overrides)
    return LassoModel(**params)


@pytest.mark.parametrize("n_clusters", [None, 2, 4])
def test_linkage_and_cut_matches_a_direct_scipy_cut(n_clusters):
    """Labels, linkage and cut height equal SciPy computed directly, ties included."""
    distance = np.array([[0.0, 0.2, 0.2, 0.9, 0.9],
                         [0.2, 0.0, 0.2, 0.9, 0.9],
                         [0.2, 0.2, 0.0, 0.9, 0.9],
                         [0.9, 0.9, 0.9, 0.0, 0.4],
                         [0.9, 0.9, 0.9, 0.4, 0.0]])
    labels, linkage, cutoff = linkage_and_cut(distance, 5, "average", 0.5, n_clusters)
    reference_linkage = hierarchy.linkage(squareform(distance), method="average")
    np.testing.assert_array_equal(linkage, reference_linkage)
    if n_clusters is None:
        assert cutoff == 0.5 * 0.9
        reference = hierarchy.fcluster(reference_linkage, 0.45, criterion="distance")
    else:
        reference = hierarchy.fcluster(reference_linkage, n_clusters, criterion="maxclust")
        merges = 5 - len(np.unique(reference))
        assert cutoff == (reference_linkage[merges - 1, 2] if merges else 0.0)
    np.testing.assert_array_equal(labels, reference)


def test_distance_and_correlation_cuts_agree_on_the_same_distance():
    """The rolling distance cut and the correlation cut give one partition for one distance."""
    _, y = _panel()
    corr = y.corr()
    geometry = _ClusterGeometry.from_model(_model(n_clusters=None, cutoff_fraction=0.6))
    from_corr = compute_clusters_from_corr_matrix(corr, **geometry.as_kwargs())
    distance = _corr_to_distance(corr.to_numpy(), distance_transform=geometry.distance_transform)
    from_distance = _cluster_distance_matrix(distance, corr, geometry)
    pd.testing.assert_series_equal(from_corr[0], from_distance[0])
    np.testing.assert_array_equal(from_corr[1], from_distance[1])
    assert float(from_corr[2]) == from_distance[2]
    assert isinstance(from_distance[2], float)


@pytest.mark.parametrize("measure, span", [("pearson", None), ("spearman", 36),
                                           ("gerber", 24)])
def test_equal_prepared_inputs_give_the_fit_dependence(measure, span):
    """Without fully missing factor rows, the rolling input equals the fit's dependence."""
    x, y = _panel()
    model = _model(dependence_measure=measure, span=span)
    _, y_np, mask = get_x_y_np(x, y, span=span)
    fit_dependence = prepared_response_dependence(
        y_np, mask, y.columns, measure, span, model.gerber_threshold)
    pd.testing.assert_frame_equal(_correlation_input(y, model), fit_dependence)


def test_fully_missing_factor_rows_remain_an_input_difference():
    """The fit masks dates without factors; the rolling input, which has none, does not."""
    x, y = _panel(missing_factor_rows=True)
    model = _model()
    _, y_np, fit_mask = get_x_y_np(x, y, span=model.span)
    dummy = pd.DataFrame(0.0, index=y.index, columns=["dummy"])
    _, _, rolling_mask = get_x_y_np(dummy, y, span=model.span)
    assert (fit_mask < rolling_mask).any()
    fit_dependence = prepared_response_dependence(
        y_np, fit_mask, y.columns, model.dependence_measure, model.span,
        model.gerber_threshold)
    assert not np.allclose(_correlation_input(y, model), fit_dependence)


def test_fit_clusters_are_the_kernel_partition_of_the_fit_preparation():
    """``clusters_`` of an HCGL fit equals the shared kernels applied to its own panel."""
    x, y = _panel(missing_factor_rows=True)
    model = _model().fit(x, y)
    _, y_np, mask = get_x_y_np(x, y, span=model.span)
    dependence = prepared_response_dependence(
        y_np, mask, y.columns, model.dependence_measure, model.span, model.gerber_threshold)
    clusters, linkage, _ = compute_clusters_from_corr_matrix(
        dependence, **_ClusterGeometry.from_model(model).as_kwargs())
    pd.testing.assert_series_equal(model.clusters_, clusters)
    np.testing.assert_array_equal(model.linkage_, linkage)


@pytest.mark.parametrize("measure, span", [("pearson", 36), ("spearman", 36)])
def test_rolling_helpers_accept_a_model_or_its_settings(measure, span):
    """Private rolling helpers take an estimator, as downstream scripts pass, or a record."""
    _, y = _panel()
    model = _model(dependence_measure=measure, span=span)
    dates = list(y.index[60::20])
    settings = _ResponseDependenceSettings.from_model(model)
    for (date_a, a), (date_b, b) in zip(_iter_correlation_inputs(y, dates, model),
                                        _iter_correlation_inputs(y, dates, settings)):
        assert date_a == date_b
        pd.testing.assert_frame_equal(a, b)
