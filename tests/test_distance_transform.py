"""distance_transform feature (0.9.0): transform values and metric
properties, exact regression of the default path against the pre-0.9.0
hardcoded ``1 - rho`` construction, monotone-invariance of rank-based
linkages, threading through LassoModel, and validation errors."""
import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import squareform

from factorlasso.cluster_utils import (
    DEFAULT_DISTANCE_TRANSFORM,
    DistanceTransform,
    _corr_to_distance,
    compute_clusters_from_corr_matrix,
)
from factorlasso.lasso_estimator import LassoModel, LassoModelType


def _random_corr(n_assets: int = 30, n_obs: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_obs, n_assets))
    cols = [f"a{i:03d}" for i in range(n_assets)]
    return pd.DataFrame(np.corrcoef(x, rowvar=False), index=cols, columns=cols)


def _block_corr(seed: int = 1) -> pd.DataFrame:
    """4 blocks of 6 assets with within-block rho ~ 0.75, one block anti-correlated."""
    rng = np.random.default_rng(seed)
    n_obs = 800
    g = rng.standard_normal((n_obs, 4))
    g[:, 3] = -0.5 * g[:, 0] + np.sqrt(1 - 0.25) * g[:, 3]  # negative-rho block
    cols, series = [], []
    for kb in range(4):
        for i in range(6):
            series.append(np.sqrt(0.75) * g[:, kb] + 0.5 * rng.standard_normal(n_obs))
            cols.append(f"b{kb}_{i}")
    x = np.column_stack(series)
    return pd.DataFrame(np.corrcoef(x, rowvar=False), index=cols, columns=cols)


# ── transform values and metric properties ──────────────────────────────


def test_transform_values_at_anchor_correlations():
    corr = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]])
    expected_at_zero = {
        DistanceTransform.ONE_MINUS_RHO: 1.0,
        DistanceTransform.CHORD: np.sqrt(2.0),
        DistanceTransform.ARCCOS: np.pi / 2.0,
    }
    for transform, value in expected_at_zero.items():
        d = _corr_to_distance(corr.to_numpy(), distance_transform=transform)
        assert d[0, 1] == pytest.approx(value)
        assert d[0, 0] == 0.0 and d[1, 1] == 0.0
    corr_neg = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert _corr_to_distance(
        corr_neg, distance_transform=DistanceTransform.ONE_MINUS_RHO)[0, 1] == pytest.approx(2.0)
    assert _corr_to_distance(
        corr_neg, distance_transform=DistanceTransform.CHORD)[0, 1] == pytest.approx(2.0)
    assert _corr_to_distance(
        corr_neg, distance_transform=DistanceTransform.ARCCOS)[0, 1] == pytest.approx(np.pi)


def test_chord_equals_euclidean_distance_between_unit_vectors():
    # ||u_i - u_j||^2 = 2 (1 - rho_ij) for unit-norm centred vectors
    rng = np.random.default_rng(42)
    x = rng.standard_normal((500, 8))
    x = x - x.mean(axis=0)
    u = x / np.linalg.norm(x, axis=0)
    corr = np.corrcoef(x, rowvar=False)
    chord = _corr_to_distance(corr, distance_transform=DistanceTransform.CHORD)
    for i in range(8):
        for j in range(i + 1, 8):
            assert chord[i, j] == pytest.approx(
                np.linalg.norm(u[:, i] - u[:, j]), abs=1e-10)


def test_out_of_range_correlations_are_clipped():
    # floating-point overshoot at rho = 1 must not produce NaN under arccos
    corr = np.array([[1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]])
    for transform in DistanceTransform:
        d = _corr_to_distance(corr, distance_transform=transform)
        assert np.isfinite(d).all()
        assert d[0, 1] == pytest.approx(0.0, abs=1e-5)


# ── regression: default path identical to the pre-0.9.0 construction ────


def test_default_transform_reproduces_hardcoded_one_minus_rho():
    # pre-0.9.0 body: np.clip(1 - corr, 0, 2) with a zeroed diagonal
    for seed in range(10):
        corr = _random_corr(seed=seed)
        legacy = np.clip(1.0 - corr.to_numpy(), 0.0, 2.0)
        np.fill_diagonal(legacy, 0.0)
        new = _corr_to_distance(corr.to_numpy(),
                                distance_transform=DEFAULT_DISTANCE_TRANSFORM)
        np.testing.assert_array_equal(new, legacy)


def test_default_clusters_linkage_cutoff_unchanged():
    # end-to-end: labels, linkage, and cutoff match the legacy construction
    for corr in (_random_corr(seed=3), _block_corr(seed=1)):
        clusters, linkage, cutoff = compute_clusters_from_corr_matrix(corr)
        legacy = np.clip(1.0 - corr.fillna(0.0).to_numpy(), 0.0, 2.0)
        np.fill_diagonal(legacy, 0.0)
        pdist = squareform(legacy, checks=False)
        legacy_linkage = spc.linkage(pdist, method='ward')
        legacy_cutoff = 0.5 * np.max(pdist)
        legacy_idx = spc.fcluster(legacy_linkage, legacy_cutoff, 'distance')
        np.testing.assert_array_equal(linkage, legacy_linkage)
        assert cutoff == legacy_cutoff
        np.testing.assert_array_equal(clusters.to_numpy(), legacy_idx)


def test_explicit_default_equals_omitted_argument():
    corr = _block_corr(seed=2)
    c0, l0, k0 = compute_clusters_from_corr_matrix(corr)
    c1, l1, k1 = compute_clusters_from_corr_matrix(
        corr, distance_transform=DistanceTransform.ONE_MINUS_RHO)
    c2, l2, k2 = compute_clusters_from_corr_matrix(
        corr, distance_transform='one_minus_rho')
    assert c0.equals(c1) and c0.equals(c2)
    np.testing.assert_array_equal(l0, l1)
    np.testing.assert_array_equal(l0, l2)
    assert k0 == k1 == k2


# ── monotone invariance of rank-based linkages ──────────────────────────


def test_rank_based_linkages_build_identical_merge_trees():
    # single/complete use order statistics, so any strictly monotone
    # transform yields the identical merge tree (heights differ, pairs don't)
    corr = _block_corr(seed=1)
    for method in ('single', 'complete'):
        trees = []
        for transform in DistanceTransform:
            _, linkage, _ = compute_clusters_from_corr_matrix(
                corr, linkage_method=method, distance_transform=transform)
            trees.append(linkage[:, :2].astype(int))
        for tree in trees[1:]:
            np.testing.assert_array_equal(trees[0], tree)


def test_ward_partitions_agree_at_matched_granularity():
    # at a fixed cluster count the three transforms produce the same
    # partition on block structure — the documented robustness property
    corr = _block_corr(seed=1)
    labels = []
    for transform in DistanceTransform:
        d = _corr_to_distance(corr.to_numpy(), distance_transform=transform)
        linkage = spc.linkage(squareform(d, checks=False), method='ward')
        labels.append(spc.fcluster(linkage, 4, 'maxclust'))
    for lab in labels[1:]:
        # identical partitions up to label permutation
        _, inv0 = np.unique(labels[0], return_inverse=True)
        _, inv1 = np.unique(lab, return_inverse=True)
        mapping = {}
        for a, b in zip(inv0, inv1):
            assert mapping.setdefault(a, b) == b
        assert len(set(mapping.values())) == len(mapping)


# ── LassoModel threading ────────────────────────────────────────────────


def _hcgl_panel(seed: int = 0):
    rng = np.random.default_rng(seed)
    t, m, n = 300, 4, 12
    x = pd.DataFrame(rng.standard_normal((t, m)),
                     columns=[f'f{j}' for j in range(m)])
    beta = rng.standard_normal((n, m))
    y = pd.DataFrame(x.to_numpy() @ beta.T + 0.5 * rng.standard_normal((t, n)),
                     columns=[f'a{i}' for i in range(n)])
    return x, y


def test_lasso_model_default_fit_unchanged():
    # a fit without the new argument must match a fit passing it explicitly
    x, y = _hcgl_panel()
    m0 = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                    reg_lambda=1e-4).fit(x=x, y=y)
    m1 = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                    reg_lambda=1e-4,
                    distance_transform=DistanceTransform.ONE_MINUS_RHO).fit(x=x, y=y)
    assert m0.clusters_.equals(m1.clusters_)
    np.testing.assert_allclose(m0.coef_.to_numpy(), m1.coef_.to_numpy())


def test_lasso_model_threads_transform_into_clusters():
    # the fitted clusters_ must come from the requested transform
    x, y = _hcgl_panel()
    span = None
    for transform in (DistanceTransform.CHORD, 'arccos'):
        model = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                           reg_lambda=1e-4, span=span,
                           distance_transform=transform).fit(x=x, y=y)
        expected, _, _ = compute_clusters_from_corr_matrix(
            y.corr(), distance_transform=transform)
        assert model.clusters_.equals(expected)


def test_get_set_params_roundtrip():
    model = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                       distance_transform=DistanceTransform.CHORD)
    params = model.get_params()
    assert params['distance_transform'] == DistanceTransform.CHORD
    model.set_params(distance_transform='arccos')
    assert DistanceTransform(model.distance_transform) == DistanceTransform.ARCCOS


# ── validation ──────────────────────────────────────────────────────────


def test_invalid_transform_raises_in_cluster_function():
    corr = _random_corr(n_assets=5)
    with pytest.raises(ValueError, match="distance_transform must be one of"):
        compute_clusters_from_corr_matrix(corr, distance_transform='geodesic')


def test_invalid_transform_raises_before_single_asset_short_circuit():
    corr = pd.DataFrame([[1.0]], index=['solo'], columns=['solo'])
    with pytest.raises(ValueError, match="distance_transform must be one of"):
        compute_clusters_from_corr_matrix(corr, distance_transform='geodesic')


def test_invalid_transform_raises_at_lasso_model_construction():
    # LassoModel validates hyperparameters in __post_init__, so a bad
    # transform fails at construction, before any fit
    with pytest.raises(ValueError, match="distance_transform must be one of"):
        LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                   distance_transform='geodesic')
