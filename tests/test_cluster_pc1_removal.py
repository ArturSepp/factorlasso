"""Tests for dominant-common-mode removal in clustering correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlasso import (
    ClusterCorrelationTransform,
    apply_cluster_correlation_transform,
    compute_clusters_from_corr_matrix,
    remove_first_principal_component,
)


def _weighted_correlation(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the weighted correlation of complete observations."""
    normalized = weights / np.sum(weights)
    centered = values - normalized @ values
    variances = np.sum(normalized[:, None] * centered**2, axis=0)
    standardized = centered / np.sqrt(variances)
    return standardized.T @ (normalized[:, None] * standardized)


def _observation_space_reference(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove PC1 scores from weighted standardized observations."""
    normalized = weights / np.sum(weights)
    centered = values - normalized @ values
    variances = np.sum(normalized[:, None] * centered**2, axis=0)
    standardized = centered / np.sqrt(variances)
    corr = standardized.T @ (normalized[:, None] * standardized)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    loading = eigenvectors[:, -1]
    scores = standardized @ loading
    residuals = standardized - np.outer(scores, loading)
    residual_corr = _weighted_correlation(residuals, normalized)
    return corr, residual_corr


def _adjusted_rand(labels: np.ndarray, truth: np.ndarray) -> float:
    """Compute adjusted Rand without adding a scikit-learn dependency."""
    rows = np.unique(labels, return_inverse=True)[1]
    columns = np.unique(truth, return_inverse=True)[1]
    table = np.zeros((rows.max() + 1, columns.max() + 1), dtype=int)
    np.add.at(table, (rows, columns), 1)

    def choose_two(values):
        """Return the elementwise number of unordered pairs."""
        return values * (values - 1) / 2.0

    same_both = np.sum(choose_two(table))
    same_rows = np.sum(choose_two(table.sum(axis=1)))
    same_columns = np.sum(choose_two(table.sum(axis=0)))
    total = choose_two(table.sum())
    expected = same_rows * same_columns / total
    maximum = 0.5 * (same_rows + same_columns)
    return float((same_both - expected) / (maximum - expected))


@pytest.mark.parametrize("span", [None, 36])
def test_matrix_deflation_matches_observation_space_reference(span):
    """Matrix and observation-space PC1 removal must agree to 1e-12."""
    rng = np.random.default_rng(20260816)
    common = rng.standard_normal((320, 1))
    blocks = rng.standard_normal((320, 3))
    loadings = np.linspace(0.5, 1.4, 9)[None, :]
    values = common @ loadings + 0.65 * blocks[:, np.repeat(range(3), 3)]
    values += 0.25 * rng.standard_normal(values.shape)
    if span is None:
        weights = np.ones(len(values))
    else:
        decay = 1.0 - 2.0 / (span + 1.0)
        weights = decay ** np.arange(len(values) - 1, -1, -1)
    corr, reference = _observation_space_reference(values, weights)
    labels = [f"asset_{i}" for i in range(corr.shape[0])]

    result = remove_first_principal_component(
        pd.DataFrame(corr, index=labels, columns=labels)
    )

    np.testing.assert_allclose(result.correlation, reference, rtol=0.0, atol=1e-12)
    assert result.correlation.index.tolist() == labels
    assert result.correlation.columns.tolist() == labels


def test_none_dispatch_is_an_exact_bypass():
    """The default transform must not copy or numerically touch its input."""
    corr = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], columns=["b", "a"], index=["b", "a"])

    transformed = apply_cluster_correlation_transform(corr)

    assert transformed is corr
    assert transformed.to_numpy().tobytes() == corr.to_numpy().tobytes()


def test_rank_one_plus_blocks_recovers_residual_structure():
    """Removing a strong common mode must improve matched-count block recovery."""
    rng = np.random.default_rng(7)
    observations = 1600
    truth = np.repeat(np.arange(3), 4)
    common = rng.standard_normal((observations, 1))
    block_factors = rng.standard_normal((observations, 3))
    market_loadings = np.tile([0.2, 0.6, 1.0, 1.4], 3)[None, :]
    values = common @ market_loadings
    values += 0.2 * block_factors[:, truth]
    values += 0.5 * rng.standard_normal(values.shape)
    corr = pd.DataFrame(np.corrcoef(values, rowvar=False))

    raw = compute_clusters_from_corr_matrix(corr, n_clusters=3)[0]
    residual_corr = remove_first_principal_component(corr).correlation
    residual = compute_clusters_from_corr_matrix(residual_corr, n_clusters=3)[0]

    assert _adjusted_rand(residual.to_numpy(), truth) == pytest.approx(1.0)
    assert _adjusted_rand(residual.to_numpy(), truth) > _adjusted_rand(raw.to_numpy(), truth)


def test_transform_properties_and_eigenvector_sign_invariance():
    """Output must be a labelled correlation and deflation must ignore eigenvector sign."""
    corr = pd.DataFrame(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]],
        index=["z", "x", "y"],
        columns=["z", "x", "y"],
    )
    result = remove_first_principal_component(corr)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    loading = eigenvectors[:, -1]

    np.testing.assert_array_equal(
        np.outer(loading, loading), np.outer(-loading, -loading)
    )
    np.testing.assert_allclose(result.correlation, result.correlation.T, atol=0.0)
    np.testing.assert_array_equal(np.diag(result.correlation), np.ones(3))
    assert np.max(np.abs(result.correlation.to_numpy())) <= 1.0
    assert result.removed_eigenvalue == pytest.approx(eigenvalues[-1])
    assert result.removed_variance_share == pytest.approx(eigenvalues[-1] / 3.0)
    assert result.eigengap == pytest.approx(eigenvalues[-1] - eigenvalues[-2])
    assert result.dominant_component_unique


def test_edge_cases_are_total_and_missing_pairs_are_counted():
    """Single, tied, perfectly common, and missing-pair matrices have frozen behavior."""
    single = pd.DataFrame([[1.0]], index=["only"], columns=["only"])
    single_result = remove_first_principal_component(single)
    pd.testing.assert_frame_equal(single_result.correlation, single)
    assert single_result.removed_variance_share == 0.0

    identity = pd.DataFrame(np.identity(3), columns=list("abc"), index=list("abc"))
    identity_result = remove_first_principal_component(identity)
    pd.testing.assert_frame_equal(identity_result.correlation, identity)
    assert not identity_result.dominant_component_unique
    assert len(identity_result.isolated_assets) == 1

    common = pd.DataFrame(np.ones((3, 3)), columns=list("abc"), index=list("abc"))
    common_result = remove_first_principal_component(common)
    pd.testing.assert_frame_equal(common_result.correlation, identity, atol=1e-12)
    assert set(common_result.isolated_assets) == set(common.columns)

    missing = pd.DataFrame(
        [[1.0, np.nan, 0.2], [np.nan, 1.0, 0.1], [0.2, 0.1, 1.0]],
        columns=list("abc"),
        index=list("abc"),
    )
    missing_result = remove_first_principal_component(missing)
    assert missing_result.missing_offdiagonal_pairs == 1
    assert np.isfinite(missing_result.correlation.to_numpy()).all()


@pytest.mark.parametrize(
    "matrix, match",
    [
        (pd.DataFrame(np.ones((2, 3))), "square"),
        (
            pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["a", "b"], columns=["b", "a"]),
            "labels",
        ),
        (pd.DataFrame([[1.0, 1.1], [1.1, 1.0]]), "outside"),
        (pd.DataFrame([[np.nan, 0.2], [0.2, 1.0]]), "diagonal"),
    ],
)
def test_malformed_correlation_matrices_raise(matrix, match):
    """Malformed correlation inputs must fail with a diagnostic message."""
    with pytest.raises(ValueError, match=match):
        remove_first_principal_component(matrix)


def test_invalid_dispatch_transform_raises():
    """Unknown transform names must enumerate the supported choices."""
    corr = pd.DataFrame(np.identity(2))
    with pytest.raises(ValueError, match="cluster correlation transform"):
        apply_cluster_correlation_transform(corr, "remove_everything")


def test_string_dispatch_matches_direct_transform():
    """The public string convention must match the enum and direct helper."""
    corr = pd.DataFrame([[1.0, 0.7], [0.7, 1.0]], columns=list("ab"), index=list("ab"))
    direct = remove_first_principal_component(corr).correlation
    string = apply_cluster_correlation_transform(corr, "remove_pc1")
    enum = apply_cluster_correlation_transform(
        corr, ClusterCorrelationTransform.REMOVE_PC1
    )
    pd.testing.assert_frame_equal(string, direct)
    pd.testing.assert_frame_equal(enum, direct)


def _fit_panel(seed: int = 17) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a deterministic factor and response panel for integration tests."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=180)
    x = pd.DataFrame(rng.standard_normal((180, 3)), index=index, columns=list("xyz"))
    common = rng.standard_normal((180, 1))
    blocks = rng.standard_normal((180, 3))
    truth = np.repeat(np.arange(3), 3)
    y = common @ np.linspace(0.5, 1.3, 9)[None, :]
    y += 0.55 * blocks[:, truth] + 0.25 * rng.standard_normal((180, 9))
    return x, pd.DataFrame(y, index=index, columns=[f"a{i}" for i in range(9)])


def _same_partition(left: pd.Series, right: pd.Series) -> bool:
    """Return whether two label series encode the same partition."""
    common = left.index.intersection(right.index)
    if len(common) != len(left) or len(common) != len(right):
        return False
    left_values = left.loc[common].to_numpy()
    right_values = right.loc[common].to_numpy()
    return bool(
        np.array_equal(
            left_values[:, None] == left_values[None, :],
            right_values[:, None] == right_values[None, :],
        )
    )


def test_lasso_model_threads_transform_and_default_is_exact():
    """Direct cluster discovery must thread de-PC1 while NONE remains exact."""
    from factorlasso import DependenceMeasure, LassoModel, LassoModelType
    from factorlasso.dependence_utils import compute_dependence_matrix

    x, y = _fit_panel()
    base = dict(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-4,
        n_clusters=3,
        demean=False,
    )
    implicit = LassoModel(**base).fit(x, y)
    explicit = LassoModel(
        **base, cluster_correlation_transform=ClusterCorrelationTransform.NONE
    ).fit(x, y)
    pd.testing.assert_series_equal(implicit.clusters_, explicit.clusters_)
    np.testing.assert_array_equal(implicit.linkage_, explicit.linkage_)
    assert implicit.cutoff_ == explicit.cutoff_
    np.testing.assert_array_equal(implicit.coef_, explicit.coef_)

    model = LassoModel(
        **base, cluster_correlation_transform=ClusterCorrelationTransform.REMOVE_PC1
    ).fit(x, y)
    corr = compute_dependence_matrix(
        y.to_numpy(), dependence_measure=DependenceMeasure.PEARSON, span=None
    )
    corr = pd.DataFrame(corr, index=y.columns, columns=y.columns)
    expected = compute_clusters_from_corr_matrix(
        remove_first_principal_component(corr).correlation, n_clusters=3
    )
    assert _same_partition(model.clusters_, expected[0])
    np.testing.assert_array_equal(model.linkage_, expected[1])
    assert model.cutoff_ == expected[2]


def test_lasso_model_transform_get_set_and_validation():
    """The declarative field must clone, round-trip, and reject unknown values."""
    from factorlasso import LassoModel

    model = LassoModel(cluster_correlation_transform="remove_pc1")
    assert model.get_params()["cluster_correlation_transform"] == "remove_pc1"
    copied = model.copy()
    assert copied.cluster_correlation_transform == "remove_pc1"
    model.set_params(cluster_correlation_transform=ClusterCorrelationTransform.NONE)
    assert model.cluster_correlation_transform == ClusterCorrelationTransform.NONE
    with pytest.raises(ValueError, match="cluster_correlation_transform must be one of"):
        LassoModel(cluster_correlation_transform="remove_pc2")


def test_rolling_depc1_matches_date_by_date_fit_with_exact_eligibility():
    """Unsmoothed rolling de-PC1 must equal independent point-in-time fits."""
    from factorlasso import LassoModel, LassoModelType, compute_rolling_smoothed_clusters

    x, y = _fit_panel()
    dates = list(y.index[[89, 129, 169]])
    eligibility = pd.DataFrame(True, index=dates, columns=y.columns)
    eligibility.loc[dates[0], ["a7", "a8"]] = False
    model = LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        cluster_correlation_transform="remove_pc1",
        span=36,
        n_clusters=3,
    )
    rolling = compute_rolling_smoothed_clusters(
        y, dates, model, eligibility=eligibility
    )
    for date in dates:
        assets = eligibility.columns[eligibility.loc[date]]
        fitted = model.copy().fit(x.loc[:date], y.loc[:date, assets])
        assert _same_partition(rolling.clusters[date], fitted.clusters_)
        np.testing.assert_array_equal(rolling.linkages[date], fitted.linkage_)
        assert rolling.cutoffs[date] == fitted.cutoff_


def test_rolling_depc1_is_causal_and_ineligible_assets_have_no_influence():
    """Future observations and excluded columns cannot alter a current residual partition."""
    from factorlasso import LassoModel, LassoModelType, compute_rolling_smoothed_clusters

    _, y = _fit_panel(seed=22)
    date = y.index[129]
    model = LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        cluster_correlation_transform="remove_pc1",
        span=36,
        n_clusters=3,
    )
    included = y.columns[:-1]
    eligibility = pd.DataFrame(False, index=[date], columns=y.columns)
    eligibility.loc[date, included] = True
    full = compute_rolling_smoothed_clusters(
        y, [date], model, eligibility=eligibility
    )
    restricted = compute_rolling_smoothed_clusters(y[included], [date], model)
    assert _same_partition(full.clusters[date], restricted.clusters[date])
    np.testing.assert_array_equal(full.linkages[date], restricted.linkages[date])
    assert full.cutoffs[date] == restricted.cutoffs[date]

    perturbed = y.copy()
    perturbed.loc[perturbed.index > date] *= 1e6
    future = compute_rolling_smoothed_clusters(
        perturbed, [date], model, eligibility=eligibility
    )
    assert _same_partition(full.clusters[date], future.clusters[date])
    np.testing.assert_array_equal(full.linkages[date], future.linkages[date])


@pytest.mark.parametrize("defect", ["dates", "columns", "dtype"])
def test_rolling_exact_eligibility_validation(defect):
    """Eligibility must be Boolean and exactly cover dates and response columns."""
    from factorlasso import LassoModel, compute_rolling_smoothed_clusters

    _, y = _fit_panel()
    dates = list(y.index[[89, 129]])
    eligibility = pd.DataFrame(True, index=dates, columns=y.columns)
    if defect == "dates":
        eligibility = eligibility.iloc[:1]
    elif defect == "columns":
        eligibility = eligibility.iloc[:, :-1]
    else:
        eligibility = eligibility.astype(int)
    with pytest.raises(ValueError, match="eligibility"):
        compute_rolling_smoothed_clusters(y, dates, LassoModel(), eligibility=eligibility)
