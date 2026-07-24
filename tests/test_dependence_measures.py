"""dependence measures and the n_clusters cut (0.10.0): Gerber against its
published definition, the EWMA generalisation and its recursion, exact
regression of the Pearson default, robustness under contamination,
portability of the maxclust cut, LassoModel threading, and validation."""
import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import squareform

from factorlasso.cluster_utils import compute_clusters_from_corr_matrix
from factorlasso.dependence_utils import (
    DEFAULT_DEPENDENCE_MEASURE,
    DependenceMeasure,
    _normalised_ewm_weights,
    compute_dependence_matrix,
    compute_gerber_matrix,
)
from factorlasso.ewm_utils import compute_ewm_covar
from factorlasso.lasso_estimator import LassoModel, LassoModelType


def _block_panel(seed: int = 0, contamination: float = 0.0, n_blocks: int = 5,
                 per_block: int = 6, n_obs: int = 600):
    """block factor panel, optionally contaminated with fat-tailed shocks."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n_obs, n_blocks))
    series = []
    for kb in range(n_blocks):
        for _ in range(per_block):
            series.append(np.sqrt(0.7) * factors[:, kb]
                          + np.sqrt(0.3) * rng.standard_normal(n_obs))
    a = np.column_stack(series)
    if contamination > 0.0:
        rows = rng.choice(n_obs, size=int(contamination * n_obs), replace=False)
        a[rows] += 4.0 * rng.standard_t(df=2, size=(len(rows), a.shape[1]))
    return a, np.repeat(np.arange(n_blocks), per_block)


def _ari(labels_a, labels_b) -> float:
    """adjusted Rand index, computed without a scikit-learn import."""
    table = pd.crosstab(pd.Series(np.asarray(labels_a)),
                        pd.Series(np.asarray(labels_b))).to_numpy()
    n = table.sum()

    def comb2(x):
        return (x * (x - 1) / 2).sum()

    index = comb2(table)
    exp_a, exp_b = comb2(table.sum(axis=1)), comb2(table.sum(axis=0))
    expected = exp_a * exp_b / (n * (n - 1) / 2)
    maximum = 0.5 * (exp_a + exp_b)
    return float((index - expected) / (maximum - expected))


# ── Gerber against the published definition ─────────────────────────────


def test_gerber_matches_published_ratio():
    # 7 UU, 1 DD, 0 UD, 2 DU, 3 NN over 13 observations
    pairs = ([(1.0, 1.0)] * 7 + [(-1.0, -1.0)] * 1
             + [(-1.0, 1.0)] * 2 + [(0.05, 0.05)] * 3)
    a = np.array(pairs)
    a = a / a.std(axis=0)          # unit sigma, so the threshold is c
    g = compute_gerber_matrix(a, span=None, gerber_threshold=0.5)
    assert g[0, 1] == pytest.approx((7 + 1 - 0 - 2) / (13 - 3))


def test_gerber_threshold_zero_counts_every_observation():
    # with c = 0 nothing is noise, so the denominator is the full sample and
    # the statistic collapses to the mean product of return signs. Gerber
    # compares raw returns against c * sigma, so no centring enters here.
    rng = np.random.default_rng(3)
    a = rng.standard_normal((200, 4))
    g = compute_gerber_matrix(a, gerber_threshold=0.0)
    sign = np.sign(a)
    expected = (sign.T @ sign) / a.shape[0]
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_allclose(g, expected, atol=1e-12)


def test_gerber_ewma_converges_to_equal_weight():
    # deviation from the published statistic is O(1/span)
    rng = np.random.default_rng(1)
    a = rng.standard_normal((300, 8))
    base = compute_gerber_matrix(a, span=None)
    errs = [np.max(np.abs(compute_gerber_matrix(a, span=s) - base))
            for s in (1e3, 1e5, 1e7)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-5


def test_gerber_ewma_equals_first_order_recursion():
    # the O(1)-per-update roll-forward form must reproduce the batch result
    rng = np.random.default_rng(0)
    a = rng.standard_normal((400, 10))
    threshold = 0.5
    for span in (24.0, 60.0, 120.0):
        lam = 1.0 - 2.0 / (span + 1.0)
        weights = _normalised_ewm_weights(a.shape[0], span)
        mean = weights @ a
        sigma = np.sqrt(weights @ (a - mean) ** 2)
        h = threshold * sigma
        num = np.zeros((a.shape[1], a.shape[1]))
        den = np.zeros_like(num)
        for t in range(a.shape[0]):
            up = (a[t] >= h).astype(float)
            dn = (a[t] <= -h).astype(float)
            s, p = up - dn, up + dn
            num = lam * num + (1.0 - lam) * np.outer(s, s)
            den = lam * den + (1.0 - lam) * (1.0 - np.outer(1.0 - p, 1.0 - p))
        recursive = np.where(den > 0.0, num / den, 0.0)
        np.fill_diagonal(recursive, 1.0)
        batch = compute_gerber_matrix(a, span=span, gerber_threshold=threshold)
        np.testing.assert_allclose(batch, recursive, atol=1e-9)


def test_gerber_bounds_symmetry_and_nan_handling():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((250, 12))
    a[:40, 3] = np.nan          # a late-starting series
    a[100:110, 7] = np.nan      # a mid-sample gap
    for span in (None, 60.0):
        g = compute_gerber_matrix(a, span=span)
        assert np.isfinite(g).all()
        np.testing.assert_allclose(g, g.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(g), 1.0)
        assert g.min() >= -1.0 and g.max() <= 1.0


def test_gerber_is_not_centring_invariant():
    # documented asymmetry against Pearson: thresholds apply to levels
    rng = np.random.default_rng(9)
    a = rng.standard_normal((300, 5))
    shifted = a + 2.0
    np.testing.assert_allclose(compute_dependence_matrix(a),
                               compute_dependence_matrix(shifted), atol=1e-10)
    assert not np.allclose(compute_gerber_matrix(a),
                           compute_gerber_matrix(shifted), atol=1e-3)


# ── the Pearson default is unchanged ────────────────────────────────────


def test_pearson_default_reproduces_prior_paths():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((250, 9))
    a[:20, 4] = np.nan
    np.testing.assert_array_equal(
        compute_dependence_matrix(a, span=None),
        pd.DataFrame(a).corr().to_numpy())
    np.testing.assert_array_equal(
        compute_dependence_matrix(a, span=60.0),
        compute_ewm_covar(a=a, span=60.0, is_corr=True))
    assert DEFAULT_DEPENDENCE_MEASURE == DependenceMeasure.PEARSON


def test_spearman_equals_pearson_on_ranks():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((200, 6))
    ranks = pd.DataFrame(a).rank().to_numpy()
    np.testing.assert_allclose(
        compute_dependence_matrix(a, DependenceMeasure.SPEARMAN),
        pd.DataFrame(ranks).corr().to_numpy(), atol=1e-12)
    np.testing.assert_allclose(
        compute_dependence_matrix(a, 'spearman', span=60.0),
        compute_ewm_covar(a=ranks, span=60.0, is_corr=True), atol=1e-12)


def test_spearman_invariant_to_monotone_transform():
    rng = np.random.default_rng(6)
    a = np.abs(rng.standard_normal((300, 5))) + 0.1
    np.testing.assert_allclose(
        compute_dependence_matrix(a, 'spearman'),
        compute_dependence_matrix(np.log(a), 'spearman'), atol=1e-12)


# ── robustness, the reason the axis exists ──────────────────────────────


@pytest.mark.parametrize('measure', ['spearman', 'gerber'])
def test_robust_measures_beat_pearson_under_contamination(measure):
    # at a matched cluster count, both robust measures recover the true
    # block structure better than Pearson when 5% of rows are contaminated
    scores = {'pearson': [], measure: []}
    for seed in range(6):
        a, truth = _block_panel(seed=seed, contamination=0.05)
        for name in scores:
            m = compute_dependence_matrix(a, name)
            corr = pd.DataFrame(m)
            clusters, _, _ = compute_clusters_from_corr_matrix(corr, n_clusters=5)
            scores[name].append(_ari(truth, clusters.to_numpy()))
    assert np.mean(scores[measure]) > np.mean(scores['pearson'])
    assert np.mean(scores[measure]) > 0.8


def test_all_measures_agree_on_clean_data():
    a, truth = _block_panel(seed=0, contamination=0.0)
    for measure in DependenceMeasure:
        m = compute_dependence_matrix(a, measure)
        clusters, _, _ = compute_clusters_from_corr_matrix(
            pd.DataFrame(m), n_clusters=5)
        assert _ari(truth, clusters.to_numpy()) == pytest.approx(1.0)


# ── n_clusters: the portable cut ────────────────────────────────────────


def test_n_clusters_returns_requested_count():
    a, _ = _block_panel(seed=1)
    corr = pd.DataFrame(np.corrcoef(a, rowvar=False))
    for k in (2, 3, 5, 8):
        clusters, _, _ = compute_clusters_from_corr_matrix(corr, n_clusters=k)
        assert clusters.nunique() <= k


def test_n_clusters_makes_measures_comparable():
    # the fractional cut does not port across measures; maxclust does
    a, _ = _block_panel(seed=2, contamination=0.02)
    counts_frac, counts_max = set(), set()
    for measure in DependenceMeasure:
        corr = pd.DataFrame(compute_dependence_matrix(a, measure))
        by_frac, _, _ = compute_clusters_from_corr_matrix(corr, cutoff_fraction=0.5)
        by_max, _, _ = compute_clusters_from_corr_matrix(corr, n_clusters=5)
        counts_frac.add(by_frac.nunique())
        counts_max.add(by_max.nunique())
    assert len(counts_frac) > 1, "fractional cut should differ across measures"
    assert counts_max == {5}, "maxclust must equalise the granularity"


def test_n_clusters_overrides_cutoff_fraction():
    a, _ = _block_panel(seed=3)
    corr = pd.DataFrame(np.corrcoef(a, rowvar=False))
    first, _, _ = compute_clusters_from_corr_matrix(
        corr, cutoff_fraction=0.2, n_clusters=4)
    second, _, _ = compute_clusters_from_corr_matrix(
        corr, cutoff_fraction=0.9, n_clusters=4)
    assert first.equals(second)


def test_n_clusters_reported_cutoff_reproduces_the_partition():
    # the returned cutoff is the last accepted merge height, so cutting
    # the same linkage at that height must give the same labels back
    a, _ = _block_panel(seed=4)
    corr = pd.DataFrame(np.corrcoef(a, rowvar=False))
    clusters, linkage, cutoff = compute_clusters_from_corr_matrix(corr, n_clusters=5)
    replayed = spc.fcluster(linkage, cutoff, 'distance')
    assert len(np.unique(replayed)) == clusters.nunique()
    assert _ari(clusters.to_numpy(), replayed) == pytest.approx(1.0)


def test_n_clusters_none_is_the_prior_behaviour():
    a, _ = _block_panel(seed=5)
    corr = pd.DataFrame(np.corrcoef(a, rowvar=False))
    clusters, linkage, cutoff = compute_clusters_from_corr_matrix(corr)
    dist = np.clip(1.0 - corr.to_numpy(), 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    pdist = squareform(dist, checks=False)
    legacy_linkage = spc.linkage(pdist, method='ward')
    legacy_cutoff = 0.5 * np.max(pdist)
    np.testing.assert_array_equal(linkage, legacy_linkage)
    assert cutoff == legacy_cutoff
    np.testing.assert_array_equal(
        clusters.to_numpy(), spc.fcluster(legacy_linkage, legacy_cutoff, 'distance'))


# ── LassoModel threading ────────────────────────────────────────────────


def _fit_panel(seed: int = 0):
    rng = np.random.default_rng(seed)
    t, m, n = 300, 4, 12
    x = pd.DataFrame(rng.standard_normal((t, m)),
                     columns=[f'f{j}' for j in range(m)])
    beta = rng.standard_normal((n, m))
    y = pd.DataFrame(x.to_numpy() @ beta.T + 0.5 * rng.standard_normal((t, n)),
                     columns=[f'a{i}' for i in range(n)])
    return x, y


def test_lasso_model_defaults_unchanged():
    x, y = _fit_panel()
    base = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                      reg_lambda=1e-4).fit(x=x, y=y)
    explicit = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                          reg_lambda=1e-4,
                          dependence_measure=DependenceMeasure.PEARSON,
                          n_clusters=None).fit(x=x, y=y)
    assert base.clusters_.equals(explicit.clusters_)
    np.testing.assert_allclose(base.coef_.to_numpy(), explicit.coef_.to_numpy())


@pytest.mark.parametrize('measure', ['pearson', 'spearman', 'gerber'])
def test_lasso_model_threads_measure_into_clusters(measure):
    x, y = _fit_panel()
    # demean=False so the estimator clusters on the raw panel: Gerber
    # thresholds levels, so it is not invariant to centring the way
    # Pearson and Spearman are
    model = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                       reg_lambda=1e-4, dependence_measure=measure,
                       n_clusters=4, demean=False).fit(x=x, y=y)
    expected_corr = compute_dependence_matrix(
        y.to_numpy(), dependence_measure=measure, span=None)
    expected, _, _ = compute_clusters_from_corr_matrix(
        pd.DataFrame(expected_corr, index=y.columns, columns=y.columns),
        n_clusters=4)
    assert model.clusters_.equals(expected)


def test_lasso_model_span_reaches_the_dependence_measure():
    # a span change must move the Gerber clustering correlation
    x, y = _fit_panel(seed=2)
    uniform = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                         reg_lambda=1e-4, dependence_measure='gerber',
                         n_clusters=4, demean=False).fit(x=x, y=y)
    windowed = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                          reg_lambda=1e-4, dependence_measure='gerber',
                          n_clusters=4, demean=False, span=30).fit(x=x, y=y)
    expected = compute_dependence_matrix(
        y.to_numpy(), dependence_measure='gerber', span=30)
    realised, _, _ = compute_clusters_from_corr_matrix(
        pd.DataFrame(expected, index=y.columns, columns=y.columns), n_clusters=4)
    assert windowed.clusters_.equals(realised)
    assert uniform.clusters_ is not None


def test_get_set_params_roundtrip():
    model = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                       dependence_measure=DependenceMeasure.GERBER,
                       gerber_threshold=0.7, n_clusters=6)
    params = model.get_params()
    assert params['dependence_measure'] == DependenceMeasure.GERBER
    assert params['gerber_threshold'] == 0.7
    assert params['n_clusters'] == 6
    model.set_params(dependence_measure='spearman', n_clusters=3)
    assert DependenceMeasure(model.dependence_measure) == DependenceMeasure.SPEARMAN
    assert model.n_clusters == 3


# ── validation ──────────────────────────────────────────────────────────


def test_invalid_measure_raises():
    with pytest.raises(ValueError, match="dependence_measure must be one of"):
        compute_dependence_matrix(np.zeros((10, 3)), 'kendall')
    with pytest.raises(ValueError, match="dependence_measure must be one of"):
        LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                   dependence_measure='kendall')


@pytest.mark.parametrize('threshold', [-0.1, 1.5])
def test_invalid_gerber_threshold_raises(threshold):
    with pytest.raises(ValueError, match=r"gerber_threshold must lie in \[0, 1\]"):
        compute_gerber_matrix(np.zeros((10, 3)), gerber_threshold=threshold)
    with pytest.raises(ValueError, match=r"gerber_threshold must lie in \[0, 1\]"):
        LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                   gerber_threshold=threshold)


def test_invalid_n_clusters_raises():
    corr = pd.DataFrame(np.eye(5))
    with pytest.raises(ValueError, match="n_clusters must be at least 1"):
        compute_clusters_from_corr_matrix(corr, n_clusters=0)
    with pytest.raises(ValueError, match="n_clusters must not exceed"):
        compute_clusters_from_corr_matrix(corr, n_clusters=99)
    with pytest.raises(ValueError, match="n_clusters must be an integer"):
        compute_clusters_from_corr_matrix(corr, n_clusters=2.5)
    with pytest.raises(ValueError, match="n_clusters must be at least 1"):
        LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                   n_clusters=-1)


def test_non_2d_input_raises():
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        compute_dependence_matrix(np.zeros(10))
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        compute_gerber_matrix(np.zeros(10))
