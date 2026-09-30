"""Four residual structures, independent block means, and portfolio selection."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import factorlasso as fl


@pytest.fixture
def snapshot():
    """PSD residual dependence with distinct means, negative pairs, and zero-risk cash."""
    names = pd.Index(['a', 'b', 'c', 'd', 'e', 'cash'])
    loadings = np.array([.8, .6, .4, -.5, .7, .9])
    corr = np.outer(loadings, loadings)
    np.fill_diagonal(corr, 1.)
    dates = pd.date_range('2020-01-31', periods=3, freq='ME')
    prepared = fl.ResidualCorrelationData(
        correlation=pd.DataFrame(corr, index=names, columns=names),
        residual_returns=pd.DataFrame(np.ones((3, 6)), index=dates, columns=names),
        asset_metadata=pd.DataFrame(index=names), frequency='ME', span=3.,
        observation_date=dates[-1], estimation_date=dates[-1],
    )
    return fl.CurrentFactorCovarData(
        x_covar=pd.DataFrame([[.04]], index=['f'], columns=['f']),
        y_betas=pd.DataFrame({'f': [.5] * 6}, index=names),
        y_variances=pd.DataFrame({'residual_var': [.01, .04, .09, .04, .01, 0.]},
                                 index=names),
        clusters=pd.Series(['ME:1', 'ME:1', 'ME:1', 'ME:2', 'ME:2', 'ME:1'], index=names),
        estimation_date=dates[-1], residual_correlation=prepared,
    )


def test_four_public_choices():
    """The public enum names each supported covariance structure."""
    assert {x.value for x in fl.ResidualType} == {
        'orthogonal', 'empirical', 'exposure_cluster', 'residual_cluster'}


@pytest.mark.parametrize('kind', ['exposure_cluster', 'residual_cluster'])
@pytest.mark.parametrize('rho', [0., .5, 1.])
def test_independent_block_reference(snapshot, kind, rho):
    """Signed block means preserve marginals and PSD, excluding cash from clustering."""
    corr = snapshot.residual_correlation.correlation.iloc[:-1, :-1]
    if kind == 'exposure_cluster':
        labels = snapshot.clusters.loc[corr.index]
    else:
        labels, _, _ = fl.compute_clusters_from_corr_matrix(
            corr, cutoff_fraction=.6, linkage_method='ward', distance_transform='one_minus_rho')
    target = np.eye(6)
    for _, members in labels.groupby(labels):
        ids = corr.index.get_indexer(members.index)
        block = corr.loc[members.index, members.index].to_numpy()
        n = len(ids)
        if n > 1:
            # Independent row-sum formula, rather than the implementation's upper triangle.
            mean = (block.sum() - np.trace(block)) / (n * (n - 1))
            target[np.ix_(ids, ids)] = rho * mean
    np.fill_diagonal(target, 1.)
    v = snapshot.y_variances.residual_var.to_numpy()
    expected = target * np.sqrt(np.outer(v, v))
    actual = snapshot.get_residual_covar(residual_type=kind, residual_corr_weight=rho)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_array_equal(np.diag(actual), v)
    assert np.linalg.eigvalsh(actual).min() >= -1e-14
    total = snapshot.get_y_covar(residual_type=kind, residual_corr_weight=rho)
    np.testing.assert_allclose(total, snapshot.get_y_covar(0.) + expected, atol=1e-15)
    np.testing.assert_array_equal(np.diag(total), np.diag(snapshot.get_y_covar()))
    if kind == 'exposure_cluster' and rho:
        assert actual.loc['d', 'e'] < 0
        assert actual.loc['a', 'd'] == 0


@pytest.mark.parametrize('kind', ['exposure_cluster', 'residual_cluster'])
def test_selection_rename_persistence_and_asof_preserve_full_universe(snapshot, kind, tmp_path):
    """Portfolio selection must not recompute block means or recut residual clusters."""
    full = snapshot.get_y_covar(residual_type=kind)
    mapping = {'b': 'B', 'a': 'A'}
    expected = full.loc[list(mapping), list(mapping)].rename(index=mapping, columns=mapping)
    selected = snapshot.filter_on_tickers(mapping)
    pd.testing.assert_frame_equal(selected.get_y_covar(residual_type=kind), expected)
    pd.testing.assert_frame_equal(snapshot.get_y_covar(assets=['b', 'a'], residual_type=kind),
                                  full.loc[['b', 'a'], ['b', 'a']])
    path = tmp_path / 'selected.xlsx'
    selected.save(str(path))
    loaded = fl.CurrentFactorCovarData.load(str(path))
    pd.testing.assert_frame_equal(loaded.get_y_covar(residual_type=kind), expected)
    rolling = fl.RollingFactorCovarData({snapshot.estimation_date: loaded})
    later = pd.Timestamp('2020-04-15')
    pd.testing.assert_frame_equal(rolling.get_y_covars(residual_type=kind, dates=[later])[later],
                                  expected)
    twice = loaded.filter_on_tickers(['A', 'B'])
    pd.testing.assert_frame_equal(twice.get_y_covar(residual_type=kind),
                                  expected.loc[['A', 'B'], ['A', 'B']])


@pytest.mark.parametrize('kind', ['exposure_cluster', 'residual_cluster'])
def test_causal_validation_and_zero_weights(snapshot, kind):
    """No prepared state is required for zero risk/retention; positive risk is causal."""
    missing = replace(snapshot, residual_correlation=None, clusters=None)
    pd.testing.assert_frame_equal(missing.get_y_covar(residual_type=kind, residual_corr_weight=0.),
                                  missing.get_y_covar())
    pd.testing.assert_frame_equal(missing.get_y_covar(0., residual_type=kind),
                                  missing.get_y_covar(0.))
    with pytest.raises(ValueError, match='prepared'):
        missing.get_y_covar(residual_type=kind)
    with pytest.raises(ValueError, match='available'):
        replace(snapshot, estimation_date=pd.Timestamp('2020-02-29')).get_y_covar(
            residual_type=kind)
    with pytest.raises(ValueError, match='nonnegative'):
        snapshot.get_y_covar(-1., residual_type=kind)
    with pytest.raises(ValueError, match='residual_corr_weight'):
        snapshot.get_y_covar(residual_type=kind, residual_corr_weight=1.1)


def test_missing_exposure_labels_and_small_universes(snapshot):
    """Exposure labels are required only for eligible assets; singleton targets are diagonal."""
    with pytest.raises(ValueError, match='cluster'):
        replace(snapshot, clusters=None).get_y_covar(residual_type='exposure_cluster')
    labels = snapshot.clusters.copy()
    labels.loc['a'] = np.nan
    with pytest.raises(ValueError, match='cluster'):
        replace(snapshot, clusters=labels).get_y_covar(residual_type='exposure_cluster')
    labels.loc['a'] = 'ME:1'
    labels.loc['cash'] = np.nan
    replace(snapshot, clusters=labels).get_y_covar(residual_type='exposure_cluster')
    for count in (0, 1):
        variances = snapshot.y_variances.copy()
        variances.iloc[count:, variances.columns.get_loc('residual_var')] = 0.
        small = replace(snapshot, y_variances=variances, clusters=None)
        for kind in ('exposure_cluster', 'residual_cluster'):
            pd.testing.assert_frame_equal(small.get_y_covar(residual_type=kind),
                                          small.get_y_covar())
