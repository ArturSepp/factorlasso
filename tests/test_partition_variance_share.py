"""
Tests for factorlasso.residual_diagnostics.partition_variance_share.

Every reference value is computed a second way: the share by hand and by a least-squares fit on
group indicators, the permutation floor by exact enumeration of all label arrangements, and the
panel path against the single cross-section path.

Run:  python -m pytest test_partition_variance_share.py -q
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from factorlasso import partition_variance_share


def _anova_adjusted_r2(x: np.ndarray, labels: np.ndarray) -> float:
    """Adjusted R^2 of an OLS fit of x on an intercept and K - 1 group indicators."""
    codes, uniques = pd.factorize(labels)
    k = uniques.shape[0]
    design = np.column_stack([np.ones_like(x)] + [(codes == j).astype(float) for j in range(1, k)])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    resid = x - design @ coef
    r2 = 1.0 - resid @ resid / np.sum((x - x.mean()) ** 2)
    n = x.shape[0]
    return 1.0 - (1.0 - r2) * (n - 1) / (n - k)


def test_hand_computed_two_groups():
    x = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    z = pd.Series(["a", "a", "a", "b", "b", "b"])
    out = partition_variance_share(x, z).iloc[0]
    # grand mean 6.5, group means 2 and 11: SSB = 6 * 4.5^2, SST = 2 * (5.5^2 + 4.5^2 + 3.5^2)
    share = 121.5 / 125.5
    assert out["share"] == pytest.approx(share, abs=1e-14)
    assert out["floor"] == pytest.approx(0.2, abs=1e-15)
    assert out["adjusted_share"] == pytest.approx((share - 0.2) / 0.8, abs=1e-14)
    assert (out["n_names"], out["n_groups"]) == (6, 2)


def test_adjusted_share_equals_anova_adjusted_r2():
    rng = np.random.default_rng(3)
    x = rng.standard_t(4, 200)
    labels = rng.integers(0, 17, 200)
    out = partition_variance_share(pd.Series(x), pd.Series(labels)).iloc[0]
    assert out["adjusted_share"] == pytest.approx(_anova_adjusted_r2(x, labels), abs=1e-12)


@pytest.mark.parametrize("sizes", [(3, 2, 1), (2, 2, 2), (4, 1, 1), (1, 1, 1, 3)])
def test_floor_is_exact_mean_over_all_permutations(sizes):
    rng = np.random.default_rng(11)
    x = rng.standard_normal(sum(sizes))
    base = np.repeat(np.arange(len(sizes)), sizes)
    arrangements = {tuple(p) for p in itertools.permutations(base)}
    shares = [
        partition_variance_share(pd.Series(x), pd.Series(np.array(a))).iloc[0]["share"]
        for a in arrangements
    ]
    n, k = len(base), len(sizes)
    reported = partition_variance_share(pd.Series(x), pd.Series(base)).iloc[0]["floor"]
    assert np.mean(shares) == pytest.approx((k - 1) / (n - 1), abs=1e-12)
    assert np.mean(shares) == pytest.approx(reported, abs=1e-12)


def test_invariant_to_location_and_scale():
    rng = np.random.default_rng(5)
    x = pd.Series(rng.standard_normal(50))
    z = pd.Series(rng.integers(0, 6, 50))
    base = partition_variance_share(x, z)
    moved = partition_variance_share(3.0 + 7.0 * x, z)
    pd.testing.assert_frame_equal(base, moved, atol=1e-12, rtol=0.0)


def test_panel_rows_match_single_cross_sections_with_missing_data():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    names = [f"s{i}" for i in range(12)]
    values = pd.DataFrame(rng.standard_normal((4, 12)), index=dates, columns=names)
    labels = pd.DataFrame(rng.integers(0, 4, (4, 12)).astype(float), index=dates, columns=names)
    values.iloc[1, 2] = np.nan
    labels.iloc[2, 5] = np.nan
    out = partition_variance_share(values, labels)
    for date in dates:
        keep = values.loc[date].notna() & labels.loc[date].notna()
        single = partition_variance_share(values.loc[date][keep], labels.loc[date][keep]).iloc[0]
        pd.testing.assert_series_equal(out.loc[date], single, check_names=False, atol=1e-14)
    assert out["n_names"].tolist() == [12, 11, 11, 12]


def test_static_labels_broadcast_over_dates_and_align_on_names():
    rng = np.random.default_rng(9)
    names = list("abcdefgh")
    values = pd.DataFrame(rng.standard_normal((3, 8)), columns=names)
    labels = pd.Series([0, 0, 1, 1, 2, 2, 3, 3], index=names)
    shuffled = labels.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(
        partition_variance_share(values, labels),
        partition_variance_share(values, shuffled),
    )


def test_labels_missing_a_date_give_an_empty_row():
    values = pd.DataFrame(np.arange(8.0).reshape(2, 4), index=[1, 2], columns=list("abcd"))
    labels = pd.DataFrame([[0, 0, 1, 1]], index=[1], columns=list("abcd"))
    out = partition_variance_share(values, labels)
    assert out.loc[2, "n_names"] == 0
    assert np.isnan(out.loc[2, "share"])


def test_degenerate_cross_sections():
    constant = partition_variance_share(pd.Series([2.0, 2.0, 2.0]), pd.Series([0, 1, 1]))
    assert np.isnan(constant.iloc[0]["share"])
    single = partition_variance_share(pd.Series([1.0]), pd.Series([0]))
    assert np.isnan(single.iloc[0]["floor"])
    singletons = partition_variance_share(pd.Series([1.0, 2.0, 4.0]), pd.Series([0, 1, 2])).iloc[0]
    assert singletons["share"] == pytest.approx(1.0) and singletons["floor"] == 1.0
    assert np.isnan(singletons["adjusted_share"])


def test_input_validation():
    x = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="Series labels"):
        partition_variance_share(x, pd.DataFrame([[0, 1]]))
    with pytest.raises(ValueError, match="DataFrame or Series"):
        partition_variance_share(np.array([1.0, 2.0]), pd.Series([0, 1]))
    with pytest.raises(ValueError, match="share no names"):
        partition_variance_share(x, pd.Series([0, 1], index=["p", "q"]))
    with pytest.raises(ValueError, match="repeated names"):
        partition_variance_share(
            pd.DataFrame([[1.0, 2.0]], columns=["a", "a"]), pd.Series([0], index=["a"])
        )
