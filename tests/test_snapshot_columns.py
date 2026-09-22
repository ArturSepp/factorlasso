"""``CurrentFactorCovarData.get_snapshot`` names its columns once and its requirements up front.

Without stored residuals the alpha column falls back to the in-sample alpha under the name
``stat_alpha``, so the table has the same columns in both cases; a ``y_variances`` frame that lacks
a required column raises ``ValueError`` naming it instead of a bare ``KeyError``.
"""

import numpy as np
import pandas as pd
import pytest

from factorlasso import CurrentFactorCovarData, VarianceColumns

FACTORS = ["f0", "f1"]
ASSETS = ["a0", "a1", "a2"]


def _snapshot(columns, residuals=False):
    rng = np.random.default_rng(5)
    x_covar = pd.DataFrame(np.diag([0.04, 0.01]), index=FACTORS, columns=FACTORS)
    betas = pd.DataFrame(rng.standard_normal((3, 2)), index=ASSETS, columns=FACTORS)
    values = {
        VarianceColumns.RESIDUAL_VARS.value: [0.01, 0.02, 0.03],
        VarianceColumns.R2.value: [0.5, 0.6, 0.7],
        VarianceColumns.INSAMPLE_ALPHA.value: [0.001, -0.002, 0.003],
    }
    y_variances = pd.DataFrame({name: values[name] for name in columns}, index=ASSETS)
    resid = None
    if residuals:
        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        resid = pd.DataFrame(0.01 * rng.standard_normal((24, 3)), index=dates, columns=ASSETS)
    return CurrentFactorCovarData(x_covar=x_covar, y_betas=betas, y_variances=y_variances,
                                  residuals=resid)


ALL = [VarianceColumns.RESIDUAL_VARS.value, VarianceColumns.R2.value,
       VarianceColumns.INSAMPLE_ALPHA.value]
EXPECTED = FACTORS + ["r2", "stat_alpha", "insample_alpha", "total_vol", "sys_vol", "resid_vol"]


def test_snapshot_without_residuals_has_one_alpha_column_per_name():
    snapshot = _snapshot(ALL).get_snapshot()
    assert list(snapshot.columns) == EXPECTED
    assert snapshot.columns.is_unique
    # the fallback: stat_alpha carries the in-sample alpha
    pd.testing.assert_series_equal(snapshot["stat_alpha"], snapshot["insample_alpha"],
                                   check_names=False)


def test_snapshot_with_residuals_has_the_same_columns():
    snapshot = _snapshot(ALL, residuals=True).get_snapshot(alpha_span=12)
    assert list(snapshot.columns) == EXPECTED
    assert not np.allclose(snapshot["stat_alpha"], snapshot["insample_alpha"])


@pytest.mark.parametrize(
    "missing", [VarianceColumns.R2.value, VarianceColumns.INSAMPLE_ALPHA.value]
)
def test_missing_required_column_is_named(missing):
    columns = [name for name in ALL if name != missing]
    with pytest.raises(ValueError, match=missing):
        _snapshot(columns).get_snapshot()
