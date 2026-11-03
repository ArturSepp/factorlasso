"""Independent checks of the example's cutoff, selection and raw forecast composition."""

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/docs/sparse_factor_model.py'
pytestmark = pytest.mark.skipif(not EXAMPLE.is_file(), reason='examples are checkout documentation')


@pytest.fixture(scope='module')
def recipe():
    """Load the canonical example without running its historical figure calculations."""
    spec = importlib.util.spec_from_file_location('forecast_documentation_example', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inputs(recipe):
    """Use the existing fixed synthetic panel and explicitly supplied same-horizon inputs."""
    x, y = recipe.make_panel()
    cutoff = x.index[59]
    premium = pd.Series([.003, .001, .002, 0, 0, 0, 0, 0], index=x.columns)
    alpha = pd.Series(0., index=y.columns)
    return x, y, cutoff, premium, alpha


def reference_fit(x, y, penalty):
    """Raw-level intercepted augmented LS and direct Ward distances, independent of the recipe."""
    xx, yy = x.to_numpy(), y.to_numpy()
    months = x.index.year * 12 + x.index.month
    w = (1 - 2 / 25) ** (months[-1] - months.to_numpy())
    w = np.asarray(w) / np.mean(w)
    xm, ym = np.average(xx, axis=0, weights=w), np.average(yy, axis=0, weights=w)
    xs = np.sqrt(np.average((xx-xm)**2, axis=0, weights=w))
    ys = np.sqrt(np.average((yy-ym)**2, axis=0, weights=w))
    cov = (yy-ym).T @ (w[:, None] * (yy-ym)) / w.sum()
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    distance = np.sqrt(2 * np.maximum(1 - corr, 0))
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method='ward')
    groups = fcluster(tree, .6 * tree[-1, 2], criterion='distance')
    design = np.column_stack([np.ones(len(x)), xx / xs])
    root = np.sqrt(w / len(x))[:, None]
    individual = np.linalg.lstsq(root * design, root * yy / ys, rcond=None)[0]
    target = np.empty((y.shape[1], x.shape[1]))
    for group in np.unique(groups):
        members = groups == group
        target[members] = individual[1:, members].mean(axis=1)
    if np.isposinf(penalty):
        beta = target * ys[:, None] / xs
        alpha = np.average(yy - xx @ beta.T, axis=0, weights=w)
    else:
        extra = np.column_stack([np.zeros(x.shape[1]), np.sqrt(penalty)*np.eye(x.shape[1])])
        solution = np.linalg.lstsq(np.vstack([root*design, extra]),
                                   np.vstack([root*yy/ys, np.sqrt(penalty)*target.T]),
                                   rcond=None)[0]
        beta, alpha = solution[1:].T * ys[:, None] / xs, solution[0] * ys
    return beta, alpha, ys, groups


def reference_selection(recipe, x, y):
    """Compute every blocked loss without using example preparation or selection code."""
    losses = np.zeros((len(recipe.FORECAST_PENALTIES), 3))
    for column, stop in enumerate([30, 40, 50]):
        for row, penalty in enumerate(recipe.FORECAST_PENALTIES):
            beta, alpha, scale, _ = reference_fit(x.iloc[:stop], y.iloc[:stop], penalty)
            pred = x.iloc[stop:stop+10].to_numpy() @ beta.T + alpha
            errors = (y.iloc[stop:stop+10].to_numpy() - pred) / scale
            losses[row, column] = np.mean(errors**2)
    means = losses.mean(axis=1)
    chosen = max(p for p, value in zip(recipe.FORECAST_PENALTIES, means)
                 if value <= means.min() + 1e-12)
    return losses, chosen


def run(recipe, x, y, cutoff, premium, alpha):
    """Invoke the example with an explicit information date for each supplied vector."""
    return recipe.forecast_recipe(x, y, cutoff, premium, alpha, cutoff, cutoff)


def test_full_selection_and_raw_forecast_match_intercepted_reference(recipe):
    x, y, cutoff, premium, alpha = inputs(recipe)
    result = run(recipe, x, y, cutoff, premium, alpha)
    losses, chosen = reference_selection(recipe, x.iloc[:60], y.iloc[:60])
    np.testing.assert_allclose(result.validation_losses, losses, rtol=1e-8, atol=1e-10)
    assert result.penalty == chosen
    beta, intercept, _, groups = reference_fit(x.iloc[:60], y.iloc[:60], chosen)
    np.testing.assert_allclose(result.coefficients, beta, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(result.regression_intercept, intercept, atol=1e-10)
    np.testing.assert_array_equal(result.window.groups, groups)
    np.testing.assert_allclose(result.forecast, beta @ premium.to_numpy(), atol=1e-10)
    # A nonzero admitted alpha changes the forecast additively, not the fitted loadings.
    admitted = pd.Series(np.linspace(-.001, .001, len(alpha)), index=alpha.index)
    actual = recipe.assemble_expected_return(result.coefficients, premium, admitted)
    np.testing.assert_allclose(actual, result.forecast + admitted, atol=1e-15)


def test_later_data_do_not_change_selection_or_forecasts_and_inputs_are_owned(recipe):
    x, y, cutoff, premium, alpha = inputs(recipe)
    original_x, original_y = x.copy(deep=True), y.copy(deep=True)
    before = run(recipe, x, y, cutoff, premium, alpha)
    x.iloc[60:] = np.nan
    y.iloc[60:] = np.inf
    after = run(recipe, x, y, cutoff, premium, alpha)
    assert before.penalty == after.penalty
    pd.testing.assert_frame_equal(before.validation_losses, after.validation_losses)
    pd.testing.assert_frame_equal(before.coefficients, after.coefficients)
    pd.testing.assert_series_equal(before.forecast, after.forecast)
    pd.testing.assert_series_equal(before.hcgl_forecast, after.hcgl_forecast)
    pd.testing.assert_frame_equal(x.iloc[:60], original_x.iloc[:60])
    pd.testing.assert_frame_equal(y.iloc[:60], original_y.iloc[:60])


def test_validation_scores_change_but_earlier_training_state_does_not(recipe):
    x, y, cutoff, premium, alpha = inputs(recipe)
    before = run(recipe, x, y, cutoff, premium, alpha)
    changed = y.copy()
    changed.iloc[30:40] += .25
    after = run(recipe, x, changed, cutoff, premium, alpha)
    first = recipe.prepare_forecast_window(x.iloc[:30], y.iloc[:30])
    second = recipe.prepare_forecast_window(x.iloc[:30], changed.iloc[:30])
    pd.testing.assert_frame_equal(first.gate.slopes, second.gate.slopes)
    assert not np.allclose(before.validation_losses[30], after.validation_losses[30])


def test_selection_requires_all_folds_and_uses_largest_near_tie(recipe):
    losses = pd.DataFrame([[1, 1, 1], [1+5e-13]*3, [.01, .01, np.nan], [2, 2, 2]],
                          index=[0., .1, 1., np.inf])
    assert recipe.select_forecast_penalty(losses) == .1
    with pytest.raises(RuntimeError, match='every validation block'):
        recipe.select_forecast_penalty(losses * np.nan)
    # A first-minimum/skip-missing mutant selects the incomplete candidate instead.
    assert losses.mean(axis=1).idxmin() != recipe.select_forecast_penalty(losses)


def test_a_failed_fit_invalidates_the_whole_candidate(recipe, monkeypatch):
    x, y, cutoff, premium, alpha = inputs(recipe)
    original = recipe.fit_forecast_window

    def failing(window, penalty):
        if len(window.x) == 30 and penalty == .1:
            raise RuntimeError('injected numerical fit failure')
        return original(window, penalty)

    monkeypatch.setattr(recipe, 'fit_forecast_window', failing)
    result = run(recipe, x, y, cutoff, premium, alpha)
    assert np.isnan(result.validation_losses.loc[.1, 30])
    assert np.isfinite(result.validation_losses.loc[.1, [40, 50]]).all()
    assert result.penalty != .1


def test_independent_scores_detect_injected_training_leakage(recipe, monkeypatch):
    x, y, cutoff, premium, alpha = inputs(recipe)
    reference, _ = reference_selection(recipe, x.iloc[:60], y.iloc[:60])
    original = recipe.prepare_forecast_window

    def leaked(x_prefix, y_prefix):
        return original(x.iloc[:60], y.iloc[:60])

    monkeypatch.setattr(recipe, 'prepare_forecast_window', leaked)
    wrong = run(recipe, x, y, cutoff, premium, alpha)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(wrong.validation_losses, reference, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('defect', ['premium_order', 'alpha_order', 'premium_duplicate',
                                  'premium_nan', 'alpha_inf', 'date_order', 'date_duplicate',
                                  'date_missing', 'future_premium', 'missing_alpha_date'])
def test_information_dates_and_labels_are_rejected(recipe, defect):
    x, y, cutoff, premium, alpha = inputs(recipe)
    premium_asof, alpha_asof = cutoff, cutoff
    if defect == 'premium_order':
        premium = premium.iloc[::-1]
    elif defect == 'alpha_order':
        alpha = alpha.iloc[::-1]
    elif defect == 'premium_duplicate':
        premium.index = [premium.index[0]] * len(premium)
    elif defect == 'premium_nan':
        premium.iloc[0] = np.nan
    elif defect == 'alpha_inf':
        alpha.iloc[0] = np.inf
    elif defect == 'date_order':
        x, y = x.iloc[::-1], y.iloc[::-1]
    elif defect in ('date_duplicate', 'date_missing'):
        dates = x.index.to_list()
        dates[1] = dates[0] if defect == 'date_duplicate' else pd.NaT
        x.index = y.index = dates
    elif defect == 'future_premium':
        premium_asof = cutoff + pd.Timedelta(days=1)
    else:
        alpha_asof = pd.NaT
    with pytest.raises(ValueError):
        recipe.forecast_recipe(x, y, cutoff, premium, alpha, premium_asof, alpha_asof)
