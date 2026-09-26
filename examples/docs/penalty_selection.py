"""Canonical example for docs/penalty_selection.md.

Sixteen responses load sparsely on four observed factors over 240 months. In a second panel, six
of the responses also load on a fifth factor that the model is not given. The same penalty grid
is scored in two ways on expanding-window folds: by held-out R-squared (``LassoModelCV``) and by
the diagonality of held-out residuals (``LassoModelDiagonalityCV``). The script also checks that
the path solver reproduces per-penalty solves. Synthetic data.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261003
N_OBS = 240
N_RESPONSES = 16
N_SPLITS = 4
FACTOR_VOL = 0.04
RESIDUAL_VOL = 0.02
OMITTED_LOADING = 0.8
LAMBDAS = np.geomspace(1e-6, 10**-2.5, 15)          # four per decade; stops before the fit empties
FACTORS = ["f1", "f2", "f3", "f4"]


def true_loadings() -> pd.DataFrame:
    """One loading of 1.0 per response and, for every second response, a second one of 0.3."""
    beta = np.zeros((N_RESPONSES, len(FACTORS)))
    for k in range(N_RESPONSES):
        beta[k, k % 4] = 1.0
        if k % 2 == 0:
            beta[k, (k + 1) % 4] = 0.3
    return pd.DataFrame(beta, index=[f"y{k + 1:02d}" for k in range(N_RESPONSES)],
                        columns=FACTORS)


def make_panel(omitted: bool, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly factor and response returns; with ``omitted``, y01 to y06 load on a hidden factor."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2006-01-31", periods=N_OBS, freq="ME")
    factors = FACTOR_VOL * rng.standard_normal((N_OBS, len(FACTORS) + 1))
    beta = true_loadings()
    hidden = np.zeros(N_RESPONSES)
    if omitted:
        hidden[:6] = OMITTED_LOADING
    noise = RESIDUAL_VOL * rng.standard_normal((N_OBS, N_RESPONSES))
    y = factors[:, :4] @ beta.T.to_numpy() + np.outer(factors[:, 4], hidden) + noise
    x = pd.DataFrame(factors[:, :4], index=dates, columns=FACTORS)
    return x, pd.DataFrame(y, index=dates, columns=beta.index)


def select_both(x: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """Select reg_lambda on one grid by held-out R-squared and by held-out residual diagonality."""
    base = fl.LassoModel(model_type=fl.LassoModelType.LASSO)
    by_r2 = fl.LassoModelCV(
        lambdas=LAMBDAS, n_splits=N_SPLITS, base_model=base,
    ).fit(x=x, y=y)
    by_diagonality = fl.LassoModelDiagonalityCV(
        lambdas=LAMBDAS, n_splits=N_SPLITS, base_model=base,
    ).fit(x=x, y=y)
    return by_r2, by_diagonality


def selection_curves(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Held-out R-squared, sphericity and threshold per penalty, with the two selections."""
    by_r2, by_diagonality = select_both(x, y)
    curves = pd.DataFrame({
        "held_out_r2": by_r2.cv_scores_.mean(axis=1),
        "sphericity": by_diagonality.diagnostics_["fold_sphericity_mean"],
        "threshold": by_diagonality.diagnostics_["threshold"],
        "passes": by_diagonality.diagnostics_["passes"],
        "n_nonzero": by_diagonality.diagnostics_["n_nonzero"],
    })
    curves.attrs.update(r2_lambda=by_r2.best_lambda_,
                        diagonality_lambda=by_diagonality.best_lambda_,
                        passed=by_diagonality.passed_,
                        missing=by_diagonality.missing_factors_)
    return curves


def path_matches_single_solves(x: pd.DataFrame, y: pd.DataFrame) -> float:
    """Largest gap between the path solver and one solve per penalty, for supplied groups."""
    groups = pd.Series([k // 4 for k in range(N_RESPONSES)], index=y.columns)
    group_loadings = fl.set_group_loadings(group_data=groups).to_numpy()
    x_np, y_np = (x - x.mean()).to_numpy(), (y - y.mean()).to_numpy()
    path = fl.solve_group_lasso_path(
        x=x_np, y=y_np, group_loadings=group_loadings, reg_lambdas=LAMBDAS,
    )
    gaps = []
    for reg_lambda, result in zip(LAMBDAS, path):
        single = fl.solve_group_lasso_cvx_problem(
            x=x_np, y=y_np, group_loadings=group_loadings, reg_lambda=float(reg_lambda),
        )
        gaps.append(np.abs(result.estimated_beta - single.estimated_beta).max())
    return float(max(gaps))


def path_and_loop_agree(x: pd.DataFrame, y: pd.DataFrame) -> tuple[float, float, float]:
    """LassoModelCV on supplied groups, with and without the path solve."""
    groups = pd.Series([k // 4 for k in range(N_RESPONSES)], index=y.columns)
    base = fl.LassoModel(model_type=fl.LassoModelType.GROUP_LASSO, group_data=groups)
    selectors = [
        fl.LassoModelCV(lambdas=LAMBDAS, n_splits=N_SPLITS, base_model=base,
                        use_lambda_path=use_path, refit=False).fit(x=x, y=y)
        for use_path in (False, True)
    ]
    gap = float((selectors[0].cv_scores_ - selectors[1].cv_scores_).abs().to_numpy().max())
    return selectors[0].best_lambda_, selectors[1].best_lambda_, gap


def main() -> None:
    curves = {name: selection_curves(*make_panel(omitted))
              for name, omitted in (("complete", False), ("omitted factor", True))}
    for name, table in curves.items():
        print(f"{name}: held-out R2 selects {table.attrs['r2_lambda']:.1e}, "
              f"diagonality selects {table.attrs['diagonality_lambda']:.1e}, "
              f"passed {table.attrs['passed']}")
        print(table.round(3))

    # complete panel: diagonality takes the sparsest penalty that passes, R2 a denser one
    complete = curves["complete"]
    assert complete.attrs["passed"]
    assert complete.attrs["diagonality_lambda"] == complete.index[complete["passes"]].max()
    assert complete.attrs["diagonality_lambda"] > complete.attrs["r2_lambda"]
    assert np.isclose(complete.attrs["r2_lambda"], 1e-4)
    assert np.isclose(complete.attrs["diagonality_lambda"], 10**-3.5)
    r2_row = complete.loc[complete.attrs["r2_lambda"]]
    diag_row = complete.loc[complete.attrs["diagonality_lambda"]]
    assert [round(r2_row["held_out_r2"], 3), round(diag_row["held_out_r2"], 3)] == [0.78, 0.768]
    assert [r2_row["n_nonzero"], diag_row["n_nonzero"]] == [45.0, 26.0]   # true support: 24
    next_step = complete.index[complete.index > complete.attrs["diagonality_lambda"]].min()
    assert complete.loc[next_step, "n_nonzero"] == 23.5 and not complete.loc[next_step, "passes"]

    # omitted factor: R2 barely moves its choice; no penalty passes, and the report names the block
    omitted = curves["omitted factor"]
    assert not omitted.attrs["passed"] and not omitted["passes"].any()
    assert np.isclose(omitted.attrs["r2_lambda"], 10**-3.75)
    assert omitted["sphericity"].min() > 3 * omitted["threshold"].iloc[0]
    missing = omitted.attrs["missing"]
    print(missing)
    assert set(missing.loc[missing["component"] == 1, "series"]) == {f"y0{k}" for k in range(1, 7)}
    assert round(float(missing["eigenvalue"].iloc[0]), 1) == 4.6
    assert round(float(omitted.loc[omitted.attrs["r2_lambda"], "held_out_r2"]), 3) == 0.675
    assert round(float(omitted.loc[omitted.attrs["r2_lambda"], "n_nonzero"])) == 44
    assert omitted["sphericity"].min() > 440
    assert round(float(complete["threshold"].iloc[0]), 1) == 146.6

    # the path solve reproduces one solve per penalty, and LassoModelCV selects the same penalty
    x, y = make_panel(omitted=False)
    assert path_matches_single_solves(x, y) < 1e-6
    loop_lambda, path_lambda, score_gap = path_and_loop_agree(x, y)
    assert loop_lambda == path_lambda and score_gap < 1e-6


if __name__ == "__main__":
    main()
