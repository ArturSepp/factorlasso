"""
Worked example for ``docs/residual_diagnostics.md``: is the residual covariance diagonal?
===========================================================================================

Eight synthetic monthly return series load sparsely on four factors. The same panel is fitted
twice with :class:`factorlasso.LassoModel`: once on all four factors, and once with the fourth
factor withheld. The first residual panel is consistent with a diagonal covariance; the second is
not, and no penalty repairs it.

Every number quoted in the article is asserted here against a reference computed a different way:
the chi-square threshold from ``scipy.stats``, the Marchenko-Pastur edge from its closed form, the
sphericity statistic from a direct sum over ``numpy.corrcoef``, the leading component from
``numpy.linalg.eigh``, and the size of the test from simulated diagonal panels.

Synthetic data, fixed seed, no network, no files written. Solver: CVXPY with CLARABEL.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

import factorlasso as fl

SEED = 20260920
N_OBS = 240                       # monthly observations
FACTOR_NAMES = ["equity", "rates", "credit", "commodity"]
WITHHELD_FACTOR = "commodity"
FACTOR_VOL = 0.04                 # per-month factor volatility, decimal returns
RESIDUAL_VOL = 0.02               # per-month idiosyncratic volatility, decimal returns
REG_LAMBDA = 1e-4
SIGNIFICANCE = 0.05
PENALTY_GRID = np.logspace(-2.0, -6.0, 9)
N_NULL_PANELS = 2000

# True loadings, responses by factors. Four series carry the factor that is later withheld.
TRUE_BETA = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.0, 0.0, 0.6],
    [0.0, 0.8, 0.0, 0.0],
    [0.0, 0.7, 0.3, 0.0],
    [0.3, 0.0, 0.9, 0.5],
    [0.2, 0.0, 0.7, 0.0],
    [0.0, 0.0, 0.0, 0.9],
    [0.5, -0.2, 0.0, 0.7],
])


def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the factor panel ``x`` (T x 4) and the response panel ``y`` (T x 8)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2006-01-31", periods=N_OBS, freq="ME")
    x = pd.DataFrame(
        FACTOR_VOL * rng.standard_normal((N_OBS, len(FACTOR_NAMES))),
        index=dates,
        columns=FACTOR_NAMES,
    )
    noise = RESIDUAL_VOL * rng.standard_normal((N_OBS, TRUE_BETA.shape[0]))
    y = pd.DataFrame(
        x.to_numpy() @ TRUE_BETA.T + noise,
        index=dates,
        columns=[f"asset_{i + 1}" for i in range(TRUE_BETA.shape[0])],
    )
    return x, y


def fit_and_diagnose(
    x: pd.DataFrame,
    y: pd.DataFrame,
    reg_lambda: float = REG_LAMBDA,
) -> tuple[fl.LassoModel, fl.Sparsity, pd.DataFrame, fl.ResidualDiagnostics]:
    """Fit a LASSO factor model and test its in-sample residuals for diagonality."""
    model = fl.LassoModel(model_type=fl.LassoModelType.LASSO, reg_lambda=reg_lambda).fit(x=x, y=y)
    sparsity = fl.effective_sparsity(model.coef_)
    residuals = y - model.predict(x)
    diagnostics = fl.diagnose_residuals(
        residuals,
        n_fitted_per_asset=sparsity.per_asset,
        significance=SIGNIFICANCE,
    )
    return model, sparsity, residuals, diagnostics


def penalty_path(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Tabulate the in-sample diagnostics over a grid of penalties, sparsest model first."""
    rows = []
    for reg_lambda in PENALTY_GRID:
        model, sparsity, _, diagnostics = fit_and_diagnose(x, y, reg_lambda=float(reg_lambda))
        rows.append({
            "reg_lambda": float(reg_lambda),
            # An absolute cut: a fully collapsed fit defeats the default relative tolerance.
            "n_loadings": fl.effective_sparsity(model.coef_, tol=1e-3, rtol=0.0).n_nonzero,
            "sphericity": diagnostics.sphericity,
            "threshold": diagnostics.threshold,
            "raw_offdiag_ss": diagnostics.raw_offdiag_ss,
            "n_above_edge": diagnostics.n_above_edge,
            "passes": diagnostics.passes,
        })
    return pd.DataFrame(rows).set_index("reg_lambda")


def null_rejection_rates(nu: float, n_series: int, seed: int = SEED + 1) -> pd.Series:
    """Rejection frequencies of the two criteria on simulated exactly-diagonal panels."""
    rng = np.random.default_rng(seed)
    upper = np.triu_indices(n_series, 1)
    threshold = stats.chi2.ppf(1.0 - SIGNIFICANCE, len(upper[0]))
    edge = (1.0 + np.sqrt(n_series / nu)) ** 2
    rejects = np.zeros(3)
    for _ in range(N_NULL_PANELS):
        corr = np.corrcoef(rng.standard_normal((N_OBS, n_series)), rowvar=False)
        by_sphericity = nu * np.sum(corr[upper] ** 2) > threshold
        by_edge = np.linalg.eigvalsh(corr)[-1] > edge
        rejects += [by_sphericity, by_edge, by_sphericity or by_edge]
    return pd.Series(rejects / N_NULL_PANELS, index=["sphericity", "edge", "either"])


def partition_shares(
    residual_panels: dict,
    labels: pd.Series,
    seed: int = SEED + 2,
) -> pd.DataFrame:
    """Per-date adjusted partition share of residual panels, for a partition and a shuffled one."""
    rng = np.random.default_rng(seed)
    shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
    columns = {}
    for panel_name, panel in residual_panels.items():
        for partition_name, partition in (("carriers", labels), ("shuffled", shuffled)):
            share = fl.partition_variance_share(panel, partition)
            columns[f"{panel_name}, {partition_name}"] = share["adjusted_share"]
    return pd.DataFrame(columns)


def main() -> None:
    """Run the example, verify each quoted number independently, and print the summary."""
    x, y = make_panel()
    n_series = y.shape[1]
    n_pairs = n_series * (n_series - 1) // 2

    # --- 1. Complete factor set: the residual covariance is indistinguishable from diagonal ---
    model, sparsity, residuals, complete = fit_and_diagnose(x, y)

    # effective_sparsity against a bare non-zero count and a hand-applied relative tolerance
    magnitudes = np.abs(model.coef_.to_numpy())
    assert int((magnitudes != 0.0).sum()) == magnitudes.size       # interior-point dust everywhere
    assert sparsity.n_nonzero == int((magnitudes > 1e-4 * magnitudes.max()).sum())
    assert sparsity.n_nonzero == 19 and not sparsity.is_rank_deficient
    gap = fl.suggest_tolerance(model.coef_)
    assert gap["gap_lo"] < sparsity.tol_used < gap["gap_hi"]       # the default cut is in the gap

    # threshold, edge and statistic against independent references
    assert np.isclose(complete.threshold, stats.chi2.ppf(1.0 - SIGNIFICANCE, n_pairs))
    assert np.isclose(complete.threshold, fl.null_threshold(n_pairs, significance=SIGNIFICANCE))
    assert np.isclose(complete.nu, N_OBS - sparsity.per_asset - 1.0)
    assert np.isclose(complete.mp_edge, (1.0 + np.sqrt(n_series / complete.nu)) ** 2)
    assert np.isclose(complete.mp_edge, fl.marchenko_pastur_edge(n_series, complete.nu))
    corr = np.corrcoef(residuals.to_numpy(), rowvar=False)
    assert np.allclose(fl.residual_correlation(residuals).to_numpy(), corr)
    direct_sum = sum(corr[i, j] ** 2 for i in range(n_series) for j in range(i + 1, n_series))
    assert np.isclose(complete.sphericity, complete.nu * direct_sum)
    assert np.isclose(fl.raw_offdiagonal_mass(residuals), direct_sum)
    assert np.isclose(complete.top_eigenvalue, np.linalg.eigvalsh(corr)[-1])
    assert complete.passes and complete.n_above_edge == 0
    assert 28.5 < complete.sphericity < 30.0 < complete.threshold      # quoted: 29.3 against 41.3
    assert 1.28 < complete.top_eigenvalue < 1.32 < complete.mp_edge    # quoted: 1.30 against 1.40

    # --- 2. One factor withheld: the test fails and names the series that carry the factor ---
    x_short = x.drop(columns=WITHHELD_FACTOR)
    _, sparsity_short, residuals_short, omitted = fit_and_diagnose(x_short, y)
    assert not omitted.passes and omitted.n_above_edge == 1
    assert 550.0 < omitted.sphericity < 570.0                          # quoted: 559
    assert 2.82 < omitted.top_eigenvalue < 2.92                        # quoted: 2.87

    components = fl.missing_factor_components(
        residuals_short, n_fitted_per_asset=sparsity_short.per_asset
    )
    carriers = [f"asset_{i + 1}" for i in np.flatnonzero(TRUE_BETA[:, 3])]
    assert sorted(components["series"]) == sorted(carriers)
    values, vectors = np.linalg.eigh(np.corrcoef(residuals_short.to_numpy(), rowvar=False))
    leading = vectors[:, -1] * np.sign(vectors[np.argmax(np.abs(vectors[:, -1])), -1])
    reported = components.set_index("series")["loading"]
    positions = [y.columns.get_loc(series) for series in reported.index]
    assert np.allclose(reported.to_numpy(), leading[positions])

    # --- 3. No penalty repairs a missing factor; with the full set the statistic flattens ---
    path_complete = penalty_path(x, y)
    path_omitted = penalty_path(x_short, y)
    assert not path_omitted["passes"].any()
    passing = path_complete.index[path_complete["passes"]]
    assert np.isclose(passing.max(), 10.0 ** -3.5)                     # sparsest passing penalty
    flat = path_complete.loc[passing, "sphericity"]
    assert flat.max() - flat.min() < 2.5                               # flat region: 29.3 to 31.2
    assert path_complete["n_loadings"].iloc[0] == 0                    # collapsed at the top

    # --- 4. Size of the test on exactly diagonal panels of this shape (simulation) ---
    rates = null_rejection_rates(nu=complete.nu, n_series=n_series)
    assert 0.03 < rates["sphericity"] < 0.07
    assert rates["edge"] < rates["sphericity"] and rates["either"] < 0.08

    print("Complete factor set")
    print(f"  kept loadings      : {sparsity.n_nonzero} of {sparsity.n_total}"
          f" (bare non-zero count: {int((magnitudes != 0.0).sum())})")
    print(f"  degrees of freedom : {complete.nu:.1f}")
    print(f"  sphericity         : {complete.sphericity:.1f} against {complete.threshold:.1f}")
    print(f"  largest eigenvalue : {complete.top_eigenvalue:.2f} against {complete.mp_edge:.2f}")
    print(f"  passes             : {complete.passes}")
    print(f"Factor '{WITHHELD_FACTOR}' withheld")
    print(f"  sphericity         : {omitted.sphericity:.0f} against {omitted.threshold:.1f}")
    print(f"  largest eigenvalue : {omitted.top_eigenvalue:.2f} against {omitted.mp_edge:.2f}")
    print(f"  passes             : {omitted.passes}; components above the edge:"
          f" {omitted.n_above_edge}")
    print(components.round(2).to_string(index=False))
    print("Penalty path, complete factor set")
    print(path_complete[["n_loadings", "sphericity", "passes"]].round(1).to_string())
    print("Rejection frequency on simulated diagonal panels")
    print(rates.round(3).to_string())

    # --- 5. Partition share: the carriers of the withheld factor form a block in the residuals ---
    labels = pd.Series(np.where(y.columns.isin(carriers), "carriers", "others"), index=y.columns)
    shares = partition_shares({"complete": residuals, "withheld": residuals_short}, labels)
    print("Mean adjusted partition share of the residuals")
    print(shares.mean().round(2).to_string())

    # the share of one cross-section against the R-squared of a regression on group indicators
    cross_section = residuals_short.iloc[0].to_numpy()
    groups = sorted(set(labels))
    indicators = np.column_stack([(labels == group).to_numpy(float) for group in groups])
    fitted = indicators @ np.linalg.lstsq(indicators, cross_section, rcond=None)[0]
    total = np.sum((cross_section - cross_section.mean()) ** 2)
    one = fl.partition_variance_share(residuals_short.iloc[0], labels)
    assert np.isclose(one["share"].iloc[0], np.sum((fitted - cross_section.mean()) ** 2) / total)
    # the floor (K - 1)/(N - 1) = 1/7 is the exact mean over all 70 size-preserving relabellings
    exact = [
        fl.partition_variance_share(
            residuals_short.iloc[0],
            pd.Series(np.where(np.isin(np.arange(n_series), chosen), "a", "b"), index=y.columns),
        )["share"].iloc[0]
        for chosen in itertools.combinations(range(n_series), int(labels.eq("carriers").sum()))
    ]
    assert len(exact) == 70 and np.isclose(np.mean(exact), 1.0 / 7.0)
    assert np.isclose(one["floor"].iloc[0], 1.0 / 7.0)
    # the carriers carry a block only while their factor is withheld
    means = shares.mean()
    assert 0.19 < means["withheld, carriers"] < 0.22                  # quoted: 0.21
    assert abs(means["complete, carriers"]) < 0.02                     # quoted: 0.01
    assert abs(means["complete, shuffled"]) < 0.03                     # quoted: 0.02
    assert abs(means["withheld, shuffled"]) < 0.06                     # quoted: 0.04


if __name__ == "__main__":
    main()
