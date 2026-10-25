"""Monte Carlo check of the pooled-slope precision formula and the gate variance estimators.

Design. A single predictor x_t ~ N(0, 1), held fixed across replications so the check
is of the conditional formula, and m responses y_tk = a x_t + e_tk with
Cov(e_t) = sigma^2 [(1 - rho) I + rho 1 1'] and independent dates t = 1..T.
For every replication the script computes the pooled slope (article Equation 2),
the date-score variance (article Equation 4, with the T/(T-1) multiplier), the
historical independent-cell variance SSR / {(T m - 1) D} with D = m x'x (the
package's unweighted ``independent`` convention), and the m member slopes.

Targets, per (m, rho) cell:
  se_ratio_empirical      sd(member slope) / sd(pooled slope) across replications
  se_ratio_formula        sqrt(m / {1 + (m - 1) rho})          (article Equation 10)
  date_variance_ratio     mean(V_date) / Var(pooled slope)      (should be near 1)
  indep_variance_ratio    mean(V_indep) / Var(pooled slope)     (near 1 / {1 + (m - 1) rho})
  gate_retention_pooled   P(|t_date| >= tau)                    pooled date-score gate
  gate_retention_single   P(|t_date| >= tau)                    singleton date-score gate

Units. x and e are in standard-normal units; a is the common member slope in those
units. T = 80 matches the simulation studies. Nothing here fits a penalised model
and nothing here uses response clustering: the cluster is prespecified.

Outputs. A CSV of the table above in the external results directory and the main-text
figure ``Fig1_pooling_precision.pdf`` (plus PNG) in ``paper/figures``.
"""
# packages
from __future__ import annotations
import argparse
from enum import Enum
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# project
if __package__:
    from .paths import PAPER_DIR, RESULTS_DIR
else:
    from paths import PAPER_DIR, RESULTS_DIR


class Columns(str, Enum):
    M = "m"
    RHO = "rho_e"
    SE_RATIO_EMP = "se_ratio_empirical"
    SE_RATIO_FORMULA = "se_ratio_formula"
    DATE_VAR_RATIO = "date_variance_ratio"
    INDEP_VAR_RATIO = "indep_variance_ratio"
    RETAIN_POOLED = "gate_retention_pooled"
    RETAIN_SINGLE = "gate_retention_single"
    REPLICATIONS = "replications"


def simulate_cell(m: int,
                  rho: float,
                  x: np.ndarray,
                  n_replications: int = 20000,  # Monte Carlo draws per cell
                  slope: float = 0.2,           # common member slope a, standard units
                  sigma: float = 1.0,           # marginal residual sd
                  tau: float = 2.0,             # gate threshold on |t|
                  seed: int = 20261025
                  ) -> Dict[str, float]:
    """Return the precision and calibration summaries for one (m, rho) cell."""
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho_e must lie in [0, 1), got {rho}")
    rng = np.random.default_rng(seed + 1000 * m + int(round(1000 * rho)))
    t_len = len(x)
    xx = float(x @ x)
    d_pooled = m * xx
    # Equicorrelated residuals: e = sqrt(rho) z 1' + sqrt(1-rho) w, scaled by sigma.
    z = rng.standard_normal((n_replications, t_len, 1))
    w = rng.standard_normal((n_replications, t_len, m))
    e = sigma * (np.sqrt(max(rho, 0.0)) * z + np.sqrt(1.0 - rho) * w)
    y = slope * x[None, :, None] + e                         # (R, T, m)
    y_sum = y.sum(axis=2)                                     # (R, T)
    b_pooled = (y_sum @ x) / d_pooled                         # (R,)
    # Date-score variance, article Equation 4.
    u = x[None, :] * (y_sum - b_pooled[:, None] * m * x[None, :])
    v_date = (t_len / (t_len - 1.0)) * (u ** 2).sum(axis=1) / d_pooled ** 2
    # Historical independent-cell variance: pooled SSR over T m - 1 cells, divided by D.
    ssr = (y ** 2).sum(axis=(1, 2)) - b_pooled ** 2 * d_pooled
    v_indep = np.maximum(ssr, 0.0) / ((t_len * m - 1.0) * d_pooled)
    # Member slopes and the singleton date-score gate on the first member.
    b_members = np.einsum("rtk,t->rk", y, x) / xx            # (R, m)
    r1 = y[:, :, 0] - b_members[:, [0]] * x[None, :]
    u1 = x[None, :] * r1
    v1 = (t_len / (t_len - 1.0)) * (u1 ** 2).sum(axis=1) / xx ** 2
    t_pooled = b_pooled / np.sqrt(v_date)
    t_single = b_members[:, 0] / np.sqrt(v1)
    var_pooled = float(b_pooled.var(ddof=1))
    sd_member = float(b_members.std(ddof=1, axis=0).mean())
    return {
        Columns.M.value: m,
        Columns.RHO.value: rho,
        Columns.SE_RATIO_EMP.value: sd_member / np.sqrt(var_pooled),
        Columns.SE_RATIO_FORMULA.value: float(np.sqrt(m / (1.0 + (m - 1.0) * rho))),
        Columns.DATE_VAR_RATIO.value: float(v_date.mean() / var_pooled),
        Columns.INDEP_VAR_RATIO.value: float(v_indep.mean() / var_pooled),
        Columns.RETAIN_POOLED.value: float((np.abs(t_pooled) >= tau).mean()),
        Columns.RETAIN_SINGLE.value: float((np.abs(t_single) >= tau).mean()),
        Columns.REPLICATIONS.value: n_replications,
    }


def run_grid(ms: List[int],
             rhos: List[float],
             t_len: int = 80,                 # observations, matches the simulation studies
             n_replications: int = 20000,
             slope: float = 0.2,
             tau: float = 2.0,
             seed: int = 20261025
             ) -> pd.DataFrame:
    """Hold x fixed and evaluate every (m, rho) cell."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(t_len)
    x = x - x.mean()
    rows = [simulate_cell(m=m, rho=rho, x=x, n_replications=n_replications,
                          slope=slope, tau=tau, seed=seed)
            for rho in rhos for m in ms]
    return pd.DataFrame(rows)


def plot_figure(table: pd.DataFrame, figure_path: Path) -> None:
    """Two panels: precision ratio against the formula, and variance-estimator calibration."""
    rhos = sorted(table[Columns.RHO.value].unique())
    ms = sorted(table[Columns.M.value].unique())
    colors = ["#1b1b1b", "#245875", "#4f8a6b", "#aa634d"]
    fig, axes = plt.subplots(1, 2, figsize=(5.17, 2.7), layout="constrained")
    ax = axes[0]
    m_grid = np.linspace(1, max(ms), 200)
    for color, rho in zip(colors, rhos):
        d = table[table[Columns.RHO.value] == rho].sort_values(Columns.M.value)
        ax.plot(m_grid, np.sqrt(m_grid / (1.0 + (m_grid - 1.0) * rho)), "-", color=color, lw=1.0)
        ax.plot(d[Columns.M.value], d[Columns.SE_RATIO_EMP.value], "o", color=color, ms=4,
                label=rf"$\rho_e={rho:g}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("cluster size $m$")
    ax.set_ylabel("se(member) / se(pooled)")
    ax.set_title("(a) precision ratio of pooling", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax = axes[1]
    for color, rho in zip(colors, rhos):
        d = table[table[Columns.RHO.value] == rho].sort_values(Columns.M.value)
        ax.plot(d[Columns.M.value], d[Columns.DATE_VAR_RATIO.value], "o-", color=color, ms=4, lw=1.0)
        ax.plot(d[Columns.M.value], d[Columns.INDEP_VAR_RATIO.value], "s--", color=color, ms=4, lw=1.0)
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("cluster size $m$")
    ax.set_ylabel("estimated / true variance")
    ax.set_title("(b) variance-estimator calibration", fontsize=9)
    for a in axes:
        a.tick_params(labelsize=8.5)
        a.xaxis.label.set_size(9)
        a.yaxis.label.set_size(9)
    fig.savefig(figure_path)
    fig.savefig(figure_path.with_suffix(".png"), dpi=200)
    plt.close(fig)


class LocalTest(Enum):
    FULL_GRID = 1
    SMOKE = 2


def run_local_test(local_test: LocalTest,
                   results_dir: Path,
                   figure_dir: Path
                   ) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if local_test == LocalTest.FULL_GRID:
        table = run_grid(ms=[1, 2, 4, 8, 16], rhos=[0.0, 0.3, 0.6, 0.9])
    elif local_test == LocalTest.SMOKE:
        table = run_grid(ms=[1, 4], rhos=[0.0, 0.6], n_replications=200)
    else:
        raise NotImplementedError(f"{local_test}")
    table.to_csv(results_dir / "pooling_precision_check.csv", index=False)
    plot_figure(table=table, figure_path=figure_dir / "Fig1_pooling_precision.pdf")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=PAPER_DIR / "paper" / "figures")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_local_test(local_test=LocalTest.SMOKE if args.smoke else LocalTest.FULL_GRID,
                   results_dir=args.results_dir,
                   figure_dir=args.figure_dir)
