"""Prior-misspecification sweep figure for the JSS 2026 paper (T2.3).

Consumes ``simulations/results_calibrated/prior_sensitivity.csv`` and writes
``paper/figures/prior_sensitivity_sweep.{pdf,png}``. Left panel: the recovered
credit loading rises monotonically with the credit prior and crosses the
calibration truth. Right panel: coefficient error stays bounded across the same
band, including the wrong-sign region, so a misspecified prior is harmless.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[0] / "simulations" / "results_calibrated" / "prior_sensitivity.csv"
FIGDIR = HERE.parents[0] / "paper" / "figures"
DPI = 150
FL = "#2471a3"
TRUE_CREDIT = 0.36
PROD_PRIOR = 0.294  # production credit prior mean (multiplier 1.0)


def main() -> None:
    d = pd.read_csv(CSV).sort_values("credit_prior_mean")
    x = d["credit_prior_mean"].to_numpy()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: recovered credit loading vs prior
    ax0.axvspan(x.min(), 0.0, color="0.92", zorder=0)
    ax0.text(x.min() + 0.01, 0.55, "wrong sign", fontsize=8, color="0.45", va="top")
    ax0.fill_between(x, d.credit_recovery - d.credit_recovery_se,
                     d.credit_recovery + d.credit_recovery_se, color=FL, alpha=0.18)
    ax0.plot(x, d.credit_recovery, "-o", color=FL, ms=4, label="recovered credit β")
    ax0.axhline(TRUE_CREDIT, ls="--", color="0.3", lw=1)
    ax0.text(x.max(), TRUE_CREDIT, f" true β={TRUE_CREDIT:.2f}", va="bottom",
             ha="right", fontsize=8, color="0.3")
    ax0.axvline(PROD_PRIOR, ls=":", color="0.5", lw=1)
    ax0.text(PROD_PRIOR, 0.02, " production\n prior", fontsize=7.5, color="0.5", va="bottom")
    ax0.set_xlabel("credit prior mean $\\beta^{0}$")
    ax0.set_ylabel("recovered credit loading")
    ax0.set_title("Recovery responds to the prior", fontsize=11)
    ax0.grid(alpha=0.25)

    # right: beta-MSE and covariance error stay bounded
    ax1.axvspan(x.min(), 0.0, color="0.92", zorder=0)
    ax1.fill_between(x, d.beta_mse - d.beta_mse_se, d.beta_mse + d.beta_mse_se,
                     color=FL, alpha=0.18)
    ax1.plot(x, d.beta_mse, "-o", color=FL, ms=4, label="$\\beta$-MSE")
    ax1.plot(x, d.cov_err, "-s", color="0.45", ms=3.5, label="covariance error")
    ax1.set_ylim(0.0, max(0.25, float(d.beta_mse.max()) + 0.05))
    ax1.set_xlabel("credit prior mean $\\beta^{0}$")
    ax1.set_ylabel("error")
    ax1.set_title("Coefficient error stays bounded", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.25)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"prior_sensitivity_sweep.{ext}", dpi=DPI, bbox_inches="tight")
    print("wrote", FIGDIR / "prior_sensitivity_sweep.pdf")


if __name__ == "__main__":
    main()
