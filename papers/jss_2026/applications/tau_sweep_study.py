"""Gate-threshold (tau) sensitivity sweep for the JSS 2026 paper (T2.2).

Sweeps the noise-floor gate threshold tau (``auto_sign_threshold_t``) of the
deployed HCGL+SIGN+PRIOR production configuration on the calibrated 102-asset
ETF DGP, holding everything else at the production profile. For each seed the
oracle lambda is selected once at the production tau=1.0, then held fixed while
tau is swept, so the figure isolates the gate. Records support F1, sign
agreement, beta-MSE, and the abstention rate (fraction of cells the gate leaves
with no sign constraint). Writes ``simulations/results_calibrated/tau_sweep.csv``
and ``paper/figures/tau_sweep.{pdf,png}``.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import papers.jss_2026.applications.etf_simulation_study as S
from papers.jss_2026.simulations import metrics as M
from factorlasso.lasso_estimator import LassoModel, LassoModelType as MT

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parents[0] / "simulations" / "results_calibrated"
FIGDIR = HERE.parents[0] / "paper" / "figures"
TAUS = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
SEEDS = range(101, 111)   # 10 seeds
T = 112
PROD_TAU = 1.0
FL = "#2471a3"
DPI = 150


def prod_kwargs(tau, d):
    return dict(model_type=MT.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                cutoff_fraction=S.PROD_CUTOFF,
                auto_sign_constraints=True, auto_sign_threshold_t=tau,
                **S.PROD_ADAPT,
                factors_beta_loading_signs=d["sign"], factors_beta_prior=d["prior"])


def fit_metrics(kwargs, lam, X_tr, Y_tr, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(reg_lambda=lam, **kwargs).fit(x=X_tr, y=Y_tr)
    bh = m.coef_.reindex(index=d["tickers"], columns=S.FACT).values
    ds = m.derived_signs_.reindex(index=d["tickers"], columns=S.FACT).values
    abst = float(np.mean(ds == 0))
    return (M.beta_mse_normalised(d["B"], bh),
            M.support_recovery_f1(d["B"], bh),
            M.sign_agreement_rate(d["B"], bh), abst)


def main():
    X, Y, uni = S.load_data(DATA, DATA / "futures_risk_factors.csv")
    d = S.build_dgp(X, Y, uni)
    rows = []
    for seed in SEEDS:
        (X_tr, Y_tr), _ = S.panel(d, seed, T)
        grid = S.lambda_grid(X_tr, Y_tr, False, 8)
        best = None
        for lam in grid:
            bm = fit_metrics(prod_kwargs(PROD_TAU, d), float(lam), X_tr, Y_tr, d)[0]
            if best is None or bm < best[1]:
                best = (float(lam), bm)
        lam_star = best[0]
        for tau in TAUS:
            bm, f1, sa, ab = fit_metrics(prod_kwargs(tau, d), lam_star, X_tr, Y_tr, d)
            rows.append(dict(seed=seed, tau=tau, beta_mse=bm, support_f1=f1,
                             sign_agree=sa, abstention=ab))
        print(f"seed {seed} done (lambda*={lam_star:.2e})")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tau_sweep.csv", index=False)

    g = df.groupby("tau")
    mean, se = g.mean(numeric_only=True), g.std(ddof=1, numeric_only=True) / np.sqrt(g.size().values[:, None] if False else len(SEEDS) ** 0.5)
    tau = mean.index.to_numpy()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: support F1 and sign agreement vs tau
    for col, mk, lab in [("support_f1", "o", "support $F_1$"), ("sign_agree", "s", "sign agreement")]:
        ax0.fill_between(tau, mean[col] - se[col], mean[col] + se[col], alpha=0.15,
                         color=FL if col == "support_f1" else "0.45")
        ax0.plot(tau, mean[col], "-" + mk, ms=4,
                 color=FL if col == "support_f1" else "0.45", label=lab)
    ax0.axvline(PROD_TAU, ls=":", color="0.5", lw=1)
    ax0.text(PROD_TAU, ax0.get_ylim()[0], " production\n $\\tau=1$", fontsize=7.5, color="0.5", va="bottom")
    ax0.set_xlabel("gate threshold $\\tau$"); ax0.set_ylabel("rate")
    ax0.set_title("Selection quality is flat across $\\tau$", fontsize=11)
    ax0.legend(fontsize=8, loc="best"); ax0.grid(alpha=0.25)
    # right: beta-MSE and abstention vs tau
    ax1.fill_between(tau, mean.beta_mse - se.beta_mse, mean.beta_mse + se.beta_mse, alpha=0.15, color=FL)
    ax1.plot(tau, mean.beta_mse, "-o", ms=4, color=FL, label="$\\beta$-MSE")
    ax1.set_xlabel("gate threshold $\\tau$"); ax1.set_ylabel("$\\beta$-MSE", color=FL)
    ax1.set_title("Error stable; abstention rises monotonically", fontsize=11)
    ax1.grid(alpha=0.25)
    axb = ax1.twinx()
    axb.plot(tau, mean.abstention, "--^", ms=4, color="0.45", label="abstention rate")
    axb.set_ylabel("abstention rate", color="0.45"); axb.set_ylim(0, 1)
    ax1.axvline(PROD_TAU, ls=":", color="0.5", lw=1)
    h0, l0 = ax1.get_legend_handles_labels(); hb, lb = axb.get_legend_handles_labels()
    ax1.legend(h0 + hb, l0 + lb, fontsize=8, loc="center right")

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"tau_sweep.{ext}", dpi=DPI, bbox_inches="tight")
    print("wrote", FIGDIR / "tau_sweep.pdf")
    print(mean[["support_f1", "sign_agree", "beta_mse", "abstention"]].round(3).to_string())


if __name__ == "__main__":
    main()
