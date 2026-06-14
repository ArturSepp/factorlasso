"""
ETF study — multi-asset application and matched simulation (JSS 2026).

Runs on the public ETF panel produced by ``fetch_etf_panel.py`` plus the MATF
factor NAVs, so the empirical application and the simulation share **one**
multi-asset universe — the structure the HCGL estimator is built for.

Two halves, both written to ``--figdir`` / ``--outdir``:

EMPIRICAL (real ETF excess returns)
    For the credit-linked ETFs (IG / HY / EM bond sleeves), the estimated
    Credit beta as a function of regularisation strength, under shrink-to-zero
    HCGL vs prior-centred HCGL. On real data, shrink-to-zero collapses the
    credit beta to ~0 (absorbed into the equity beta) while prior-centring
    holds it at the economic prior. No ground-truth beta exists empirically;
    the OLS estimate is shown as the unregularised reference.
        -> etf_credit_beta_vs_lambda.{pdf,png}
        -> etf_empirical_credit_betas.csv  (per-ETF OLS / HCGL0 / PRIOR)

SIMULATION (ETF-calibrated DGP)
    A controlled DGP calibrated to the same universe: true loadings are the
    production sub-class betas mapped onto each ETF with per-fund perturbation
    (genuine within-cluster heterogeneity), the factor covariance is estimated
    from the factor NAVs over the panel window (Credit-Equity ~0.84), and
    idiosyncratic noise is set to match each ETF's empirical R^2. Five
    estimators are scored at oracle-λ on economic attribution (the metric that
    matters), not raw β-MSE.
        -> etf_sim_credit_beta_vs_lambda.{pdf,png}
        -> etf_sim_attribution_table.csv

Usage (from repository root)::

    python -m papers.jss_2026.applications.run_etf_study \
        --data-dir papers/jss_2026/applications/data \
        --factor-nav papers/jss_2026/applications/data/futures_risk_factors.csv
    # add --quick for 3 simulation seeds
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[3]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from papers.jss_2026.simulations import matf_calibration as cal
from papers.jss_2026.simulations import metrics as M

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from factorlasso import LassoModel, LassoModelType as MT  # noqa: E402

FACT = cal.FACTORS
CREDIT_CLASSES = {"Global IG Bonds", "Global HY Bonds", "EM Bonds"}
BONDEQ = {"Government Bonds", "Global IG Bonds", "Global HY Bonds", "EM Bonds",
          "Other Fixed Income", "North America", "Europe", "Japan",
          "Asia Ex-Japan", "EM ex-Asia", "US Sectors", "REITs"}
# ETF sub-class -> production-universe asset whose beta is the true loading
SUBCLASS_TO_PROD = {
    "Government Bonds": "G_Gov", "Global IG Bonds": "G_IG", "Global HY Bonds": "G_HY",
    "EM Bonds": "EM_HC", "Other Fixed Income": "G_IG", "North America": "MSCI_US",
    "Europe": "MSCI_EU", "Japan": "MSCI_JP", "Asia Ex-Japan": "MSCI_AxJ",
    "EM ex-Asia": "MSCI_EMxA", "US Sectors": "MSCI_US", "Hedge Funds": "HF",
    "Private Equity": "PrivEq", "Private Debt": "PrivCred",
    "Commodities EX-Precious": "RealAst", "Commodities Precious": "RealAst",
}
CREDIT_PRIORS = {"Global IG Bonds": 0.20, "Global HY Bonds": 0.40, "EM Bonds": 0.30}
# Production configuration (LGT MATF-CMA deployment, EWMA span excluded for
# the article — uniform weights throughout): HCGL cutoff 0.40, noise-floor
# gate τ = 1.0, adaptive reweighting with floor 0.5. The empirical contrast
# below holds this profile fixed and varies only the economic prior.
PROD_FL = dict(model_type=None, cutoff_fraction=0.40,
               auto_sign_constraints=True, auto_sign_threshold_t=1.0,
               auto_sign_adaptive_weights=True, auto_sign_adaptive_gamma=1.0,
               auto_sign_adaptive_floor=0.5)
DPI = 150
COLORS = {"LASSO (→0)": "#c0392b", "GRP-HCGL (→0)": "#e67e22",
          "HCGL+PRIOR": "#2471a3", "HCGL+SIGN+PRIOR": "#1e8449"}


def load_data(data_dir: Path, factor_nav: Path):
    nav = (pd.read_csv(factor_nav, parse_dates=["date"]).set_index("date")
             .rename(columns={"Private Equity": "PrivateEquity", "Rates Vol": "RatesVol"})[FACT])
    X = np.log(nav.resample("ME").last()).diff()
    Y = pd.read_csv(data_dir / "etf_excess_logreturns.csv", index_col=0, parse_dates=True)
    ix = X.index.intersection(Y.index)
    X, Y = X.loc[ix], Y.loc[ix]
    uni = (pd.read_csv(Path(__file__).resolve().parent / "etf_universe.csv")
             .set_index("ticker").reindex(Y.columns))
    return X, Y, uni


def sign_prior(uni: pd.DataFrame, tickers: list[str]):
    sign = pd.DataFrame(np.nan, index=tickers, columns=FACT)
    prior = pd.DataFrame(0.0, index=tickers, columns=FACT)
    for tk in tickers:
        sac = uni.loc[tk, "sub_asset_class"]
        if sac in BONDEQ:
            sign.loc[tk, ["Equity", "Rates", "Credit", "Carry"]] = 1
            sign.loc[tk, "PrivateEquity"] = 0
            sign.loc[tk, "RatesVol"] = -1
        elif sac in {"Private Equity", "Private Debt"}:
            sign.loc[tk, ["Equity", "Rates", "Credit", "Carry", "PrivateEquity"]] = 1
            sign.loc[tk, "RatesVol"] = -1
        elif sac in {"Commodities EX-Precious", "Commodities Precious"}:
            sign.loc[tk, "Commodities"] = 1
            sign.loc[tk, "PrivateEquity"] = 0
        else:
            sign.loc[tk, "PrivateEquity"] = 0
        if sac in CREDIT_PRIORS:
            prior.loc[tk, "Credit"] = CREDIT_PRIORS[sac]
    return sign, prior


def true_beta(uni: pd.DataFrame, tickers: list[str], perturb_seed: int = 0) -> np.ndarray:
    prodB = pd.DataFrame(cal.BETA, index=cal.ASSETS, columns=FACT)
    rows = []
    for tk in tickers:
        sac = uni.loc[tk, "sub_asset_class"]
        if sac == "REITs":
            rows.append((0.5 * prodB.loc["MSCI_US"] + 0.5 * prodB.loc["G_Gov"]).values)
        elif sac == "Currencies":
            v = np.zeros(len(FACT)); v[FACT.index("Fx")] = 0.9 if tk == "UUP" else -0.9
            rows.append(v)
        else:
            rows.append(prodB.loc[SUBCLASS_TO_PROD[sac]].values)
    B0 = np.vstack(rows)
    rng = np.random.default_rng(perturb_seed)
    nz = np.abs(B0) > 1e-9
    B = B0.copy()
    B[nz] = B0[nz] * (1 + rng.normal(0, 0.15, size=B0.shape)[nz])   # within-cluster heterogeneity
    return B


def _lambda_path(Xv, Yv, grouped, gn):
    XtY = Xv.T @ Yv
    lmax = ((8.0 / len(Xv)) * np.max(np.linalg.norm(XtY, axis=0)) if grouped
            else (2.0 / len(Xv)) * np.abs(XtY).max())
    return gn * lmax


def empirical(X, Y, uni, sign, prior, figdir: Path, outdir: Path):
    tickers = list(Y.columns)
    credit = [t for t in tickers if uni.loc[t, "sub_asset_class"] in CREDIT_CLASSES]
    ci, ei = FACT.index("Credit"), FACT.index("Equity")
    Xc, Yc = X - X.mean(), Y - Y.mean()
    Bo = pd.DataFrame(np.linalg.lstsq(Xc.values, Yc.values, rcond=None)[0].T,
                      index=tickers, columns=FACT)
    ols_c = float(Bo.loc[credit, "Credit"].mean())
    prior_c = float(prior.loc[credit, "Credit"].mean())

    gn = np.geomspace(1.0, 1e-5, 16)
    path = _lambda_path(X.values, Y.values, True, gn)
    curve = []
    kw = dict(PROD_FL, model_type=MT.HIERARCHICAL_CLUSTER_GROUP_LASSO,
              factors_beta_loading_signs=sign)
    kw_fcgl = dict(PROD_FL, model_type=MT.CLUSTER_FACTOR_GROUP_LASSO,
                   factors_beta_loading_signs=sign)
    for lam in path:
        b0 = LassoModel(reg_lambda=float(lam), **kw).fit(x=X, y=Y).coef_.reindex(index=tickers, columns=FACT)
        bP = LassoModel(reg_lambda=float(lam), factors_beta_prior=prior, **kw).fit(x=X, y=Y).coef_.reindex(index=tickers, columns=FACT)
        bF = LassoModel(reg_lambda=float(lam), factors_beta_prior=prior, **kw_fcgl).fit(x=X, y=Y).coef_.reindex(index=tickers, columns=FACT)
        curve.append((lam / path[0], float(b0.loc[credit, "Credit"].mean()),
                      float(bP.loc[credit, "Credit"].mean()),
                      float(bF.loc[credit, "Credit"].mean())))
    cur = pd.DataFrame(curve, columns=["lambda_norm", "HCGL0", "HCGL_PRIOR", "FCGL_PRIOR"])

    # per-ETF table at a moderate shrinkage
    lam_mod = 0.02 * path[0]
    b0m = LassoModel(reg_lambda=float(lam_mod), **kw).fit(x=X, y=Y).coef_.reindex(index=tickers, columns=FACT)
    bPm = LassoModel(reg_lambda=float(lam_mod), factors_beta_prior=prior, **kw).fit(x=X, y=Y).coef_.reindex(index=tickers, columns=FACT)
    # FCGL prior fit: identical config to the HCGL prior fit, cluster-by-factor
    # block penalty instead of the row penalty.
    bFm = LassoModel(reg_lambda=float(lam_mod), factors_beta_prior=prior, **kw_fcgl).fit(x=X, y=Y).coef_.reindex(index=tickers, columns=FACT)
    tbl = pd.DataFrame({
        "sub_asset_class": uni.loc[credit, "sub_asset_class"].values,
        "OLS_credit": Bo.loc[credit, "Credit"].values,
        "HCGL0_credit": b0m.loc[credit, "Credit"].values,
        "PRIOR_credit": bPm.loc[credit, "Credit"].values,
        "FCGL_credit": bFm.loc[credit, "Credit"].values,
        "OLS_equity": Bo.loc[credit, "Equity"].values,
        "HCGL0_equity": b0m.loc[credit, "Equity"].values,
        "PRIOR_equity": bPm.loc[credit, "Equity"].values,
    }, index=credit).round(3)
    outdir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(outdir / "etf_empirical_credit_betas.csv")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(cur.lambda_norm, cur.HCGL0, marker="o", ms=3.5, lw=1.8, color=COLORS["LASSO (→0)"], label="HCGL (→0)")
    ax.plot(cur.lambda_norm, cur.HCGL_PRIOR, marker="o", ms=3.5, lw=1.8, color=COLORS["HCGL+SIGN+PRIOR"], label="HCGL + PRIOR")
    ax.plot(cur.lambda_norm, cur.FCGL_PRIOR, marker="s", ms=3.5, lw=1.8, color=COLORS["HCGL+PRIOR"], label="FCGL + PRIOR")
    ax.axhline(ols_c, ls="--", lw=1.3, color="0.35", label=f"OLS mean credit β = {ols_c:.2f}")
    ax.axhline(prior_c, ls=":", lw=1.1, color=COLORS["HCGL+SIGN+PRIOR"], alpha=0.7, label=f"economic prior mean = {prior_c:.2f}")
    ax.axhline(0, ls=":", lw=0.9, color="0.7")
    ax.set_xscale("log"); ax.set_xlim(cur.lambda_norm.max() * 1.3, cur.lambda_norm.min() / 1.3)
    ax.set_xlabel("normalised regularisation strength  λ / λ$_{max}$  (→ stronger shrinkage)")
    ax.set_ylabel(f"mean credit β over {len(credit)} IG/HY/EM bond ETFs")
    ax.set_title("Empirical credit attribution vs regularisation — real ETF panel\n"
                 f"(Credit–Equity factor corr {X.corr().loc['Equity','Credit']:.2f})", fontsize=10)
    ax.legend(fontsize=8, loc="center left", framealpha=0.9); ax.grid(alpha=0.25); fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "etf_credit_beta_vs_lambda.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(figdir / "etf_credit_beta_vs_lambda.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[empirical] {len(credit)} credit ETFs | OLS mean credit β {ols_c:.2f} -> "
          f"HCGL0 collapses to {cur.HCGL0.iloc[0]:.2f}, PRIOR holds {cur.HCGL_PRIOR.iloc[0]:.2f}")
    return tbl


def simulation(X, Y, uni, sign, prior, seeds, figdir: Path, outdir: Path):
    tickers = list(Y.columns); N = len(tickers)
    SIGMA_F = (X.cov() * 12).values
    B = true_beta(uni, tickers)
    ci, ei = FACT.index("Credit"), FACT.index("Equity")
    Xc, Yc = X - X.mean(), Y - Y.mean()
    Bo = np.linalg.lstsq(Xc.values, Yc.values, rcond=None)[0].T
    r2 = np.clip(1 - ((Yc.values - Xc.values @ Bo.T) ** 2).sum(0) / (Yc.values ** 2).sum(0), 0.05, 0.95)
    sys_var = np.einsum("ij,jk,ik->i", B, SIGMA_F, B)
    resid_var = sys_var * (1 - r2) / r2
    credit = [t for t in tickers if uni.loc[t, "sub_asset_class"] in CREDIT_CLASSES]
    cidx = np.array([tickers.index(t) for t in credit])
    true_c = float(B[cidx, ci].mean())
    T = len(X)

    def panel(seed):
        rng = np.random.default_rng(seed)
        Xs = rng.standard_normal((T, len(FACT))) @ np.linalg.cholesky(SIGMA_F / 12).T
        eps = rng.standard_normal((T, N)) * np.sqrt(resid_var / 12)
        return pd.DataFrame(Xs, columns=FACT), pd.DataFrame(Xs @ B.T + eps, columns=tickers)

    # Clustered rows run the production profile (cutoff 0.40, gate τ = 1.0,
    # adaptive floor 0.5, uniform weights); the LASSO row stays the plain
    # L1 reference.
    hc = dict(PROD_FL, model_type=MT.HIERARCHICAL_CLUSTER_GROUP_LASSO)
    cfg = {
        "LASSO (→0)": dict(model_type=MT.LASSO),
        "GRP-HCGL (→0)": dict(hc),
        "HCGL+SIGN": dict(hc, factors_beta_loading_signs=sign),
        "HCGL+PRIOR": dict(hc, factors_beta_prior=prior),
        "HCGL+SIGN+PRIOR": dict(hc, factors_beta_loading_signs=sign, factors_beta_prior=prior),
    }
    gn = np.geomspace(1.0, 1e-5, 12)
    rows = []
    for seed in seeds:
        Xs, Ys = panel(seed)
        for nm, kw in cfg.items():
            grouped = kw["model_type"] != MT.LASSO
            for lam in _lambda_path(Xs.values, Ys.values, grouped, gn):
                try:
                    bh = LassoModel(reg_lambda=float(lam), **kw).fit(x=Xs, y=Ys).coef_.reindex(index=tickers, columns=FACT).values
                except Exception:
                    continue
                a = M.compute_attribution(B, bh, factor_cov=SIGMA_F, target_factor_idx=ci, leak_factor_idx=ei, asset_idx=cidx)
                rows.append(dict(config=nm, seed=seed, lambda_norm=float(lam / _lambda_path(Xs.values, Ys.values, grouped, gn)[0]),
                                 beta_mse=M.beta_mse_normalised(B, bh), credit_recovery=a["recovery"],
                                 credit_abs_err=a["abs_error"], equity_leak=a["leakage"], risk_share_err=a["risk_share_error"]))
    L = pd.DataFrame(rows)
    order = list(cfg)
    oc = L.loc[L.groupby(["config", "seed"])["beta_mse"].idxmin()]
    agg = oc.groupby("config").mean(numeric_only=True).drop(columns=["seed", "lambda_norm"]).reindex(order)
    agg.to_csv(outdir / "etf_sim_attribution_table.csv")

    cur = L.groupby(["config", "lambda_norm"])["credit_recovery"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for c in ["LASSO (→0)", "GRP-HCGL (→0)", "HCGL+PRIOR", "HCGL+SIGN+PRIOR"]:
        d = cur[cur.config == c].sort_values("lambda_norm")
        ax.plot(d.lambda_norm, d.credit_recovery, marker="o", ms=3, lw=1.7, color=COLORS[c], label=c)
    ax.axhline(true_c, ls="--", lw=1.3, color="0.35", label=f"true mean credit β = {true_c:.2f}")
    ax.axhline(0, ls=":", lw=0.8, color="0.7")
    ax.set_xscale("log"); ax.set_xlim(cur.lambda_norm.max() * 1.3, cur.lambda_norm.min() / 1.3)
    ax.set_xlabel("normalised regularisation strength  λ / λ$_{max}$  (→ stronger shrinkage)")
    ax.set_ylabel(f"mean credit β over {len(credit)} credit assets")
    ax.set_title("Simulated credit attribution vs regularisation — ETF-calibrated DGP\n"
                 "(recovery against known truth)", fontsize=10)
    ax.legend(fontsize=8, loc="center left", framealpha=0.9); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(figdir / "etf_sim_credit_beta_vs_lambda.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(figdir / "etf_sim_credit_beta_vs_lambda.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return agg, true_c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--data-dir", type=Path, default=here / "data")
    ap.add_argument("--factor-nav", type=Path, default=here / "data" / "futures_risk_factors.csv")
    ap.add_argument("--figdir", type=Path, default=here.parents[0] / "paper" / "figures")
    ap.add_argument("--outdir", type=Path, default=here.parents[0] / "simulations" / "results_calibrated")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="3 simulation seeds")
    args = ap.parse_args()

    X, Y, uni = load_data(args.data_dir, args.factor_nav)
    print(f"panel: {Y.shape[1]} ETFs x {len(Y)} months ({Y.index.min():%Y-%m}..{Y.index.max():%Y-%m}); "
          f"Credit-Equity factor corr {X.corr().loc['Equity','Credit']:.2f}")
    sign, prior = sign_prior(uni, list(Y.columns))
    args.outdir.mkdir(parents=True, exist_ok=True)

    empirical(X, Y, uni, sign, prior, args.figdir, args.outdir)
    n = 3 if args.quick else args.seeds
    agg, true_c = simulation(X, Y, uni, sign, prior, range(42, 42 + n), args.figdir, args.outdir)
    print(f"\n[simulation] ETF-calibrated DGP, true mean credit β = {true_c:.2f} (oracle-λ):")
    cols = ["credit_recovery", "credit_abs_err", "equity_leak", "risk_share_err", "beta_mse"]
    print(f"{'config':18s}" + "".join(f"{c.split('_')[0][:7]:>9s}" for c in cols))
    for c in agg.index:
        r = agg.loc[c]
        print(f"{c:18s}" + "".join(f"{r[k]:9.3f}" for k in cols))
    print(f"\nfigures -> {args.figdir}\ntables  -> {args.outdir}")


if __name__ == "__main__":
    main()
