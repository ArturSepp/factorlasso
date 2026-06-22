"""
ETF-calibrated simulation study — factorlasso vs competitor packages (JSS 2026).

A Monte Carlo benchmark on a data-generating process calibrated to the public
multi-asset ETF panel (the same universe as the empirical application), where
the true loadings are known, so every estimator can be scored against ground
truth — statistically AND economically.

DGP (reuses ``run_etf_study`` calibration helpers)
    N=102 assets, M=9 factors. True β = production sub-class loadings mapped
    onto each ETF with per-fund perturbation; factor covariance from the factor
    NAVs over the panel window (Credit-Equity ~0.84); idiosyncratic variance
    set to match each ETF's empirical R^2. Generative clusters = the 18 sub-
    asset classes (where the betas are assigned); the 4 asset classes are the
    coarse economic grouping. Each replication draws a train window and an
    equal test window.

Method tiers
    competitors  : OLS, sklearn Lasso, skglm GroupLasso, asgl SGL
    factorlasso  : LASSO -> HCGL -> +sign-derivation -> +adaptive -> sparse-group
                   (+ true-cluster oracle, isolating cluster *discovery*)
    economic     : HCGL + economic credit prior; + economic signs + prior
                   (no competitor analog — no other package takes an economic prior)

Clustered factorlasso rows run the production configuration
(``cutoff_fraction = 0.40``, gate ``τ = 1.0``, adaptive floor ``0.5``),
set a priori from the LGT MATF-CMA deployment — not tuned on this DGP.
The production EWMA span is excluded (stationary DGP).

Metrics
    statistical  : beta_mse (normalised), support_f1, sign_agree, oos_r2,
                   cluster_ari_sac (vs sub-classes), cluster_ari_ac (vs asset
                   classes), runtime_s
    economic     : credit_recovery, credit_abs_err, equity_leak,
                   risk_share_err, rp_rmse (implied-return β·λ RMSE)
    covariance   : cov_err (systematic-covariance Frobenius, relative)

Protocol
    Two λ-selectors per method: ``oracle`` (argmin β-MSE — best case for each
    competitor, so any factorlasso edge is conservative) and ``bic`` (feasible,
    what a practitioner can actually compute). Sample sizes T in {60,112,240}
    test small-sample behaviour. Monte Carlo over seeds; tables are means.

Usage (from repository root)::

    python -m papers.jss_2026.applications.etf_simulation_study \
        --data-dir   papers/jss_2026/applications/data \
        --factor-nav papers/jss_2026/applications/data/futures_risk_factors.csv \
        --seeds 50 --sample-sizes 60,112,240
    # --quick : 2 seeds, 6-point grid, T=112 only (smoke test)
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
from time import perf_counter

import numpy as np
import pandas as pd

from papers.jss_2026.applications.run_etf_study import (
    load_data, sign_prior, true_beta, CREDIT_CLASSES, FACT,
)
from papers.jss_2026.simulations import metrics as M
from papers.jss_2026.simulations.estimators import ESTIMATORS, EstimatorResult

warnings.filterwarnings("ignore")
from factorlasso import LassoModel, LassoModelType as MT  # noqa: E402

SR_PRIOR = np.array([0.45, 0.30, 0.30, 0.30, 0.10, 0.10, 0.60, 0.30, 0.00])
# ── Production configuration (LGT MATF-CMA deployment) ────────────────
# The clustered factorlasso rows run the structural production settings:
# cutoff_fraction = 0.40, noise-floor gate τ = 1.0, adaptive reweighting
# with floor 0.5. The production EWMA span is deliberately excluded: the
# DGP is stationary, so time-decayed weights are pure efficiency loss and
# would handicap every estimator equally without informing the comparison.
# Reference rows (OLS, FL LASSO, the true-cluster oracle) and external
# competitors are untouched.
PROD_CUTOFF = 0.40
PROD_SIGN = dict(auto_sign_constraints=True, auto_sign_threshold_t=1.0)
PROD_ADAPT = dict(auto_sign_adaptive_weights=True, auto_sign_adaptive_gamma=1.0,
                  auto_sign_adaptive_floor=0.5)
L1_ANCHORED = {"OLS", "sklearn_lasso", "factorlasso_lasso"}
_TOL = 1e-6

ROSTER = [
    ("OLS", "OLS"),
    ("sklearn_lasso", "sklearn Lasso"),
    ("skglm_grouplasso", "skglm GroupLasso"),
    ("asgl_sgl", "asgl SGL"),
    ("adelie_grp", "adelie group LASSO"),
    ("factorlasso_lasso", "FL LASSO"),
    ("factorlasso_grp_hcgl", "FL HCGL"),
    ("factorlasso_grp_hcgl_sign", "FL HCGL+sign"),
    ("factorlasso_grp_hcgl_sign_adapt", "FL HCGL+sign+adapt"),
    ("factorlasso_fcgl_sign_adapt", "FL FCGL+sign+adapt"),
    ("factorlasso_sgl_hcgl_sign_adapt", "FL SGL+sign+adapt"),
    ("factorlasso_grp_oracle", "FL oracle (clusters)"),
    ("factorlasso_grp_oracle_sign_adapt", "FL ORACLE+sign+adapt"),
    ("factorlasso_econ_prior", "FL HCGL+PRIOR"),
    ("factorlasso_econ_sign_prior", "FL HCGL+SIGN+PRIOR"),
    ("factorlasso_fcgl_sign_prior", "FL FCGL+SIGN+PRIOR"),
    ("factorlasso_econ_oracle_sign_prior", "FL ORACLE+SIGN+PRIOR"),
]
METRICS = ["beta_mse", "support_f1", "sign_agree",
           "oos_r2", "cluster_ari_sac",
           "cluster_ari_ac", "credit_recovery", "equity_leak", "risk_share_err",
           "rp_rmse", "cov_err", "runtime_s"]
HDR = dict(beta_mse="betaMSE", support_f1="suppF1", sign_agree="signAgr",
           oos_r2="oosR2",
           cluster_ari_sac="ARIsac", cluster_ari_ac="ARIac", credit_recovery="crRecov",
           equity_leak="eqLeak", risk_share_err="riskSh", rp_rmse="rpRMSE",
           cov_err="covErr", runtime_s="rt(s)")


def build_dgp(X, Y, uni):
    tickers = list(Y.columns)
    SIGMA_F = (X.cov() * 12).values
    B = true_beta(uni, tickers)
    sign, prior = sign_prior(uni, tickers)
    Xc, Yc = X - X.mean(), Y - Y.mean()
    Bo = np.linalg.lstsq(Xc.values, Yc.values, rcond=None)[0].T
    r2 = np.clip(1 - ((Yc.values - Xc.values @ Bo.T) ** 2).sum(0) / (Yc.values ** 2).sum(0), 0.05, 0.95)
    sys_var = np.einsum("ij,jk,ik->i", B, SIGMA_F, B)
    resid_var = sys_var * (1 - r2) / r2
    clusters = uni["sub_asset_class"].reindex(tickers)              # generative clusters (oracle)
    ari_sac = pd.factorize(uni["sub_asset_class"].reindex(tickers))[0]
    ari_ac = pd.factorize(uni["asset_class"].reindex(tickers))[0]
    lam_true = SR_PRIOR * np.sqrt(np.diag(SIGMA_F))
    cidx = np.array([i for i, t in enumerate(tickers) if uni.loc[t, "sub_asset_class"] in CREDIT_CLASSES])
    return dict(tickers=tickers, B=B, SIGMA_F=SIGMA_F, resid_var=resid_var, clusters=clusters,
                ari_sac=ari_sac, ari_ac=ari_ac, sign=sign, prior=prior, lam_true=lam_true,
                cidx=cidx, ci=FACT.index("Credit"), ei=FACT.index("Equity"))


def panel(d, seed, T):
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(d["SIGMA_F"] / 12)
    sd = np.sqrt(d["resid_var"] / 12)
    def draw(n):
        Xs = rng.standard_normal((n, len(FACT))) @ chol.T
        eps = rng.standard_normal((n, len(d["tickers"]))) * sd
        return pd.DataFrame(Xs, columns=FACT), pd.DataFrame(Xs @ d["B"].T + eps, columns=d["tickers"])
    return draw(T), draw(T)


def make_econ_fit(model_kwargs):
    def fit(X_tr, Y_tr, reg_lambda, *, true_clusters=None):
        t0 = perf_counter()
        m = LassoModel(reg_lambda=reg_lambda, **model_kwargs).fit(x=X_tr, y=Y_tr)
        rt = perf_counter() - t0
        ic = m.alpha_const_ if m.alpha_const_ is not None else m.intercept_
        ex = {"clusters_hat": m.clusters_} if m.clusters_ is not None else {}
        return EstimatorResult("econ", reg_lambda, m.coef_, ic, rt, ex)
    return fit


def fit_ols(X_tr, Y_tr, reg_lambda, *, true_clusters=None):
    t0 = perf_counter()
    Xc, Yc = X_tr - X_tr.mean(), Y_tr - Y_tr.mean()
    B = np.linalg.lstsq(Xc.values, Yc.values, rcond=None)[0].T
    rt = perf_counter() - t0
    bh = pd.DataFrame(B, index=Y_tr.columns, columns=X_tr.columns)
    ic = Y_tr.mean() - (X_tr.mean().values @ B.T)
    return EstimatorResult("OLS", reg_lambda, bh, pd.Series(ic, index=Y_tr.columns), rt)


def get_fit(name, d):
    if name == "OLS":
        return fit_ols
    # Clustered factorlasso rows run the production configuration locally;
    # the shared ESTIMATORS registry keeps package defaults for the
    # synthetic ablation of Sections 6.1-6.4 (the "package profile").
    base = dict(model_type=MT.HIERARCHICAL_CLUSTER_GROUP_LASSO, cutoff_fraction=PROD_CUTOFF)
    if name == "factorlasso_grp_hcgl":
        return make_econ_fit(dict(base))
    if name == "factorlasso_grp_hcgl_sign":
        return make_econ_fit(dict(base, **PROD_SIGN))
    if name == "factorlasso_grp_hcgl_sign_adapt":
        return make_econ_fit(dict(base, **PROD_SIGN, **PROD_ADAPT))
    if name == "factorlasso_fcgl_sign_adapt":
        # FCGL twin of HCGL+sign+adapt: identical config, cluster-by-factor
        # block penalty instead of the per-row penalty.
        return make_econ_fit(dict(model_type=MT.FACTOR_CLUSTER_GROUP_LASSO,
                                  cutoff_fraction=PROD_CUTOFF, **PROD_SIGN, **PROD_ADAPT))
    if name == "factorlasso_sgl_hcgl_sign_adapt":
        return make_econ_fit(dict(base, l1_weight=0.1, **PROD_SIGN, **PROD_ADAPT))
    if name == "factorlasso_grp_oracle_sign_adapt":
        # D.1 ablation: same sign+adaptive machinery as the HCGL+sign+adapt
        # row, but the sign pooling unit is the TRUE sub-asset-class taxonomy
        # (group_data) rather than the HCGL-discovered partition. Isolates
        # whether cluster DISCOVERY buys anything over the known taxonomy for
        # the mechanism that consumes the partition.
        return make_econ_fit(dict(model_type=MT.GROUP_LASSO, group_data=d["clusters"],
                                  **PROD_SIGN, **PROD_ADAPT))
    if name == "factorlasso_econ_prior":
        return make_econ_fit(dict(base, **PROD_SIGN, **PROD_ADAPT,
                                  factors_beta_prior=d["prior"]))
    if name == "factorlasso_econ_sign_prior":
        return make_econ_fit(dict(base, **PROD_SIGN, **PROD_ADAPT,
                                  factors_beta_loading_signs=d["sign"],
                                  factors_beta_prior=d["prior"]))
    if name == "factorlasso_fcgl_sign_prior":
        # FCGL twin of HCGL+SIGN+PRIOR (the deployed configuration): identical
        # sign+prior config, cluster-by-factor block penalty.
        return make_econ_fit(dict(model_type=MT.FACTOR_CLUSTER_GROUP_LASSO,
                                  cutoff_fraction=PROD_CUTOFF, **PROD_SIGN, **PROD_ADAPT,
                                  factors_beta_loading_signs=d["sign"],
                                  factors_beta_prior=d["prior"]))
    if name == "factorlasso_econ_oracle_sign_prior":
        # D.1 with-prior counterpart: the deployed HCGL+SIGN+PRIOR
        # configuration with the sign pooling unit swapped from the
        # HCGL-discovered partition to the true sub-asset-class taxonomy.
        # Tests whether discovery changes the deployed result.
        return make_econ_fit(dict(model_type=MT.GROUP_LASSO, group_data=d["clusters"],
                                  **PROD_SIGN, **PROD_ADAPT,
                                  factors_beta_loading_signs=d["sign"],
                                  factors_beta_prior=d["prior"]))
    return ESTIMATORS[name]


def lambda_grid(X_tr, Y_tr, l1, n):
    XtY = X_tr.values.T @ Y_tr.values
    lmax = (np.abs(XtY).max() / len(X_tr)) if l1 else (np.linalg.norm(XtY, axis=0).max() / len(X_tr))
    return lmax * np.geomspace(1.0, 1e-7, n)


def _bic(X_tr, Y_tr, bh, intercept, tickers):
    resid = Y_tr.values - X_tr.values @ bh.T - intercept.reindex(tickers).values
    rss = float((resid ** 2).sum())
    nobs = Y_tr.size
    df = int((np.abs(bh) > _TOL).sum()) + len(tickers)         # loadings + intercepts
    return nobs * np.log(rss / nobs + 1e-300) + df * np.log(nobs)


def score(name, res, d, X_te, Y_te):
    B, bh = d["B"], res.beta_hat.reindex(index=d["tickers"], columns=FACT).values
    y_pred = (X_te.values @ bh.T) + res.intercept_hat.reindex(d["tickers"]).values
    att = M.compute_attribution(B, bh, factor_cov=d["SIGMA_F"], target_factor_idx=d["ci"],
                                leak_factor_idx=d["ei"], asset_idx=d["cidx"])
    Ss, Sh = B @ d["SIGMA_F"] @ B.T, bh @ d["SIGMA_F"] @ bh.T
    cov_err = float(np.linalg.norm(Sh - Ss) / np.linalg.norm(Ss))
    ari_sac = ari_ac = np.nan
    ch = res.extra.get("clusters_hat")
    if ch is not None:
        ch = np.asarray(ch)
        try:
            ari_sac = M.cluster_recovery_ari(d["ari_sac"], ch)
            ari_ac = M.cluster_recovery_ari(d["ari_ac"], ch)
        except Exception:
            pass
    return dict(method=name, beta_mse=M.beta_mse_normalised(B, bh),
                support_f1=M.support_recovery_f1(B, bh), sign_agree=M.sign_agreement_rate(B, bh),
                oos_r2=M.oos_r2(Y_te.values, y_pred), cluster_ari_sac=ari_sac, cluster_ari_ac=ari_ac,
                runtime_s=res.runtime, credit_recovery=att["recovery"], credit_abs_err=att["abs_error"],
                equity_leak=att["leakage"], risk_share_err=att["risk_share_error"],
                rp_rmse=M.factor_rp_rmse(B, bh, d["lam_true"]), cov_err=cov_err)


def run(d, seeds, n_lambda, sample_sizes, selectors):
    rows = []
    for T in sample_sizes:
        for seed in seeds:
            (X_tr, Y_tr), (X_te, Y_te) = panel(d, seed, T)
            for name, _ in ROSTER:
                fit = get_fit(name, d)
                grid = lambda_grid(X_tr, Y_tr, name in L1_ANCHORED, 1 if name == "OLS" else n_lambda)
                fits = []
                for k, lam in enumerate(grid):
                    try:
                        res = fit(X_tr, Y_tr, float(lam), true_clusters=d["clusters"])
                    except Exception:
                        continue
                    bh = res.beta_hat.reindex(index=d["tickers"], columns=FACT).values
                    fits.append((k, res, M.beta_mse_normalised(d["B"], bh),
                                 _bic(X_tr, Y_tr, bh, res.intercept_hat, d["tickers"])))
                if not fits:
                    continue
                chosen = {"oracle": min(fits, key=lambda f: f[2]),
                          "bic": min(fits, key=lambda f: f[3])}
                for sel in selectors:
                    k, res, _, _ = chosen[sel]
                    r = score(name, res, d, X_te, Y_te)
                    r.update(seed=seed, T=T, selector=sel,
                             oracle_at_edge=bool(k in (0, len(grid) - 1)) and name != "OLS")
                    rows.append(r)
    return pd.DataFrame(rows)


def _print_table(agg, T, sel):
    print(f"\n=== T={T}, selector={sel} ===")
    print(f"{'method':24s}" + "".join(f"{HDR[m]:>9s}" for m in METRICS))
    print("-" * (24 + 9 * len(METRICS)))
    label = dict(ROSTER)
    for name, _ in ROSTER:
        if name not in agg.index:
            continue
        r = agg.loc[name]
        print(f"{label[name]:24s}" + "".join(
            "     -- " if pd.isna(r[m]) else f"{r[m]:9.3f}" for m in METRICS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--data-dir", type=Path, default=here / "data")
    ap.add_argument("--factor-nav", type=Path, default=here / "data" / "futures_risk_factors.csv")
    ap.add_argument("--outdir", type=Path, default=here.parents[0] / "simulations" / "results_calibrated")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--n-lambda", type=int, default=14)
    ap.add_argument("--sample-sizes", type=str, default="60,112,240")
    ap.add_argument("--selectors", type=str, default="oracle,bic")
    ap.add_argument("--quick", action="store_true", help="2 seeds, 6-pt grid, T=112")
    ap.add_argument("--seed-start", type=int, default=101, help="first seed (distinct per chunk)")
    ap.add_argument("--append", action="store_true", help="append to existing raw CSV (chunked runs)")
    args = ap.parse_args()

    X, Y, uni = load_data(args.data_dir, args.factor_nav)
    d = build_dgp(X, Y, uni)
    n_seeds = 2 if args.quick else args.seeds
    n_lambda = 6 if args.quick else args.n_lambda
    sizes = [112] if args.quick else [int(s) for s in args.sample_sizes.split(",")]
    selectors = [s.strip() for s in args.selectors.split(",")]
    print(f"DGP: N={len(d['tickers'])}, M={len(FACT)}, Credit-Equity factor corr "
          f"{X.corr().loc['Equity','Credit']:.2f}; true mean credit β = "
          f"{d['B'][d['cidx'], d['ci']].mean():.2f}; production profile: cutoff={PROD_CUTOFF}, tau={PROD_SIGN['auto_sign_threshold_t']}, adaptive floor={PROD_ADAPT['auto_sign_adaptive_floor']}")
    print(f"Monte Carlo: {n_seeds} seeds, {n_lambda}-pt grid, T in {sizes}, selectors {selectors}")

    seeds = range(args.seed_start, args.seed_start + n_seeds)
    L_new = run(d, seeds, n_lambda, sizes, selectors)
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_path = args.outdir / "etf_competitor_study_raw.csv"
    if args.append and raw_path.exists():
        L = pd.concat([pd.read_csv(raw_path), L_new], ignore_index=True) \
              .drop_duplicates(subset=["method", "seed", "T", "selector"], keep="last")
    else:
        L = L_new
    L.to_csv(raw_path, index=False)
    grp = L.groupby(["T", "selector", "method"])[METRICS]
    n_seed = L.groupby(["T", "selector", "method"])["seed"].nunique()
    agg_mean = grp.mean()
    agg_se = grp.std(ddof=1).div(np.sqrt(n_seed), axis=0)  # standard error across seeds
    agg_se.columns = [f"{c}_se" for c in agg_se.columns]
    agg = agg_mean.join(agg_se)
    agg.to_csv(args.outdir / "etf_competitor_study_summary.csv")
    by_T = L.groupby("T")["seed"].nunique().to_dict()
    print(f"ran seeds {min(seeds)}–{max(seeds)}; raw CSV now holds seeds/T = {by_T}")
    if args.append:
        for T in sizes:                                   # brief sanity, not full tables
            for sel in selectors:
                if (T, sel) in agg.index.droplevel(2).unique():
                    s = agg.loc[(T, sel)]
                    fl = s.loc["factorlasso_econ_sign_prior"] if "factorlasso_econ_sign_prior" in s.index else None
                    sk = s.loc["skglm_grouplasso"] if "skglm_grouplasso" in s.index else None
                    if fl is not None and sk is not None:
                        print(f"  T={T} {sel:6s}: FL+SIGN+PRIOR betaMSE={fl.beta_mse:.3f} crRecov={fl.credit_recovery:.3f} | "
                              f"skglm betaMSE={sk.beta_mse:.3f} crRecov={sk.credit_recovery:.3f}")
    else:
        for T in sizes:
            for sel in selectors:
                if (T, sel) in agg.index.droplevel(2).unique():
                    _print_table(agg.loc[(T, sel)].reindex([n for n, _ in ROSTER]), T, sel)
    edge = L[L["oracle_at_edge"]]
    if len(edge):
        print(f"[warn] oracle-λ at grid edge for {sorted(edge.method.unique())}")
    print(f"raw -> {raw_path}\nsummary -> {args.outdir/'etf_competitor_study_summary.csv'}")


if __name__ == "__main__":
    main()
