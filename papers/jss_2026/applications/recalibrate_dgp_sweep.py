"""Integrity test: does cluster-factor's identification edge survive within-cluster heterogeneity?

The article DGP builds the true loadings from a shared per-sub-class center
(``cal.BETA``, the row-grouped HCGL production snapshot) plus a +-15%
multiplicative perturbation. Cluster-factor's penalty pools each cluster x
factor block to one value, so a DGP whose truth is *block-constant* would
flatter it. The honest question is whether cluster-factor still wins on
identification (beta-MSE, support) when the truth carries genuine cross-asset
heterogeneity within each cluster, and how that edge decays as the
heterogeneity grows.

This harness:
  1. Re-anchors the ground-truth loadings on a choice of center:
       --anchor ols   : OLS loadings on the real ETF panel (rawest, most
                        heterogeneous, favours neither pooling mode)
       --anchor hcgl  : the canonical per-sub-class cal.BETA center (article default)
  2. Sweeps the within-cluster heterogeneity sigma (the +-X% perturbation),
     so sigma=0 is block-constant (cluster-factor's ideal) and larger sigma is
     genuinely heterogeneous (cluster-factor's hard case).
  3. Runs row-grouped HCGL, cluster-factor, and the plain-LASSO baseline at
     each sigma, and reports beta-MSE / support-F1 / sign-agree / credit-recovery.

Decision rule the output speaks to:
  - If cluster-factor's beta-MSE and support edge over row-grouped PERSISTS at
    sigma in {0.15, 0.30}, the production result is robust and not a
    block-constant artifact.
  - If the edge only appears near sigma=0, cluster-factor is exploiting a truth
    that matches its bias, and the paper should not lead with it.
  - The sign-derivation should beat plain LASSO at every sigma regardless.

Run (from repo root)::

    C:\\Python\\FactorLasso312\\Scripts\\python.exe ^
        papers\\jss_2026\\applications\\recalibrate_dgp_sweep.py ^
        --anchor ols --sigmas 0.0 0.15 0.30 0.45 --seeds 15 --seed-start 101
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse

import numpy as np
import pandas as pd

from papers.jss_2026.applications.run_etf_study import (
    load_data, sign_prior, true_beta, SUBCLASS_TO_PROD, FACT, CREDIT_CLASSES,
)
from papers.jss_2026.applications.etf_simulation_study import (
    panel, score, lambda_grid, make_econ_fit, fit_ols,
    PROD_CUTOFF, PROD_SIGN, PROD_ADAPT, _bic,
)
from papers.jss_2026.simulations import matf_calibration as cal
from papers.jss_2026.simulations import metrics as M
from papers.jss_2026.simulations.estimators import ESTIMATORS, EstimatorResult
from factorlasso import LassoModelType as MT


def ols_center(X, Y, uni, tickers):
    """Per-sub-class OLS center: fit OLS on the real panel, average within sub-class."""
    Xc, Yc = X - X.mean(), Y - Y.mean()
    Bo = pd.DataFrame(np.linalg.lstsq(Xc.values, Yc.values, rcond=None)[0].T,
                      index=tickers, columns=FACT)
    sac = uni["sub_asset_class"].reindex(tickers)
    center = Bo.groupby(sac).mean()       # one center per sub-class
    return center


def build_truth(X, Y, uni, tickers, anchor, sigma, perturb_seed):
    """Construct ground-truth (N, M) loadings: a per-sub-class center + sigma scatter.

    anchor='hcgl' reuses true_beta's center (cal.BETA mapped per sub-class).
    anchor='ols' uses the per-sub-class OLS center from the real panel.
    sigma replaces the hardcoded 0.15 multiplicative heterogeneity.
    """
    if anchor == "hcgl":
        # Reuse the article's center but with our own sigma: rebuild from prodB.
        prodB = pd.DataFrame(cal.BETA, index=cal.ASSETS, columns=FACT)
        rows = []
        for tk in tickers:
            s = uni.loc[tk, "sub_asset_class"]
            if s == "REITs":
                rows.append((0.5 * prodB.loc["MSCI_US"] + 0.5 * prodB.loc["G_Gov"]).values)
            elif s == "Currencies":
                v = np.zeros(len(FACT)); v[FACT.index("Fx")] = 0.9 if tk == "UUP" else -0.9
                rows.append(v)
            else:
                rows.append(prodB.loc[SUBCLASS_TO_PROD[s]].values)
        B0 = np.vstack(rows)
    else:  # ols
        center = ols_center(X, Y, uni, tickers)
        sac = uni["sub_asset_class"].reindex(tickers)
        B0 = np.vstack([center.loc[sac[tk]].values for tk in tickers])

    rng = np.random.default_rng(perturb_seed)
    nz = np.abs(B0) > 1e-9
    B = B0.copy()
    if sigma > 0:
        B[nz] = B0[nz] * (1 + rng.normal(0, sigma, size=B0.shape)[nz])
    return B


def build_dgp_custom(X, Y, uni, B):
    """Mirror etf_simulation_study.build_dgp but with an externally supplied truth B."""
    tickers = list(Y.columns)
    SIGMA_F = (X.cov() * 12).values
    sign, prior = sign_prior(uni, tickers)
    Xc, Yc = X - X.mean(), Y - Y.mean()
    Bo = np.linalg.lstsq(Xc.values, Yc.values, rcond=None)[0].T
    r2 = np.clip(1 - ((Yc.values - Xc.values @ Bo.T) ** 2).sum(0) / (Yc.values ** 2).sum(0),
                 0.05, 0.95)
    sys_var = np.einsum("ij,jk,ik->i", B, SIGMA_F, B)
    resid_var = sys_var * (1 - r2) / r2
    clusters = uni["sub_asset_class"].reindex(tickers)
    ari_sac = pd.factorize(uni["sub_asset_class"].reindex(tickers))[0]
    ari_ac = pd.factorize(uni["asset_class"].reindex(tickers))[0]
    SR_PRIOR = np.array([0.45, 0.30, 0.30, 0.30, 0.10, 0.10, 0.60, 0.30, 0.00])
    lam_true = SR_PRIOR * np.sqrt(np.diag(SIGMA_F))
    cidx = np.array([i for i, t in enumerate(tickers)
                     if uni.loc[t, "sub_asset_class"] in CREDIT_CLASSES])
    return dict(tickers=tickers, B=B, SIGMA_F=SIGMA_F, resid_var=resid_var, clusters=clusters,
                ari_sac=ari_sac, ari_ac=ari_ac, sign=sign, prior=prior, lam_true=lam_true,
                cidx=cidx, ci=FACT.index("Credit"), ei=FACT.index("Equity"))


# Estimators compared at each sigma. Matched config; only model_type differs for
# the two HCGL modes. Plain LASSO is the baseline the sign-derivation must beat.
def fit_for(name, d, with_prior=False):
    base = dict(cutoff_fraction=PROD_CUTOFF, **PROD_SIGN, **PROD_ADAPT)
    if with_prior:
        base = dict(base, factors_beta_loading_signs=d["sign"],
                    factors_beta_prior=d["prior"])
    if name == "row":
        return make_econ_fit(dict(model_type=MT.HIERARCHICAL_CLUSTER_GROUP_LASSO, **base))
    if name == "cluster_factor":
        return make_econ_fit(dict(model_type=MT.FACTOR_CLUSTER_GROUP_LASSO, **base))
    if name == "sklearn_lasso":
        return ESTIMATORS["sklearn_lasso"]
    raise ValueError(name)


METHODS = ["row", "cluster_factor", "sklearn_lasso"]
SHOW = ["beta_mse", "support_f1", "sign_agree", "credit_recovery"]


def run_sigma(X, Y, uni, anchor, sigma, seeds, n_lambda, T, with_prior=False):
    rows = []
    for seed in seeds:
        # Truth perturbation seed is offset so each seed gets a distinct truth,
        # mirroring the article's per-seed DGP draw.
        B = build_truth(X, Y, uni, list(Y.columns), anchor, sigma, perturb_seed=1000 + seed)
        d = build_dgp_custom(X, Y, uni, B)
        (X_tr, Y_tr), (X_te, Y_te) = panel(d, seed, T)
        for name in METHODS:
            fit = fit_for(name, d, with_prior=with_prior)
            l1 = (name == "sklearn_lasso")
            grid = lambda_grid(X_tr, Y_tr, l1, n_lambda)
            fits = []
            for k, lam in enumerate(grid):
                try:
                    res = fit(X_tr, Y_tr, float(lam), true_clusters=d["clusters"])
                except Exception:
                    continue
                bh = res.beta_hat.reindex(index=d["tickers"], columns=FACT).values
                fits.append((k, res, M.beta_mse_normalised(d["B"], bh)))
            if not fits:
                continue
            k, res, _ = min(fits, key=lambda f: f[2])      # oracle selector
            r = score(name, res, d, X_te, Y_te)
            r.update(seed=seed, sigma=sigma, method=name)
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=["ols", "hcgl"], default="ols")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.15, 0.30, 0.45])
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed-start", type=int, default=101)
    ap.add_argument("--n-lambda", type=int, default=10)
    ap.add_argument("--T", type=int, default=112)
    ap.add_argument("--with-prior", action="store_true",
                    help="inject economic sign+prior into both grouped modes "
                         "(mirrors production HCGL+SIGN+PRIOR)")
    _here = Path(__file__).resolve().parent
    ap.add_argument("--data-dir", type=Path, default=_here / "data")
    ap.add_argument("--factor-nav", type=Path,
                    default=_here / "data" / "futures_risk_factors.csv")
    args = ap.parse_args()

    X, Y, uni = load_data(args.data_dir, args.factor_nav)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    all_rows = []
    for sigma in args.sigmas:
        all_rows += run_sigma(X, Y, uni, args.anchor, sigma, seeds, args.n_lambda,
                               args.T, with_prior=args.with_prior)
    df = pd.DataFrame(all_rows)
    tag = "_prior" if args.with_prior else ""
    out = f"recalibrate_sweep_{args.anchor}{tag}_detail.csv"
    df.to_csv(out, index=False)

    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n{'='*88}")
    print(f"Heterogeneity sweep — anchor={args.anchor}, T={args.T}, "
          f"seeds {args.seed_start}..{args.seed_start+args.seeds-1}, oracle selector")
    print(f"truth = per-sub-class {args.anchor.upper()} center + sigma scatter "
          f"(sigma=0 block-constant)")
    print(f"{'='*88}")
    for sigma in args.sigmas:
        sub = df[np.isclose(df["sigma"], sigma)]
        print(f"\n  within-cluster heterogeneity sigma = {sigma:.2f}")
        print(f"    {'method':16s}" + "".join(f"{m:>16s}" for m in SHOW))
        print("    " + "-" * (16 + 16 * len(SHOW)))
        for name in METHODS:
            g = sub[sub["method"] == name]
            if not len(g):
                continue
            cells = "".join(f"{g[m].mean():>9.4f}({se(g[m]):.3f})" for m in SHOW)
            print(f"    {name:16s}{cells}")
        # explicit cluster_factor vs row delta on the two identification metrics
        rg = sub[sub["method"] == "row"]
        cf = sub[sub["method"] == "cluster_factor"]
        if len(rg) and len(cf):
            for m in ["beta_mse", "support_f1"]:
                dlt = cf[m].mean() - rg[m].mean()
                pooled = np.sqrt(se(cf[m])**2 + se(rg[m])**2)
                sig = " (>2SE)" if pooled > 0 and abs(dlt) > 2 * pooled else ""
                better = ("cf better" if (dlt < 0 if m == "beta_mse" else dlt > 0) else "row better")
                print(f"      Δ {m:12s} cf-row = {dlt:+.4f}{sig}  [{better}]")
    print(f"\nPer-seed detail -> {out}")


if __name__ == "__main__":
    main()
