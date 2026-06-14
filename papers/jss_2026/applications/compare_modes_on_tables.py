"""Compare row-grouped vs cluster-factor HCGL on the article's simulation tables.

This harness reuses the calibrated DGP, the panel simulator, the lambda grid,
the selector logic, and every metric from ``etf_simulation_study.py`` (the
module that produces Table 5 / tab:competitor of the article). For each of the
key HCGL configurations it runs a matched pair that is identical except for the
group-penalty mode:

    HIERARCHICAL_CLUSTER_GROUP_LASSO       (row)            vs
    CLUSTER_FACTOR_GROUP_LASSO (cluster x factor)

so any difference in the reported metrics is attributable to the penalty
geometry alone, on the same 102-asset / 9-factor calibrated structure and the
same metrics as the published table.

Pairs compared (matched config, only model_type differs):
    HCGL+sign+adapt          : PROD_SIGN + PROD_ADAPT, prior 0
    HCGL+SIGN+PRIOR          : PROD_SIGN + PROD_ADAPT + economic sign + prior
    SGL+sign+adapt           : as HCGL+sign+adapt with l1_weight=0.1

Run (from repository root)::

    C:\\Python\\FactorLasso312\\Scripts\\python.exe ^
        papers\\jss_2026\\applications\\compare_modes_on_tables.py ^
        --seeds 15 --seed-start 101
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse

import numpy as np
import pandas as pd

# Reuse the article study's own machinery verbatim.
from papers.jss_2026.applications.etf_simulation_study import (
    build_dgp, panel, score, lambda_grid, make_econ_fit,
    PROD_CUTOFF, PROD_SIGN, PROD_ADAPT, METRICS, HDR, L1_ANCHORED, _bic,
)
from papers.jss_2026.applications.run_etf_study import load_data, FACT
from papers.jss_2026.simulations import metrics as M
from factorlasso import LassoModelType as MT

# Matched-config pairs. Each entry: (label, base_kwargs_without_model_type).
# The harness runs each twice — once row-grouped, once cluster-factor.
PAIRS = [
    ("HCGL+sign+adapt",
     lambda d: dict(cutoff_fraction=PROD_CUTOFF, **PROD_SIGN, **PROD_ADAPT)),
    ("SGL+sign+adapt",
     lambda d: dict(cutoff_fraction=PROD_CUTOFF, l1_weight=0.1, **PROD_SIGN, **PROD_ADAPT)),
    ("HCGL+SIGN+PRIOR",
     lambda d: dict(cutoff_fraction=PROD_CUTOFF, **PROD_SIGN, **PROD_ADAPT,
                    factors_beta_loading_signs=d["sign"], factors_beta_prior=d["prior"])),
]

MODES = [
    ("row", MT.HIERARCHICAL_CLUSTER_GROUP_LASSO),
    ("cluster_factor", MT.CLUSTER_FACTOR_GROUP_LASSO),
]

# Metrics worth showing for the comparison (subset of the table's columns most
# sensitive to penalty geometry, plus runtime to expose the separability cost).
SHOW = ["beta_mse", "support_f1", "sign_agree", "credit_recovery",
        "equity_leak", "risk_share_err", "cov_err", "runtime_s"]


def run_pair(d, label, kw_fn, seeds, n_lambda, sample_sizes, selectors):
    rows = []
    for T in sample_sizes:
        for seed in seeds:
            (X_tr, Y_tr), (X_te, Y_te) = panel(d, seed, T)
            for mode_name, mt in MODES:
                fit = make_econ_fit(dict(model_type=mt, **kw_fn(d)))
                grid = lambda_grid(X_tr, Y_tr, False, n_lambda)
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
                    r = score(f"{label}|{mode_name}", res, d, X_te, Y_te)
                    r.update(seed=seed, T=T, selector=sel, pair=label, mode=mode_name)
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed-start", type=int, default=101)
    ap.add_argument("--n-lambda", type=int, default=12)
    ap.add_argument("--sample-sizes", type=int, nargs="+", default=[112])
    ap.add_argument("--selectors", type=str, nargs="+", default=["oracle"])
    _here = Path(__file__).resolve().parent
    ap.add_argument("--data-dir", type=Path, default=_here / "data")
    ap.add_argument("--factor-nav", type=Path,
                    default=_here / "data" / "futures_risk_factors.csv")
    args = ap.parse_args()

    X, Y, uni = load_data(args.data_dir, args.factor_nav)
    d = build_dgp(X, Y, uni)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    all_rows = []
    for label, kw_fn in PAIRS:
        all_rows += run_pair(d, label, kw_fn, seeds, args.n_lambda,
                             args.sample_sizes, args.selectors)
    df = pd.DataFrame(all_rows)

    out = "compare_modes_on_tables_detail.csv"
    df.to_csv(out, index=False)

    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    for T in args.sample_sizes:
        for sel in args.selectors:
            sub = df[(df["T"] == T) & (df["selector"] == sel)]
            print(f"\n{'='*92}\nT={T}, selector={sel}  "
                  f"(seeds {args.seed_start}..{args.seed_start+args.seeds-1}, "
                  f"calibrated 102-asset / 9-factor DGP)\n{'='*92}")
            for label, _ in PAIRS:
                p = sub[sub["pair"] == label]
                row = p[p["mode"] == "row"]
                cf = p[p["mode"] == "cluster_factor"]
                print(f"\n  {label}")
                print(f"    {'metric':16s}{'row':>12s}{'(se)':>9s}"
                      f"{'cluster_fac':>12s}{'(se)':>9s}{'Δ(cf-row)':>11s}")
                print("    " + "-" * 69)
                for m in SHOW:
                    rm, rs = row[m].mean(), se(row[m])
                    cm, cs = cf[m].mean(), se(cf[m])
                    delta = cm - rm
                    pooled = np.sqrt(rs**2 + cs**2)
                    flag = ""
                    if pooled > 0 and abs(delta) > 2 * pooled:
                        flag = "  **" if delta < 0 or m in ("support_f1", "sign_agree", "credit_recovery") else "  ~~"
                    print(f"    {m:16s}{rm:12.4f}{rs:9.4f}{cm:12.4f}{cs:9.4f}{delta:11.4f}{flag}")
    print(f"\nPer-seed detail written to {out}")
    print("** marks a >2 pooled-SE difference (favourable to cluster_factor for "
          "error metrics, or higher for recovery metrics).")


if __name__ == "__main__":
    main()
