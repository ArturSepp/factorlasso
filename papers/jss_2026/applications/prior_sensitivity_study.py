"""Prior-misspecification sweep (JSS revision, reviewer Concern 2).

The headline credit-recovery result uses a prior centred near the
calibration truth. This script stress-tests that prior: it scales the
credit-prior block by a multiplier m in a wide band around the baseline
(including m < 0, a wrong-signed prior) and reports how credit recovery,
normalised beta-MSE, and covariance error of the
``FL HCGL+SIGN+PRIOR`` configuration degrade as the prior moves away from
the truth. The DGP, the production fit configuration, the sign overlay,
and the metric definitions are imported unchanged from the calibrated
benchmark, so only the prior varies.

The baseline credit prior is (IG, HY, EM) = (0.20, 0.40, 0.30); at
multiplier m the prior block is m * baseline. m = 1 is the paper's
configuration, m = 0 drops the credit prior (pure sign+adaptive), and
m < 0 supplies an economically wrong-signed prior.

Usage (from repository root)::

    python papers/jss_2026/applications/prior_sensitivity_study.py --seeds 15
    # writes simulations/results_calibrated/prior_sensitivity.csv
    # and prints the LaTeX table body
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

warnings.filterwarnings("ignore")

from papers.jss_2026.applications.etf_simulation_study import (
    build_dgp, panel, make_econ_fit, lambda_grid, PROD_CUTOFF, PROD_SIGN, PROD_ADAPT,
)
from papers.jss_2026.applications.run_etf_study import (
    load_data, sign_prior, FACT,
)
from factorlasso import LassoModelType as MT
import papers.jss_2026.simulations.metrics as M

# Prior-scale multipliers. 1.0 is the paper configuration; 0.0 drops the
# credit prior; negative values supply a wrong-signed prior; >1 overshoots.
MULTIPLIERS = [-1.0, -0.5, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0]


def _credit_recovery(B_hat, d):
    ci = d["ci"]
    return float(B_hat[d["cidx"], ci].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed-start", type=int, default=101)
    ap.add_argument("--T", type=int, default=112)
    here = Path(__file__).resolve().parent
    ap.add_argument("--outdir", type=Path,
                    default=here.parents[0] / "simulations" / "results_calibrated")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    X, Y, uni = load_data(base / "data", base / "data" / "futures_risk_factors.csv")
    d = build_dgp(X, Y, uni)
    tickers = d["tickers"]
    sign, prior0 = sign_prior(uni, tickers)
    B_true = d["B"]
    sst = float((B_true ** 2).sum())

    seeds = range(args.seed_start, args.seed_start + args.seeds)
    rows = []
    for m in MULTIPLIERS:
        prior_m = prior0.copy()
        prior_m["Credit"] = prior0["Credit"] * m            # scale credit block
        rec, bmse, cov = [], [], []
        for sd in seeds:
            (X_tr, Y_tr), (X_te, Y_te) = panel(d, sd, args.T)
            fit = make_econ_fit(dict(
                model_type=MT.HIERARCHICAL_CLUSTER_GROUP_LASSO, cutoff_fraction=PROD_CUTOFF,
                **PROD_SIGN, **PROD_ADAPT,
                factors_beta_loading_signs=sign, factors_beta_prior=prior_m))
            # oracle-lambda over the same grid the benchmark uses
            best = None
            for lam in lambda_grid(X_tr, Y_tr, False, 14):
                res = fit(X_tr, Y_tr, lam)
                bh = res.beta_hat.reindex(index=tickers, columns=FACT).values
                e = float(((bh - B_true) ** 2).sum() / sst)
                if best is None or e < best[0]:
                    best = (e, bh)
            bh = best[1]
            rec.append(_credit_recovery(bh, d))
            bmse.append(best[0])
            Sigma = bh @ d["SIGMA_F"] @ bh.T + np.diag(d["resid_var"])
            Strue = B_true @ d["SIGMA_F"] @ B_true.T + np.diag(d["resid_var"])
            cov.append(float(np.linalg.norm(Sigma - Strue) / np.linalg.norm(Strue)))
        rows.append(dict(
            multiplier=m,
            credit_prior_mean=float(prior_m["Credit"][prior_m["Credit"] != 0].mean())
            if (prior_m["Credit"] != 0).any() else 0.0,
            credit_recovery=np.mean(rec), credit_recovery_se=np.std(rec) / np.sqrt(len(rec)),
            beta_mse=np.mean(bmse), beta_mse_se=np.std(bmse) / np.sqrt(len(bmse)),
            cov_err=np.mean(cov), cov_err_se=np.std(cov) / np.sqrt(len(cov)),
        ))
        print(f"m={m:+.2f}  prior_credit={rows[-1]['credit_prior_mean']:+.2f}  "
              f"recov={rows[-1]['credit_recovery']:.3f}  "
              f"bMSE={rows[-1]['beta_mse']:.3f}  covErr={rows[-1]['cov_err']:.3f}")

    df = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "prior_sensitivity.csv", index=False)
    print(f"\nwrote {args.outdir / 'prior_sensitivity.csv'}")

    print("\nLaTeX body (true credit loading = "
          f"{B_true[d['cidx'], d['ci']].mean():.2f}):")
    for _, r in df.iterrows():
        print(f"\t\t\t${r['multiplier']:+.2f}$ & ${r['credit_prior_mean']:+.2f}$ & "
              f"${r['credit_recovery']:.2f}$ & ${r['beta_mse']:.2f}$ & "
              f"${r['cov_err']:.2f}$ \\\\")


if __name__ == "__main__":
    main()
