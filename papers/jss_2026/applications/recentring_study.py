"""Re-centring vs prior-centred shrinkage (T2.6).

A natural alternative to the prior-centred penalty is to subtract the prior's
contribution from the response, Y' = Y - X b0, fit a zero-centred model, and add
b0 back. At the cell level the two are algebraically identical when unconstrained:
argmin ||Y - X b||^2 + lam ||b - b0||_1 = b0 + argmin ||Y' - X b~||^2 + lam ||b~||_1,
and the sign constraint does not break it here because the production prior sits
between zero and the truth. Under cluster discovery the equivalence does break:
re-centring changes the response correlation and hence the Ward partition, so the
cluster-pooled sign derivation and adaptive weights act on a different grouping.
This script measures both, on the calibrated DGP.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

import papers.jss_2026.applications.etf_simulation_study as S
from papers.jss_2026.simulations import metrics as M
from factorlasso.lasso_estimator import LassoModel, LassoModelType as MT

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parents[0] / "simulations" / "results_calibrated"
SEEDS = range(101, 111)   # 10 seeds
T = 112
LAM = 1e-4


def _fit(Xc, Yc, prior, signs, model):
    kw = dict(model_type=model, reg_lambda=LAM, demean=False, warmup_period=None,
              factors_beta_prior=prior)
    if model != MT.LASSO:
        kw.update(cutoff_fraction=S.PROD_CUTOFF, auto_sign_constraints=True,
                  auto_sign_threshold_t=1.0, **S.PROD_ADAPT)
    if signs is not None:
        kw["factors_beta_loading_signs"] = signs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LassoModel(**kw).fit(x=Xc, y=Yc)
    return m.coef_.reindex(index=Yc.columns, columns=S.FACT), m.clusters_


def _recovery(bh, d):
    b = bh.reindex(index=d["tickers"], columns=S.FACT).values
    return M.compute_attribution(d["B"], b, factor_cov=d["SIGMA_F"],
                                 target_factor_idx=d["ci"], leak_factor_idx=d["ei"],
                                 asset_idx=d["cidx"])["recovery"]


def main():
    try:
        from sklearn.metrics import adjusted_rand_score
    except Exception:
        adjusted_rand_score = None
    X, Y, uni = S.load_data(DATA, DATA / "futures_risk_factors.csv")
    d = S.build_dgp(X, Y, uni)
    prior = d["prior"].reindex(columns=S.FACT)
    signs = d["sign"].reindex(columns=S.FACT)
    rows = []
    for seed in SEEDS:
        (Xtr, Ytr), _ = S.panel(d, seed, T)
        Xc = Xtr[S.FACT] - Xtr[S.FACT].mean()
        Yc = Ytr - Ytr.mean()
        pa = prior.reindex(index=Ytr.columns)
        Yp = Yc - pd.DataFrame(Xc.values @ pa.values.T, index=Yc.index, columns=Yc.columns)
        zero = pa * 0.0

        # cell level (LASSO): unconstrained and sign-constrained
        bd_u, _ = _fit(Xc, Yc, pa, None, MT.LASSO)
        br_u, _ = _fit(Xc, Yp, zero, None, MT.LASSO)
        br_u = br_u.add(pa, fill_value=0.0)
        bd_s, _ = _fit(Xc, Yc, pa, signs, MT.LASSO)
        br_s, _ = _fit(Xc, Yp, zero, signs, MT.LASSO)
        br_s = br_s.add(pa, fill_value=0.0)

        # cluster discovery (HCGL, the deployed config)
        hd, cd = _fit(Xc, Yc, pa, signs, MT.HIERARCHICAL_CLUSTER_GROUP_LASSO)
        hr, cr = _fit(Xc, Yp, zero, signs, MT.HIERARCHICAL_CLUSTER_GROUP_LASSO)
        hr = hr.add(pa, fill_value=0.0)
        ari = (adjusted_rand_score(np.asarray(cd), np.asarray(cr))
               if adjusted_rand_score is not None and cd is not None and cr is not None else np.nan)

        rows.append(dict(
            seed=seed,
            lasso_db_unconstrained=float((bd_u - br_u).abs().to_numpy().max()),
            lasso_db_signed=float((bd_s - br_s).abs().to_numpy().max()),
            hcgl_db=float((hd - hr).abs().to_numpy().max()),
            hcgl_cluster_ari=ari,
            hcgl_recovery_prior_centred=_recovery(hd, d),
            hcgl_recovery_recentred=_recovery(hr, d),
        ))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "recentring.csv", index=False)
    m = df.mean(numeric_only=True)
    print(f"mean over {len(SEEDS)} seeds:")
    print(f"  LASSO  max|db| unconstrained        = {m.lasso_db_unconstrained:.2e}")
    print(f"  LASSO  max|db| with sign constraint  = {m.lasso_db_signed:.2e}")
    print(f"  HCGL   max|db| (deployed config)     = {m.hcgl_db:.3f}")
    print(f"  HCGL   cluster ARI (Y vs Y')         = {m.hcgl_cluster_ari:.3f}")
    print(f"  HCGL   credit recovery: prior {m.hcgl_recovery_prior_centred:.3f} "
          f"vs re-centred {m.hcgl_recovery_recentred:.3f}")


if __name__ == "__main__":
    main()
