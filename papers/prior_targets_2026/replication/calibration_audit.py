"""Read-only legacy calibration audit before the new multi-asset evidence study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import (
    HERE, REPO, check_snapshot, digest, load_panels, read_json, save_json, validate_root,
)


def main():
    """Reproduce saved OLS rows at original seeds and freeze calibration evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    check_snapshot(root)
    sys.path.insert(0, str(root / "source_snapshot/src"))
    out = root / "E3_calibration_audit_v1"
    out.mkdir()
    files = [
        "applications/run_etf_study.py", "applications/etf_simulation_study.py",
        "simulations/matf_calibration.py", "simulations/estimators.py",
        "simulations/metrics.py", "applications/etf_universe.csv",
        "simulations/results_calibrated/etf_competitor_study_raw.csv",
        "simulations/results_calibrated/etf_competitor_study_summary.csv",
    ]
    legacy = REPO / "papers/jss_2026"
    hashes = {name: digest(legacy / name) for name in files}
    for name in files:
        dest = out / "legacy_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy / name, dest)
    shutil.copy2(HERE / "calibration_audit.py", out / "calibration_audit.py")
    # The original modules install broad warning filters; confine that side effect.
    with warnings.catch_warnings():
        from papers.jss_2026.applications import etf_simulation_study as old
        from papers.jss_2026.applications.run_etf_study import true_beta
    config = read_json(root / "source_snapshot/protocol.json")
    x, y = load_panels(root, config)
    uni = pd.read_csv(root / "source_snapshot/data/etf_universe.csv").set_index("ticker")
    uni = uni.reindex(y.columns)
    assert np.isfinite(x).all().all() and np.isfinite(y).all().all()
    dgp = old.build_dgp(x, y, uni)
    assert len(dgp["tickers"]) == 102 and len(dgp["cidx"]) == 17
    np.testing.assert_array_equal(dgp["B"], true_beta(uni, list(y.columns), perturb_seed=0))
    np.savez(out / "calibration.npz", beta=dgp["B"], sigma_annual=dgp["SIGMA_F"],
             residual_var_annual=dgp["resid_var"], prior=dgp["prior"].to_numpy(),
             signs=dgp["sign"].to_numpy(), credit_indices=dgp["cidx"],
             tickers=np.array(dgp["tickers"]), factors=np.array(old.FACT))
    old_rows = pd.read_csv(legacy / files[-2])
    rows = []
    metrics = ["beta_mse", "credit_recovery", "credit_abs_err", "oos_r2", "cov_err"]
    reference_rows = old_rows[(old_rows.method == "OLS")
                             & (old_rows.selector == "bic")
                             & old_rows.seed.isin(range(101, 106))]
    assert len(reference_rows) == 15
    max_delta = 0.
    for _, row in reference_rows.iterrows():
        train, test = old.panel(dgp, int(row.seed), int(row["T"]))
        fitted = old.fit_ols(*train, 0.)
        recalculated = old.score("OLS", fitted, dgp, *test)
        # Independent intercept-augmented OLS confirms the estimation calculation.
        design = np.column_stack([np.ones(len(train[0])), train[0].to_numpy()])
        direct = np.linalg.lstsq(design, train[1].to_numpy(), rcond=None)[0]
        np.testing.assert_allclose(direct[1:].T, fitted.beta_hat, atol=1e-10)
        for metric in metrics:
            delta = float(recalculated[metric] - row[metric])
            max_delta = max(max_delta, abs(delta))
            rows.append(dict(seed=int(row.seed), n=int(row["T"]), metric=metric,
                             saved=float(row[metric]), reproduced=float(recalculated[metric]),
                             difference=delta))
    pd.DataFrame(rows).to_csv(out / "legacy_ols_reproduction.csv", index=False)
    pd.DataFrame(dgp["B"], index=dgp["tickers"], columns=old.FACT).to_csv(
        out / "calibration_betas.csv")
    # Read-only inspection must not rewrite historical results or their helpers.
    assert hashes == {name: digest(legacy / name) for name in files}
    credit = [dgp["tickers"][i] for i in dgp["cidx"]]
    report = dict(assets=len(y.columns), factors=len(x.columns), observations=len(x),
                  first=str(x.index[0].date()), last=str(x.index[-1].date()),
                  credit_assets=credit, credit_count=len(credit),
                  mean_true_credit_beta=float(dgp["B"][dgp["cidx"], dgp["ci"]].mean()),
                  equity_credit_correlation=float(x.corr().loc["Equity", "Credit"]),
                  original_seeds=[101, 105], sizes=[60, 112, 240],
                  reproduced_ols_rows=len(reference_rows), compared_values=len(rows),
                  max_abs_legacy_metric_difference=max_delta,
                  baseline_reproduced=bool(max_delta < 1e-10),
                  untouched_legacy_hashes=hashes,
                  limitations="Calibration and economic prior share a production source. "
                  "Frozen NAV factors are not a certified publicly rebuildable vintage. "
                  "OLS is reproduced here; legacy penalised/group rows are not rerun.")
    save_json(out / "summary.json", report)
    save_json(out / "manifest.json", {
        str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

