"""Timing-scaling exhibit for the JSS paper (Section 4).

Per-fit wall-clock seconds across panel widths N in {50, 100, 500} at
fixed T = 112, M = 9 (the dimensions of the empirical application).
Clustered factorlasso rows run the production configuration
(cutoff_fraction = 0.40, gate tau = 1.0, adaptive floor 0.5); the
competitor packages are applied per asset, exactly as in the calibrated
benchmark. All estimators are timed at the common operating point
lambda / lambda_max = 0.02. The first fit of every estimator at every N
is discarded as a warm-up (skglm JIT-compiles via numba on first call);
the reported figure is the median of the subsequent repeats.

Usage (from repository root)::

    python papers/jss_2026/simulations/timing_scaling.py
    # writes simulations/results_calibrated/timing_scaling.csv
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
from time import perf_counter

import numpy as np
import pandas as pd

from papers.jss_2026.simulations.dgp import DGPConfig, make_synthetic_panel
from papers.jss_2026.simulations.estimators import ESTIMATORS

warnings.filterwarnings("ignore")
from factorlasso import LassoModel, LassoModelType as MT  # noqa: E402

# Production configuration (matches etf_simulation_study.py)
PROD = dict(model_type=MT.HIERARCHICAL_CLUSTER_GROUP_LASSO, cutoff_fraction=0.40,
            auto_sign_constraints=True, auto_sign_threshold_t=1.0,
            auto_sign_adaptive_weights=True, auto_sign_adaptive_gamma=1.0,
            auto_sign_adaptive_floor=0.5)

ROWS = [
    ("sklearn_lasso", "sklearn Lasso (per asset)"),
    ("skglm_grouplasso", "skglm group LASSO (per asset)"),
    ("asgl_sgl", "asgl SGL (per asset)"),
    ("fl_lasso", "FL LASSO (joint)"),
    ("fl_hcgl_sign_adapt", "FL HCGL+SIGN+ADAPT (joint)"),
    ("fl_fcgl_sign_adapt", "FL FCGL+SIGN+ADAPT (joint)"),
    ("fl_sgl_sign_adapt", "FL SGL+SIGN+ADAPT (joint)"),
]


def _lambda(X, Y, frac=0.02):
    """Common operating point lambda = frac * lambda_max (group norm)."""
    XtY = X.values.T @ Y.values
    lmax = np.linalg.norm(XtY, axis=0).max() / len(X)
    return float(frac * lmax)


def _fl_fit(kwargs):
    def fit(X, Y, lam):
        m = LassoModel(reg_lambda=lam, **kwargs)
        m.fit(x=X, y=Y)
    return fit


def _competitor_fit(name):
    f = ESTIMATORS[name]

    def fit(X, Y, lam):
        f(X, Y, lam)
    return fit


def get_fit(name):
    if name == "fl_lasso":
        return _fl_fit(dict(model_type=MT.LASSO))
    if name == "fl_hcgl_sign_adapt":
        return _fl_fit(dict(PROD))
    if name == "fl_fcgl_sign_adapt":
        return _fl_fit(dict(PROD, model_type=MT.FACTOR_CLUSTER_GROUP_LASSO))
    if name == "fl_sgl_sign_adapt":
        return _fl_fit(dict(PROD, l1_weight=0.1))
    return _competitor_fit(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=str, default="50,100,500")
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed repeats after one discarded warm-up fit")
    ap.add_argument("--T", type=int, default=112)
    here = Path(__file__).resolve().parent
    ap.add_argument("--outdir", type=Path, default=here / "results_calibrated")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    for N in sizes:
        cfg = DGPConfig(N=N, M=9, T=args.T, K=max(6, N // 10),
                        sparsity="moderate", sign_mix="clean", seed=7)
        out = make_synthetic_panel(cfg)
        X, Y = out.X, out.Y
        lam = _lambda(X, Y)
        for name, label in ROWS:
            fit = get_fit(name)
            try:
                fit(X, Y, lam)                         # warm-up, discarded
                times = []
                for _ in range(args.repeats):
                    t0 = perf_counter()
                    fit(X, Y, lam)
                    times.append(perf_counter() - t0)
                med = float(np.median(times))
            except Exception as e:                     # package unavailable
                print(f"[skip] {name} at N={N}: {e}")
                med = np.nan
            rows.append(dict(N=N, method=name, label=label, seconds=med))
            print(f"N={N:4d}  {label:30s}  {med:8.3f}s")

    df = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "timing_scaling.csv", index=False)
    print(f"\nwrote {args.outdir / 'timing_scaling.csv'}")

    wide = df.pivot(index="label", columns="N", values="seconds") \
             .reindex([lb for _, lb in ROWS])
    print("\nLaTeX body:")
    for lb, r in wide.iterrows():
        cells = " & ".join(f"{v:.2f}" for v in r)
        print(f"\t\t\t{lb:34s} & {cells} \\\\")


if __name__ == "__main__":
    main()
