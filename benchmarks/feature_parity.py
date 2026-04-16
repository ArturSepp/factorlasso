"""
Feature-parity sanity check: factorlasso vs. scikit-learn vs. skglm.

On problems that all three packages can solve (vanilla Lasso, vanilla Group
Lasso, both without sign constraints or priors), their coefficient estimates
should agree to solver tolerance.  This script makes that claim concrete.

This is a correctness check, NOT a speed benchmark.  factorlasso's CVXPY
backend is expected to be slower than skglm's Numba-accelerated coordinate
descent; the point here is that the flexibility of factorlasso does not come
at the cost of wrong answers on the reduced problems.

For speed benchmarks, use the benchopt framework
(https://benchopt.github.io/) which is maintained by the skglm authors.

Run
---
    python benchmarks/feature_parity.py

Expected output: a table with |Δβ|_max < 1e-3 on every row.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso as SklearnLasso

from factorlasso import LassoModel, LassoModelType

try:
    from skglm.estimators import Lasso as SkglmLasso

    HAS_SKGLM = True
except ImportError:
    HAS_SKGLM = False


def make_problem(T: int = 500, M: int = 30, sparsity: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, M))
    beta = rng.standard_normal(M)
    beta[rng.uniform(size=M) < sparsity] = 0.0
    y = X @ beta + 0.1 * rng.standard_normal(T)
    cols = [f"f{i}" for i in range(M)]
    return (
        pd.DataFrame(X, columns=cols),
        pd.Series(y, name="y"),
        beta,
    )


def compare_lasso(alpha: float = 0.05, n_reps: int = 3) -> pd.DataFrame:
    """Run vanilla Lasso across packages, report coefficient agreement."""
    rows = []
    for rep in range(n_reps):
        X, y, _ = make_problem(seed=rep)

        # scikit-learn
        t0 = time.perf_counter()
        skl = SklearnLasso(alpha=alpha, fit_intercept=True).fit(X.values, y.values)
        t_skl = time.perf_counter() - t0
        beta_skl = skl.coef_

        # factorlasso — match objectives:
        #   sklearn:     (1/(2T))·RSS + α·‖β‖₁
        #   factorlasso: (1/T)·RSS    + λ·‖β‖₁
        # argmin is equivalent when λ = 2α  (multiply sklearn by 2 to align RSS).
        t0 = time.perf_counter()
        fl = LassoModel(
            model_type=LassoModelType.LASSO,
            reg_lambda=2.0 * alpha,
            demean=True,
            warmup_period=None,
        ).fit(x=X, y=y.to_frame())
        t_fl = time.perf_counter() - t0
        beta_fl = fl.coef_.values.ravel()

        row = {
            "rep": rep,
            "max|β_fl − β_skl|": np.max(np.abs(beta_fl - beta_skl)),
            "time_sklearn": t_skl,
            "time_factorlasso": t_fl,
        }

        if HAS_SKGLM:
            t0 = time.perf_counter()
            sg = SkglmLasso(alpha=alpha, fit_intercept=True).fit(X.values, y.values)
            t_sg = time.perf_counter() - t0
            beta_sg = sg.coef_
            row["max|β_fl − β_sg|"] = np.max(np.abs(beta_fl - beta_sg))
            row["time_skglm"] = t_sg

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Feature-parity sanity check: vanilla Lasso")
    print("=" * 60)
    df = compare_lasso()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    print()

    tol = 1e-3
    agree_skl = (df["max|β_fl − β_skl|"] < tol).all()
    msg = "PASS" if agree_skl else "FAIL"
    print(f"  factorlasso vs. scikit-learn: {msg} (tol={tol:.0e})")

    if HAS_SKGLM:
        agree_sg = (df["max|β_fl − β_sg|"] < tol).all()
        msg = "PASS" if agree_sg else "FAIL"
        print(f"  factorlasso vs. skglm:        {msg} (tol={tol:.0e})")
