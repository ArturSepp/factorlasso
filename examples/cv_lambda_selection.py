"""
Example: Time-series cross-validation for reg_lambda selection
==============================================================

Demonstrates :class:`factorlasso.LassoModelCV` — picking the regularisation
strength by expanding-window cross-validation on a synthetic asset-factor
panel.

Why expanding-window CV?
------------------------
Random K-fold puts future observations into the training set and past
observations into the test set, leaking information forward in time and
producing optimistic R² estimates. For factor models on returns data
this is the wrong default. ``LassoModelCV`` uses expanding-window splits
(sklearn ``TimeSeriesSplit`` semantics): each successive fold trains on
a strictly larger prefix of history and scores on the immediately
following window — the same way the model is refit and used in production.

The example below sweeps a 15-point log-spaced grid of ``reg_lambda``
across 5 folds, picks the lambda with the highest mean fold R², refits
on the full dataset, and compares the CV-tuned model against an
arbitrarily-chosen default and an over-regularised baseline.
"""

import numpy as np
import pandas as pd

from factorlasso import LassoModel, LassoModelCV


def main():
    # --- 1. Synthetic factor panel ---
    rng = np.random.default_rng(2026)
    T = 260  # ~5 years of weekly observations
    factor_names = ['Equity', 'Rates', 'Credit', 'Commodity']
    asset_names = ['US_Eq', 'EU_Eq', 'EM_Eq', 'US_Govt', 'EU_Govt',
                   'IG_Credit', 'HY_Credit', 'Gold', 'Oil']
    M, N = len(factor_names), len(asset_names)

    beta_true = np.array([
        # Equity  Rates  Credit  Commodity
        [1.0,    0.0,   0.0,    0.0],    # US_Eq
        [0.9,    0.0,   0.0,    0.0],    # EU_Eq
        [1.2,    0.0,   0.1,    0.0],    # EM_Eq
        [0.0,    1.0,   0.0,    0.0],    # US_Govt
        [0.0,    0.8,   0.0,    0.0],    # EU_Govt
        [0.0,    0.3,   0.8,    0.0],    # IG_Credit
        [0.2,    0.0,   1.0,    0.0],    # HY_Credit
        [0.0,    0.0,   0.0,    0.7],    # Gold
        [0.0,    0.0,   0.0,    1.0],    # Oil
    ])

    dates = pd.date_range('2021-01-01', periods=T, freq='W-FRI')
    X = pd.DataFrame(
        0.02 * rng.standard_normal((T, M)),
        index=dates, columns=factor_names,
    )
    Y = pd.DataFrame(
        X.values @ beta_true.T + 0.005 * rng.standard_normal((T, N)),
        index=dates, columns=asset_names,
    )

    # Hold the last 20% out for honest final evaluation
    split = int(0.8 * T)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    Y_train, Y_test = Y.iloc[:split], Y.iloc[split:]

    # --- 2. Cross-validated reg_lambda selection ---
    # Inherit hyperparameters (span, demean, etc.) from a base template;
    # CV will only sweep reg_lambda.
    base = LassoModel(span=52, demean=True)

    cv = LassoModelCV(
        lambdas=np.logspace(-6, -1, 15),
        n_splits=5,
        base_model=base,
        refit=True,
    ).fit(x=X_train, y=Y_train)

    print("=== CV results ===")
    print(f"Best reg_lambda: {cv.best_lambda_:.2e}")
    print(f"Best mean fold R²: {cv.best_score_:.4f}")
    print()

    # --- 3. Score curve across the lambda grid ---
    print("=== Mean R² by reg_lambda (across 5 folds) ===")
    mean_scores = cv.cv_scores_.mean(axis=1, skipna=True)
    score_table = pd.DataFrame({
        'reg_lambda': mean_scores.index,
        'mean_R2':    mean_scores.values,
        'std_R2':     cv.cv_scores_.std(axis=1, skipna=True).values,
    })
    score_table['reg_lambda'] = score_table['reg_lambda'].map(lambda v: f"{v:.2e}")
    print(score_table.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    print()

    # --- 4. Out-of-sample comparison vs. fixed-lambda baselines ---
    naive_default = LassoModel(reg_lambda=1e-5, span=52).fit(x=X_train, y=Y_train)
    over_reg = LassoModel(reg_lambda=1e-1, span=52).fit(x=X_train, y=Y_train)

    print("=== Out-of-sample R² on held-out test window ===")
    print(f"  CV-tuned (λ={cv.best_lambda_:.1e}): {cv.score(X_test, Y_test):.4f}")
    print(f"  Default  (λ=1e-5)          : {naive_default.score(X_test, Y_test):.4f}")
    print(f"  Over-reg (λ=1e-1)          : {over_reg.score(X_test, Y_test):.4f}")
    print()

    # --- 5. Inspect the refitted best model ---
    print("=== Estimated β at best λ (sparsity recovered) ===")
    print(cv.best_model_.coef_.round(3))


if __name__ == '__main__':
    main()
