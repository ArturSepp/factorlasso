"""
Example: ``alpha_const_`` (economic intercept) vs. ``intercept_`` (solver
output) on a single-asset factor model
=========================================================================

When ``demean=True`` (the default), ``factorlasso`` removes the conditional
mean from both ``y`` and ``X`` before solving — span=None subtracts the
sample mean, span=integer subtracts a trailing one-sided EWMA mean — and
then fits the no-intercept model

    y_demeaned ≈ X_demeaned · β

The fitted ``LassoModel`` exposes two distinct quantities. Their naming
preserves backward compatibility with pre-0.3.4 code while making the
financial concept of "alpha" available cleanly:

* ``model.alpha_const_`` — the **economic intercept** α in the original
  ``y = α + Xβ + ε`` representation.  Reconstructed from the sample
  means of the *original* (uncentered) ``y`` and ``X`` and the fitted β,
  so the identity ``y_mean = α + x_mean · β`` holds exactly.  This is
  the quantity you want when reporting "alpha after factor exposure".

* ``model.intercept_`` — the **raw solver output**: the EWMA-weighted
  residual mean on the demeaned data the solver actually saw, equal to
  ``model.estimation_result_.alpha``.  This is a mechanical artefact of
  fitting a no-intercept model on centered data:
  * for ``span=None`` it is identically zero by the OLS first-order
    condition;
  * for ``span=integer`` it is a finite-sample EWMA-demean leftover.

  Preserved under this name for back-compat with pre-0.3.4 code.

The example below estimates a synthetic single-asset factor model where
the true α is known.  Compare the two attributes across span choices to
see why ``alpha_const_`` is the right field to read.
"""

import numpy as np
import pandas as pd

from factorlasso import LassoModel, LassoModelType


def main():
    np.random.seed(2026)
    T, M = 2000, 4
    factor_names = ['Equity', 'Rates', 'Credit', 'Commodity']

    # Drift the factor means away from zero so demeaning is non-trivial
    X = pd.DataFrame(np.random.randn(T, M), columns=factor_names)
    X['Equity'] += 0.40
    X['Rates'] -= 0.20

    beta_true = np.array([0.8, 0.1, 0.6, 0.0])
    alpha_true = 0.05
    noise = 0.10 * np.random.randn(T)
    Y = pd.DataFrame(
        (alpha_true + X.values @ beta_true + noise),
        columns=['Asset_A'],
    )

    print(f"True intercept: {alpha_true:+.4f}")
    print(f"True β:         {dict(zip(factor_names, beta_true))}")
    print()

    print(f"{'span':>6} {'β fit':>40} {'alpha_const_':>15} "
          f"{'intercept_':>13} {'note':<42}")
    print("-" * 130)

    for span in [None, 500, 200, 60, 24, 12]:
        m = LassoModel(
            model_type=LassoModelType.LASSO,
            reg_lambda=1e-6,
            span=span,
            demean=True,
            warmup_period=12,
        )
        m.fit(x=X, y=Y)
        beta_est = m.coef_.iloc[0].values
        alpha_const = float(m.alpha_const_.iloc[0])
        intercept_raw = float(m.intercept_.iloc[0])

        if span is None:
            note = "intercept_ = 0 exactly (OLS-on-centered)"
        elif span > T:
            note = "EWMA never converges → intercept_ ≠ 0"
        else:
            note = "EWMA approximates sample mean"

        beta_str = "[" + ", ".join(f"{b:+.2f}" for b in beta_est) + "]"
        span_str = str(span) if span is not None else "None"
        print(f"{span_str:>6} {beta_str:>40} {alpha_const:>+15.4f} "
              f"{intercept_raw:>+13.4f} {note:<42}")

    print()
    print("Read ``alpha_const_`` for the economic intercept α; ``intercept_`` "
          "is the raw solver")
    print("output (a residual-mean diagnostic, not the regression intercept).")


if __name__ == '__main__':
    main()
