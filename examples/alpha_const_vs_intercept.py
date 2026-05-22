"""
Example: ``alpha_const_`` (economic intercept) vs. ``intercept_`` (solver
output) on a single-asset factor model
=========================================================================

When ``demean=True`` (the default), ``factorlasso`` removes the conditional
mean from both ``y`` and ``X`` before solving — span=None subtracts the
sample mean, span=integer subtracts a trailing one-sided EWMA mean — and
then fits the no-intercept model

    y_demeaned ≈ X_demeaned · β

The fitted ``LassoModel`` exposes two distinct quantities:

* ``model.alpha_const_`` — the **economic intercept** α paired
  consistently with β under the same weighted-least-squares objective.
  For ``span=None`` (uniform weights) this is the sample-mean
  reconstruction ``α = ȳ_sample − x̄_sample · β`` (= OLS intercept);
  for ``span=integer`` (EWMA weights) it is the EWMA-weighted-mean
  reconstruction, with the same weights factorlasso applies in the loss
  function.  The result is that the (α, β) pair represents one coherent
  estimator — the weighted-residual mean is zero by the first-order
  condition.

* ``model.intercept_`` — the **raw solver output**: the EWMA-weighted
  residual mean on the demeaned data the solver actually saw, equal to
  ``model.estimation_result_.alpha``.  This is a mechanical artefact of
  fitting a no-intercept model on centered data, preserved under this
  name for back-compat with pre-0.3.4 code:
  * for ``span=None`` it is identically zero by the OLS first-order
    condition;
  * for ``span=integer`` it is a finite-sample EWMA-demean leftover.

The example below estimates a synthetic single-asset factor model where
the true α is known.  It also verifies the FOC: under the same
weighting, the weighted residual mean using ``α_const`` is identically
zero.
"""

import numpy as np
import pandas as pd

from factorlasso import LassoModel, LassoModelType


def main():
    np.random.seed(2026)
    T, M = 200, 4
    factor_names = ['Equity', 'Rates', 'Credit', 'Commodity']

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
    print(f"True β:         {dict(zip(factor_names, beta_true.tolist()))}")
    print()

    print(f"{'span':>6} {'β fit':>40} {'alpha_const_':>15} "
          f"{'intercept_':>13} {'FOC (wtd resid mean)':>22}")
    print("-" * 100)

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

        # FOC check: under the same weighting as the fit, the weighted
        # mean of residuals (y - α_const - X·β) must be zero.
        if span is None:
            w = np.ones(T) / T
        else:
            lam = 1.0 - 2.0 / (span + 1.0)
            w = lam ** np.arange(T - 1, -1, -1)
            w = w / w.sum()
        resid = Y.values[:, 0] - alpha_const - X.values @ beta_est
        foc = float(w @ resid)

        beta_str = "[" + ", ".join(f"{b:+.2f}" for b in beta_est) + "]"
        span_str = str(span) if span is not None else "None"
        print(f"{span_str:>6} {beta_str:>40} {alpha_const:>+15.4f} "
              f"{intercept_raw:>+13.4f} {foc:>+22.2e}")

    print()
    print("FOC column verifies internal consistency: under each span's "
          "weighting, the (α_const, β) pair")
    print("makes the weighted residual mean exactly zero — both are "
          "estimators on the same weighted objective.")
    print()
    print("Read alpha_const_ for the economic intercept α; intercept_ is "
          "the raw solver output")
    print("(a residual-mean diagnostic on demeaned data, not the regression "
          "intercept).")


if __name__ == '__main__':
    main()
