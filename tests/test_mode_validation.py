"""Cross-implementation validation of the LASSO mode.

The package minimises ``(1/T)||X b' - Y||_F^2 + lambda ||b||_1`` whereas
scikit-learn's :class:`~sklearn.linear_model.Lasso` minimises
``(1/(2n))||X w - y||^2 + alpha ||w||_1``. With ``n = T`` the squared-error
coefficients coincide when ``alpha = lambda / 2``, so the two estimators share
a minimiser. The first test confirms the matched-penalty equality; the second
shows the naive same-numeric-lambda comparison does *not* agree, which is the
gap the manuscript appendix reconciles.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from factorlasso.lasso_estimator import LassoModel, LassoModelType


def _panel(T=400, K=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, K))
    beta = np.array([1.5, -0.8, 0.0, 0.4, 0.0])
    y = X @ beta + 0.5 * rng.standard_normal(T)
    cols = [f"f{j}" for j in range(K)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="asset"), cols


def test_lasso_matches_sklearn_under_matched_penalty():
    Xdf, y, cols = _panel()
    Xc = Xdf - Xdf.mean()
    yc = y - y.mean()
    lam = 0.05

    fl = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=lam,
                    span=None, demean=False, warmup_period=None)
    fl.fit(x=Xc, y=yc.to_frame("asset"), verbose=False)
    fl_beta = fl.coef_.iloc[0][cols].to_numpy(dtype=float)

    sk = Lasso(alpha=lam / 2.0, fit_intercept=False, tol=1e-12, max_iter=500000)
    sk.fit(Xc.to_numpy(), yc.to_numpy())

    max_abs = float(np.max(np.abs(fl_beta - sk.coef_)))
    print(f"\n[mode-validation] LASSO vs sklearn, matched penalty "
          f"(alpha=lambda/2): max|db| = {max_abs:.3e}")
    print(f"  factorlasso b = {np.round(fl_beta, 4)}")
    print(f"  sklearn     b = {np.round(sk.coef_, 4)}")
    assert max_abs < 2e-3, f"matched-penalty mismatch: max|db|={max_abs:.3e}"


def test_naive_same_lambda_comparison_does_not_agree():
    """Same numeric lambda, sklearn's own intercept: the two disagree. This is
    the 1/T-vs-1/(2n) + intercept artefact, not a correctness defect."""
    Xdf, y, cols = _panel()
    lam = 0.05

    fl = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=lam,
                    span=None, warmup_period=None)
    fl.fit(x=Xdf, y=y.to_frame("asset"), verbose=False)
    fl_beta = fl.coef_.iloc[0][cols].to_numpy(dtype=float)

    sk = Lasso(alpha=lam, fit_intercept=True, tol=1e-12, max_iter=500000)
    sk.fit(Xdf.to_numpy(), y.to_numpy())

    gap = float(np.max(np.abs(fl_beta - sk.coef_)))
    print(f"\n[mode-validation] naive same-lambda comparison: "
          f"max|db| = {gap:.3e}  (expected non-trivial)")
    assert gap > 1e-3
