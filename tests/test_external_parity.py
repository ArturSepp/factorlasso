"""
CI-enforced parity against scikit-learn and skglm.

This test asserts that on problems all three packages can solve (plain Lasso,
no sign constraints, no prior, no EWMA weighting), factorlasso's coefficient
estimates agree with scikit-learn and skglm to within solver tolerance.

It complements ``benchmarks/feature_parity.py`` — the benchmark is a
user-facing script that prints a nice table, this test is a regression guard
that fires on every PR.  Both use ``importorskip`` so the external packages
are not forced into factorlasso's install graph; the tests skip cleanly when
sklearn or skglm is not installed.

Mapping of regularisation parameters
------------------------------------
scikit-learn objective::   (1/(2T)) ‖y − Xβ‖² + α ‖β‖₁
factorlasso objective::    (1/T)     ‖y − Xβ‖² + λ ‖β‖₁

Multiplying sklearn's objective by 2 aligns the RSS term, giving the
equivalence ``λ = 2α``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sklearn_linear_model = pytest.importorskip("sklearn.linear_model")
SklearnLasso = sklearn_linear_model.Lasso

from factorlasso import LassoModel, LassoModelType  # noqa: E402

TOL = 1e-3


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(params=[0, 1, 2])
def synthetic_panel(request):
    """Three independent seeds for robustness."""
    rng = np.random.default_rng(request.param)
    T, M = 500, 30
    X = rng.standard_normal((T, M))
    beta = rng.standard_normal(M)
    beta[rng.uniform(size=M) < 0.7] = 0.0  # ~70% sparse
    y = X @ beta + 0.1 * rng.standard_normal(T)
    cols = [f"f{i}" for i in range(M)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y"), beta


def _fit_factorlasso(X: pd.DataFrame, y: pd.Series, alpha: float) -> np.ndarray:
    """Fit factorlasso and return coefficient vector aligned with sklearn."""
    model = LassoModel(
        model_type=LassoModelType.LASSO,
        reg_lambda=2.0 * alpha,   # see module docstring for the mapping
        demean=True,
        warmup_period=None,       # disable the short-history safeguard
    ).fit(x=X, y=y.to_frame())
    return model.coef_.values.ravel()


# ═══════════════════════════════════════════════════════════════════════
# factorlasso ↔ scikit-learn
# ═══════════════════════════════════════════════════════════════════════

class TestParitySklearn:
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_coef_agreement(self, synthetic_panel, alpha):
        X, y, _ = synthetic_panel
        beta_fl = _fit_factorlasso(X, y, alpha)
        beta_skl = SklearnLasso(alpha=alpha, fit_intercept=True).fit(
            X.values, y.values,
        ).coef_
        max_diff = np.max(np.abs(beta_fl - beta_skl))
        assert max_diff < TOL, (
            f"factorlasso and sklearn disagree by {max_diff:.2e} "
            f"(tol={TOL:.0e}) at alpha={alpha}"
        )

    def test_sparsity_pattern_agreement(self, synthetic_panel):
        """Active-set recovery should match up to a small threshold."""
        X, y, _ = synthetic_panel
        alpha = 0.05
        beta_fl = _fit_factorlasso(X, y, alpha)
        beta_skl = SklearnLasso(alpha=alpha, fit_intercept=True).fit(
            X.values, y.values,
        ).coef_
        # Count coefficients where one is ~0 and the other is not (disagreement)
        thresh = 1e-4
        active_fl = np.abs(beta_fl) > thresh
        active_skl = np.abs(beta_skl) > thresh
        n_mismatched = np.sum(active_fl != active_skl)
        # Allow up to 2 borderline coefficients to differ (solver tolerance)
        assert n_mismatched <= 2, (
            f"Active sets differ at {n_mismatched} coefficients"
        )


# ═══════════════════════════════════════════════════════════════════════
# factorlasso ↔ skglm
# ═══════════════════════════════════════════════════════════════════════

class TestParitySkglm:
    @pytest.fixture(autouse=True)
    def _skip_if_no_skglm(self):
        pytest.importorskip("skglm")

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_coef_agreement(self, synthetic_panel, alpha):
        from skglm.estimators import Lasso as SkglmLasso

        X, y, _ = synthetic_panel
        beta_fl = _fit_factorlasso(X, y, alpha)
        beta_sg = SkglmLasso(alpha=alpha, fit_intercept=True).fit(
            X.values, y.values,
        ).coef_
        max_diff = np.max(np.abs(beta_fl - beta_sg))
        assert max_diff < TOL, (
            f"factorlasso and skglm disagree by {max_diff:.2e} "
            f"(tol={TOL:.0e}) at alpha={alpha}"
        )
