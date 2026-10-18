"""A completed, inspectable estimator on synthetic monthly decimal returns.

Run after installing factorlasso; no checkout imports, network, files, or sibling
packages are needed. The zero-penalty case is checked against independent least
squares, and diagnostics against direct original-unit residual calculations.
"""

from io import StringIO

import numpy as np
import pandas as pd

from factorlasso import LassoEstimationResult, LassoModel, LassoModelType


def complete_estimator() -> tuple[LassoModel, dict, dict]:
    """Return the fitted estimator, copied numerical tables, and caller metadata."""
    rng = np.random.default_rng(20261018)
    dates = pd.date_range("2020-01-31", periods=72, freq="ME")
    x = pd.DataFrame(
        rng.normal(0.002, 0.025, size=(len(dates), 2)),
        index=dates, columns=["growth", "rates"],
    )
    beta = np.array([[0.8, -0.1], [0.2, 0.7], [0.4, 0.3]])
    y = pd.DataFrame(
        np.array([0.001, 0.002, -0.001]) + x.to_numpy() @ beta.T
        + rng.normal(0.0, 0.003, size=(len(dates), 3)),
        index=dates, columns=["asset_a", "asset_b", "asset_c"],
    )
    # Units and availability belong to the caller, not to inferred estimator fields.
    metadata = {
        "training_start": dates[0].isoformat(),
        "training_cutoff": dates[-1].isoformat(),
        "frequency": "monthly", "returns": "decimal simple returns",
        "annualisation": "none", "availability": "synthetic; known at each date",
    }
    model = LassoModel(
        model_type=LassoModelType.LASSO, reg_lambda=0.0,
        span=None, demean=True, solver="CLARABEL",
    )
    assert model.fit(x=x, y=y) is model
    result = model.estimation_result_
    assert isinstance(result, LassoEstimationResult)
    assert model.coef_.index.equals(y.columns)
    assert model.coef_.columns.equals(x.columns)
    assert model.x_.index.equals(dates) and model.y_.index.equals(dates)
    assert model.valid_mask_.all() and model.effective_span_ is None
    assert np.isfinite(model.coef_.to_numpy()).all()
    assert np.isfinite(model.alpha_const_.to_numpy()).all()
    assert np.isfinite(result.ss_res).all() and np.isfinite(result.r2).all()

    # Independent reference: solve an augmented least-squares system, including alpha.
    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    reference = np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]
    # Decimal-return objectives are small relative to the solver's absolute tolerance.
    np.testing.assert_allclose(model.coef_, reference[1:].T, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(model.alpha_const_, reference[0], atol=1e-7, rtol=0.0)
    prediction = model.predict(x)
    residuals = y.to_numpy() - design @ reference
    reference_residual_variance = np.mean(residuals ** 2, axis=0)
    reference_total_variance = np.var(y.to_numpy(), axis=0, ddof=0)
    np.testing.assert_allclose(prediction, design @ reference, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(result.ss_res, reference_residual_variance, atol=1e-12)
    np.testing.assert_allclose(result.ss_total, reference_total_variance, atol=1e-12)
    np.testing.assert_allclose(
        result.r2, 1.0 - reference_residual_variance / reference_total_variance, atol=1e-8,
    )
    assert np.max(np.abs(model.intercept_)) < 1e-12

    # The estimator is complete here: inspect and copy it before any downstream application.
    tables = {
        "loadings": model.coef_.copy(deep=True),
        "intercepts": model.alpha_const_.copy(deep=True),
        "diagnostics": pd.DataFrame({
            "ss_res": result.ss_res.copy(), "ss_total": result.ss_total.copy(),
            "r2": result.r2.copy(), "valid_observations": model.valid_mask_.sum(axis=0),
        }, index=model.coef_.index.copy()),
    }
    # Ordinary pandas export is explicit and does not serialize a restorable estimator.
    loadings_csv = tables["loadings"].to_csv()
    restored = pd.read_csv(StringIO(loadings_csv), index_col=0)
    pd.testing.assert_frame_equal(restored, tables["loadings"], atol=1e-14, rtol=1e-14)
    detached = tables["loadings"].copy(deep=True)
    detached.iloc[0, 0] += 1.0
    pd.testing.assert_frame_equal(model.coef_, tables["loadings"])
    return model, tables, metadata


def main() -> None:
    """Inspect a complete numerical object without constructing a portfolio or report."""
    model, tables, metadata = complete_estimator()
    print(model.summary())
    print("Training cutoff:", metadata["training_cutoff"])
    print("Units:", metadata["returns"], "; annualisation:", metadata["annualisation"])
    print(tables["loadings"].to_string())
    print(tables["diagnostics"].to_string())


if __name__ == "__main__":
    main()
