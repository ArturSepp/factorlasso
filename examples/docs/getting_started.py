"""Canonical example for docs/getting-started.md: a deterministic first fit.

A small synthetic factor model is fitted through the supported top-level API. The script asserts
the structural output the page quotes and, as an independent reference, that the fitted loadings
are close to ordinary least squares on the same demeaned arrays at the small penalty used here.
It needs no network, data file, plotting backend or sibling package.
"""

import numpy as np
import pandas as pd

import factorlasso as fl


def main() -> None:
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(120, 3)), columns=["growth", "rates", "inflation"])
    beta = np.array(
        [
            [0.8, 0.0, -0.2],
            [0.0, 0.6, 0.1],
            [-0.4, 0.2, 0.0],
            [0.3, -0.5, 0.2],
            [0.1, 0.1, 0.7],
        ]
    )
    y = pd.DataFrame(
        x.to_numpy() @ beta.T + 0.05 * rng.normal(size=(len(x), len(beta))),
        columns=[f"asset_{i}" for i in range(len(beta))],
    )

    model = fl.LassoModel(reg_lambda=1e-4).fit(x=x, y=y)
    prediction = model.predict(x)

    print(model.coef_.shape)
    print(prediction.shape)
    print(bool(np.isfinite(prediction.to_numpy()).all()))

    # --- the structural output quoted on the page ------------------------------------------
    assert model.coef_.shape == (5, 3)
    assert prediction.shape == (120, 5)
    assert list(model.coef_.index) == list(y.columns)
    assert list(model.coef_.columns) == list(x.columns)
    assert bool(np.isfinite(prediction.to_numpy()).all())

    # --- independent reference: least squares on the demeaned arrays -----------------------
    xd = x.to_numpy() - x.to_numpy().mean(axis=0)
    yd = y.to_numpy() - y.to_numpy().mean(axis=0)
    ols = np.linalg.lstsq(xd, yd, rcond=None)[0].T
    assert np.max(np.abs(model.coef_.to_numpy() - ols)) < 0.01
    assert np.allclose(
        model.alpha_const_.to_numpy(),
        y.to_numpy().mean(axis=0) - x.to_numpy().mean(axis=0) @ model.coef_.to_numpy().T,
        atol=1e-8,
    )


if __name__ == "__main__":
    main()
