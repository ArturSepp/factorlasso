"""Canonical example for docs/empirical_residual_correlation.md.

Eight responses have monthly log-return residuals over ten years, in two blocks of four whose
residuals are correlated at 0.5 within a block and not across. One response stores its
residuals in percent. The residual correlation is estimated on a quarterly common grid, checked
against a direct EWMA computation, and used to assemble the residual covariance
D = S [(1 - rho) I + rho R] S for rho in {0, 0.5, 1}. The script also exercises the availability
date and the failure on a gap. Synthetic data.
"""

import numpy as np
import pandas as pd

import factorlasso as fl

SEED = 20261010
N_MONTHS = 122
WITHIN_BLOCK = 0.5
RESIDUAL_VOL = 0.03
BETA_SPAN = 36
RHOS = [0.0, 0.5, 1.0]
NAMES = [f"a{k}" for k in range(1, 5)] + [f"b{k}" for k in range(1, 5)]
ESTIMATION_DATE = pd.Timestamp("2026-02-28")              # the fit date, two months into Q1


def make_residuals(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly residuals to February 2026, b4 stored in percent, and their metadata."""
    rng = np.random.default_rng(seed)
    block = np.full((4, 4), WITHIN_BLOCK) + (1.0 - WITHIN_BLOCK) * np.eye(4)
    corr = np.kron(np.eye(2), block)
    # Cholesky fixes the basis when the equicorrelation blocks have repeated eigenvalues.
    draws = rng.multivariate_normal(
        np.zeros(8), RESIDUAL_VOL**2 * corr, size=N_MONTHS, method="cholesky",
    )
    dates = pd.date_range("2016-01-31", periods=N_MONTHS, freq="ME")
    residuals = pd.DataFrame(draws, index=dates, columns=NAMES)
    metadata = pd.DataFrame({"frequency": "ME", "beta_span": float(BETA_SPAN),
                             "annualisation_factor": 12.0, "residual_scale": 1.0}, index=NAMES)
    residuals["b4"] *= 100.0                                 # stored in percent
    metadata.loc["b4", "residual_scale"] = 100.0
    return residuals, metadata


def estimate(residuals: pd.DataFrame, metadata: pd.DataFrame) -> fl.ResidualCorrelationData:
    """Quarterly common-grid correlation, available at the fit date."""
    return fl.estimate_residual_correlation(
        residuals, metadata, ESTIMATION_DATE, frequency="QE", periods_per_year=4.0,
    )


def direct_correlation(residuals: pd.DataFrame, metadata: pd.DataFrame, span: float) -> np.ndarray:
    """The same estimate by pandas: quarterly sums, causal EWMA centring, weighted moments."""
    raw = residuals.div(metadata["residual_scale"], axis="columns").loc[:"2025-12-31"]
    quarterly = raw.resample("QE").sum()
    centred = (quarterly - quarterly.ewm(span=span, adjust=False).mean()).iloc[1:].to_numpy()
    decay = 1.0 - 2.0 / (span + 1.0)
    weights = decay ** np.arange(len(centred) - 1, -1, -1)
    moment = (centred * weights[:, None]).T @ centred
    vol = np.sqrt(np.diag(moment))
    return moment / np.outer(vol, vol)


def residual_covariances(prepared: fl.ResidualCorrelationData) -> dict:
    """Residual covariance blocks at each correlation retention rho."""
    variances = pd.DataFrame({fl.VarianceColumns.RESIDUAL_VARS.value: 12 * RESIDUAL_VOL**2},
                             index=NAMES)
    snapshot = fl.CurrentFactorCovarData(
        x_covar=pd.DataFrame([[0.04]], index=["f"], columns=["f"]),
        y_betas=pd.DataFrame({"f": 1.0}, index=NAMES), y_variances=variances,
        estimation_date=ESTIMATION_DATE, residual_correlation=prepared,
    )
    return {rho: snapshot.get_residual_covar(residual_type=fl.ResidualType.EMPIRICAL,
                                             residual_corr_weight=rho)
            for rho in RHOS}


def block_portfolio_vol(blocks: dict) -> dict:
    """Annual residual volatility of an equal-weight portfolio of block a, per rho."""
    weights = np.r_[np.full(4, 0.25), np.zeros(4)]
    return {rho: float(np.sqrt(weights @ block.to_numpy() @ weights))
            for rho, block in blocks.items()}


def main() -> None:
    residuals, metadata = make_residuals()
    prepared = estimate(residuals, metadata)

    # --- the estimate: quarterly grid, converted span, observation before availability ---------
    monthly_decay = 1.0 - 2.0 / (BETA_SPAN + 1.0)
    quarterly_decay = monthly_decay**3
    assert np.isclose(prepared.span, (1.0 + quarterly_decay) / (1.0 - quarterly_decay))
    assert prepared.observation_date == pd.Timestamp("2025-12-31")
    assert prepared.estimation_date == ESTIMATION_DATE
    assert prepared.observation_count == 40
    assert np.allclose(prepared.correlation.to_numpy(),
                       direct_correlation(residuals, metadata, prepared.span))
    try:
        prepared.get_corr(pd.Timestamp("2026-01-31"))       # before the fit date
    except ValueError:
        pass
    else:
        raise AssertionError("correlation was returned before it was available")
    corr = prepared.correlation.to_numpy()
    within = corr[:4, :4][~np.eye(4, dtype=bool)].mean()
    across = corr[:4, 4:].mean()
    print(round(prepared.span, 2), round(within, 2), round(across, 2))

    # --- a gap fails instead of being filled ---------------------------------------------------
    gapped = residuals.copy()
    gapped.iloc[60, 0] = np.nan
    try:
        estimate(gapped, metadata)
    except ValueError:
        pass
    else:
        raise AssertionError("a gapped history was accepted")

    # --- assembly: D = S [(1 - rho) I + rho R] S ------------------------------------------------
    blocks = residual_covariances(prepared)
    vol = np.full(8, np.sqrt(12) * RESIDUAL_VOL)
    for rho, block in blocks.items():
        expected = np.diag(vol) @ ((1.0 - rho) * np.eye(8) + rho * corr) @ np.diag(vol)
        assert np.allclose(block.to_numpy(), expected)
        assert np.array_equal(np.diag(block.to_numpy()),         # the stored diagonal, bit for bit
                              np.full(8, 12 * RESIDUAL_VOL**2))
        assert np.linalg.eigvalsh(block.to_numpy()).min() > 0.0
    orthogonal = fl.CurrentFactorCovarData(
        x_covar=pd.DataFrame([[0.04]], index=["f"], columns=["f"]),
        y_betas=pd.DataFrame({"f": 1.0}, index=NAMES),
        y_variances=pd.DataFrame({fl.VarianceColumns.RESIDUAL_VARS.value: 12 * RESIDUAL_VOL**2},
                                 index=NAMES),
    ).get_residual_covar()
    assert np.array_equal(blocks[0.0].to_numpy(), orthogonal.to_numpy())

    portfolio = block_portfolio_vol(blocks)
    print({rho: round(100 * v, 2) for rho, v in portfolio.items()})

    # quoted values
    assert round(prepared.span, 1) == 12.0
    assert [round(within, 2), round(across, 2)] == [0.49, -0.29]
    assert [round(100 * portfolio[rho], 2) for rho in RHOS] == [5.2, 6.84, 8.15]


if __name__ == "__main__":
    main()
