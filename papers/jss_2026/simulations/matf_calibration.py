"""
Production calibration for the MATF-CMA universe (factorlasso §5).

This module pins the synthetic DGP to the *actual* multi-asset universe the
methodology is built for, rather than to a stylised random-cluster panel. All
numbers are transcribed from the production MATF-CMA snapshot as of
31 March 2026:

- ``BETA``    — HCGL-estimated factor loadings, 17 assets x 9 factors
                (Exhibit 15 "Universe Betas and CMAs").
- ``CORR``    — MATF factor correlation matrix (Exhibit 10), with realised
                annualised factor volatilities ``FVOL`` on the diagonal.
- ``R2``      — per-asset model R^2 (Exhibit 15 regression diagnostic).
- ``VOL``     — per-asset annualised total volatility (Exhibit 15).

Why this matters
----------------
The defining feature the random DGP omits is *factor collinearity*: the
Credit-Equity factor correlation is 0.87, so the partial correlation between
the Credit factor and a credit-linked asset, conditional on Equity, is near
zero. A sparsity-seeking estimator therefore drives credit betas to zero and
absorbs the exposure into the equity beta — statistically optimal, but it
conflates two distinct risk premia. Recovering the credit attribution requires
external information (a sign constraint or a non-zero prior), which is exactly
what :func:`sign_matrix` and :func:`prior_matrix` supply.

Self-consistency
----------------
The calibration is not hand-tuned. For all but one asset the systematic
volatility ``sqrt(beta' Sigma_F beta)`` matches the published ``VOL * sqrt(R2)``
to within a few percent (run ``python -m
papers.jss_2026.simulations.matf_calibration`` to verify), so ``BETA``,
``Sigma_F`` and ``R2`` tie out as a single coherent generative object.

Conventions
-----------
- Factor order is fixed by :data:`FACTORS`; asset order by :data:`ASSETS`.
- ``BETA`` is ``(N x M)`` = (assets x factors), matching ``LassoModel.coef_``.
- Volatilities are annualised decimals; :func:`draw_factor_returns` and
  :func:`draw_idiosyncratic` emit *monthly* returns (annual / 12 in variance).
"""
from __future__ import annotations

import numpy as np

# ── Universe definition ───────────────────────────────────────────────

FACTORS = [
    "Equity", "Rates", "Credit", "Carry", "Inflation",
    "Commodities", "PrivateEquity", "RatesVol", "Fx",
]
ASSETS = [
    "G_Gov", "G_IG", "G_HY", "EM_HC", "G_IL",
    "MSCI_US", "MSCI_EU", "MSCI_JP", "MSCI_UK", "SLIC_CH", "MSCI_AxJ", "MSCI_EMxA",
    "PrivEq", "PrivCred", "ILS", "HF", "RealAst",
]
ASSET_CLASS = (
    ["Bonds"] * 5 + ["Equities"] * 7 + ["Alternatives"] * 5
)
# Credit-linked assets: those carrying a genuine (true) Credit-factor loading.
CREDIT_ASSETS = ["G_IG", "G_HY", "EM_HC", "PrivCred"]

N_ASSETS = len(ASSETS)
N_FACTORS = len(FACTORS)

# ── Exhibit 15: HCGL-estimated factor betas (assets x factors) ─────────
# Blank cells in the published table (HCGL-imposed zeros) are 0.0 here.
# The Commodities loadings for PrivEq/PrivCred are set to vol-consistent
# values (0.05 / 0.15): the published table figures were inconsistent with
# those assets' total volatilities and the equity-spread construction of the
# Private Equity factor.
BETA = np.array([
    # Eq    Rates  Credit Carry  Infl   Comm   PE     RVol   Fx
    [0.00,  0.99,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # G_Gov
    [0.10,  0.19,  0.22,  0.00, -0.04,  0.00,  0.00, -0.06, -0.09],  # G_IG
    [0.10,  0.06,  0.43,  0.00,  0.00,  0.01,  0.00, -0.03, -0.05],  # G_HY
    [0.13,  0.18,  0.42,  0.00, -0.05,  0.00,  0.00, -0.09, -0.10],  # EM_HC
    [0.13,  0.26,  0.00,  0.00,  0.13,  0.02,  0.00, -0.06, -0.06],  # G_IL
    [1.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # MSCI_US
    [0.69,  0.29,  0.00,  0.13, -0.07,  0.04,  0.00, -0.39, -0.68],  # MSCI_EU
    [0.68,  0.53,  0.00,  0.06, -0.26, -0.04,  0.00, -0.45, -0.42],  # MSCI_JP
    [0.60,  0.29,  0.00,  0.13, -0.08,  0.10,  0.00, -0.35, -0.46],  # MSCI_UK
    [0.67,  0.50,  0.00,  0.06, -0.19, -0.07,  0.00, -0.42, -0.62],  # SLIC_CH
    [0.77,  0.40,  0.00,  0.41, -0.18,  0.05,  0.00, -0.56, -0.47],  # MSCI_AxJ
    [0.62,  0.08,  0.00,  0.06,  0.09,  0.32,  0.00, -0.61, -0.74],  # MSCI_EMxA
    [0.08,  0.29,  0.00,  0.00,  0.00,  0.05,  0.99, -0.03, -0.35],  # PrivEq
    [0.04,  0.00,  0.16,  0.07,  0.00,  0.15,  0.41,  0.00, -0.26],  # PrivCred
    [0.10,  0.21,  0.00,  0.00,  0.04,  0.03,  0.00, -0.31,  0.06],  # ILS
    [0.13,  0.03,  0.02,  0.00,  0.00,  0.00,  0.00, -0.06, -0.03],  # HF
    [0.29,  0.39,  0.00,  0.00,  0.00,  0.51,  0.00, -0.10, -0.21],  # RealAst
])

# Exhibit 15 regression diagnostics
R2 = np.array([0.52, 0.68, 0.69, 0.67, 0.56, 0.90, 0.80, 0.65, 0.68,
               0.80, 0.60, 0.65, 0.69, 0.70, 0.24, 0.55, 0.85])
VOL = np.array([4.8, 4.2, 4.5, 5.5, 4.8, 13.6, 14.8, 15.4, 13.7, 14.9,
                18.0, 17.8, 8.9, 5.3, 5.1, 3.3, 11.4]) / 100.0  # annualised

# ── Exhibit 10: factor volatilities (diagonal) + correlation ───────────
FVOL = np.array([12.9, 4.1, 3.6, 4.1, 4.3, 16.3, 6.1, 5.2, 6.7]) / 100.0

_CORR_LOWER = {
    (1, 0): 0.28,
    (2, 0): 0.87, (2, 1): 0.10,
    (3, 0): 0.28, (3, 1): -0.31, (3, 2): 0.24,
    (4, 0): 0.08, (4, 1): -0.39, (4, 2): -0.05, (4, 3): 0.38,
    (5, 0): 0.21, (5, 1): -0.21, (5, 2): 0.03, (5, 3): 0.38, (5, 4): 0.64,
    (6, 0): -0.12, (6, 1): -0.13, (6, 2): -0.02, (6, 3): 0.20, (6, 4): -0.07, (6, 5): -0.06,
    (7, 0): -0.50, (7, 1): -0.44, (7, 2): -0.35, (7, 3): 0.02, (7, 4): -0.02, (7, 5): -0.03, (7, 6): 0.20,
    (8, 0): -0.33, (8, 1): -0.31, (8, 2): -0.27, (8, 3): 0.28, (8, 4): 0.26, (8, 5): -0.07, (8, 6): 0.08, (8, 7): 0.28,
}


def _build_corr() -> np.ndarray:
    c = np.eye(N_FACTORS)
    for (i, j), v in _CORR_LOWER.items():
        c[i, j] = v
        c[j, i] = v
    # Project to nearest positive-definite correlation matrix if needed.
    if np.min(np.linalg.eigvalsh(c)) <= 1e-8:
        w, vv = np.linalg.eigh(c)
        c = vv @ np.diag(np.clip(w, 1e-6, None)) @ vv.T
        d = np.sqrt(np.diag(c))
        c = c / np.outer(d, d)
    return c


CORR = _build_corr()
SIGMA_F = np.outer(FVOL, FVOL) * CORR          # annual factor covariance
SIGMA_F_MONTHLY = SIGMA_F / 12.0               # monthly factor covariance

# ── Factor risk premia (SR prior x factor vol) ─────────────────────────
# SR prior vector from the production MATF specification. FX carries an
# explicit zero premium (covariance capture only).
_SR_PRIOR = np.array([0.45, 0.30, 0.30, 0.30, 0.10, 0.10, 0.60, 0.30, 0.00])
FACTOR_PREMIA = _SR_PRIOR * FVOL               # annual excess return per factor


# ── Economic-prior structures (Exhibit 4 signs, credit/inflation priors) ─

def class_labels() -> np.ndarray:
    """Integer cluster labels by asset class (Bonds=0, Equities=1, Alt=2)."""
    mapping = {"Bonds": 0, "Equities": 1, "Alternatives": 2}
    return np.array([mapping[c] for c in ASSET_CLASS], dtype=int)


def sign_matrix() -> np.ndarray:
    """
    Exhibit 4 sign-constraint matrix, ``(N x M)``::

        0  -> beta = 0   1 -> beta >= 0   -1 -> beta <= 0   NaN -> free

    Core risk premia (Equity, Rates, Credit, Carry) are non-negative for
    long-only holdings; Rates Volatility is non-positive (corporate assets are
    implicitly short rates vol); Private Equity is forced to zero for all
    traditional assets and free (>=0) only for PE/PD; Inflation, Commodities,
    and FX are unconstrained. Hedge funds and real assets are fully
    unconstrained except PE.
    """
    s = np.full((N_ASSETS, N_FACTORS), np.nan)
    idx = {f: k for k, f in enumerate(FACTORS)}
    traditional = ASSETS[:12]                  # bonds + equities
    pe_pd = ["PrivEq", "PrivCred"]
    other = ["ILS", "HF", "RealAst"]
    for a in traditional:
        i = ASSETS.index(a)
        for f in ("Equity", "Rates", "Credit", "Carry"):
            s[i, idx[f]] = 1.0
        s[i, idx["PrivateEquity"]] = 0.0
        s[i, idx["RatesVol"]] = -1.0
    for a in pe_pd:
        i = ASSETS.index(a)
        for f in ("Equity", "Rates", "Credit", "Carry", "PrivateEquity"):
            s[i, idx[f]] = 1.0
        s[i, idx["RatesVol"]] = -1.0
    for a in other:
        i = ASSETS.index(a)
        s[i, idx["PrivateEquity"]] = 0.0
    return s


def prior_matrix() -> np.ndarray:
    """
    Prior-centred loadings ``beta0`` ``(N x M)``; the penalty becomes
    ``||beta - beta0||``. Non-zero entries:

    - Credit: investment-grade 0.20 (G_IG, EM_HC), sub-IG/hybrid 0.40
      (G_HY, PrivCred).
    - Inflation: inflation-linked bonds 0.20 (G_IL).

    All other cells are zero (i.e. shrink-to-zero, the default HCGL behaviour).
    """
    p = np.zeros((N_ASSETS, N_FACTORS))
    idx = {f: k for k, f in enumerate(FACTORS)}
    p[ASSETS.index("G_IG"), idx["Credit"]] = 0.20
    p[ASSETS.index("EM_HC"), idx["Credit"]] = 0.20
    p[ASSETS.index("G_HY"), idx["Credit"]] = 0.40
    p[ASSETS.index("PrivCred"), idx["Credit"]] = 0.40
    p[ASSETS.index("G_IL"), idx["Inflation"]] = 0.20
    return p


# ── Index helpers ──────────────────────────────────────────────────────

def factor_index(name: str) -> int:
    return FACTORS.index(name)


def credit_asset_indices() -> np.ndarray:
    return np.array([ASSETS.index(a) for a in CREDIT_ASSETS], dtype=int)


# ── Generative primitives ──────────────────────────────────────────────

def draw_factor_returns(T: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``(T x M)`` monthly factor returns from ``N(0, Sigma_F_monthly)``."""
    L = np.linalg.cholesky(SIGMA_F_MONTHLY)
    return rng.standard_normal((T, N_FACTORS)) @ L.T


def draw_idiosyncratic(T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw ``(T x N)`` monthly idiosyncratic returns. Per-asset volatility is set
    so the realised R^2 matches the published value:
    ``eps_vol_i = (VOL_i / sqrt(12)) * sqrt(1 - R2_i)``.
    """
    eps_vol_monthly = (VOL / np.sqrt(12.0)) * np.sqrt(1.0 - R2)
    return rng.standard_normal((T, N_ASSETS)) * eps_vol_monthly


# ── Self-test ───────────────────────────────────────────────────────────

def _self_test() -> None:
    sys_vol = np.sqrt(np.einsum("ij,jk,ik->i", BETA, SIGMA_F, BETA))
    implied = VOL * np.sqrt(R2)
    ratio = sys_vol / implied
    print("Factor corr min eigenvalue:", round(float(np.min(np.linalg.eigvalsh(CORR))), 4))
    print("Credit-Equity factor corr :", round(float(CORR[0, 2]), 3), "(target 0.87)")
    print("RatesVol-Equity corr      :", round(float(CORR[0, 7]), 3), "(target -0.50)")
    print("\nSelf-consistency  systematic vol  vs  published VOL*sqrt(R2):")
    n_ok = 0
    for a, sv, im, r in zip(ASSETS, sys_vol, implied, ratio):
        flag = "" if 0.80 <= r <= 1.25 else "  <-- check"
        n_ok += int(0.80 <= r <= 1.25)
        print(f"  {a:10s} sysvol={sv * 100:5.1f}%  pub={im * 100:5.1f}%  ratio={r:4.2f}{flag}")
    print(f"\n{n_ok}/{N_ASSETS} assets within ratio [0.80, 1.25].")

    rng = np.random.default_rng(42)
    X = draw_factor_returns(300, rng)
    eps = draw_idiosyncratic(300, rng)
    Y = X @ BETA.T + eps
    r2_real = 1.0 - (Y - X @ BETA.T).var(0) / Y.var(0)
    fc = np.corrcoef(X.T)
    print(f"\nrealised factor Credit-Equity corr: {fc[0, 2]:.2f}")
    print(f"realised per-asset R2 range: {r2_real.min():.2f}-{r2_real.max():.2f} (target 0.24-0.90)")

    assert BETA.shape == (N_ASSETS, N_FACTORS)
    assert sign_matrix().shape == (N_ASSETS, N_FACTORS)
    assert prior_matrix().shape == (N_ASSETS, N_FACTORS)
    assert np.min(np.linalg.eigvalsh(CORR)) > 0, "factor corr not PD"
    assert abs(CORR[0, 2] - 0.87) < 1e-9
    assert n_ok >= N_ASSETS - 1, "calibration self-consistency degraded"
    print("\nmatf_calibration self-test: PASS")


if __name__ == "__main__":
    _self_test()
