"""
sign_pooling_robustness.py — robustness studies for the simulation section.

Three studies, each writing a CSV to ``results/`` consumed by ``exhibits.py``:

    CORRELATED      sign recovery, flip, and false-sign vs predictor correlation
                    (factor_corr sweep)               -> correlated_predictors.csv -> Figure S5
    RHOCUT          HCGL sensitivity to the clustering cutoff fraction
                                                       -> rhocut_sensitivity.csv
    GATE_CORRECTION default vs design-effect-corrected gate at the base config
                                                       -> gate_correction.csv

The correlated study stresses the marginal-sign-agreement condition (A6): with
correlated predictors the marginal pooled slope can disagree in sign with the
partial coefficient, which turns a derived sign into a wrong hard constraint.

The gate-correction study divides the pooled standard error by the design effect
1 + (|C|-1) rho_bar, where rho_bar is the within-cluster residual correlation. The
uncorrected gate reproduces ``factorlasso`` element-for-element (asserted in
``validate_gate``); the correction removes the null over-dispersion.

Run: python sign_pooling_robustness.py
"""
# packages
import os
import csv
import numpy as np
import pandas as pd
import cvxpy as cp
from enum import Enum
from dataclasses import replace
from typing import List, Tuple
from sklearn.linear_model import Lasso
# project
import sign_pooling_simulation as S
from sign_pooling_simulation import DgpConfig, simulate, _zscore, SignMethod, fast_derive_signs, score_signs
# factorlasso
from factorlasso.cluster_utils import compute_clusters_from_corr_matrix

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, 'results')

_METHODS = (SignMethod.LASSO, SignMethod.GROUP_FIXED, SignMethod.HCGL)


# ----------------------------------------------------------------- gate
def _cluster_gate(xj: np.ndarray, yC: np.ndarray, tau: float, deff: bool) -> float:
    """pooled gated sign for one predictor over a response cluster.

    Reproduces factorlasso's closed-form pooled gate
    (beta_j = x_j' (sum_k y_k) / (|C| ||x_j||^2),
     SE = sqrt(sigma2 / (|C| ||x_j||^2)), df = |C|(T-1)) when ``deff`` is False.
    When ``deff`` is True the SE is multiplied by sqrt(1 + (|C|-1) rho_bar), the
    design effect of a clustered mean, with rho_bar the mean within-cluster
    residual correlation (floored at zero).
    """
    T, q = yC.shape
    xx = float(xj @ xj)
    D = q * xx
    beta = float(xj @ yC.sum(axis=1)) / D
    ssr = max(float((yC * yC).sum()) - beta * beta * D, 0.0)
    df = max(q * (T - 1), 1.0)
    sigma2 = ssr / df
    se = np.sqrt(sigma2 / D) if (sigma2 > 0 and D > 0) else np.inf
    t = beta / se if se > 0 else 0.0
    if deff and q > 1:
        slopes = (xj @ yC) / xx                       # (q,) per-response slopes
        resid = yC - np.outer(xj, slopes)             # (T, q)
        corr = np.corrcoef(resid, rowvar=False)
        rho = (corr.sum() - q) / (q * (q - 1))         # mean off-diagonal
        t = t / np.sqrt(1.0 + (q - 1) * max(rho, 0.0))
    return float(np.sign(beta)) if abs(t) >= tau else 0.0


def gate_sign_matrix(x: pd.DataFrame, y: pd.DataFrame, labels: np.ndarray,
                     tau: float = 0.75, deff: bool = False,
                     estimate_clusters: bool = False, cutoff: float = 0.5) -> np.ndarray:
    """pooled gated sign matrix (N, M), optionally design-effect corrected."""
    X = x.values
    Y = y.values
    N, M = Y.shape[1], X.shape[1]
    if estimate_clusters:
        resp, _, _ = compute_clusters_from_corr_matrix(y.corr(), cutoff)
        lab = resp.values
    else:
        lab = labels
    out = np.zeros((N, M), dtype=float)
    for c in np.unique(lab):
        members = np.where(lab == c)[0]
        yC = Y[:, members]
        out[members] = np.array([_cluster_gate(X[:, j], yC, tau, deff) for j in range(M)])
    return out


def validate_gate(reps: int = 30) -> None:
    """assert the uncorrected gate matches factorlasso element-for-element."""
    cfg = DgpConfig()
    for est, meth in ((False, SignMethod.GROUP_FIXED), (True, SignMethod.HCGL)):
        bad = 0
        for seed in range(reps):
            x, y, _, lab = simulate(cfg, seed)
            pkg = fast_derive_signs(meth, x, y, lab, threshold_t=0.75, cutoff_fraction=0.5)
            mine = gate_sign_matrix(x, y, lab, tau=0.75, deff=False,
                                    estimate_clusters=est, cutoff=0.5)
            bad += int((pkg != mine).sum())
        if bad:
            raise AssertionError(f"gate mismatch vs factorlasso ({meth.value}): {bad} cells")
    print("validate_gate: uncorrected gate reproduces factorlasso (0 mismatches).")


# ----------------------------------------------------------------- studies
def _agg(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr) / np.sqrt(len(arr)))


def study_correlated(reps: int = 200,
                     grid: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)) -> pd.DataFrame:
    """sign recovery / flip / false-sign vs predictor correlation."""
    base = DgpConfig()
    rows = []
    for fc in grid:
        cfg = replace(base, factor_corr=fc)
        for meth in _METHODS:
            rec, flp, fls = [], [], []
            for seed in range(reps):
                x, y, ts, lab = simulate(cfg, seed)
                sc = score_signs(fast_derive_signs(meth, x, y, lab, threshold_t=0.75), ts)
                rec.append(sc.sign_recovery)
                flp.append(sc.flip)
                fls.append(sc.false_sign)
            rm, rse = _agg(rec)
            fm, fse = _agg(flp)
            sm, sse = _agg(fls)
            rows.append(dict(factor_corr=fc, method=meth.value,
                             recovery=rm, recovery_se=rse, flip=fm, flip_se=fse,
                             false_sign=sm, false_sign_se=sse))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_RES, 'correlated_predictors.csv'), index=False)
    return df


def study_rhocut(reps: int = 200,
                 grid: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """HCGL sign recovery / flip / false-sign vs the clustering cutoff fraction."""
    base = DgpConfig()
    rows = []
    for cut in grid:
        rec, flp, fls = [], [], []
        for seed in range(reps):
            x, y, ts, lab = simulate(base, seed)
            sc = score_signs(fast_derive_signs(SignMethod.HCGL, x, y, lab,
                                               threshold_t=0.75, cutoff_fraction=cut), ts)
            rec.append(sc.sign_recovery)
            flp.append(sc.flip)
            fls.append(sc.false_sign)
        rm, rse = _agg(rec)
        rows.append(dict(rho_cut=cut, recovery=rm, recovery_se=rse,
                         flip=_agg(flp)[0], false_sign=_agg(fls)[0]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_RES, 'rhocut_sensitivity.csv'), index=False)
    return df


def study_gate_correction(reps: int = 200) -> pd.DataFrame:
    """default vs design-effect-corrected gate at the base configuration."""
    base = DgpConfig()
    rows = []
    for est, mlabel in ((False, 'known-cluster'), (True, 'HCGL')):
        for deff, glabel in ((False, 'default'), (True, 'deff_corrected')):
            rec, flp, fls = [], [], []
            for seed in range(reps):
                x, y, ts, lab = simulate(base, seed)
                d = gate_sign_matrix(x, y, lab, tau=0.75, deff=deff,
                                     estimate_clusters=est, cutoff=0.5)
                sc = score_signs(d, ts)
                rec.append(sc.sign_recovery)
                flp.append(sc.flip)
                fls.append(sc.false_sign)
            rows.append(dict(method=mlabel, gate=glabel,
                             recovery=_agg(rec)[0], flip=_agg(flp)[0], false_sign=_agg(fls)[0]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_RES, 'gate_correction.csv'), index=False)
    return df



# ---------------------------------------------- cooperative-LASSO benchmark
def _simulate_full(cfg: DgpConfig, seed: int):
    """base DGP, additionally returning the noise-free signal and the raw responses."""
    rng = np.random.default_rng(seed)
    per = cfg.n_responses // cfg.n_clusters
    labels = np.repeat(np.arange(cfg.n_clusters), per)
    if cfg.factor_corr > 0.0:
        sigma = (1.0 - cfg.factor_corr) * np.eye(cfg.n_factors) + cfg.factor_corr * np.ones((cfg.n_factors, cfg.n_factors))
        x = rng.standard_normal((cfg.n_obs, cfg.n_factors)) @ np.linalg.cholesky(sigma).T
    else:
        x = rng.standard_normal((cfg.n_obs, cfg.n_factors))
    loadings = np.zeros((cfg.n_responses, cfg.n_factors))
    for k in range(cfg.n_clusters):
        active = rng.choice(cfg.n_factors, cfg.n_active, replace=False)
        signs = rng.choice([-1.0, 1.0], size=cfg.n_active)
        for i in np.where(labels == k)[0]:
            mag = cfg.coef_low + (cfg.coef_high - cfg.coef_low) * rng.random(cfg.n_active)
            loadings[i, active] = signs * mag
    signal = x @ loadings.T
    noise_sd = signal.std(axis=0) * np.sqrt((1.0 - cfg.r2) / cfg.r2)
    y = signal + rng.standard_normal((cfg.n_obs, cfg.n_responses)) * noise_sd
    x_df = pd.DataFrame(_zscore(x), columns=[f"f{j}" for j in range(cfg.n_factors)])
    y_df = pd.DataFrame(_zscore(y), columns=[f"y{i}" for i in range(cfg.n_responses)])
    return x_df, y_df, np.sign(loadings), labels, signal, y


def _simulate_rho_resid(cfg: DgpConfig, rho_resid: float, seed: int):
    """base DGP with within-cluster equicorrelated residuals of correlation rho_resid."""
    rng = np.random.default_rng(seed)
    per = cfg.n_responses // cfg.n_clusters
    labels = np.repeat(np.arange(cfg.n_clusters), per)
    x = rng.standard_normal((cfg.n_obs, cfg.n_factors))
    loadings = np.zeros((cfg.n_responses, cfg.n_factors))
    for k in range(cfg.n_clusters):
        active = rng.choice(cfg.n_factors, cfg.n_active, replace=False)
        signs = rng.choice([-1.0, 1.0], size=cfg.n_active)
        for i in np.where(labels == k)[0]:
            mag = cfg.coef_low + (cfg.coef_high - cfg.coef_low) * rng.random(cfg.n_active)
            loadings[i, active] = signs * mag
    signal = x @ loadings.T
    noise_sd = signal.std(axis=0) * np.sqrt((1.0 - cfg.r2) / cfg.r2)
    a, b = np.sqrt(max(rho_resid, 0.0)), np.sqrt(1.0 - max(rho_resid, 0.0))
    eps = np.zeros((cfg.n_obs, cfg.n_responses))
    for k in range(cfg.n_clusters):
        idx = np.where(labels == k)[0]
        u = rng.standard_normal((cfg.n_obs, 1))
        v = rng.standard_normal((cfg.n_obs, len(idx)))
        eps[:, idx] = a * u + b * v
    y = signal + eps * noise_sd
    x_df = pd.DataFrame(_zscore(x), columns=[f"f{j}" for j in range(cfg.n_factors)])
    y_df = pd.DataFrame(_zscore(y), columns=[f"y{i}" for i in range(cfg.n_responses)])
    return x_df, y_df, np.sign(loadings), labels


def _beta_target(x_df: pd.DataFrame, signal: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
    """partial-coefficient target: OLS of the standardized noise-free signal on X."""
    yn = (signal - y_raw.mean(axis=0)) / y_raw.std(axis=0)
    B, *_ = np.linalg.lstsq(x_df.values, yn, rcond=None)
    return B.T


def _fit_hard(X: np.ndarray, Y: np.ndarray, sign_mat: np.ndarray,
              alphas: np.ndarray, Bt: np.ndarray) -> np.ndarray:
    """per-response sign-constrained LASSO under sign_mat; penalty by oracle beta-MSE."""
    N, M = Y.shape[1], X.shape[1]
    best = None
    for a in alphas:
        Bh = np.zeros((N, M))
        for k in range(N):
            s = sign_mat[k]
            act = np.where(s != 0)[0]
            if len(act) == 0:
                continue
            m = Lasso(alpha=a, positive=True, fit_intercept=False, max_iter=1200, tol=1e-3)
            m.fit(X[:, act] * s[act], Y[:, k])
            Bh[k, act] = s[act] * m.coef_
        e = float(np.mean((Bh - Bt) ** 2))
        if best is None or e < best[0]:
            best = (e, Bh)
    return best[1]


def _fit_coop(X: np.ndarray, Y: np.ndarray, labels: np.ndarray,
              lams: np.ndarray, Bt: np.ndarray) -> np.ndarray:
    """cooperative-LASSO fit per cluster (Chiquet et al. 2012); penalty by oracle beta-MSE.

    Per cluster: min_B ||Y_C - X B||^2 + lam (sum_m ||P_m||_2 + ||N_m||_2),
    B = P - N, P, N >= 0, the sign-coherent cooperative penalty.
    """
    N, M = Y.shape[1], X.shape[1]
    clusters = [np.where(labels == c)[0] for c in np.unique(labels)]
    best = None
    for lam in lams:
        Bh = np.zeros((N, M))
        for mem in clusters:
            Yc = Y[:, mem]
            q = Yc.shape[1]
            P = cp.Variable((M, q), nonneg=True)
            Nn = cp.Variable((M, q), nonneg=True)
            B = P - Nn
            pen = cp.sum(cp.norm(P, 2, axis=1)) + cp.sum(cp.norm(Nn, 2, axis=1))
            prob = cp.Problem(cp.Minimize(cp.sum_squares(Yc - X @ B) + lam * pen))
            try:
                prob.solve(solver=cp.ECOS, abstol=1e-4, reltol=1e-4, max_iters=80)
            except Exception:
                try:
                    prob.solve(solver=cp.SCS, eps=1e-3)
                except Exception:
                    pass
            if B.value is not None:
                Bh[mem, :] = B.value.T
        e = float(np.mean((Bh - Bt) ** 2))
        if best is None or e < best[0]:
            best = (e, Bh)
    return best[1]


def _fitted_sign_metrics(Bh: np.ndarray, Bt: np.ndarray,
                         true_sign: np.ndarray) -> Tuple[float, float, float, float]:
    """recovery, flip, abstention on active cells, and beta-MSE, from fitted-coefficient signs."""
    nz = true_sign != 0
    recovery = float(np.mean(np.sign(Bh[nz]) == true_sign[nz]))
    flip = float(np.mean((Bh[nz] != 0) & (np.sign(Bh[nz]) != true_sign[nz])))
    abstain = float(np.mean(Bh[nz] == 0))
    beta_mse = float(np.mean((Bh - Bt) ** 2))
    return recovery, flip, abstain, beta_mse


def study_cooperative(reps: int = 40,
                      grid: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8),
                      alphas: np.ndarray = np.logspace(-3, -0.5, 4),
                      lams: np.ndarray = np.logspace(-0.5, 1.5, 4)) -> pd.DataFrame:
    """hard sign-constrained fit vs cooperative LASSO (fitted signs) vs predictor correlation.

    Writes coop_benchmark.csv (Table S4). Both fit on known clusters; each penalty
    is selected by oracle beta-MSE; the sign is read from the fitted coefficient.
    Uses cvxpy, so the default replication count is below the other studies.
    """
    base = DgpConfig()
    rows = []
    for fc in grid:
        cfg = replace(base, factor_corr=fc)
        H, C = [], []
        for seed in range(reps):
            x, y, ts, lab, signal, y_raw = _simulate_full(cfg, seed)
            Bt = _beta_target(x, signal, y_raw)
            X, Y = x.values, y.values
            smat = gate_sign_matrix(x, y, lab, tau=0.75, deff=False, estimate_clusters=False)
            H.append(_fitted_sign_metrics(_fit_hard(X, Y, smat, alphas, Bt), Bt, ts))
            C.append(_fitted_sign_metrics(_fit_coop(X, Y, lab, lams, Bt), Bt, ts))
        Hm, Cm = np.mean(H, axis=0), np.mean(C, axis=0)
        rows.append(dict(factor_corr=fc, method='hard-constrained',
                         recovery=Hm[0], flip=Hm[1], abstain=Hm[2], beta_mse=Hm[3], reps=reps))
        rows.append(dict(factor_corr=fc, method='cooperative',
                         recovery=Cm[0], flip=Cm[1], abstain=Cm[2], beta_mse=Cm[3], reps=reps))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_RES, 'coop_benchmark.csv'), index=False)
    return df


def study_rho_resid(reps: int = 100,
                    grid: Tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)) -> pd.DataFrame:
    """known-cluster recovery / abstention / false-sign vs within-cluster residual correlation.

    Writes rho_resid_attenuation.csv (Table S3). Shows the design-effect attenuation
    of the pooling gain as the within-cluster residual correlation rises.
    """
    base = DgpConfig()
    m = base.n_responses // base.n_clusters
    rows = []
    for rb in grid:
        rec, ab, fs = [], [], []
        for seed in range(reps):
            x, y, ts, lab = _simulate_rho_resid(base, rb, seed)
            d = gate_sign_matrix(x, y, lab, tau=0.75, deff=False, estimate_clusters=False)
            act, nul = ts != 0, ts == 0
            rec.append(float(np.mean(d[act] == ts[act])))
            ab.append(float(np.mean(d[act] == 0)))
            fs.append(float(np.mean(d[nul] != 0)))
        rate_factor = float(np.sqrt(m / (1.0 + (m - 1) * rb)))
        rows.append(dict(rho_bar=rb, recovery=float(np.mean(rec)), abstain=float(np.mean(ab)),
                         false_sign=float(np.mean(fs)), rate_factor=rate_factor, reps=reps))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(_RES, 'rho_resid_attenuation.csv'), index=False)
    return df


class Study(str, Enum):
    CORRELATED = 'correlated'
    RHOCUT = 'rhocut'
    GATE_CORRECTION = 'gate_correction'
    COOPERATIVE = 'cooperative'
    RHO_RESID = 'rho_resid'


def run_local_test(study: Study, reps: int = 200) -> pd.DataFrame:
    """dispatch one robustness study."""
    if study is Study.CORRELATED:
        return study_correlated(reps)
    if study is Study.RHOCUT:
        return study_rhocut(reps)
    if study is Study.GATE_CORRECTION:
        return study_gate_correction(reps)
    if study is Study.COOPERATIVE:
        return study_cooperative()
    if study is Study.RHO_RESID:
        return study_rho_resid()
    raise ValueError(f"unknown study, got {study!r}")


if __name__ == '__main__':
    os.makedirs(_RES, exist_ok=True)
    validate_gate()
    for st in Study:
        print(f"running {st.value} ...")
        out = run_local_test(st, reps=200)
        print(out.to_string(index=False))
