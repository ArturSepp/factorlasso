"""
sign_pooling_simulation.py

Monte Carlo engine for the simulation section of the cluster-pooled sign paper.
It demonstrates that gated cluster-pooled sign derivation recovers signs more
reliably than per-response derivation when the response cluster structure is known
or recoverable, and it maps the regime where the advantage vanishes (low cluster
recoverability), which is the regime the eQTL application sits in.

DGP (factor model):
    Y = X Lambda' + E,   X in R^{n x p},  Lambda in R^{N x p},  E ~ N(0, sigma^2 I)
    Lambda has cluster-coherent signs (responses in a cluster share the loading sign
    on the cluster's active factors) and genuine nulls (zero loadings elsewhere).
    sigma is set so each response has population R^2 = r2.
    The recovery target is the marginal loading sign, sign(Lambda_{ij}), which is
    what the univariate sign-derivation estimates.

Studies (Study enum):
    BASE_TABLE        full metric set for all methods at the base config   -> Table S1
    RECOVERABILITY    sign recovery vs cluster recoverability (r2 sweep)    -> Figure S1
    GATE_ROC          false-sign vs sensitivity across the gate threshold   -> Figure S2
    REGIME_MAP        sign recovery across an (r2, n) grid                   -> Table S2
    SIGN_CONSISTENCY  sign recovery vs n, empirical consistency             -> Figure S4

Run: python sign_pooling_simulation.py   (executes a quick SMOKE test)
"""
# packages
import os
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Optional
# factorlasso
from factorlasso import LassoModel, LassoModelType
from factorlasso.sign_constraints import (
    _compute_sign_matrix_per_response,
    derive_sign_constraints,
)
from factorlasso.cluster_utils import compute_clusters_from_corr_matrix


# --------------------------------------------------------------------- DGP

@dataclass(frozen=True)
class DgpConfig:
    """configuration of the factor DGP for cluster-pooled sign recovery."""
    n_responses: int = 60      # N, responses (genes / assets)
    n_clusters: int = 6        # K, response clusters
    n_factors: int = 20        # p, predictors / factors
    n_obs: int = 80            # n, observations
    n_active: int = 5          # active factors per cluster
    r2: float = 0.10           # per-response population signal share
    factor_corr: float = 0.0   # equicorrelation among factors (0.0 = orthogonal)
    coef_low: float = 0.5      # lower bound of |loading|
    coef_high: float = 1.5     # upper bound of |loading|

    def __post_init__(self) -> None:
        if self.n_responses % self.n_clusters != 0:
            raise ValueError(f"n_responses must divide by n_clusters, got {self.n_responses} and {self.n_clusters}")
        if self.n_active > self.n_factors:
            raise ValueError(f"n_active exceeds n_factors, got {self.n_active} > {self.n_factors}")
        if not 0.0 < self.r2 < 1.0:
            raise ValueError(f"r2 must lie in (0, 1), got {self.r2!r}")
        if not 0.0 <= self.factor_corr < 1.0:
            raise ValueError(f"factor_corr must lie in [0, 1), got {self.factor_corr!r}")


def _zscore(m: np.ndarray) -> np.ndarray:
    """standardize columns to zero mean and unit variance."""
    sd = m.std(axis=0)
    return (m - m.mean(axis=0)) / np.where(sd == 0.0, 1.0, sd)


def simulate(cfg: DgpConfig, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """draw one replicate (x, y, true_sign, labels) from the factor DGP.

    Parameters
    ----------
    cfg : DgpConfig
    seed : int
        replicate seed.

    Returns
    -------
    x : pd.DataFrame, shape (n, p)        standardized factors, columns f0..f{p-1}
    y : pd.DataFrame, shape (n, N)        standardized responses, columns y0..y{N-1}
    true_sign : np.ndarray, shape (N, p)  sign(Lambda) in {-1, 0, +1}
    labels : np.ndarray, shape (N,)       cluster label of each response
    """
    rng = np.random.default_rng(seed)
    per = cfg.n_responses // cfg.n_clusters
    labels = np.repeat(np.arange(cfg.n_clusters), per)
    if cfg.factor_corr > 0.0:
        sigma = (1.0 - cfg.factor_corr) * np.eye(cfg.n_factors) + cfg.factor_corr * np.ones((cfg.n_factors, cfg.n_factors))
        chol = np.linalg.cholesky(sigma)
        x = rng.standard_normal((cfg.n_obs, cfg.n_factors)) @ chol.T
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
    return x_df, y_df, np.sign(loadings), labels


# ----------------------------------------------------------------- methods

class SignMethod(str, Enum):
    """sign-derivation methods compared in the study."""
    LASSO = 'LASSO'                    # per-response sign derivation
    LASSO_ADAPTIVE = 'LASSO-adaptive'  # per-response with adaptive sign weights
    GROUP_FIXED = 'GROUP-fixed'        # cluster-pooled, clusters known / fixed (production mode)
    SGL_FIXED = 'SGL-fixed'            # sparse-group lasso, clusters known / fixed
    HCGL = 'HCGL'                      # cluster-pooled, clusters estimated
    FCGL = 'FCGL'                      # factor-cluster group lasso, clusters estimated


_POOLED_METHODS = (SignMethod.GROUP_FIXED, SignMethod.SGL_FIXED, SignMethod.HCGL, SignMethod.FCGL)


def fast_derive_signs(method: SignMethod,
                      x: pd.DataFrame,
                      y: pd.DataFrame,
                      labels: np.ndarray,
                      threshold_t: float = 0.75,
                      cutoff_fraction: float = 0.5,
                      ) -> np.ndarray:
    """solve-free gated sign matrix, shape (N, p) in {-1, 0, +1}.

    The derived signs come from the closed-form gate (pooled univariate slope ->
    SSR -> t -> threshold) that runs before the cone solve and is invariant to the
    coefficient penalty. This calls factorlasso's own gate functions directly and
    is element-for-element identical to reading ``LassoModel.fit().derived_signs_``
    for LASSO (per-response), GROUP_FIXED (known response clusters), and HCGL
    (Ward-discovered response clusters), at a fraction of the cost.
    """
    if method is SignMethod.LASSO:
        return _compute_sign_matrix_per_response(
            x.values, y.values, auto_sign_threshold_t=threshold_t)
    if method is SignMethod.GROUP_FIXED:
        resp = pd.Series(list(labels), index=list(y.columns))
    elif method is SignMethod.HCGL:
        resp, _, _ = compute_clusters_from_corr_matrix(y.corr(), cutoff_fraction)
    else:
        raise ValueError(f"fast path covers LASSO/GROUP_FIXED/HCGL, got {method!r}")
    n_resp, n_fac = y.shape[1], x.shape[1]
    out = np.zeros((n_resp, n_fac), dtype=float)
    resp_vals = resp.values
    for c in pd.unique(resp_vals):
        members = [i for i, cc in enumerate(resp_vals) if cc == c]
        pooled = np.asarray(derive_sign_constraints(
            x, y.iloc[:, members], clusters=None, auto_sign_threshold_t=threshold_t))
        for m in members:
            out[m] = pooled[0]          # pooled rows are identical within a cluster
    return out


def derive_signs(method: SignMethod,
                 x: pd.DataFrame,
                 y: pd.DataFrame,
                 labels: np.ndarray,
                 reg_lambda: float = 0.2,
                 threshold_t: float = 0.75,
                 cutoff_fraction: float = 0.5,
                 ) -> np.ndarray:
    """fit one method and return its derived sign matrix, shape (N, p) in {-1, 0, +1}.

    GROUP_FIXED and SGL_FIXED receive the true cluster labels as group_data; the
    production workflow computes clusters once on stable data and holds them fixed.
    HCGL and FCGL estimate clusters internally from the response correlation.
    """
    group_data = pd.Series([f"c{l}" for l in labels], index=list(y.columns))
    common = dict(reg_lambda=reg_lambda, demean=True,
                  auto_sign_constraints=True, auto_sign_threshold_t=threshold_t)
    # The three sign-distinct methods take the solve-free path (identical result).
    if method in (SignMethod.LASSO, SignMethod.GROUP_FIXED, SignMethod.HCGL):
        return fast_derive_signs(method, x, y, labels,
                                 threshold_t=threshold_t, cutoff_fraction=cutoff_fraction)
    if method is SignMethod.LASSO:
        model = LassoModel(model_type=LassoModelType.LASSO, **common)
    elif method is SignMethod.LASSO_ADAPTIVE:
        model = LassoModel(model_type=LassoModelType.LASSO, auto_sign_adaptive_weights=True, **common)
    elif method is SignMethod.GROUP_FIXED:
        model = LassoModel(model_type=LassoModelType.GROUP_LASSO, group_data=group_data, **common)
    elif method is SignMethod.SGL_FIXED:
        model = LassoModel(model_type=LassoModelType.GROUP_LASSO, group_data=group_data, l1_weight=0.5, **common)
    elif method is SignMethod.HCGL:
        model = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                           cutoff_fraction=cutoff_fraction, **common)
    elif method is SignMethod.FCGL:
        model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                           cutoff_fraction=cutoff_fraction, group_penalty='yuan_lin', **common)
    else:
        raise ValueError(f"unknown method, got {method!r}")
    model.fit(x=x, y=y)
    return model.derived_signs_.values


# ----------------------------------------------------------------- metrics

@dataclass(frozen=True)
class SignScore:
    """sign-recovery metrics on one replicate, conditional on the truth."""
    sign_recovery: float   # P(D == T | T != 0): true signs recovered
    flip: float            # P(D != 0 and D != T | T != 0): wrong nonzero sign
    abstain_true: float    # P(D == 0 | T != 0): gated away a true signal
    false_sign: float      # P(D != 0 | T == 0): nonzero sign on a null cell


def score_signs(derived: np.ndarray, true_sign: np.ndarray) -> SignScore:
    """compare a derived sign matrix to the truth, both in {-1, 0, +1}."""
    if derived.shape != true_sign.shape:
        raise ValueError(f"shape mismatch, got {derived.shape} and {true_sign.shape}")
    nz = true_sign != 0
    nl = ~nz
    rec = float(np.mean(derived[nz] == true_sign[nz])) if nz.any() else float('nan')
    flip = float(np.mean((derived[nz] != 0) & (derived[nz] != true_sign[nz]))) if nz.any() else float('nan')
    abst = float(np.mean(derived[nz] == 0)) if nz.any() else float('nan')
    fls = float(np.mean(derived[nl] != 0)) if nl.any() else float('nan')
    return SignScore(rec, flip, abst, fls)


def _mean_se(values: List[float]) -> Tuple[float, float]:
    """mean and standard error across replications."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr) / np.sqrt(len(arr)))


# ----------------------------------------------------------------- studies

def study_base_table(cfg: DgpConfig,
                     methods: List[SignMethod],
                     reps: int,
                     reg_lambda: float = 0.2,
                     threshold_t: float = 0.75,
                     ) -> pd.DataFrame:
    """full metric set per method at the base configuration (Table S1)."""
    acc: Dict[SignMethod, List[SignScore]] = {m: [] for m in methods}
    for seed in range(reps):
        x, y, true_sign, labels = simulate(cfg, seed)
        for m in methods:
            d = derive_signs(m, x, y, labels, reg_lambda=reg_lambda, threshold_t=threshold_t)
            acc[m].append(score_signs(d, true_sign))
    rows = []
    for m in methods:
        rec = _mean_se([s.sign_recovery for s in acc[m]])
        flp = _mean_se([s.flip for s in acc[m]])
        abt = _mean_se([s.abstain_true for s in acc[m]])
        fls = _mean_se([s.false_sign for s in acc[m]])
        rows.append(dict(method=m.value, pooled=m in _POOLED_METHODS,
                         sign_recovery=rec[0], sign_recovery_se=rec[1],
                         flip=flp[0], flip_se=flp[1],
                         abstain_true=abt[0], abstain_true_se=abt[1],
                         false_sign=fls[0], false_sign_se=fls[1]))
    return pd.DataFrame(rows)


def study_recoverability(cfg: DgpConfig,
                         r2_grid: List[float],
                         methods: List[SignMethod],
                         reps: int,
                         **kw) -> pd.DataFrame:
    """sign recovery vs cluster recoverability via an r2 sweep (Figure S1)."""
    rows = []
    for r2 in r2_grid:
        c = replace(cfg, r2=r2)
        acc: Dict[SignMethod, List[float]] = {m: [] for m in methods}
        for seed in range(reps):
            x, y, true_sign, labels = simulate(c, seed)
            for m in methods:
                acc[m].append(score_signs(derive_signs(m, x, y, labels, **kw), true_sign).sign_recovery)
        for m in methods:
            mean, se = _mean_se(acc[m])
            rows.append(dict(r2=r2, method=m.value, sign_recovery=mean, sign_recovery_se=se))
    return pd.DataFrame(rows)


def study_gate_roc(cfg: DgpConfig,
                   thresholds: List[float],
                   reps: int,
                   method: SignMethod = SignMethod.GROUP_FIXED,
                   reg_lambda: float = 0.2,
                   ) -> pd.DataFrame:
    """false-sign vs sensitivity across the gate threshold (Figure S2).

    sensitivity = sign recovery on true cells; false_sign = nonzero sign on null cells.
    """
    rows = []
    for thr in thresholds:
        sens, fals = [], []
        for seed in range(reps):
            x, y, true_sign, labels = simulate(cfg, seed)
            s = score_signs(derive_signs(method, x, y, labels, reg_lambda=reg_lambda, threshold_t=thr), true_sign)
            sens.append(s.sign_recovery)
            fals.append(s.false_sign)
        sm, sse = _mean_se(sens)
        fm, fse = _mean_se(fals)
        rows.append(dict(threshold=thr, sensitivity=sm, sensitivity_se=sse, false_sign=fm, false_sign_se=fse))
    return pd.DataFrame(rows)


def study_regime_map(cfg: DgpConfig,
                     r2_grid: List[float],
                     n_grid: List[int],
                     reps: int,
                     pooled: SignMethod = SignMethod.HCGL,
                     baseline: SignMethod = SignMethod.LASSO,
                     **kw) -> pd.DataFrame:
    """sign recovery across an (r2, n) grid, pooled vs per-response (Table S2)."""
    rows = []
    for r2 in r2_grid:
        for n in n_grid:
            c = replace(cfg, r2=r2, n_obs=n)
            p_acc, b_acc = [], []
            for seed in range(reps):
                x, y, true_sign, labels = simulate(c, seed)
                p_acc.append(score_signs(derive_signs(pooled, x, y, labels, **kw), true_sign).sign_recovery)
                b_acc.append(score_signs(derive_signs(baseline, x, y, labels, **kw), true_sign).sign_recovery)
            pm, _ = _mean_se(p_acc)
            bm, _ = _mean_se(b_acc)
            rows.append(dict(r2=r2, n_obs=n, pooled=pm, per_response=bm, gap=pm - bm))
    return pd.DataFrame(rows)


def study_sign_consistency(cfg: DgpConfig,
                           n_grid: List[int],
                           reps: int,
                           methods: Optional[List[SignMethod]] = None,
                           **kw) -> pd.DataFrame:
    """sign recovery vs n, empirical consistency (Figure S4)."""
    methods = methods or [SignMethod.GROUP_FIXED, SignMethod.LASSO]
    rows = []
    for n in n_grid:
        c = replace(cfg, n_obs=n)
        acc: Dict[SignMethod, List[float]] = {m: [] for m in methods}
        for seed in range(reps):
            x, y, true_sign, labels = simulate(c, seed)
            for m in methods:
                acc[m].append(score_signs(derive_signs(m, x, y, labels, **kw), true_sign).sign_recovery)
        for m in methods:
            mean, se = _mean_se(acc[m])
            rows.append(dict(n_obs=n, method=m.value, sign_recovery=mean, sign_recovery_se=se))
    return pd.DataFrame(rows)


# -------------------------------------------------------------- dispatcher

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, 'results')
# The sign derivation is invariant to the coefficient penalty, so on the sign metric
# LASSO-adaptive coincides with LASSO, SGL-fixed with GROUP-fixed, and FCGL with HCGL.
# We therefore report the three sign-distinct methods: per-response, pooled-known, pooled-estimated.
_ALL_METHODS = [SignMethod.LASSO, SignMethod.GROUP_FIXED, SignMethod.HCGL]


class Study(Enum):
    """selectable studies for run_local_test."""
    SMOKE = 0
    BASE_TABLE = 1
    RECOVERABILITY = 2
    GATE_ROC = 3
    REGIME_MAP = 4
    SIGN_CONSISTENCY = 5
    ALL = 6


def run_local_test(study: Study, reps: int = 200) -> Dict[str, pd.DataFrame]:
    """run one or all studies at the base configuration, save CSVs, return the frames."""
    os.makedirs(_OUT, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    if study is Study.SMOKE:
        cfg = DgpConfig(n_responses=24, n_clusters=4, n_factors=10, n_obs=60, n_active=3, r2=0.15)
        out['base_table'] = study_base_table(cfg, _ALL_METHODS, reps=3)
        out['recoverability'] = study_recoverability(
            cfg, [0.10, 0.50], [SignMethod.LASSO, SignMethod.HCGL, SignMethod.GROUP_FIXED], reps=3)
        out['gate_roc'] = study_gate_roc(cfg, [0.75, 1.5, 2.5], reps=3)
        return out

    base = DgpConfig()
    if study in (Study.BASE_TABLE, Study.ALL):
        out['base_table'] = study_base_table(base, _ALL_METHODS, reps)
    if study in (Study.RECOVERABILITY, Study.ALL):
        out['recoverability'] = study_recoverability(
            base, [0.05, 0.10, 0.20, 0.30, 0.50],
            [SignMethod.LASSO, SignMethod.HCGL, SignMethod.GROUP_FIXED], reps)
    if study in (Study.GATE_ROC, Study.ALL):
        out['gate_roc'] = study_gate_roc(base, [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], reps)
    if study in (Study.REGIME_MAP, Study.ALL):
        out['regime_map'] = study_regime_map(base, [0.10, 0.30, 0.50], [60, 120, 240], reps)
    if study in (Study.SIGN_CONSISTENCY, Study.ALL):
        out['sign_consistency'] = study_sign_consistency(base, [40, 80, 160, 320, 640], reps)

    for name, df in out.items():
        df.to_csv(os.path.join(_OUT, f"{name}.csv"), index=False)
    return out


if __name__ == '__main__':
    # Reproduce the paper's results: all five studies at R = 200 replications.
    # Pass an integer argument for a faster run, e.g. `python sign_pooling_simulation.py 50`.
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    frames = run_local_test(Study.ALL, reps=reps)
    for name, frame in frames.items():
        print(f"\n===== {name} =====")
        print(frame.round(3).to_string(index=False))
