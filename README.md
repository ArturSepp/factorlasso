# factorlasso

**Sparse multi-output regression with sign constraints, prior-centered
regularisation, hierarchical group LASSO, and sparse group LASSO — via CVXPY.**

[![PyPI](https://img.shields.io/pypi/v/factorlasso.svg)](https://pypi.org/project/factorlasso/)
[![Python](https://img.shields.io/pypi/pyversions/factorlasso.svg)](https://pypi.org/project/factorlasso/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`factorlasso` is a small, dependency-light Python package for fitting sparse
multi-output linear models

$$
Y = X\beta^\top + \varepsilon,
\qquad \beta \in \mathbb{R}^{N \times M}
$$

when four things matter:

- Some coefficients **must be zero, non-negative, or non-positive**, possibly by
  asset, by factor, or both.
- You have a **prior** β₀ and want to penalise `‖β − β₀‖`, not `‖β‖`.
- You want **structured sparsity** — groups of responses entering or leaving
  the model together — where the groups are either user-supplied or discovered
  by hierarchical clustering of the response correlation matrix (HCGL).
- You want to combine **group-level selection with within-group elementwise
  sparsity** via a tunable mix of group L2 and L1 penalties (Sparse Group
  LASSO).

It is written in pure numpy/pandas/scipy/cvxpy. No numba, no custom
coordinate descent. The solver is CVXPY (default `CLARABEL`), so problem
formulation is explicit and auditable.

---

## Installation

```bash
pip install factorlasso
```

Requires Python ≥ 3.9, CVXPY ≥ 1.3, and numpy / pandas / scipy / openpyxl.

---

## Quickstart

```python
import numpy as np
import pandas as pd
from factorlasso import LassoModel, LassoModelType

rng = np.random.default_rng(0)
T, M, N = 200, 4, 10
X = pd.DataFrame(rng.standard_normal((T, M)), columns=[f"f{i}" for i in range(M)])
Y = pd.DataFrame(rng.standard_normal((T, N)), columns=[f"y{i}" for i in range(N)])

model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-3).fit(x=X, y=Y)

model.coef_         # (N, M) estimated β
model.intercept_    # (N,) estimated α
model.predict(X)    # Ŷ
model.score(X, Y)   # mean R²
```

The API mirrors scikit-learn: `fit(x, y)`, `predict(x)`, `score(x, y)`,
`get_params()`, `set_params()`. Fitted attributes carry a trailing underscore.

---

## What makes it different

### 1. Per-element sign constraints

A `(N × M)` matrix drives the constraints. Each entry is one of
`{0, 1, -1, NaN}`: equality-to-zero, non-negative, non-positive, or free.
This lets a single fit encode structural knowledge that spans multiple
responses.

```python
signs = pd.DataFrame(np.nan, index=Y.columns, columns=X.columns)
signs.loc["y0", "f0"] = 1      # β[y0, f0] ≥ 0
signs.loc["y0", "f1"] = 0      # β[y0, f1] == 0
signs.loc["y1", "f0"] = -1     # β[y1, f0] ≤ 0

model = LassoModel(
    reg_lambda=1e-3,
    factors_beta_loading_signs=signs,
).fit(x=X, y=Y)
```

Scikit-learn's `Lasso` supports only a single `positive` flag across the whole
coefficient matrix. Arbitrary per-element sign constraints are not expressible
without a custom CVXPY problem; this is that custom problem, packaged.

### 2. Data-driven sign constraints with a noise-floor gate

Hand-coding an `(N × M)` sign matrix scales poorly. Setting
`auto_sign_constraints=True` derives signs inside `fit()` from pooled
univariate slopes computed on the same EWMA-demeaned arrays the CVXPY
solver consumes (no train/test inconsistency, automatic per-fold
derivation under `LassoModelCV`).

```python
model = LassoModel(
    reg_lambda=1e-3,
    auto_sign_constraints=True,    # derive signs from univariate slopes
    auto_sign_threshold_t=0.75,    # noise floor (default 0.75)
).fit(x=X, y=Y)

# Inspect the matrix the solver actually saw
model.derived_signs_
```

How the pooling is dispatched depends on `model_type`:

| `model_type` | Sign-derivation pooling |
|---|---|
| `LASSO` (or single-column `y`) | Per-`y`-column independent univariate fit. Rows of `derived_signs_` may differ across responses. |
| `GROUP_LASSO` | Pool `y` within each `group_data` group. All members of a group share their `derived_signs_` row. |
| `GROUP_LASSO_CLUSTERS` | Pool `y` within each HCGL asset cluster (the same clustering the solver uses). |

**The threshold gate.** `auto_sign_threshold_t` (default `0.75`) is a noise
floor on the per-column univariate t-statistic. Factors with `|t| <`
threshold have their sign pinned to `0`, forcing `β = 0` in the fit.
Rationale: under weak L1 (typical of factor models with `reg_lambda` ≪ 1
and `l1_weight = 0`), an unfiltered slope sign drawn from sampling noise
becomes a hard constraint that the solver can exploit to fit residual
variance via offsetting loading pairs (e.g. +Credit ↔ −Inflation on a
factor whose true effect is zero). The gate is **not a significance test**
— `|t| = 0.75` corresponds to two-sided `p ≈ 0.45`. It is a defensive
filter that removes only the worst noise-driven sign constraints.

Set `auto_sign_threshold_t=None` to disable the gate entirely
(reproduces v0.3.6 behaviour, every univariate sign is enforced regardless
of evidence strength).

**Explicit overrides still work.** Setting `factors_beta_loading_signs`
alongside `auto_sign_constraints=True` overlays the user's matrix on top
of the auto-derived signs per-cell — non-NaN entries win, NaN cells
inherit the auto value. Use this for asset-specific constraints that no
amount of marginal-correlation data could surface (e.g. forcing a
mandate-restricted bond fund to zero equity loading regardless of
spurious sample correlations).

**Adaptive L1 penalty weights (Zou 2006).** Set
`auto_sign_adaptive_weights=True` (default `False`) alongside
`auto_sign_constraints=True` to reweight the L1 penalty elementwise by
the inverse univariate-slope magnitude:

```python
model = LassoModel(
    reg_lambda=1e-3,
    auto_sign_constraints=True,
    auto_sign_adaptive_weights=True,   # opt in to magnitude-aware L1
    auto_sign_adaptive_gamma=1.0,      # Zou (2006) exponent γ
    auto_sign_adaptive_floor=1e-3,     # stabiliser on tiny slopes
).fit(x=X, y=Y)
```

The L1 penalty becomes

```
λ · |β_kj − β⁰_kj| / max(|β̂_uni_kj|, floor)^γ
```

where `β̂_uni_kj` is the same pooled univariate slope used to derive the
sign matrix. Strong-evidence factors (large `|β̂_uni|`) get a lighter L1
penalty and can take larger multivariate coefficients; weak-evidence
factors get a heavier penalty and are pushed harder toward the prior.
This is the Zou (2006) adaptive Lasso oracle property: the penalty
becomes magnitude-aware without being a thresholding operator. The
formulation matches Richland et al. (2025) eq. (3.3); see the technical
note `factorlasso_sign_constraints_note.tex` for the derivation.

The adaptive layer is independent of the threshold gate: cells pinned
to `β = 0` by the gate continue to be forced to zero by the hard sign
constraint, with the adaptive weight acting only on the non-pinned
cells. Default behaviour (`auto_sign_adaptive_weights=False`)
reproduces v0.3.8 fits bit-for-bit.

**Group LASSO mode (`l1_weight=0`).** In pure group-LASSO configurations
where the L1 term is inactive, the adaptive reweighting is routed
through the group L2 norms following Wang & Leng (2008)'s adaptive
group lasso. Per-cell weights are aggregated per-asset by
root-mean-square over the non-pinned factors:

```
W_k  =  sqrt( mean_{j: s_kj ≠ 0} W_kj² )
```

and each asset's contribution `‖β_k − β⁰_k‖₂` to the group penalty is
scaled by `W_k`. This is what gives the adaptive flag actual impact in
the production `GROUP_LASSO_CLUSTERS` configuration where the L1 term
is zero-weighted. Assets with uniformly strong univariate evidence
across factors get `W_k → 1` (preserved); assets with uniformly weak
evidence get `W_k > 1` (shrunk harder toward the prior). Assets with
all cells pinned by the gate fall back to `W_k = 1` (no-op).

**Related work and intellectual lineage.** The univariate-slope-as-
sign-constraint mechanism is adapted from the **uniLasso** framework of
Chatterjee, Hastie & Tibshirani (2025) and its biobank-scale follow-up
by Richland et al. (2025). Specifically, Richland et al. (2025)
eq. (3.3) imposes `sign(γ_j) = sign(β̃_j)` as a hard constraint on the
original variables — structurally identical to what
`factors_beta_loading_signs` encodes here. The broader idea of using
univariate marginal evidence to guide a multivariate fit goes back to
Zou (2006)'s adaptive Lasso, which uses univariate *magnitudes* as
adaptive penalty weights.

Two things factorlasso does *not* inherit from uniLasso, worth flagging
to avoid overclaiming:

* uniLasso's stage-2 architecture (Chatterjee et al. 2025 §2.1) fits a
  non-negative Lasso on leave-one-out fitted values used as new
  features. factorlasso instead constrains coefficients directly on
  the original variables via the CVXPY sign-constraint set — simpler
  in financial-panel sizes where `n` is typically in the hundreds,
  not the hundreds of thousands.
* The hard t-statistic noise floor (`auto_sign_threshold_t`) is not in
  uniLasso. uniLasso's LOO machinery achieves a smoother form of
  noise downweighting via stage-2 regularization on out-of-sample
  predictions. The threshold gate here is conceptually closer to
  **Sure Independence Screening** (Fan & Lv 2008): screen marginal
  evidence first, then regularize.

References:
* Chatterjee, S., Hastie, T., & Tibshirani, R. (2025). Univariate-
  guided sparse regression. *Harvard Data Science Review* 7(3).
* Fan, J., & Lv, J. (2008). Sure independence screening for ultrahigh
  dimensional feature space. *J. R. Stat. Soc. B* 70(5), 849–911.
* Richland, J., Kiiskinen, T., Wang, W., Lu, S., Narasimhan, B.,
  Hastie, T., Rivas, M., & Tibshirani, R. (2025). Univariate-guided
  sparse regression for biobank-scale high-dimensional -omics data.
  arXiv:2511.22049.
* Wang, H., & Leng, C. (2008). A note on adaptive group lasso.
  *Comput. Stat. Data Anal.* 52(12), 5277–5286.
* Zou, H. (2006). The adaptive Lasso and its oracle properties.
  *J. Amer. Stat. Assoc.* 101(476), 1418–1429.

### 3. Prior-centered regularisation

Pass a `(N × M)` DataFrame `factors_beta_prior` to penalise `‖β − β₀‖` instead
of `‖β‖`. The prior is a soft target, not a hard constraint — the penalty
tension between data fit and prior is still controlled by `reg_lambda`.

```python
prior = 0.5 * np.sign(X.corrwith(Y["y0"]).to_numpy())
# ... build an (N, M) DataFrame `prior_df` with that structure ...

model = LassoModel(
    reg_lambda=1e-3,
    factors_beta_prior=prior_df,
).fit(x=X, y=Y)
```

### 4. Hierarchical Clustering Group LASSO (HCGL)

The groups in classical group LASSO are user-specified. HCGL discovers them
from the data: EWMA correlation of the response matrix → Ward's linkage →
dendrogram cut at `cutoff_fraction × max(pdist)` → block-sparse penalty on
the resulting clusters.

```python
model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-4,
    cutoff_fraction=0.5,   # tune granularity; smaller → tighter clusters
    span=60,               # EWMA span for correlation estimate
).fit(x=X, y=Y)

model.coef_        # (N, M)
model.clusters_    # pd.Series of cluster labels per response
model.linkage_     # scipy linkage matrix
```

Useful when you suspect group structure in the responses but don't know the
partition — or when the correct partition drifts over time, so any manual
grouping would need to be refit anyway.

### 5. Sparse Group LASSO

Group LASSO selects whole groups in or out — every response inside an
"active" group gets a non-zero loading. When the discovered groups are
slightly heterogeneous (and HCGL clusters often are, especially at coarser
`cutoff_fraction`), this admits noisy within-group loadings on responses
that don't actually load on the factor.

The `l1_weight` mixing parameter α ∈ [0, 1] adds an elementwise L1 penalty
on top of the group L2 (Simon, Friedman, Hastie & Tibshirani 2013):

$$
\mathcal{P}(\beta) = (1 - \alpha)\,\lambda \sum_g w_g \, \|\beta_g - \beta_0\|_{2,1}
\;+\; \alpha\,\lambda \, \|\beta - \beta_0\|_1
$$

```python
model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-4,
    cutoff_fraction=0.65,   # coarser clusters
    l1_weight=0.10,         # α — group L2 still primary, L1 corrects within-group
).fit(x=X, y=Y)
```

The interpretation is "group-then-prune": the group L2 term still drives
group-level selection, while the L1 term zeros individual asset-factor
coefficients within active groups whose contribution is noise. Setting
`l1_weight=0.0` (the default) reduces exactly to pure group LASSO and is
backward-compatible — the L1 term is dropped from the CVX problem entirely
when α = 0, with zero runtime cost.

Typical research range: α ∈ [0.05, 0.20]. Above ~0.30 the group structure
stops driving the model and the result reverts toward plain LASSO. The
penalty is centered on the same prior `β₀` as the group term, so the two
shrinkage mechanisms compose consistently.

The L1 term respects the same per-element sign constraints and the same
prior as the group term, so all four features in this section compose: a
single fit can simultaneously enforce sign constraints, shrink toward a
prior, group-select via HCGL clusters, and apply within-group elementwise
sparsity.

---

## When to use it — and when not

**Use it when:**

- Multi-output LASSO with heterogeneous sign constraints across the coefficient
  matrix.
- You have a prior `β₀` that should shrink the fit instead of zero.
- You need discovered-group structured sparsity (HCGL).
- You need group-level selection with within-group elementwise sparsity
  (sparse group LASSO at small-to-moderate α).
- You want a small, auditable CVXPY-based tool rather than a coordinate-descent
  library with opaque internals.

**Reach for something else when:**

- Your problem is single-output elastic-net at large scale — `scikit-learn`,
  `celer`, or `skglm` will be faster and have years of battle-testing.
- You need fixed-group group LASSO at very large scale — `group-lasso` or
  `asgl` are the standard tools.
- You need sparse group LASSO at large α (close to 1.0) or at very large
  scale — specialised solvers like `asgl` or `SGL` handle proximal-operator
  acceleration that the CVXPY formulation here does not. This package's
  sparse group LASSO is intended for moderate α ∈ [0, 0.3] where the group
  structure remains primary.
- You need non-linear models, random effects, or GLM link functions.

A feature-by-feature comparison matrix is in
[`COMPARISON.md`](COMPARISON.md).

---

## Examples

Three runnable examples in [`examples/`](examples/):

- [`genomics_factor_model.py`](examples/genomics_factor_model.py) —
  QTL-style multi-response LASSO: genotype matrix → expression panel, with
  sign constraints derived from biological priors.
- [`finance_factor_model.py`](examples/finance_factor_model.py) —
  Multi-asset factor decomposition with sign constraints and HCGL clustering.
- [`cv_lambda_selection.py`](examples/cv_lambda_selection.py) —
  Time-series cross-validated `reg_lambda` selection via `LassoModelCV` with
  expanding-window splits.

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite currently has 201 tests at 98%+ coverage, including numerical parity
tests against `qis` for the EWMA primitives and against `scikit-learn` for the
LASSO path.

---

## Citation

If you use `factorlasso` in academic work, please cite both the
methodology paper documenting the framework in which the package was
developed, and the software itself:

```bibtex
@article{SeppHansenKastenholz2026MATF,
  author  = {Sepp, Artur and Hansen, Emilie and Kastenholz, Mika},
  title   = {Capital Market Assumptions Using Multi-Asset Tradable Factors:
             The {MATF-CMA} Framework},
  journal = {Journal of Portfolio Management},
  year    = {2026},
  note    = {Forthcoming.}
}

@article{SeppOssaKastenholz2026,
  author  = {Sepp, Artur and Ossa, Ivan and Kastenholz, Mika},
  title   = {Robust Optimization of Strategic and Tactical Asset Allocation
             for Multi-Asset Portfolios},
  journal = {The Journal of Portfolio Management},
  year    = {2026},
  volume  = {52},
  number  = {4},
  pages   = {86--120},
}

@software{factorlasso,
  author  = {Sepp, Artur and Kastenholz, Mika},
  title   = {factorlasso: Sparse Multi-Asset Factor Model Estimation with
             Cluster-Pooled Sign Derivation and Hierarchical Group {LASSO}
             in {Python}},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/ArturSepp/factorlasso},
}
```

---

## Contributing & feedback

Issues and pull requests welcome at
<https://github.com/ArturSepp/factorlasso>.

See [`CHANGELOG.md`](CHANGELOG.md) for release history and
[`COMPATIBILITY.md`](COMPATIBILITY.md) for the API stability policy
covering the v0.4.x line.

---

## License

MIT — see [`LICENSE`](LICENSE).
