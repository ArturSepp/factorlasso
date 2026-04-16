# `factorlasso` vs. the Python sparse-regression ecosystem

This document answers the question *"why not just use scikit-learn / skglm / groupyr?"*. The short answer: each of those packages is excellent for its niche, but none of them covers the combination of **element-wise sign constraints + informative priors + data-driven groups + ragged histories + joint covariance assembly** in a single estimator. If your problem needs any two of those features together, `factorlasso` is the only drop-in option in Python today.

All information in the tables below was verified by inspecting each package's public API (`sklearn==1.7.x`, `skglm==0.5`, `groupyr==0.3.2`, `group-lasso==1.5.0`, `asgl` current release) as of November 2026. Corrections welcome via [GitHub issues](https://github.com/ArturSepp/factorlasso/issues).

---

## **Feature matrix**

| Feature | `scikit-learn` | `skglm` | `celer` | `groupyr` | `group-lasso` | `asgl` | **`factorlasso`** |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Standard L1 (Lasso) | ✓ | ✓ | ✓ | ✓ (α=1) | — | ✓ | ✓ |
| Group LASSO (L2,1) | ✓ (MultiTask only) | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Sparse Group Lasso | — | — | — | ✓ | ✓ | ✓ | — |
| **Data-driven groups (HCGL)** | — | — | — | — | — | — | **✓** |
| **Element-wise sign constraints** | blanket only¹ | blanket only¹ | blanket only¹ | — | — | — | **✓** |
| **Prior-centered penalty** ‖β − β₀‖ | — | — | — | — | — | — | **✓** |
| Multi-output regression $(N > 1)$ | MultiTask only² | — | — | — | — | — | **✓** |
| **Ragged histories (per-response NaN mask)** | — | — | — | — | — | — | **✓** |
| **Time-weighted objective (EWMA)** | — | sample_weight³ | sample_weight³ | — | — | — | **✓** |
| **Joint covariance assembly $\Sigma_Y = \beta \Sigma_X \beta^\top + D$** | — | — | — | — | — | — | **✓** |
| sklearn `fit`/`predict`/`score` API | ✓ | ✓ | ✓ | ✓ | ✓ | partial | ✓ |
| Time-series CV estimator built-in | — | — | — | — | — | — | ✓ |

¹ "Blanket only": a single `positive=True` flag constrains *all* coefficients to be non-negative. None of these packages accept an element-wise sign matrix where individual coefficients can be non-negative, non-positive, fixed at zero, or free.

² `sklearn.linear_model.MultiTaskLasso` couples responses via an L2,1 penalty that forces *all* responses to select the same features — not what you want when response-specific sparsity is expected.

³ Generic per-observation weights are supported but not the EWMA $\lambda = 1 - 2/(\mathrm{span} + 1)$ convention with matched covariance assembly that `factorlasso` uses throughout.

---

## **Technical characteristics**

| | `scikit-learn` | `skglm` | `celer` | `groupyr` | `group-lasso` | `asgl` | **`factorlasso`** |
|--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Solver backend | Coordinate descent | Anderson-accelerated CD (Numba) | Anderson-accelerated CD | `copt` (PGD) | FISTA | CVXPY | CVXPY |
| Max feature count (benchmark regime) | $10^4$–$10^5$ | $10^6$+ | $10^6$+ | $10^5$ | $10^5$ | $10^4$ | $10^3$–$10^4$ |
| Relative speed (L1, single-output) | 1× (baseline) | 10–100×⁴ | 10–100×⁴ | ~1× | ~1× | slower | slower |
| Install deps footprint | standard | +`numba`, +`numpy<2` | +`numpy`, +`Cython` | +`copt`, +`scikit-optimize` | +`scikit-learn` | +`cvxpy` | +`cvxpy` |
| License | BSD-3 | BSD-3 | BSD-3 | BSD-3 | MIT | GPL-3 | **MIT** |
| Publication venue | JMLR 2011 | JMLR 2025 (MLOSS) | NeurIPS 2018 | JOSS 2021 | — | — | (in progress) |
| Active maintenance (2026) | very high | high | moderate | low | low | moderate | active |

⁴ Relative speed claims are quoted from the respective packages' publications; `factorlasso`'s CVXPY backend is slower on problems all packages can solve, which is a deliberate trade-off for expressiveness.

---

## **When to use each**

### Reach for `scikit-learn` when you need
- Standard L1 / L2 / Elastic Net on tabular data
- Maximum ecosystem integration (pipelines, grid search, everything interops)
- Blanket non-negativity (`positive=True`) is enough

### Reach for `skglm` when you need
- Very high-dimensional data ($p \gg n$, sparse design matrices)
- State-of-the-art speed on a modular penalty $\times$ datafit combination
- Non-convex penalties (MCP, SCAD, L0.5, L2/3)
- Survival analysis (Cox), robust regression (Huber, Quantile)

### Reach for `celer` when you need
- Vanilla Lasso / Elastic Net at the largest scales with rigorous screening rules

### Reach for `groupyr` when you need
- Sparse group lasso on pre-specified groups with a clean `alpha` / `l1_ratio` API
- Bayesian (SMBO) hyperparameter tuning

### Reach for `group-lasso` (Moe) when you need
- A minimal, pure-Python sparse group lasso with FISTA solver

### Reach for `asgl` when you need
- Adaptive sparse group lasso (weighted penalties) in quantile regression

### Reach for `factorlasso` when you need
- **Any combination of** element-wise sign constraints, informative priors, data-driven groups, multi-output regression, ragged histories, and a consistent $\Sigma_Y = \beta \Sigma_X \beta^\top + D$ assembled from the *same* $\beta$
- A drop-in scikit-learn API for problems specialized solvers can't express
- EWMA observation weighting matched to EWMA covariance estimation (same $\lambda$ convention)
- Portfolio construction, genomics, macro-econometrics, or any domain where domain knowledge encodes as sign priors or shrinkage targets

---

## **Complementarity, not competition**

`factorlasso` is deliberately positioned *downstream* of the fast specialized solvers. For the problem *"solve a vanilla Lasso on a sparse million-feature matrix"*, use `skglm` or `celer`. For the problem *"solve a multi-output penalized regression where half the coefficients have sign priors, a third have informative priors, responses have ragged histories, and I need the implied covariance matrix to be mutually consistent with the point estimates"*, no fast specialized solver exists — the CVXPY backend is the practical trade-off for expressing the full constraint set.

A natural workflow is **`skglm` for feature pre-screening** (reduce $M$ from thousands to tens) **→ `factorlasso` for the constrained final estimation** on the pre-screened features. Both packages compose cleanly with scikit-learn pipelines.

---

## **Benchmark: feature-parity sanity check**

On problems all packages can solve (plain Lasso and plain Group Lasso without constraints or priors), `factorlasso` produces coefficients within solver tolerance (`atol=1e-4`) of `scikit-learn` and `skglm` reference implementations. The script `benchmarks/feature_parity.py` runs the comparison end-to-end and prints a side-by-side table. This is a correctness check, not a speed benchmark — for speed, defer to the `benchopt` ecosystem.

---

## **Corrections and updates**

Packages change. If any cell in the feature matrix is out of date — a feature added, a maintainer returned, a benchmark improved — please open a PR or issue. The goal of this document is to give honest, current guidance on where `factorlasso` fits in the ecosystem, not to claim permanent advantage.
