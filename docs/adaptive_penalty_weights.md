---
myst:
  html_meta:
    description: >-
      Adaptive penalty weights in factorlasso: cell weights from pooled univariate slopes in the
      manner of the adaptive LASSO, their root-mean-square aggregation into group and block
      weights, the floor and exponent, and a worked example with an exact reparameterisation check.
---

# Adaptive penalty weights

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Adaptive penalty weights scale the penalty of each loading by how strong its univariate evidence
is. A cell whose pooled univariate slope is large pays a lighter penalty; a cell with weak
evidence pays a heavier one. The weights reuse the slopes that the
[sign derivation](gated_cluster_pooled_signs.md) computes, and enter the L1 term cell by cell and
the group terms through a root-mean-square aggregation.

## Overview

The LASSO penalises every loading at the same rate, so it shrinks large loadings as much as small
ones and needs a large penalty to remove weak ones. The adaptive LASSO of Zou (2006) weights the
penalty of each coefficient by the inverse of a first-stage estimate, which reduces the bias on
large coefficients and strengthens the selection of small ones; Wang and Leng (2008) carry the
idea to group penalties. factorlasso takes its first-stage estimates from the pooled univariate
slopes of the sign derivation, which are available before the fit and are computed on the same
arrays. The binary gate of the sign derivation decides which cells may be non-zero at all; the
weights decide, continuously, how hard the remaining cells are pulled towards their centre.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $\hat b_{kj}$ | Pooled univariate slope of cell $(k, j)$, `sign_slopes_` | Same units as the loadings |
| $s_{kj}$ | Detected sign; 0 where the gate pinned the cell | `detected_signs_` |
| $\epsilon$ | Floor, `auto_sign_adaptive_floor` | Default $10^{-3}$; 0.5 in the JSS production configuration |
| $\gamma$ | Exponent, `auto_sign_adaptive_gamma` | Default 1 |
| $W_{kj}$ | Cell weight | Dimensionless |
| $\mathcal{J}_k$ | Ungated factors of response $k$ | Cells with $s_{kj} \neq 0$ |

The weights require `auto_sign_constraints=True`, because the slopes come from the sign
derivation, and are computed on each training fold under cross-validation. Loadings are in the
units of the returns, so a floor is a statement about loading magnitudes: a floor of 0.5 means
that slopes below 0.5 all receive the same, largest weight.

## Methodology

### Cell weights

For an ungated cell the weight is

$$
W_{kj} = \frac{1}{\max(\lvert \hat b_{kj} \rvert, \epsilon)^{\gamma}} ,
$$

and a gated cell, whose loading is fixed at zero anyway, receives the placeholder weight 1. The
L1 term of the penalty becomes

$$
\lambda \alpha \sum_{k, j} W_{kj} \lvert \beta_{kj} - \beta_{0,kj} \rvert ,
$$

with $\alpha$ the `l1_weight` of the group models and $\alpha = 1$ for the LASSO. The floor bounds
the weights by $\epsilon^{-\gamma}$; with the production floor of 0.5 and $\gamma = 1$ no weight
exceeds 2, and a slope near 1 leaves the penalty unchanged.

For one response the weighted problem is an ordinary LASSO in disguise. Writing
$\theta_j = W_{kj} \beta_{kj}$ and dividing factor $j$ by $W_{kj}$ turns the weighted penalty into
the plain one, so the adaptive fit equals the plain fit of the rescaled design, divided back by
the weights (Zou, 2006). The worked example uses this identity as its independent check.

### Group and block weights

The group penalties weight whole rows or blocks, so the cell weights are aggregated by a
root-mean-square over the ungated cells. For HCGL each response row $k$ receives

$$
W_k = \left( \frac{1}{\lvert \mathcal{J}_k \rvert} \sum_{j \in \mathcal{J}_k} W_{kj}^2 \right)^{1/2},
$$

which multiplies the row norm $\lVert \beta_{k\cdot} - \beta_{0,k\cdot} \rVert_2$ of that response in
the group term; a row whose cells are all gated keeps $W_k = 1$. FCGL aggregates in the same way
over the ungated members of each cluster-by-factor block. The root-mean-square is the aggregation
that pairs with a Euclidean norm, and it returns exactly 1 when every slope has unit magnitude, so
the cluster scaling $\sqrt{\lvert g \rvert / G}$ of the group penalty is unchanged in that case
(Sepp and Kastenholz, 2026b, Section 2.5).

## Worked example

The canonical script
[`examples/docs/adaptive_penalty_weights.py`](../examples/docs/adaptive_penalty_weights.py) fits
six responses on six independent factors over 60 months. Each response has one loading of 1.0, one
of 0.3 and four zeros. The LASSO with derived signs at the gate $\tau = 1$ is fitted with and
without adaptive weights, with the production floor 0.5 and $\gamma = 1$:

```python
def fit(x: pd.DataFrame, y: pd.DataFrame, adaptive: bool,
        model_type: fl.LassoModelType = fl.LassoModelType.LASSO) -> fl.LassoModel:
    """LASSO with gated derived signs, with or without adaptive penalty weights."""
    model = fl.LassoModel(
        model_type=model_type,
        reg_lambda=REG_LAMBDA,
        auto_sign_constraints=True,
        auto_sign_threshold_t=TAU,
        auto_sign_adaptive_weights=adaptive,
        auto_sign_adaptive_gamma=GAMMA,
        auto_sign_adaptive_floor=FLOOR,
    )
    return model.fit(x=x, y=y)
```

The fitted weights, `sign_penalty_weights_`, lie between 0.94 and 2: cells whose slope is near 1
keep a weight near 1, and cells with a slope below the floor, the small loadings and most null
cells that pass the gate, carry weight 2. The script recomputes every weight from the fitted
slopes and checks the adaptive fit against the plain LASSO on the rescaled design, response by
response:

```python
for k, asset in enumerate(y.columns):
    w = weights.loc[asset].to_numpy()
    rescaled = fl.LassoModel(
        model_type=fl.LassoModelType.LASSO,
        reg_lambda=REG_LAMBDA,
        factors_beta_loading_signs=adaptive.derived_signs_.loc[[asset]],
    ).fit(x=x / w, y=y[[asset]])
    theta = rescaled.coef_.loc[asset].to_numpy()
```

At `reg_lambda` $= 2 \times 10^{-4}$ the plain LASSO keeps 7 non-zero loadings on the 24 true zeros
and the adaptive fit 2. The small loadings are shrunk by 0.15 on average instead of 0.09, and the
large loadings by about 0.07 in both fits. With the production floor, the adaptive weights act
mainly by doubling the penalty on weak cells.

![The adaptive weight as a function of the univariate slope for three exponents and two floors, and the fitted loadings of a plain and an adaptive LASSO grouped by true loading](images/adaptive_penalty_weights.png)

*Synthetic teaching exhibit. Left: $W = 1/\max(\lvert b \rvert, \epsilon)^{\gamma}$ for
$\gamma \in \lbrace 0.5, 1, 2 \rbrace$ at the floor 0.5, and for $\gamma = 1$ at the package
default floor $10^{-3}$. Right: the 36 fitted loadings of six responses, 60 months, LASSO with
derived signs at $\tau = 1$, grouped by true loading; dashed lines mark the truth. Produced by
`tools/docs_analytics/estimation.py` from the example script.*

The script also fits FCGL with adaptive weights and checks every entry of the block weights,
`sign_block_weights_`, against the root-mean-square of the ungated cell weights of its block.

## Implementation in factorlasso

Verified with factorlasso 0.20.0 and CVXPY with the CLARABEL solver.

| Name | Role |
|---|---|
| `auto_sign_adaptive_weights` | Enable the weights. Requires `auto_sign_constraints=True`; default `False`, which leaves every penalty unweighted. |
| `auto_sign_adaptive_gamma` | The exponent $\gamma$, default 1. |
| `auto_sign_adaptive_floor` | The floor $\epsilon$, default $10^{-3}$. |
| `sign_penalty_weights_` | The fitted cell weights $W_{kj}$, response by factor. |
| `sign_block_weights_` | The FCGL block weights, one row per cluster in the column order of `set_group_loadings` and one column per factor; `None` for other models. |

The weights use the slopes of the detection layer, `sign_slopes_`, before explicit signs or prior
centres are applied: a cell whose sign an explicit constraint or a prior overrides keeps the
weight of its detected slope. The weights apply to `LASSO`, `GROUP_LASSO`, HCGL and FCGL; the HCGL
row weights are applied inside the solver and are not stored. To run the example from a checkout:

```console
python examples/docs/adaptive_penalty_weights.py
```

## Interpretation and limitations

- **The first stage is marginal.** The weights come from univariate slopes, not from a
  multivariate first-stage fit as in Zou (2006). With correlated factors a marginal slope can
  overstate or understate a cell's conditional importance, and the weight inherits the error.
- **The floor sets the range.** With the default floor $10^{-3}$ a near-zero slope receives a
  weight of up to 1000 and is effectively removed; with the production floor 0.5 the weights stay
  within a factor of two. Neither the floor nor $\gamma$ is selected by the package.
- **Oracle properties are not inherited.** The adaptive LASSO's selection consistency relies on
  a consistent first-stage estimate and conditions on the penalty sequence; the pooled marginal
  slopes and a fixed penalty do not meet them, so the weights are a heuristic reweighting here.
- **Weights and signs are coupled.** Both come from the same slopes, so a cluster that pools
  opposite exposures produces a weak slope, a zero sign or a heavy weight for all its members.

## See also

- [Gated cluster-pooled sign derivation](gated_cluster_pooled_signs.md): the slopes and the gate
  the weights reuse.
- [Group penalties: HCGL, sparse-group and FCGL](group_penalties_hcgl_fcgl.md): the row and block
  norms the aggregated weights multiply.
- [Sparse multi-output factor model](sparse_factor_model.md): the unweighted LASSO.
- [Credit attribution case study](app_multi_asset_credit_attribution.md): the production
  configuration with floor 0.5.

## References

- Sepp, A., and Kastenholz, M. (2026a). Gated Cluster-Pooled Sign Constraints for Multi-Output
  Sparse Regression. Submitted to *Computational Statistics & Data Analysis*.
  [Manuscript](../papers/sign_pooling_2026/paper/article.pdf).
- Sepp, A., and Kastenholz, M. A. (2026b). factorlasso: Sparse Multi-Output Regression with
  Cluster-Grouped Sign Constraints in Python. Submitted to the *Journal of Statistical
  Software*. [Manuscript](../papers/jss_2026/paper/article.pdf).
- Wang, H., and Leng, C. (2008). A note on adaptive group lasso. *Computational Statistics & Data
  Analysis* 52(12), 5277-5286. DOI 10.1016/j.csda.2008.05.006.
- Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the American
  Statistical Association* 101(476), 1418-1429. DOI 10.1198/016214506000000735.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
