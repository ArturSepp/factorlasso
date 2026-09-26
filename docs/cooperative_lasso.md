---
myst:
  html_meta:
    description: >-
      The cooperative LASSO in factorlasso: a group penalty on the positive and negative parts of
      each cluster-by-factor block, which favours sign-coherent clusters without imposing signs,
      with its closed form under an orthonormal design and a comparison with hard pooled signs.
---

# Cooperative LASSO

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

The cooperative LASSO is a group penalty that treats the positive and the negative loadings of a
group separately. A block whose loadings share one sign pays the ordinary group norm; a block
with mixed signs pays more. Clusters of responses are thereby encouraged, but not forced, to
agree on the sign of each exposure: the data may keep a member on the other side if the evidence
is strong enough.

## Overview

The [gated sign derivation](gated_cluster_pooled_signs.md) turns cluster-level evidence into hard
signs: a member whose true exposure opposes its cluster is pinned at zero. The FCGL block penalty
of the [group penalties](group_penalties_hcgl_fcgl.md) article ignores signs altogether. The
cooperative LASSO of Chiquet, Grandvalet and Charbonnier (2012) sits between the two. It keeps
the cluster-by-factor blocks of FCGL, splits each block into its positive and negative parts,
and applies the group norm to each part, so sign coherence is a preference of the penalty and not
a constraint.

factorlasso offers it on supplied groups, `COOPERATIVE_GROUP_LASSO`, and on clusters discovered
from the responses, `COOPERATIVE_CLUSTER_GROUP_LASSO`. The sign-pooling paper compares it with the
hard pooled sign when predictors are correlated (Sepp and Kastenholz, 2026a, Section 4.5); the
software paper lists it among the modes of one estimator (Sepp and Kastenholz, 2026b,
Section 3.1).

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $\beta_{g,j}$ | Loadings of the members of cluster $g$ on factor $j$, a block | Vector of length $\lvert g \rvert$ |
| $\beta_0$ | Prior centres, `factors_beta_prior` | Zero by default |
| $(v)_{+}$, $(v)_{-}$ | Elementwise positive and negative parts, $v = (v)_{+} - (v)_{-}$ | Both non-negative |
| $w_g$ | Cluster weight: $\sqrt{\lvert g \rvert / G}$ or $\sqrt{\lvert g \rvert}$ | `group_penalty` |
| $\alpha$ | Weight of an additional cell-wise L1 term, `l1_weight` | Default 0, the pure cooperative LASSO |

The loss, weights, masks and conventions are those of the other estimators; see the
[conventions page](conventions.md).

## Methodology

### The penalty

The cooperative LASSO minimises

$$
L(\beta) + (1 - \alpha) \lambda \sum_g w_g \sum_{j=1}^{M} \left( \lVert (\beta_{g,j} - \beta_{0,g,j})_{+} \rVert_2 + \lVert (\beta_{g,j} - \beta_{0,g,j})_{-} \rVert_2 \right) + \alpha \lambda \lVert \beta - \beta_0 \rVert_1 ,
$$

where $L$ is the weighted squared loss. For a block $v$ whose entries share one sign, one of the
two parts is zero and the penalty is $\lVert v \rVert_2$, the group LASSO norm of Yuan and Lin
(2006). For a mixed block it is $\lVert v_{+} \rVert_2 + \lVert v_{-} \rVert_2$, which exceeds $\lVert v \rVert_2$ and is at
most $\sqrt{2} \lVert v \rVert_2$, the maximum reached when the two parts have equal norm. The
block $(0.6, 0.8)$ pays 1.0, while $(0.6, -0.8)$ pays 1.4.

No sign is imposed and no gate is applied. The solver writes
$\beta - \beta_0 = P - N$ with $P, N \ge 0$ and penalises the group norms of $P$ and $N$, which is a
convex programme that CVXPY accepts; at the optimum $P$ and $N$ are the positive and negative
parts.

### An orthonormal design

When the factors are orthonormal, $X^{\top} X / T = I$, the squared loss of each response equals
$\lVert \beta_{k\cdot} - z_{k\cdot} \rVert_2^2$ up to a constant, where $z = Y^{\top} X / T$ holds
the least-squares loadings. The blocks then decouple, and each has a closed form with threshold
$c = \lambda w_g$. The group LASSO shrinks the whole block,
$\beta_{g,j} = z_{g,j} \max(0, 1 - c / (2 \lVert z_{g,j} \rVert_2))$. The cooperative LASSO applies
the same group soft threshold separately to the cells with a positive and with a negative
least-squares loading. A member that disagrees with the rest of its cluster forms a small sign
part of its own and is shrunk towards zero much faster than it would be inside the whole block,
while the coherent members lose almost nothing.

## Worked example

The canonical script [`examples/docs/cooperative_lasso.py`](../examples/docs/cooperative_lasso.py)
builds two clusters of four responses on two exactly orthonormal factors over 120 dates. In the
first cluster three members load positively on the first factor and one member, the rogue, has
a true loading of $-0.3$. Three penalties are fitted at the same `reg_lambda`:

```python
def fit_three(x: pd.DataFrame, y: pd.DataFrame, reg_lambda: float = REG_LAMBDA) -> dict:
    """Group LASSO blocks, cooperative LASSO blocks, and blocks with a hard pooled sign."""
    x_np, y_np = x.to_numpy(), y.to_numpy()
    groups = fl.set_group_loadings(group_data=CLUSTERS).to_numpy()
    group = fl.solve_group_lasso_cvx_problem(
        x=x_np, y=y_np, group_loadings=groups, reg_lambda=reg_lambda,
        block_mode="cluster_factor",
    )
    cooperative = fl.solve_cooperative_group_lasso_cvx_problem(
        x=x_np, y=y_np, group_loadings=groups, reg_lambda=reg_lambda,
    )
    hard = fl.LassoModel(
        model_type=fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        reg_lambda=reg_lambda,
        auto_sign_constraints=True,
        auto_sign_threshold_t=None,
    ).fit(x=x, y=y, external_clusters=CLUSTERS)
```

At `reg_lambda` $= 0.3$ the rogue's loading is $-0.35$ by least squares, $-0.30$ under the group
LASSO, $-0.14$ under the cooperative LASSO and exactly zero under the hard pooled sign. The first
coherent member moves from 0.648 under the group LASSO to 0.644 under the cooperative LASSO: its
positive part no longer includes the rogue, so its own norm is slightly smaller. The script checks
every block of both penalties against the closed forms above, to $10^{-4}$.

![Unit level sets of the group and cooperative penalties for two loadings, and the loading of a cluster's rogue member along the penalty grid under three penalties](images/cooperative_lasso_geometry.png)

*Synthetic teaching exhibit. Left: loadings of a two-member block with penalty 1; the cooperative
level set coincides with the circle where the signs agree and cuts across it where they differ.
Right: the rogue member's loading for 16 penalties, from 0.8 on the left to 0.05 on the right, on
two orthonormal factors and 120 dates; the dashed line is the least-squares loading. Produced by `tools/docs_analytics/estimation.py` from the example script.*

## Implementation in factorlasso

Verified with factorlasso 0.20.0 and CVXPY with the CLARABEL solver.

| Name | Role |
|---|---|
| `LassoModelType.COOPERATIVE_GROUP_LASSO` | Cooperative blocks on the groups of `group_data`. |
| `LassoModelType.COOPERATIVE_CLUSTER_GROUP_LASSO` | Cooperative blocks on clusters discovered from the responses, or passed as `external_clusters` to `fit`. |
| `solve_cooperative_group_lasso_cvx_problem` | The CVXPY programme for NumPy inputs with a group-membership matrix; it accepts `factors_beta_prior`, `group_penalty`, `l1_weight`, per-block `col_weights`, `valid_mask`, `span` and `loss_normalization`. |

The cooperative modes take no sign constraint. `LassoModel` rejects `factors_beta_loading_signs`
and `nonneg=True` for them with `ValueError`; with `auto_sign_constraints=True` they still fit, and
`derived_signs_` is `None` because derived signs would not reach the solver. `group_penalty` and
`l1_weight` have the meaning described in the [group penalties](group_penalties_hcgl_fcgl.md)
article. To run the example from a checkout:

```console
python examples/docs/cooperative_lasso.py
```

## Interpretation and limitations

- **Soft coherence can be overruled, in both directions.** A strongly supported opposite
  exposure survives, which is the purpose; a weakly supported correct one inside a mixed block is
  shrunk as hard as noise.
- **It asserts signs where the hard constraint abstains.** In the sign-pooling paper's designs
  with correlated predictors, the cooperative LASSO assigned a wrong sign to 0.101 to 0.220 of the
  active cells while the hard pooled constraint kept its flip rate at 0.006 or below and abstained
  instead. The cooperative LASSO recovered more signs and reached a lower coefficient error
  throughout (Sepp and Kastenholz, 2026a, Section 4.5). Which kind of error is worse depends on
  the use of the loadings.
- **The clusters matter as much as the penalty.** Coherence is encouraged within the blocks the
  clusters define; a cluster that mixes different exposures is penalised for being mixed.
- **The penalty is not scale-free.** Like every group penalty, it depends on the units of the
  loadings and on the cluster weights $w_g$.

## See also

- [Group penalties: HCGL, sparse-group and FCGL](group_penalties_hcgl_fcgl.md): the sign-blind
  block penalty.
- [Gated cluster-pooled sign derivation](gated_cluster_pooled_signs.md): the hard alternative.
- [UniLasso](unilasso.md): the univariate-guided mode, which keeps univariate signs through
  non-negative stage-two coefficients.
- [Conventions and glossary](conventions.md).

## References

- Chiquet, J., Grandvalet, Y., and Charbonnier, C. (2012). Sparsity with sign-coherent groups of
  variables via the cooperative-Lasso. *The Annals of Applied Statistics* 6(2), 795-830.
  DOI 10.1214/11-AOAS520.
- Sepp, A., and Kastenholz, M. (2026a). Gated Cluster-Pooled Sign Constraints for Multi-Output
  Sparse Regression. Submitted to *Computational Statistics & Data Analysis*.
  [Manuscript](../papers/sign_pooling_2026/paper/article.pdf).
- Sepp, A., and Kastenholz, M. A. (2026b). factorlasso: Sparse Multi-Output Regression with
  Cluster-Grouped Sign Constraints in Python. Submitted to the *Journal of Statistical
  Software*. [Manuscript](../papers/jss_2026/paper/article.pdf).
- Yuan, M., and Lin, Y. (2006). Model selection and estimation in regression with grouped
  variables. *Journal of the Royal Statistical Society: Series B* 68(1), 49-67.
  DOI 10.1111/j.1467-9868.2005.00532.x.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
