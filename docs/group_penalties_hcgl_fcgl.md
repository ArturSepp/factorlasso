---
myst:
  html_meta:
    description: >-
      Group penalties in factorlasso: the row-grouped HCGL penalty, the cluster-by-factor FCGL
      penalty, the sparse-group mixture and the two group-weight conventions, with a worked
      example showing which loadings each penalty removes and how to refit a selected support.
---

# Group penalties: HCGL, sparse-group and FCGL

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

A group penalty replaces the sum of absolute loadings by a sum of Euclidean norms over groups of
loadings, so that a whole group enters or leaves the model together. The package offers two
geometries on a partition of the responses into clusters: the norm of each response's loading
row (HCGL), and the norm of each cluster-by-factor block (FCGL). The partition is supplied by
the user or discovered from the correlation matrix of the responses.

## Overview

The cell-wise LASSO of the [sparse factor model](sparse_factor_model.md) decides every loading
on its own evidence. Responses of one kind, such as the funds of one asset class, usually load
on the same factors with different magnitudes. A cell-wise penalty cannot use that: for each
fund separately it keeps the large loadings and drops the small ones, so funds of one class end
up with different factor sets.

Group penalties (Yuan and Lin 2006) pool the evidence. Which loadings are pooled is a choice of
geometry, and the two geometries in the package behave very differently:

| `model_type` | Group of the norm | What is selected | Coupling across responses |
|---|---|---|---|
| `LASSO` | One cell | Single loadings | None |
| `GROUP_LASSO`, `HIERARCHICAL_CLUSTER_GROUP_LASSO` (HCGL) | One response row, all $M$ factors | Whole responses | None: the cluster enters through a weight only |
| The same with `l1_weight` $> 0$ (sparse-group) | Row norm plus cell-wise L1 | Responses, then cells | None |
| `FACTOR_CLUSTER_GROUP_LASSO` (FCGL) | One cluster-by-factor block | A factor for a whole cluster | Responses of a cluster are fitted jointly |

`GROUP_LASSO` takes the partition from `group_data`. HCGL and FCGL discover it: EWMA correlation
of the responses, a correlation-to-distance transform, Ward linkage, and a cut of the dendrogram
at `cutoff_fraction` times the largest pairwise distance, or into `n_clusters` groups.

In the worked example, twelve funds in three clusters and 48 months, FCGL recovers the
generating support exactly, 24 of 72 cells with no false and no missed loading, at a penalty
where the LASSO misses five loadings and the row penalty keeps 71.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $\beta$, `coef_` | Loading matrix | $N \times M$; $\beta_i$ is the row of response $i$ |
| $g$, $G$ | A cluster of responses and the number of clusters | A partition of the $N$ responses; $\lvert g \rvert$ is the cluster size |
| $\beta_{g,j}$ | Block of cluster $g$ and factor $j$ | The $\lvert g \rvert$ loadings of the cluster's responses on factor $j$ |
| $\beta_0$ | Prior loading matrix | Defaults to zero; see [sign constraints and priors](sign_constraints_and_priors.md) |
| $\lambda$, `reg_lambda` | Penalty strength | Units of `y` squared |
| $a$, `l1_weight` | Share of the cell-wise L1 term | In $[0, 1]$; default 0 |
| $w_g$, `group_penalty` | Group weight | `"normalized"` (default): $\sqrt{\lvert g \rvert / G}$; `"yuan_lin"`: $\sqrt{\lvert g \rvert}$ |
| `group_data` | User partition for `GROUP_LASSO` | Series indexed by response with group labels |
| `cutoff_fraction`, `n_clusters` | Dendrogram cut for the cluster modes | Default `cutoff_fraction=0.5`; `n_clusters` overrides it |
| `clusters_`, `linkage_`, `cutoff_` | Fitted partition, SciPy linkage matrix and cut distance | Set by the cluster modes |

The methods assume that the responses of a cluster share a factor support. Nothing requires
them to share magnitudes. The partition is treated as known once discovered: its sampling error
is not propagated into the loadings.

## Methodology

### Row-grouped penalty: HCGL

With the loss $L(\beta) = \frac{1}{T} \lVert W \odot (\tilde X \beta^{\top} - \tilde Y) \rVert_F^2$
of the sparse factor model, HCGL and `GROUP_LASSO` solve

$$
\hat\beta = \arg\min_{\beta \in \mathcal{C}}
L(\beta) + (1 - a) \lambda \sum_{g} w_g \sum_{i \in g} \lVert \beta_i - \beta_{0,i} \rVert_2 +
a \lambda \lVert \beta - \beta_0 \rVert_1 .
$$

The Euclidean norm of a row is not differentiable only where the whole row equals its prior. The
penalty can therefore set a whole row to its prior, but it never sets a single loading of a kept
row to zero: a kept row is dense. Within a kept row the penalty acts like a ridge term along the
row and shrinks all $M$ loadings together.

The programme is a sum over responses. The cluster of response $i$ enters only through the
scalar $w_g$, so the partition changes the fit of a response only through the size of its
cluster. With clusters of equal size the HCGL loadings do not depend on the partition at all,
which the worked example verifies. The clusters matter in HCGL through two other channels: the
cluster-pooled derivation of sign constraints under `auto_sign_constraints=True`, and the
weights themselves when cluster sizes differ.

### Sparse-group mixture

With $0 < a \le 1$ the cell-wise L1 term of Simon, Friedman, Hastie and Tibshirani (2013) is
added and the row norm is scaled by $1 - a$. The L1 term restores cell-level zeros inside a kept
row. At $a = 1$ the group term vanishes and the fit is the LASSO at the same $\lambda$.

### Cluster-by-factor penalty: FCGL

FCGL keeps the loss and the L1 term and changes the group of the norm:

$$
\hat\beta = \arg\min_{\beta \in \mathcal{C}}
L(\beta) + (1 - a) \lambda \sum_{g} w_g \sum_{j=1}^{M} \lVert \beta_{g,j} - \beta_{0,g,j} \rVert_2 +
a \lambda \lVert \beta - \beta_0 \rVert_1 .
$$

The norm is now taken down a column of the loading matrix, over the responses of one cluster. A
block $\beta_{g,j}$ is zero for all responses of the cluster or for none: the factor is
selected for the cluster. Within a kept block the magnitudes are free, shrunk together like a
ridge term along the block. A response with a small true loading keeps it because its cluster
peers carry the evidence; this is what the cell-wise LASSO cannot do.

The programme couples the responses of a cluster and is solved as one cone programme. The
partition now determines the result, so a wrong partition gives wrong blocks.

### Group weights

A norm over more cells is larger, and $w_g$ compensates for cluster size. `"yuan_lin"` is the
classical $\sqrt{\lvert g \rvert}$. `"normalized"` divides by the number of clusters,
$\sqrt{\lvert g \rvert / G}$, so that a penalty keeps a comparable strength when the discovered
number of clusters changes between estimation dates. The two differ by the constant
$\sqrt{G}$:

$$
\hat\beta^{\text{yuan-lin}}(\lambda) = \hat\beta^{\text{normalized}}(\lambda \sqrt{G}) .
$$

### Penalty scales are not comparable across geometries

A value of `reg_lambda` does not mean the same strength under the four penalties: a row norm, a
block norm and a sum of absolute values of the same loadings differ. Select the penalty
separately for each `model_type`.

## Worked example

The example uses synthetic data with a fixed seed. Twelve monthly return series form three
clusters of four: equity funds load on equity and value, bond funds on rates and credit,
real-asset funds on commodity and equity. Two of six factors are relevant for each cluster, so
24 of the 72 cells of the generating matrix are non-zero, with loadings from 0.2 to 1.2. The six
factors are uncorrelated with a volatility of 0.04 per month; idiosyncratic volatility is 0.025
per month and the history is 48 months, all in decimal units. No sign constraint and no prior
is used, so the comparison isolates the penalty.

```python
MODEL_TYPES = {
    "LASSO": (fl.LassoModelType.LASSO, 0.0),
    "HCGL": (fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, 0.0),
    "sparse HCGL": (fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO, L1_WEIGHT),
    "FCGL": (fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO, 0.0),
}
```

```python
def fit(
    x: pd.DataFrame,
    y: pd.DataFrame,
    name: str,
    reg_lambda: float,
    group_penalty: str = "normalized",
) -> fl.LassoModel:
    """Fit one of the four penalties; the cluster modes discover the partition from ``y``."""
    model_type, l1_weight = MODEL_TYPES[name]
    model = fl.LassoModel(
        model_type=model_type,
        reg_lambda=reg_lambda,
        l1_weight=l1_weight,               # 0 is the pure group penalty, 1 the pure LASSO
        group_penalty=group_penalty,       # w_g = sqrt(|g| / G); "yuan_lin" gives sqrt(|g|)
    )
    return model.fit(x=x, y=y)
```

Both cluster modes discover three clusters that equal the generating groups, with the default
`cutoff_fraction=0.5`.

### What each penalty keeps

[![Four heatmaps of a twelve by six loading matrix with the three clusters separated by lines. The true matrix has two non-zero columns per cluster. The LASSO keeps 19 cells and misses five small ones. HCGL keeps 71 of 72 cells. FCGL keeps exactly the 24 true cells.](images/group_penalties_selection.png)](images/group_penalties_selection.png)

**Figure 1.** Synthetic teaching exhibit. Loadings of twelve funds (rows, three clusters of four
separated by lines) on six factors (columns) at one common `reg_lambda` $= 10^{-3}$; a dot marks
every loading above $10^{-3}$ in magnitude and the colour gives its value. From the left: the
generating matrix; the LASSO, which keeps the large loadings and misses five small ones; HCGL,
whose row penalty keeps 71 of 72 cells; FCGL, which keeps exactly the generating support. The
penalty scales of the three fits are not comparable; the tables below give each penalty at its
own best point. Select the image for the full-resolution view.

| At `reg_lambda` $= 10^{-3}$ | Kept | False | Missed | Loading RMSE |
|---|---|---|---|---|
| LASSO | 19 | 0 | 5 | 0.173 |
| HCGL | 71 | 47 | 0 | 0.149 |
| Sparse HCGL, `l1_weight` $= 0.3$ | 40 | 17 | 1 | 0.148 |
| FCGL | 24 | 0 | 0 | 0.112 |

The five loadings the LASSO misses are true loadings of 0.2 to 0.5: the value loadings of three
equity funds and the equity loadings of two real-asset funds. Taken one cell at a time they
are not distinguishable from noise on 48 months. FCGL keeps them because the block norm pools
the four funds of the cluster.

Along a grid of thirteen penalties from $10^{-2}$ to $10^{-5}$, each penalty at its own best
point:

| Penalty | Fewest support errors on the grid (false plus missed) | Smallest loading RMSE on the grid | Kept at the smallest RMSE |
|---|---|---|---|
| LASSO | 5 at $10^{-3}$ | 0.089 at $1.8 \times 10^{-4}$ | 52 |
| HCGL | 24 at $5.6 \times 10^{-3}$, the empty model | 0.105 at $1.8 \times 10^{-4}$ | 72 |
| Sparse HCGL | 12 at $3.2 \times 10^{-3}$ | 0.097 at $3.2 \times 10^{-4}$ | 57 |
| FCGL | 0 at $10^{-3}$ | 0.078 at $3.2 \times 10^{-4}$ | 61 |
| Ordinary least squares | 48 | 0.109 | 72 |

Two results stand out. The row penalty has no support to recover: its smallest loading error,
0.105, is barely below the 0.109 of least squares, and its fewest support errors occur at the
empty model.
HCGL alone is a shrinkage device; the cell-level sparsity of an HCGL model comes from `l1_weight`
or from the gate of the derived sign constraints. And for every penalty, the point of smallest
loading error keeps many false loadings, because shrinkage bias on the true loadings costs more
than small false ones: the same tension between prediction and support as in the
[sparse factor model](sparse_factor_model.md).

### Select with FCGL, then refit

The tension is resolved in two stages: select the support at the penalty that recovers it, then
re-estimate the kept cells without shrinkage. A sign matrix with zeros on the dropped cells and
NaN elsewhere does the second stage with no new machinery:

```python
def refit_on_support(x: pd.DataFrame, y: pd.DataFrame, selected: fl.LassoModel) -> fl.LassoModel:
    """Re-estimate without shrinkage the cells ``selected`` kept; its zeros become constraints."""
    kept = np.abs(selected.coef_.to_numpy()) > ZERO_TOLERANCE
    support = pd.DataFrame(np.where(kept, np.nan, 0.0), index=y.columns, columns=x.columns)
    return fl.LassoModel(reg_lambda=1e-12, factors_beta_loading_signs=support).fit(x=x, y=y)
```

The refit on the FCGL support has a loading error of 0.060, against 0.112 for the FCGL fit that
selected it, 0.078 for the best FCGL fit on the grid, and 0.109 for least squares on all cells.
This is the relaxed LASSO of Meinshausen (2007) with a group selector in the first stage.

### Checks

| Check | Result |
|---|---|
| Discovered clusters of HCGL and FCGL against the generating groups | identical partition |
| `LassoModel` in HCGL mode against `solve_group_lasso_cvx_problem` with `block_mode="row"` on `set_group_loadings(clusters_)` | equal within $10^{-7}$ |
| `LassoModel` in FCGL mode against the same solver with `block_mode="cluster_factor"` | equal within $10^{-7}$ |
| FCGL with `"yuan_lin"` at $\lambda / \sqrt{3}$ against `"normalized"` at $\lambda$ | equal within $10^{-4}$ |
| FCGL blocks at $10^{-3}$ | each of the 18 blocks has 0 or 4 kept cells, and the kept blocks are the generating ones |
| HCGL rows at $10^{-2.5}$ | three rows are entirely zero; every other row keeps at least five of six cells |
| HCGL under a scrambled partition with the same cluster sizes | loadings equal within $10^{-6}$ |
| FCGL under the same scrambled partition | loadings differ by more than 0.05 |
| HCGL with `l1_weight=1` against the LASSO at the same penalty | equal within $10^{-3}$ |

A kept HCGL row can show five kept cells instead of six when one loading falls below the
counting tolerance by chance; the penalty did not select it out.

## Implementation in factorlasso

All names below are exported from the top-level package and documented in the
[API reference](api.rst).

| Public name | Role |
|---|---|
| `solve_group_lasso_cvx_problem` | The CVXPY programme for NumPy inputs. `group_loadings` is an $N \times G$ indicator matrix; `block_mode` is `"row"` (HCGL, `GROUP_LASSO`) or `"cluster_factor"` (FCGL); `group_penalty`, `l1_weight`, sign and prior matrices as in the estimator; `penalty_weights`, `row_weights` and `col_weights` carry adaptive weights, for rows in the sense of Wang and Leng (2008). |
| `set_group_loadings` | Converts a Series of group labels indexed by response into the indicator matrix the solver takes. |

The estimator-level arguments of `LassoModel` are `model_type`, `group_data`, `group_penalty`,
`l1_weight`, `cutoff_fraction`, `n_clusters`, `linkage_method`, `distance_transform` and
`dependence_measure`. `fit` also accepts `external_clusters`, a Series that replaces cluster
discovery in the HCGL and FCGL modes. The low-level path reproduces the estimator:

<!-- fragment -->
```python
import factorlasso as fl

model = fl.LassoModel(
    model_type=fl.LassoModelType.FACTOR_CLUSTER_GROUP_LASSO, reg_lambda=1e-3,
).fit(x=x, y=y)

x_np, y_np, valid_mask = fl.get_x_y_np(x=x, y=y, span=None)
group_loadings = fl.set_group_loadings(group_data=model.clusters_)
result = fl.solve_group_lasso_cvx_problem(
    x=x_np, y=y_np, group_loadings=group_loadings.to_numpy(), valid_mask=valid_mask,
    reg_lambda=1e-3, block_mode="cluster_factor",
)
result.estimated_beta          # equals model.coef_
```

This fragment assumes factor and response panels `x` and `y`. The runnable version is the
canonical script
[examples/docs/group_penalties_hcgl_fcgl.py](../examples/docs/group_penalties_hcgl_fcgl.py),
which needs only the core dependencies, runs offline, and asserts every number in the tables
above:

```console
python examples/docs/group_penalties_hcgl_fcgl.py
```

`LassoModel` raises `ValueError` at construction when `GROUP_LASSO` has no `group_data`, when
`group_penalty` is not one of the two names, and when `l1_weight` lies outside $[0, 1]$. For a
grid of penalties, `solve_group_lasso_path` and `LassoModel.fit_reg_lambda_path` build the
programme once and reuse it, which the [quickstart](quickstart.md) uses through
`LassoModelCV(use_lambda_path=True)`.

The numbers in this article were produced with factorlasso 0.20.0.dev2, CVXPY 1.9 and CLARABEL on
Python 3.12. Figure 1 is regenerated from the same script by the documentation analytics runner
described in the [documentation standard](documentation_standard.md); its
[provenance record](images/analytics_manifest.json) holds the configuration, the source identity
and the hash of the image.

## Interpretation and limitations

- **HCGL does not select factors.** Its row norm selects responses and shrinks kept rows as a
  whole. Use `l1_weight`, the sign gate of `auto_sign_constraints`, or FCGL when the factor set
  of a response is the question.
- **FCGL depends on the partition.** A response placed in the wrong cluster receives that
  cluster's factors. Inspect `clusters_` and prefer `n_clusters` or `external_clusters` when the
  discovered partition is unstable.
- **FCGL assumes a common support within a cluster.** A response with a private factor loses it
  unless the block as a whole is kept, and receives small loadings on factors only its peers
  carry.
- **Kept groups are shrunk.** Both norms bias kept loadings toward the prior. Refit the selected
  support when magnitudes matter.
- **`reg_lambda` does not transfer between penalties,** and the `"normalized"` weight changes
  its strength with the number of clusters by design. Select the penalty per configuration.
- **The exhibit favours FCGL by construction.** The generating process has exactly the structure
  FCGL assumes: equal supports within clusters, uncorrelated factors and a partition that
  clustering recovers. With supports that differ inside a cluster the ranking can change.
- **Cost.** FCGL is one coupled cone programme over all responses; the row penalty separates by
  response. For large $N$ the difference in solve time is material.

## See also

- [Sparse factor model](sparse_factor_model.md) for the loss and the cell-wise penalty.
- [Sign constraints and priors](sign_constraints_and_priors.md) for $\mathcal{C}$, $\beta_0$ and
  the zero entries used by the refit.
- [Quickstart](quickstart.md) for HCGL with derived signs and cross-validated penalty.
- [Residual diagnostics](residual_diagnostics.md) for counting kept loadings.
- [Task guides](task-guides.rst) for cluster-aware estimation recipes.
- [API reference](api.rst) for signatures.

## References

- Meinshausen, N. (2007). Relaxed Lasso. *Computational Statistics and Data Analysis* 52(1),
  374-393. DOI 10.1016/j.csda.2006.12.019.
- Sepp, A., and Kastenholz, M. (2026). factorlasso: hierarchical clustering group LASSO (HCGL)
  with cluster-pooled sign derivation for multi-asset factor models in Python. Submitted to the
  *Journal of Statistical Software*.
- Sepp, A., Ossa, I., and Kastenholz, M. (2026). Robust optimization of strategic and tactical
  asset allocation for multi-asset portfolios. *The Journal of Portfolio Management* 52(4),
  86-120.
- Simon, N., Friedman, J., Hastie, T., and Tibshirani, R. (2013). A sparse-group lasso. *Journal
  of Computational and Graphical Statistics* 22(2), 231-245. DOI 10.1080/10618600.2012.681250.
- Wang, H., and Leng, C. (2008). A note on adaptive group lasso. *Computational Statistics and
  Data Analysis* 52(12), 5277-5286. DOI 10.1016/j.csda.2008.05.006.
- Yuan, M., and Lin, Y. (2006). Model selection and estimation in regression with grouped
  variables. *Journal of the Royal Statistical Society: Series B* 68(1), 49-67.
  DOI 10.1111/j.1467-9868.2005.00532.x.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
