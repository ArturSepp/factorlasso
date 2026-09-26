---
myst:
  html_meta:
    description: >-
      Cluster discovery in factorlasso: the dependence measure (Pearson, Spearman, Gerber), the
      correlation-to-distance transform, Ward linkage and the dendrogram cut, why a fractional cut
      does not port across measures and transforms, and a worked example with crisis shocks.
---

# Cluster discovery: dependence, distance, linkage and cut

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

The cluster modes of factorlasso discover groups of responses from their co-movement: a
dependence matrix of the responses, a distance derived from it, an agglomerative tree and a cut.
Each step has a setting, and each setting can change the partition that the group penalty, the
sign pooling and the adaptive weights all use. This article follows the four steps and shows
which choices travel from one panel to another and which do not.

## Overview

`HIERARCHICAL_CLUSTER_GROUP_LASSO`, `FACTOR_CLUSTER_GROUP_LASSO` and
`COOPERATIVE_CLUSTER_GROUP_LASSO` discover their clusters inside `fit`, with the same function
that is public as `compute_clusters_from_corr_matrix`. The default is the method of the software
paper: the Pearson correlation of the responses, the dissimilarity $1 - \rho$, Ward linkage and a
cut at half the largest pairwise distance (Sepp and Kastenholz, 2026b, Section 2.3; Sepp and
Kastenholz, 2026a, Section 2.5). One partition then serves three purposes: it groups the penalty,
pools the univariate slopes for the [sign derivation](gated_cluster_pooled_signs.md), and pools
the [adaptive weights](adaptive_penalty_weights.md).

The clusters come from the responses alone. The factors do not enter, and a partition is a
statement about which responses move together, not about which share loadings.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $Y$ | Response panel, $T \times N$, de-meaned when `demean=True` | Returns; missing values allowed |
| $C$ | Dependence matrix of the responses, `dependence_measure` | Signed, unit diagonal |
| $D$ | Distance matrix, $D_{ij} = d(C_{ij})$, `distance_transform` | Zero diagonal |
| $f$ | `cutoff_fraction`, the cut as a fraction of $\max_{i<j} D_{ij}$ | In $(0, 1]$, default 0.5 |
| $K$ | `n_clusters`, a target cluster count that replaces $f$ | Optional |

Observations are weighted uniformly, or by EWMA with `cluster_correlation_span`, which defaults
to the `span` of the loss; see the [conventions page](conventions.md).

## Methodology

### Step 1: the dependence measure

`dependence_measure` selects how $C$ is estimated from the responses:

- `"pearson"`, the default: the product-moment correlation.
- `"spearman"`: the Pearson correlation of the ranks, invariant to monotone transforms of each
  series and robust to outliers.
- `"gerber"`: the Gerber statistic of Gerber et al. (2022), a count of joint moves.

The Gerber statistic sets a threshold $c \sigma_i$ per series, with $c$ given by
`gerber_threshold` (default 0.5), and counts the dates on which both series move beyond their
thresholds:

$$
g_{ij} = \frac{n^{UU}_{ij} + n^{DD}_{ij} - n^{UD}_{ij} - n^{DU}_{ij}}{T - n^{NN}_{ij}} ,
$$

where $U$ and $D$ mark a move above $c \sigma$ or below $-c \sigma$, and $n^{NN}_{ij}$ counts the
dates on which neither series moves beyond its threshold. A large move counts once, like any
other, and a small one not at all. The equal-weight Gerber matrix is positive semidefinite
(Gerber et al., 2023); the EWMA-weighted one is not covered by that proof.

All three measures are signed, which the package requires: the partition pools signs, and an unsigned
measure would put responses with opposite exposures in one cluster. With an EWMA span every
measure weights its observations, the Gerber counts included. The statistic is computed on the
de-meaned panel the solver sees, and unlike the correlations it is not invariant to centring.

### Step 2: the distance transform

`distance_transform` maps $\rho$ to a distance:

| `DistanceTransform` | $d(\rho)$ | Property |
|---|---|---|
| `ONE_MINUS_RHO`, the default | $1 - \rho$ | A dissimilarity, not a metric |
| `CHORD` | $\sqrt{2 (1 - \rho)}$ | The Euclidean distance between standardised series (Mantegna, 1999); López de Prado (2016) uses half of it |
| `ARCCOS` | $\arccos \rho$ | The angle between them, a metric |

All three are decreasing in $\rho$, so linkages that use only the order of the distances, single
and complete, build the same tree under each. Ward, average, centroid and median linkage use the
magnitudes, and the tree can change.

### Step 3: Ward linkage

`linkage_method` defaults to Ward's criterion (Ward, 1963), which merges at each step the two
clusters whose union adds least to the within-cluster sum of squares. The criterion is exact for
Euclidean distances, so it is exact under `CHORD`; under the default $1 - \rho$ the package
applies it as a stable heuristic, as the software paper states (Sepp and Kastenholz, 2026b,
Section 2.3). Ward merge heights grow with cluster size and can exceed every pairwise distance.

### Step 4: the cut

By default the tree is cut at height $f \cdot \max_{i<j} D_{ij}$. The fraction is calibrated to
the scale of $D$, so it does not carry over:

- **Across transforms.** The same $f$ under `CHORD` cuts at a different pairwise correlation. The
  conversion $f \mapsto \sqrt f$ keeps the implied pairwise threshold under `CHORD`, and
  $\arccos(1 - f (1 - \rho_{\min})) / \arccos(\rho_{\min})$ under `ARCCOS`, where $\rho_{\min}$ is
  the smallest off-diagonal correlation. Under Ward the merge heights are not pairwise distances,
  so the conversion typically, not always, keeps the partition.
- **Across measures and panels.** The Gerber statistic and the rank correlation are smaller than
  the Pearson correlation on the same data, by a factor that depends on the data, and a weakly
  correlated panel places every pairwise distance above the cut.

`n_clusters` replaces the height by a count: the tree is cut into at most $K$ clusters. It is
the portable choice whenever partitions are compared across measures, transforms or panels.

## Worked example

The canonical script [`examples/docs/cluster_discovery.py`](../examples/docs/cluster_discovery.py)
simulates 240 months of 12 responses in three blocks of four, $a$, $b$ and $c$. Each block has its
own factor with 1.5% monthly volatility; all responses load 0.5 on a common factor with 2%
volatility, and the residual volatility is 2%. On six dates a joint shock of 25% hits the eight
responses of blocks $b$ and $c$, as a crisis would. The dependence matrix and the partition come
from two functions:

```python
def dependence(y: pd.DataFrame, measure: str) -> pd.DataFrame:
    """The clustering dependence matrix of the de-meaned responses, as LassoModel builds it."""
    values = fl.compute_dependence_matrix((y - y.mean()).to_numpy(), dependence_measure=measure)
    return pd.DataFrame(values, index=y.columns, columns=y.columns)


def discover(y: pd.DataFrame, measure: str, transform: str = "one_minus_rho",
             cutoff_fraction: float = 0.5, n_clusters: int | None = None) -> tuple:
    """Partition, linkage and cut height for one dependence measure and distance transform."""
    return fl.compute_clusters_from_corr_matrix(
        dependence(y, measure),
        cutoff_fraction=cutoff_fraction,
        linkage_method="ward",
        distance_transform=transform,
        n_clusters=n_clusters,
    )
```

| Measure | Mean within block $b$ | Mean between $b$ and $c$ | Three clusters |
|---|---|---|---|
| Pearson | 0.81 | 0.74 | $b$ and $c$ merged, $a$ split in two |
| Spearman | 0.45 | 0.20 | $a$, $b$, $c$ |
| Gerber | 0.24 | 0.09 | $a$, $b$, $c$ |

Six dates out of 240 lift the Pearson correlation between $b$ and $c$ to 0.74, close to the
correlation within a block, and the three-cluster cut merges the two shocked blocks. The ranks and
the co-movement counts give each shock the weight of one ordinary date, and both measures recover
the blocks. The script checks the Gerber matrix against a pair-by-pair count of the published
formula to $10^{-12}$.

The default fractional cut is another matter. At $f = 0.5$ the Pearson tree has five clusters,
and the Spearman and Gerber trees isolate every response, because all their pairwise distances
exceed half the largest. On the Spearman matrix $f = 0.75$ recovers the three blocks under
$1 - \rho$, while the same fraction gives 10 clusters under `CHORD` and 6 under `ARCCOS`; the
conversion $f \mapsto \sqrt f$ reproduces the $1 - \rho$ partition under `CHORD` at every fraction
on the grid. Even at $f = 1$ the tree keeps three clusters, since the two top Ward merges lie
above the largest pairwise distance. Single linkage builds the same tree under all three
transforms, and `LassoModel` with `dependence_measure="spearman"` and `n_clusters=3` finds the
same partition inside `fit`.

![Ward dendrograms of the shocked three-block panel under Pearson and Spearman dependence cut at three clusters, and the cluster count against the cutoff fraction under three distance transforms](images/cluster_discovery_dendrograms.png)

*Synthetic teaching exhibit. Left and middle: Ward trees under $1 - \rho$ for the Pearson and
Spearman matrices, with the cut at three clusters. Right: realised cluster count on the Spearman
matrix against `cutoff_fraction` for the three transforms; the dashed line marks the default 0.5
and the dotted line the three true blocks. Produced by `tools/docs_analytics/clustering.py` from
the example script.*

## Implementation in factorlasso

Verified with factorlasso 0.20.0.

| Name | Role |
|---|---|
| `compute_clusters_from_corr_matrix` | Partition, linkage matrix and cut height from a labelled dependence matrix; the function `fit` calls. |
| `compute_dependence_matrix` | The dependence matrix of a panel under a measure and an optional EWMA span; with uniform weights, missing values are handled pairwise. |
| `compute_gerber_matrix` | The Gerber statistic alone, with EWMA-weighted counts when a span is given. |
| `DependenceMeasure`, `DistanceTransform` | The admissible measures and transforms; plain strings are accepted. |

The `LassoModel` settings of the discovery step:

| Parameter | Default | Role |
|---|---|---|
| `dependence_measure` | `"pearson"` | Step 1. |
| `gerber_threshold` | 0.5 | Threshold $c$ of the Gerber statistic; ignored by the other measures. |
| `distance_transform` | `ONE_MINUS_RHO` | Step 2. |
| `linkage_method` | `"ward"` | Step 3: any SciPy method from single to Ward. |
| `cutoff_fraction` | 0.5 | Step 4, the fractional cut. |
| `n_clusters` | `None` | Step 4, the count cut; replaces `cutoff_fraction` when set. |
| `cluster_correlation_span` | `None` | EWMA span of the discovery step; `None` uses the `span` of the loss. A different value also sets the de-meaning of the panel used for discovery. |
| `cluster_correlation_span_freq_dict` | `None` | Per-frequency discovery spans for multi-frequency pipelines, resolved by the caller before `fit`, like `span_freq_dict`. |

The settings apply only to the three discovery modes. `external_clusters` passed to `fit`
bypasses discovery for HCGL and FCGL, and `clusters_`, `linkage_` and `cutoff_` hold the result.
To run the example from a checkout:

```console
python examples/docs/cluster_discovery.py
```

## Interpretation and limitations

- **A partition is a grouping device, not a finding.** The software paper reads the step as a
  correlation-clustering heuristic that groups the penalty and pools signs; it makes no claim
  that the clusters are the true structure (Sepp and Kastenholz, 2026b, Section 2.3). Check the
  partition against the group structure you expect before relying on it.
- **Robust measures trade information for robustness.** The rank correlation discards
  magnitudes and the Gerber statistic ignores moves inside the threshold; both lower the measured
  dependence, which is why the fractional cut shatters them here.
- **The fractional cut is not portable.** Recalibrate `cutoff_fraction` when the measure, the
  transform or the universe changes, or use `n_clusters`.
- **Correlation is not exposure.** Two responses with the same loadings and very different
  residual variances can be weakly correlated and land in different clusters.
- **A common mode can dominate.** When one factor drives most responses, the tree can sort them
  by their exposure to it; [common-mode removal](common_mode_removal.md) treats that case.

## See also

- [Group penalties](group_penalties_hcgl_fcgl.md): what the partition groups.
- [Gated cluster-pooled sign derivation](gated_cluster_pooled_signs.md): what the partition pools.
- [Dominant common-mode removal](common_mode_removal.md) and
  [causal smoothing of rolling clusters](rolling_cluster_smoothing.md): the next steps of the
  discovery pipeline.

## References

- Gerber, S., Javid, B., Markowitz, H., Sargen, P., and Starer, D. (2022). The Gerber statistic: a
  robust co-movement measure for portfolio optimization. *The Journal of Portfolio Management*
  48(2), 87-102.
- Gerber, S., Markowitz, H., Ernst, P., Miao, Y., Javid, B., and Sargen, P. (2023). Proofs that
  the Gerber statistic is positive semidefinite. arXiv:2305.05663.
- López de Prado, M. (2016). Building diversified portfolios that outperform out of sample.
  *The Journal of Portfolio Management* 42(4), 59-69. DOI 10.3905/jpm.2016.42.4.059.
- Mantegna, R. N. (1999). Hierarchical structure in financial markets. *The European Physical
  Journal B* 11, 193-197. DOI 10.1007/s100510050929.
- Sepp, A., and Kastenholz, M. (2026a). Gated Cluster-Pooled Sign Constraints for Multi-Output
  Sparse Regression. Submitted to *Computational Statistics & Data Analysis*.
  [Manuscript](../papers/sign_pooling_2026/paper/article.pdf).
- Sepp, A., and Kastenholz, M. A. (2026b). factorlasso: Sparse Multi-Output Regression with
  Cluster-Grouped Sign Constraints in Python. Submitted to the *Journal of Statistical
  Software*. [Manuscript](../papers/jss_2026/paper/article.pdf).
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the
  American Statistical Association* 58(301), 236-244. DOI 10.1080/01621459.1963.10500845.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
