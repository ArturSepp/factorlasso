---
myst:
  html_meta:
    description: >-
      Dominant common-mode removal in factorlasso: removing the largest eigencomponent of the
      clustering correlation before cluster discovery, its audit record, when it restores sector
      structure hidden by a market factor, and when it removes a sector instead.
---

# Dominant common-mode removal

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

When one factor drives most responses, their correlation matrix is dominated by it, and a tree
built on that matrix groups the responses by how strongly they load on the common factor rather
than by what else they share. `cluster_correlation_transform="remove_pc1"` removes the largest
eigencomponent of the correlation matrix before cluster discovery. It is off by default, and this
article shows when it helps and when it harms.

## Overview

The [cluster discovery](cluster_discovery.md) step builds a tree from the correlation of the
responses. Under a strong market factor, two responses with high market betas are highly
correlated whatever their sectors, and two low-beta responses are weakly correlated even within a
sector. The largest eigenvalue of the correlation matrix then carries most of the trace, a
pattern that Plerou et al. (2002) identify with a market-wide mode, and the sector structure sits
in the modes below it.

Removing the rank-one component of that eigenvalue and rescaling to unit diagonal leaves a
residual dependence matrix in which the sectors can emerge. The transform acts only on the matrix
used to discover clusters: it does not residualise the responses, the loadings or the assembled
covariance. A changed partition still changes the fit indirectly, because the group penalty and
the sign pooling use it.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $R$ | Clustering dependence matrix, $N \times N$ | Signed, unit diagonal; missing pairs set to zero |
| $\lambda_1 \ge \lambda_2$ | Its two largest eigenvalues | $\sum_k \lambda_k = N$ |
| $v_1$ | Unit eigenvector of $\lambda_1$ | |
| $\tilde R$ | Residual dependence matrix after the removal | Unit diagonal |

The dependence matrix is the one set by `dependence_measure` and `cluster_correlation_span`; see
[cluster discovery](cluster_discovery.md).

## Methodology

### The deflation

`remove_first_principal_component` subtracts the dominant rank-one component,

$$
Q = R - \lambda_1 v_1 v_1^{\top}, \qquad Q_{ii} = 1 - \lambda_1 v_{1i}^2 ,
$$

and restandardises it to unit diagonal, $\tilde R_{ij} = Q_{ij} / \sqrt{Q_{ii} Q_{jj}}$. A
response whose residual variance $Q_{ii}$ is at the numerical floor, one explained entirely by
the removed mode, is kept as an isolated series. $Q$ has $v_1$ in its null space, so $\tilde R$ is
singular: its smallest eigenvalue is zero.

### Why it restores sectors

With a market factor $m$, sector factors and uncorrelated residuals, the covariance of two
responses $i$ and $j$ in different sectors is $\beta_i \beta_j \sigma_m^2$, and in the same sector
it adds the sector variance. When $\sigma_m$ is large, the market term dominates both, the
correlation between two responses rises with both betas, and the tree sorts the responses by
beta. The dominant eigenvector is then close to the vector of market exposures, and the residual
$\tilde R$ keeps mainly the sector terms. The removal does not know which factor is the market: it
removes whatever mode is largest.

### The audit record

The function returns a `ClusterCorrelationTransformResult` with the residual matrix and the
quantities that say whether the removal was appropriate:

| Field | Meaning |
|---|---|
| `removed_eigenvalue`, `removed_variance_share` | $\lambda_1$, and $\lambda_1 / N$, its share of the trace |
| `eigengap`, `dominant_component_unique` | $\lambda_1 - \lambda_2$, and whether it exceeds the numerical tolerance |
| `minimum_residual_variance`, `isolated_assets` | The smallest $Q_{ii}$, and the responses at the floor |
| `missing_offdiagonal_pairs` | Pairs without a correlation, set to zero before the deflation |

A large share and a large gap describe a dominant common mode. A small gap means the first mode
is one of several of similar size, such as sectors, and removing it takes one of them away.

## Worked example

The canonical script [`examples/docs/common_mode_removal.py`](../examples/docs/common_mode_removal.py)
simulates 240 months of 12 responses in three sectors of four. Each sector has its own factor
with 2% monthly volatility and the residual volatility is 2%. All responses load on a market
factor, with betas of 0.4, 0.8, 1.2 and 1.6 inside each sector. The partitions come from one
function:

```python
def partitions(y: pd.DataFrame) -> tuple:
    """Three-cluster Ward partitions of the correlation and of its common-mode residual."""
    corr = y.corr()
    removal = fl.remove_first_principal_component(corr)
    raw, _, _ = fl.compute_clusters_from_corr_matrix(corr, n_clusters=3)
    residual, _, _ = fl.compute_clusters_from_corr_matrix(removal.correlation, n_clusters=3)
    return raw, residual, removal
```

With a market volatility of 6% per month, the largest eigenvalue is 9.03, a share of 0.75 of the
trace, and the next two are 0.86 and 0.85. Cut into three clusters, the correlation tree isolates
two responses with beta 0.4 and puts the other ten together. After the removal, the residual
spectrum starts with 3.10, 2.90 and 1.32, one mode per sector, and the three clusters are the
three sectors. `LassoModel` with `cluster_correlation_transform="remove_pc1"` finds the same
partition inside `fit`. The script checks the residual matrix against the deflation formula and
the audit fields against a direct eigendecomposition.

Across 50 panels per level, the correlation recovers the sectors in every panel up to a market
volatility of 3%, in 16% of panels at 5% and in 2% at 6%. With the removal, recovery is complete
from a market volatility of 1% upwards. Without a market factor it falls to 40%: the largest
eigenvalue is then a sector mode, with a share of 0.23 of the trace, and removing it merges that
sector into the others.

![The eigenvalues of the response correlation matrix before and after the dominant mode is removed, and the share of panels clustered by sector against the market volatility with and without the removal](images/common_mode_spectrum.png)

*Synthetic teaching exhibit. Left: eigenvalues by rank for one panel with 6% monthly market
volatility. Right: share of 50 panels whose three Ward clusters equal the three sectors, against
the market volatility; the dotted line marks the panel on the left. Produced by
`tools/docs_analytics/clustering.py` from the example script.*

## Implementation in factorlasso

Verified with factorlasso 0.20.0.

| Name | Role |
|---|---|
| `cluster_correlation_transform` | The `LassoModel` setting: `"none"`, the default, or `"remove_pc1"`, applied to the dependence matrix before the distance, linkage and cut. |
| `ClusterCorrelationTransform` | The admissible values, `NONE` and `REMOVE_PC1`. |
| `remove_first_principal_component` | The deflation with its audit record, for inspection. |
| `ClusterCorrelationTransformResult` | The residual matrix and the audit fields above. |
| `apply_cluster_correlation_transform` | The dispatch `fit` uses; `NONE` returns the input object itself, untouched. |

The setting applies in the three cluster-discovery modes and in
`compute_rolling_smoothed_clusters`, where the asset universe of each date is selected before the
transform. To run the example from a checkout:

```console
python examples/docs/common_mode_removal.py
```

## Interpretation and limitations

- **Read the audit record before using the result.** The removal is right when one mode
  dominates: a large share of the trace and a large gap to the second eigenvalue. Without one, it
  removes a genuine group.
- **It is a diagnostic, not a default.** The package default stays `NONE`; use the removal to
  test whether a partition reflects exposure to one factor, and compare the partitions with and
  without it.
- **Only one mode is removed.** Two strong common factors leave the second in the residual. The
  function does not separate the noise bulk from the signal, as the random-matrix filtering of
  Laloux et al. (1999) does, and it has no null model for the communities that remain, as
  MacMahon and Garlaschelli (2015) develop.
- **The cut scale changes.** Residual correlations can be negative, so distances under $1 - \rho$
  reach beyond one and a `cutoff_fraction` calibrated on the raw matrix does not carry over; the
  example cuts by `n_clusters`.

## See also

- [Cluster discovery](cluster_discovery.md): the dependence matrix, the distance and the cut.
- [Residual diagnostics](residual_diagnostics.md): the eigenvalue spectrum of residuals and the
  Marchenko-Pastur edge.
- [Causal smoothing of rolling clusters](rolling_cluster_smoothing.md): the transform inside a
  rolling estimation.

## References

- Laloux, L., Cizeau, P., Bouchaud, J.-P., and Potters, M. (1999). Noise dressing of financial
  correlation matrices. *Physical Review Letters* 83(7), 1467-1470.
  DOI 10.1103/PhysRevLett.83.1467.
- MacMahon, M., and Garlaschelli, D. (2015). Community detection for correlation matrices.
  *Physical Review X* 5, 021006. DOI 10.1103/PhysRevX.5.021006.
- Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., Guhr, T., and Stanley, H. E.
  (2002). Random matrix approach to cross correlations in financial data. *Physical Review E* 65,
  066126. DOI 10.1103/PhysRevE.65.066126.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
