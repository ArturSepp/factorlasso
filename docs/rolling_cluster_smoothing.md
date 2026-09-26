---
myst:
  html_meta:
    description: >-
      Causal smoothing of rolling clusters in factorlasso: why re-estimated partitions churn while
      the correlation estimate moves slowly, the hold, partition-bonus and similarity-EWMA
      smoothers, and what they cost in lag on a panel with a migrating response.
---

# Causal smoothing of rolling clusters

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

A rolling estimation discovers its clusters again on every date. The correlation estimate behind
them moves slowly, but a partition is a discrete function of it: a response that sits between two
clusters can change cluster on one date and change back on the next. `compute_rolling_smoothed_clusters`
re-estimates partitions causally, with one of three smoothers that damp such churn, and this
article measures what each removes and what it costs.

## Overview

With an EWMA span $s$, the correlation estimate is updated each date by

$$
C_t = \lambda C_{t-1} + (1 - \lambda) x_t x_t^{\top}, \qquad 1 - \lambda = \frac{2}{s + 1} ,
$$

for standardised observations $x_t$, so one date moves it by a fraction $2 / (s + 1)$, 5.4% for
a 36-month span. The loadings estimated with the same span move by similar fractions. The
partition does not move by fractions. A response whose distances to two clusters are nearly equal
changes cluster whenever one date's increment tips the balance, and the next increment can tip it
back. The rest of the partition can stay where it was; the churn is concentrated on the
responses near a boundary. The rolling-Ward working paper documents the scale of the effect on
equity panels: a weekly EWMA of span 156 replaces about 5% of its weight between two monthly
estimation dates, while the unsmoothed Ward partition reassigns roughly a quarter of the assets
(Sepp, 2026).

The smoothers make the partition depend on its own history, in the spirit of evolutionary
clustering, which trades faithfulness to the current data against stability over time
(Chakrabarti, Kumar and Tomkins, 2006). Each partition still uses only data up to its date.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $t$ | Estimation date, in `estimation_dates` | Each date sees only `y.loc[:t]` |
| $C_t$, $D_t$ | Dependence matrix and distance at $t$, as in [cluster discovery](cluster_discovery.md) | Span `cluster_correlation_span`, else `span` |
| $P_{t-1}$ | The partition of the previous date | Labels are arbitrary |
| $\delta$ | `smoother_delta`, the distance discount for previous peers | Distance units, default 0.05 |
| $\lambda_s$ | `smoother_lambda`, the weight on the previous similarity state | In $[0, 1)$, default 0.7 |
| anchors | Dates of `recluster_freq`, a pandas frequency such as `"QE"` | Optional except for `HOLD` |

Cluster numbers are assigned afresh on every date, so two partitions are compared by their
memberships, which pairs of responses share a cluster, and not by their labels.

## Methodology

### The four smoothers

`cluster_smoother_type` selects the rule by which the partition $P_t$ is formed:

| `ClusterSmootherType` | Partition at date $t$ |
|---|---|
| `NONE` | The tree of $D_t$, cut as configured: the partition `fit` would find on `y.loc[:t]`. |
| `HOLD` | Re-estimated only at the last estimation date on or before each anchor and held until the next; a response that enters meanwhile joins the held cluster with the highest mean current correlation. |
| `PARTITION_BONUS` | The tree of $D_t$ after pairs that shared a cluster in $P_{t-1}$ have their distance reduced by $\delta$, floored at zero. |
| `SIMILARITY_EWMA` | The tree of a smoothed similarity $S_t = (1 - \lambda_s) C_t + \lambda_s S_{t-1}$; pairs new to the universe start from $C_t$. |

The partition bonus is a history cost of the kind Chakrabarti, Kumar and Tomkins (2006) add to
the clustering objective, expressed here as a fixed discount on distances. The similarity EWMA
follows the approach of tracking the proximities over time and clustering the tracked matrix,
which Xu, Kliger and Hero (2014) develop with a smoothing weight estimated from the data; here the
weight is fixed. Its memory is that of an EWMA of span $(1 + \lambda_s) / (1 - \lambda_s)$, about
5.7 dates at the default, on top of the span of $C_t$. The defaults of both are implementation
choices.

### Calibrating the partition bonus

A membership change should need more evidence than the sampling noise of the correlation
estimate provides. The rolling-Ward working paper derives that noise for an EWMA correlation of
span $s$, $\sqrt{1 + \kappa} (1 - \rho^2) / \sqrt{s}$ with $\kappa$ the excess kurtosis of the
returns, and sets the bonus at its one-sided quantile (Sepp, 2026):

$$
\delta^{*} = z_{1-\alpha} \sqrt{1 + \kappa} \frac{1 - \bar\rho^{2}}{\sqrt{s}} ,
$$

where $\bar\rho$ is a typical correlation within a cluster and $\alpha$ the tolerated rate of
false flips for a response with no margin. The package does not compute $\delta^{*}$; the
example below evaluates it by hand.

With `recluster_freq` set, `PARTITION_BONUS` and `SIMILARITY_EWMA` also update their state and
partition only at anchors and hold the partition in between. `HOLD` requires it, and `NONE`
rejects it.

### Causality and the universe

At every date the function truncates the panel to `y.loc[:t]` before any correlation or smoother
state is computed, and a response enters only once it has `warmup_period` observations. An
optional Boolean `eligibility` frame restricts the universe of each date further, point in time,
without filling membership forward. A later observation therefore never changes an earlier
partition.

## Worked example

The canonical script
[`examples/docs/rolling_cluster_smoothing.py`](../examples/docs/rolling_cluster_smoothing.py)
simulates 240 months of 12 responses in three blocks of four, with block volatility 3% and
residual volatility 2% per month. One response, $a4$, migrates: its loading moves linearly from
block $a$ to block $b$ between months 60 and 180, so the two loadings are equal in January 2016.
Partitions are re-estimated monthly from month 36, 204 dates, with a 36-month EWMA correlation
cut at three clusters. The within-block correlation is 0.69, so with Gaussian returns and
$\alpha = 0.05$ the calibrated bonus is $\delta^{*} = 1.645 \times (1 - 0.69^2) / 6 = 0.14$,
nearly three times the default; it enters as a fifth configuration:

```python
SMOOTHERS = {
    "none": {"cluster_smoother_type": fl.ClusterSmootherType.NONE},
    "hold": {"cluster_smoother_type": fl.ClusterSmootherType.HOLD, "recluster_freq": "QE"},
    "partition bonus": {"cluster_smoother_type": fl.ClusterSmootherType.PARTITION_BONUS,
                        "smoother_delta": 0.05},
    "bonus at noise floor": {"cluster_smoother_type": fl.ClusterSmootherType.PARTITION_BONUS,
                             "smoother_delta": CALIBRATED_DELTA},
    "similarity EWMA": {"cluster_smoother_type": fl.ClusterSmootherType.SIMILARITY_EWMA,
                        "smoother_lambda": 0.7},
}
```

```python
def rolling_partitions(y: pd.DataFrame, smoother: str) -> fl.RollingClusterData:
    """Monthly causal partitions under one smoother, with the model's discovery settings."""
    model = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        span=SPAN,
        n_clusters=N_CLUSTERS,
        **SMOOTHERS[smoother],
    )
    return fl.compute_rolling_smoothed_clusters(y, list(y.index[FIRST_ESTIMATION:]), model)
```

The other eleven responses never change cluster under any smoother; all membership changes are
switches of $a4$. On this panel:

| Smoother | Membership changes | Label changes | Final switch, months after the crossing |
|---|---|---|---|
| None | 3 | 40 | 21 |
| Hold | 3 | 28 | 23 |
| Partition bonus | 1 | 40 | 20 |
| Bonus at the noise floor | 1 | 37 | 22 |
| Similarity EWMA | 1 | 18 | 21 |

Without smoothing, $a4$ moves to block $b$ in March 2017, back in June and finally in October. The
partition bonuses and the similarity EWMA move it once, between September and November 2017.
Holding at
quarter ends keeps all three switches, because the flip-flop lasts longer than a quarter, and
delays the last to December. The
cluster labels change on up to 40 of the 203 transitions although the memberships change on at
most three.

Over ten panels, $a4$ switches 2.8 times on average without smoothing, 2.0 times with hold, and
1.4 times with the default partition bonus or the similarity EWMA; one switch is the truth. The
final switch comes 21.2, 22.4, 21.5 and 22.6 months after the crossing. Most of that lag is the
correlation estimate's own: a 36-month EWMA weights observations with a mean age of 17.5 months,
and the default smoothers add at most about a month and a half to it. The bonus at the noise
floor switches exactly once in every panel, at a mean lag of 24.8 months: it removes the
remaining flip-flops for another three months of delay.

The script checks that the unsmoothed partitions equal those a `LassoModel` fit finds on
`y.loc[:t]` at three dates, that a run on a truncated panel reproduces the earlier partitions of
the full run, and the bonus and EWMA updates against their formulas.

![The cluster of a migrating response at each monthly estimation date under four causal smoothers, and the mean number of its cluster switches over ten panels with the lag of its final switch](images/rolling_smoothing_churn.png)

*Synthetic teaching exhibit. Left: the cluster of $a4$ at each estimation date, low when it
shares block $a$'s cluster and high when it shares block $b$'s; the shaded span is the migration
and the dashed line the date of equal loadings. Right: switches of $a4$, mean over ten panels,
with the mean lag of the final switch; hatched and dashed, the bonus at the noise floor.
Produced by `tools/docs_analytics/clustering.py` from the
example script.*

## Implementation in factorlasso

Verified with factorlasso 0.20.0.

| Name | Role |
|---|---|
| `compute_rolling_smoothed_clusters` | Causal partitions for a list of dates from a response panel and a `LassoModel` that declares the discovery and smoother settings; the model is neither fitted nor changed. |
| `RollingClusterData` | Per-date `clusters`, `linkages` and `cutoffs`, and `co_association`, the share of each response's current peers that shared its cluster over the trailing six dates. |
| `ClusterSmootherType` | `NONE`, `HOLD`, `PARTITION_BONUS`, `SIMILARITY_EWMA`. |
| `smooth_similarity_ewma` | One update of the similarity state. |
| `apply_partition_distance_bonus` | The distance discount for previous peers. |

| Parameter | Default | Role |
|---|---|---|
| `cluster_smoother_type` | `NONE` | The smoother. |
| `smoother_delta` | 0.05 | $\delta$ for `PARTITION_BONUS`, in the units of the distance transform. |
| `smoother_lambda` | 0.7 | $\lambda_s$ for `SIMILARITY_EWMA`. |
| `recluster_freq` | `None` | Anchor frequency; required by `HOLD`, optional for the bonus and the EWMA, rejected by `NONE`. |

`LassoModel.fit` does not smooth: a single fit sees a single date. A rolling estimation computes
the partitions with this function and passes each date's partition to `fit` as
`external_clusters`, which HCGL and FCGL accept. To run the example from a checkout:

```console
python examples/docs/rolling_cluster_smoothing.py
```

## Interpretation and limitations

- **Smoothing trades flip-flops for delay.** Here the delay is small next to the lag of the
  correlation estimate, but a larger $\delta$, a larger $\lambda_s$ or a coarser anchor frequency
  holds a genuine change back for longer.
- **The settings are not portable.** $\delta$ is in distance units and changes meaning with the
  distance transform and the dependence measure; $\lambda_s$ compounds with the span of $C_t$.
  The noise-floor calibration ties $\delta$ to the span and the correlation level, but it is
  derived for the $1 - \rho$ distance and a flat cut.
- **Compare memberships, not labels.** Labels are renumbered on every date. Persistent cluster
  identities across dates are the subject of the offline cluster lineage, which is not causal.
- **One boundary response is the easy case.** With many responses near boundaries, or clusters
  that split and merge, the smoothers interact with the cut in ways this example does not test.

## See also

- [Cluster discovery](cluster_discovery.md): the dependence matrix, the distance and the cut that
  every date applies.
- [Dominant common-mode removal](common_mode_removal.md): applied on each date after the universe
  is selected.
- [EWMA weighting](ewma_weighting_and_ragged_histories.md): the spans and weights behind $C_t$.

## References

- Chakrabarti, D., Kumar, R., and Tomkins, A. (2006). Evolutionary clustering. In *Proceedings of
  the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 554-560.
  DOI 10.1145/1150402.1150467.
- Sepp, A. (2026). *Rolling-Ward clustering: noise-calibrated stability for rolling
  correlation-based clusters*. Working paper; link to be added.
- Xu, K. S., Kliger, M., and Hero, A. O. (2014). Adaptive evolutionary clustering. *Data Mining
  and Knowledge Discovery* 28, 304-336. DOI 10.1007/s10618-012-0302-x.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
