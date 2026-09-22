---
myst:
  html_meta:
    description: >-
      Factor covariance assembly in factorlasso: the decomposition of the response covariance into
      loadings, factor covariance and residual variances, the containers that store and roll it,
      and a worked example against the sample covariance for 40 series.
---

# Factor covariance assembly

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Factor covariance assembly builds the $N \times N$ covariance matrix of the responses from three
estimated parts: the loading matrix, the covariance of the factors, and the residual variances.
`CurrentFactorCovarData` stores the parts at one estimation date and assembles the matrix on
request; `RollingFactorCovarData` holds a dated sequence of such snapshots and returns the one
that was available at a query date.

## Overview

Under the factor model $Y_t = \alpha + \beta X_t + \varepsilon_t$ with residuals uncorrelated with
the factors, the covariance of the responses is

$$
\Sigma_y = \beta \Sigma_x \beta^{\top} + D ,
$$

where $\Sigma_x$ is the $M \times M$ factor covariance and $D$ the residual covariance. With a
diagonal $D$ the matrix has $NM + M(M+1)/2 + N$ free parameters, against $N(N+1)/2$ for an
unrestricted covariance: 210 against 820 for 40 series and four factors. The restriction is the
point. Rosenberg and McKibben (1973) introduced the decomposition for equity risk, and Fan, Fan
and Lv (2008) show that, when the factor structure holds, the factor-based estimator has little
advantage over the sample covariance for the matrix itself and a substantial one for its
inverse, which is what portfolio construction uses.

The worked example reproduces that result at small scale. For 40 series and 48 months the
relative error of the two estimators is about equal, 0.32 against 0.34, while the minimum-variance
portfolio built from the sample covariance has a realised volatility of 10.1% against 4.9% from
the assembled matrix and 4.4% for the population optimum.

The containers do not estimate anything. The caller supplies the parts, from `LassoModel` or
from any other source, and chooses their units. They were developed for the factor risk models
of Sepp, Hansen and Kastenholz (2026) and Sepp, Ossa and Kastenholz (2026). Linear shrinkage of
the sample covariance (Ledoit and Wolf 2004) is the main alternative way to obtain a
well-conditioned matrix; it is not part of this package and is not compared here.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $\beta$, `y_betas` | Loading matrix | DataFrame $N \times M$, responses by factors |
| $\Sigma_x$, `x_covar` | Factor covariance | DataFrame $M \times M$ in the variance units the caller chooses, for example annualised |
| `y_variances` | Per-response diagnostics | DataFrame indexed by response; the column `residual_var` is required and holds the diagonal of $D$ in the units of `x_covar`; `r2` and `insample_alpha` are needed by `get_snapshot`; `ewma_var` and `cluster` are optional |
| $w$, `residual_var_weight` | Multiplier on the whole residual block | Default 1; 0 returns the systematic part |
| `residual_type` | Structure of $D$ | `ResidualType.ORTHOGONAL` (default, diagonal) or `ResidualType.EMPIRICAL` |
| `residuals` | Residual panel, optional | $T \times N$ in units the caller declares; used by `estimate_alpha` |
| `estimation_date` | Date of the snapshot, optional | Used by the rolling container and the empirical residual mode |
| `clusters`, `linkages`, `cutoffs` | Cluster metadata, optional | Labels prefixed by frequency, such as `ME:3`; a stacked linkage table; one cut distance per frequency |
| `derived_signs` | Sign matrix of the fit, optional | For audit; see [sign constraints and priors](sign_constraints_and_priors.md) |

The decomposition assumes that factors and residuals are uncorrelated and, in the default mode,
that residuals are uncorrelated with each other. The second assumption is testable:
[residual diagnostics](residual_diagnostics.md) tests it, and the empirical residual mode relaxes
it. All parts must share one variance unit. The containers never annualise or rescale.

## Methodology

### Assembly

`get_y_covar` returns

$$
\hat\Sigma_y = \hat\beta \hat\Sigma_x \hat\beta^{\top} + w \hat D ,
\qquad
\hat D = \operatorname{diag}(\hat\sigma_{\varepsilon,1}^2, \ldots, \hat\sigma_{\varepsilon,N}^2)
$$

in the default orthogonal mode. The first term has rank at most $M$ and is positive
semi-definite when $\hat\Sigma_x$ is. With positive residual variances and $w > 0$ the sum is
positive definite, and its smallest eigenvalue is at least the smallest residual variance times
$w$. The matrix is therefore always invertible, for any ratio of $N$ to $T$, which the sample
covariance is not.

In the empirical mode the residual block is

$$
\hat D = S \left[ (1 - \rho) I + \rho R \right] S ,
$$

where $S$ is the diagonal matrix of residual standard deviations, $R$ a residual correlation
matrix prepared by `estimate_residual_correlation`, and $\rho$ the retention
`residual_corr_weight` in $[0, 1]$. At $\rho = 0$ it is the orthogonal matrix. The diagonal of
$\hat D$ is the same in both modes.

### Volatility decomposition

The diagonal of the assembly splits the variance of each response,

$$
\hat\sigma_{y,i}^2 = \hat\beta_i^{\top} \hat\Sigma_x \hat\beta_i + \hat\sigma_{\varepsilon,i}^2 ,
$$

and `get_model_vols` returns the square roots of the total, the systematic and the residual
part. The systematic share of variance is the model's $R^2$ under the stored factor covariance.

### Estimating the parts with LassoModel

For a fitted `LassoModel`, `coef_` is $\hat\beta$ and `estimation_result_.ss_res` is the weighted
mean squared residual per response, with the weights of the loss. With unit weights it is the
population variance of the residuals, with divisor $T$. `estimation_result_.ss_total` and `r2`
are the matching total variance and $R^2$. The factor covariance is not part of the fit: the
caller estimates it, with a sample covariance as in the example below or with `compute_ewm_covar`
for an EWMA estimate whose span matches the loss.

### Rolling snapshots without look-ahead

`RollingFactorCovarData` maps estimation dates to snapshots. A query by date returns the latest
snapshot whose key is not after the query date, and raises `ValueError` when none exists. A
covariance requested between two estimation dates is the earlier one, held. Retrieval never
refits.

## Worked example

The example uses synthetic data with a fixed seed. Forty monthly return series in four groups of
ten load on four factors with a pairwise correlation of 0.3 and a volatility of 0.04 per month.
Series $i$ of group $k$ loads between 0.6 and 1.2 on factor $k$ and between 0.0 and 0.5 on
factor $k+1$, so 76 of the 160 cells of the generating matrix are non-zero. Idiosyncratic
volatility runs from 0.02 to 0.04 per month within each group. Returns are decimal per-period
returns, and every variance is annualised by a factor of 12.

One function fits the loadings and collects the parts:

```python
def estimate_covar_data(
    x: pd.DataFrame,
    y: pd.DataFrame,
    model_type: fl.LassoModelType = fl.LassoModelType.LASSO,
) -> tuple[fl.LassoModel, fl.CurrentFactorCovarData]:
    """Fit the loadings and collect every input of ``Sigma_y`` in one snapshot, annualised."""
    model = fl.LassoModel(model_type=model_type, reg_lambda=REG_LAMBDA).fit(x=x, y=y)
    result = model.estimation_result_
    y_variances = pd.DataFrame(
        {
            fl.VarianceColumns.EWMA_VARIANCE.value: PERIODS_PER_YEAR * result.ss_total,
            fl.VarianceColumns.RESIDUAL_VARS.value: PERIODS_PER_YEAR * result.ss_res,
            fl.VarianceColumns.INSAMPLE_ALPHA.value: PERIODS_PER_YEAR * model.alpha_const_.values,
            fl.VarianceColumns.R2.value: result.r2,
        },
        index=model.coef_.index,
    )
    covar_data = fl.CurrentFactorCovarData(
        x_covar=PERIODS_PER_YEAR * x.cov(ddof=0),
        y_betas=model.coef_,
        y_variances=y_variances,
        estimation_date=x.index[-1],
        residuals=PERIODS_PER_YEAR * (y - model.predict(x)),       # annualised, like the alphas
    )
    return model, covar_data
```

### One snapshot of 120 months

| Quantity | Value |
|---|---|
| `get_y_covar()` against $\hat\beta \hat\Sigma_x \hat\beta^{\top} + \operatorname{diag}(\hat\sigma_\varepsilon^2)$ computed in NumPy | equal |
| Stored residual variance against 12 times the population variance of $Y - \hat Y$ | equal |
| Smallest eigenvalue of $\hat\Sigma_y$ | 0.0048 |
| Rank of `get_y_covar(residual_var_weight=0.0)` | 4 |
| `get_model_vols()` for `asset_01`: total, systematic, residual | 0.125, 0.099, 0.076 |
| Squared total volatility against the diagonal of $\hat\Sigma_y$, and against the sum of the squared parts | equal |
| `get_y_covar(assets=subset)` and `filter_on_tickers(subset).get_y_covar()` against the matching block | equal |
| Relative Frobenius error against the population covariance: assembled, sample | 0.206, 0.218 |
| Condition number: assembled, sample | 99, 254 |

`get_snapshot()` returns one table per response: the loadings, `r2`, `stat_alpha`,
`insample_alpha`, and the three volatilities. `stat_alpha` is the last value of an EWMA of the
stored residuals with span `alpha_span`, so it carries their units; the example stores annualised
residuals to match the annualised `insample_alpha`. Without stored residuals `stat_alpha` falls
back to the in-sample alpha, so the table has the same columns either way.

### Rolling container

```python
rolling = fl.RollingFactorCovarData()
for n_obs in (96, 120):
    _, snapshot_data = estimate_covar_data(x.iloc[:n_obs], y.iloc[:n_obs])
    rolling.add(snapshot_data.estimation_date, snapshot_data)
```

The container holds snapshots at the 96th and the 120th month. `get_y_covars()` returns a
dictionary with one matrix per snapshot. `get_y_covars(dates=...)` with the 101st month-end
returns the matrix of the 96th: the 120-month fit did not exist then. `get_residual_vars()`,
`get_r2()`, `get_total_vols()` and `get_beta(factor)` return panels indexed by estimation date.

### Cluster metadata

A snapshot built from an HCGL fit can carry the partition and the dendrogram. The four helper
functions return them in the form SciPy and a later fit expect:

| Helper | Input | Output in the example |
|---|---|---|
| `get_clusters_by_freq` | `clusters` with labels such as `ME:3` | `{"ME": Series}` with the prefix removed; equal to `clusters_` of the fit |
| `get_linkage_array` | `linkages`, a frequency code | The $(N-1) \times 4$ SciPy linkage array of that frequency |
| `get_linkages_by_freq` | `linkages` | `{"ME": array}`; the same array |
| `get_cutoffs_by_freq` | `cutoffs` | `{"ME": float}`; cutting the returned linkage at this distance with `scipy.cluster.hierarchy.fcluster` reproduces the four clusters of the fit |

On construction, a `clusters` Series is copied into the `cluster` column of `y_variances`, so
that the assignment survives `save`, `load` and `filter_on_tickers`.

### Why assemble: history length against 40 series

[![Two panels against months of history, 48 to 240. Left: relative Frobenius error of the assembled factor covariance and of the sample covariance, close to each other and falling from about 0.33 to 0.14. Right: realised annualised volatility of the minimum-variance portfolio; 10.1 percent at 48 months for the sample covariance against 4.9 percent for the assembled matrix, both approaching the population optimum of 4.4 percent.](images/factor_covariance_assembly_history.png)](images/factor_covariance_assembly_history.png)

**Figure 1.** Synthetic teaching exhibit. Means over 16 redrawn panels for each history length;
40 series and four factors. Left: relative Frobenius error of each estimator against the
population covariance. Right: annualised volatility, under the population covariance, of the
fully invested minimum-variance portfolio computed from each estimator; the dashed line is the
volatility of the population-optimal portfolio, 4.4%. Blue circles: assembled factor covariance
with LASSO loadings at `reg_lambda` $= 10^{-5}$. Orange squares: sample covariance. Select the
image for the full-resolution view.

| Months | Relative error: assembled, sample | Condition number: assembled, sample | Minimum-variance volatility: assembled, sample |
|---|---|---|---|
| 48 | 0.323, 0.342 | 161, 6594 | 4.94%, 10.06% |
| 72 | 0.307, 0.321 | 161, 870 | 4.70%, 6.46% |
| 120 | 0.216, 0.229 | 141, 350 | 4.57%, 5.33% |
| 240 | 0.140, 0.149 | 121, 185 | 4.46%, 4.74% |

The Frobenius errors are close at every length. Most of that error is the sampling error of the
$4 \times 4$ factor covariance, scaled by the loadings, and both estimators carry it. The
difference is in the small eigenvalues, which the Frobenius norm hardly sees and the inverse
magnifies. With 48 months the sample covariance has a condition number of 6594. The
minimum-variance weights $\hat\Sigma^{-1} \mathbf{1} / (\mathbf{1}^{\top} \hat\Sigma^{-1} \mathbf{1})$
load on its smallest, noise-dominated eigenvectors, and the portfolio realises 10.1% where 4.4%
was attainable. The assembled matrix keeps its condition number near 150 and loses half a
percentage point. At 240 months the two are close. The minimum-variance portfolio here is a
closed-form diagnostic of the inverse, not an allocation method of this package.

## Implementation in factorlasso

All names below are exported from the top-level package and documented in the
[API reference](api.rst).

| Public name | Role |
|---|---|
| `CurrentFactorCovarData` | Frozen dataclass for one snapshot. `get_y_covar`, `get_residual_covar` and the property `y_covar` assemble; `get_model_vols`, `estimate_alpha` and `get_snapshot` report; `filter_on_tickers` subsets or renames responses; `save` and `load` write and read one Excel workbook. |
| `RollingFactorCovarData` | Dated snapshots. `add`, `get_latest`, indexing by date and iteration over dates; `get_y_covars`, `get_residual_covars`, `get_x_covars`, `get_y_betas` return dictionaries by date; `get_residual_vars`, `get_ewma_vars`, `get_r2`, `get_systematic_vars`, `get_total_vols`, `get_residual_vols`, `get_alphas`, `get_factor_var` and `get_beta` return panels. |
| `VarianceColumns` | String enum of the `y_variances` column names: `ewma_var`, `residual_var`, `insample_alpha`, `r2`, `stat_alpha`, `total_vol`, `sys_vol`, `resid_vol`, `cluster`. |
| `ResidualType` | `ORTHOGONAL` for a diagonal residual block, `EMPIRICAL` for a residual correlation scaled by current residual volatilities. |
| `get_clusters_by_freq`, `get_linkages_by_freq`, `get_linkage_array`, `get_cutoffs_by_freq` | Recover per-frequency cluster labels, SciPy linkage arrays and cut distances from a snapshot. |

<!-- fragment -->
```python
import factorlasso as fl

model, covar_data = estimate_covar_data(x, y)
covar_data.get_y_covar()                              # beta Sigma_x beta' + D
covar_data.get_y_covar(residual_var_weight=0.0)       # systematic part only
covar_data.get_y_covar(assets=["asset_01", "asset_02"])
covar_data.get_model_vols()                           # total_vol, sys_vol, resid_vol
covar_data.save("factor_covar.xlsx")
restored = fl.CurrentFactorCovarData.load("factor_covar.xlsx")
```

This fragment assumes factor and response panels `x` and `y` and the function above. The
runnable version is the canonical script
[examples/docs/factor_covariance_assembly.py](../examples/docs/factor_covariance_assembly.py),
which needs only the core dependencies, runs offline, writes no file, and asserts every number in
the tables above:

```console
python examples/docs/factor_covariance_assembly.py
```

`get_residual_covar` raises `ValueError` when the indices of `y_betas` and `y_variances`
disagree, when the empirical mode is requested without a prepared residual correlation, and when
`residual_corr_weight` is outside $[0, 1]$ or differs from 1 in the orthogonal mode.
`get_snapshot` raises `ValueError` naming the missing column when `y_variances` lacks `r2` or
`insample_alpha`, and `estimate_alpha` raises `ValueError` when no residuals are stored.

The numbers in this article were produced with factorlasso 0.20.0.dev2, CVXPY 1.9 and CLARABEL on
Python 3.12. Figure 1 is regenerated from the same script by the documentation analytics runner
described in the [documentation standard](documentation_standard.md); its
[provenance record](images/analytics_manifest.json) holds the configuration, the source identity
and the hash of the image.

## Interpretation and limitations

- **The units are the caller's.** The containers multiply and add what they are given. A factor
  covariance in annual units with residual variances in monthly units assembles without an
  error and is wrong.
- **The factor covariance is the dominant error.** With few factors the error of $\hat\Sigma_y$
  in a matrix norm is mostly the error of $\hat\Sigma_x$. A longer or better factor covariance
  estimate helps more than a better penalty.
- **A diagonal $D$ is an assumption.** Omitted common variation makes the assembled matrix
  understate correlations between the responses that share it. Test with
  [residual diagnostics](residual_diagnostics.md) before relying on the default mode.
- **Shrunk loadings understate systematic risk.** A strong penalty biases loadings toward zero,
  the residual variance rises to compensate, and total variance is roughly preserved while
  correlations fall. The example uses a weak penalty for that reason.
- **In-sample residual variances are optimistic.** `ss_res` is not corrected for the loadings
  fitted per response; with $k$ kept loadings and $T$ observations the downward bias is of order
  $k/T$.
- **Held snapshots age.** Between estimation dates the rolling container returns the last
  snapshot unchanged, including its factor covariance.
- **The exhibit is one generating process** with an exact four-factor structure. When the factor
  set misses a common driver, the advantage over the sample covariance shrinks and can reverse
  for the pairs the missing driver connects.

## See also

- [Quickstart](quickstart.md) for the assembly at the end of a full workflow.
- [Sparse factor model](sparse_factor_model.md) for $\hat\beta$, `ss_res` and the weights behind
  them.
- [Residual diagnostics](residual_diagnostics.md) for the test of a diagonal $D$.
- [Group penalties: HCGL and FCGL](group_penalties_hcgl_fcgl.md) for the clusters stored with a
  snapshot.
- [Task guides](task-guides.rst) for the empirical residual correlation and rolling estimation.
- [API reference](api.rst) for signatures.

## References

- Fan, J., Fan, Y., and Lv, J. (2008). High dimensional covariance matrix estimation using a
  factor model. *Journal of Econometrics* 147(1), 186-197.
- Ledoit, O., and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance
  matrices. *Journal of Multivariate Analysis* 88(2), 365-411.
  DOI 10.1016/S0047-259X(03)00096-4.
- Rosenberg, B., and McKibben, W. (1973). The prediction of systematic and specific risk in
  common stocks. *Journal of Financial and Quantitative Analysis* 8(2), 317-333.
  DOI 10.2307/2330027.
- Sepp, A., Hansen, E., and Kastenholz, M. (2026). Capital market assumptions and strategic asset
  allocation using multi-asset tradable factors. *The Journal of Portfolio Management*,
  forthcoming.
- Sepp, A., Ossa, I., and Kastenholz, M. (2026). Robust optimization of strategic and tactical
  asset allocation for multi-asset portfolios. *The Journal of Portfolio Management* 52(4),
  86-120.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
