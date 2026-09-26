---
myst:
  html_meta:
    description: >-
      factorlasso quickstart: select the penalty by cross-validation, fit a hierarchical-cluster
      group LASSO with derived sign constraints, read loadings, clusters and signs, test the
      residuals, and assemble the factor covariance matrix, on one reproducible synthetic panel.
---

# Quickstart: from return panels to a factor covariance

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-22](https://github.com/ArturSepp/factorlasso/commit/fe2063860f701ac5a71951161bf391a1285d503d)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

This page runs the package once from end to end on a synthetic panel with known loadings: select
the penalty, fit the loadings, read the fit, test the residuals, and assemble the covariance
matrix. Each step names the article that defines the method. For installation and the smallest
possible fit, start with [installation and first fit](getting-started.md).

## The model and the five steps

For $N$ response series $Y_t$ and $M$ factor series $X_t$ observed over $T$ periods, the package
estimates

$$
Y_t = \alpha + \beta X_t + \varepsilon_t,
\qquad
\Sigma_y = \beta \Sigma_x \beta^{\top} + D,
$$

where $\beta$ is the $N \times M$ loading matrix, $\Sigma_x$ the factor covariance and $D$ the
residual covariance, diagonal by default. The loadings minimise a weighted squared loss plus a
penalty that removes loadings the data do not support:

$$
\hat\beta = \arg\min_{\beta \in \mathcal{C}}
\frac{1}{T} \lVert W \odot (X \beta^{\top} - Y) \rVert_F^2 + \lambda P(\beta - \beta_0).
$$

$W$ holds the observation weights, $\mathcal{C}$ the sign constraints, $\beta_0$ an optional
prior, and $P$ the penalty chosen by `model_type`.

| Step | Question | Public entry point | Method |
|---|---|---|---|
| 1 | How strong should the penalty be? | `LassoModelCV` | Expanding-window cross-validation |
| 2 | Which loadings does each series carry? | `LassoModel` | [Sparse factor model](sparse_factor_model.md), [group penalties](group_penalties_hcgl_fcgl.md) |
| 3 | Which signs are admissible? | `auto_sign_constraints`, `factors_beta_loading_signs` | [Sign constraints and priors](sign_constraints_and_priors.md) |
| 4 | Is anything systematic left in the residuals? | `diagnose_residuals` | [Residual diagnostics](residual_diagnostics.md) |
| 5 | What covariance matrix does the model imply? | `CurrentFactorCovarData` | [Factor covariance assembly](factor_covariance_assembly.md) |

## Settings of the example

| Setting | Value |
|---|---|
| Data | Synthetic, seed 20260921; not market or paper-replication evidence |
| Sample | 120 month-end observations from 2016-01-31 |
| Factors | equity, rates, credit, commodity; monthly volatilities 0.045, 0.015, 0.020, 0.050; equity and credit 60% correlated |
| Responses | 12 funds in three groups of four: equity funds, bond funds, real-asset funds |
| True loadings | 20 of 48 cells are non-zero; every group shares one support |
| Idiosyncratic volatility | 0.015, 0.005 and 0.025 per month by group |
| Returns | Decimal per-period returns; the covariance step annualises by 12 |
| Estimator | `HIERARCHICAL_CLUSTER_GROUP_LASSO`, `auto_sign_constraints=True`, uniform observation weights, CLARABEL |
| Penalty grid | Nine values from $10^{-6}$ to $10^{-2}$, four expanding-window folds |

The generating loadings are known, so every estimate below can be compared with the truth. The
panel is drawn by `make_panel`:

```python
def make_panel(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw the factor panel ``x`` (T x 4) and the response panel ``y`` (T x 12)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-31", periods=N_OBS, freq="ME")
    factor_covar = FACTOR_CORR * np.outer(FACTOR_VOL, FACTOR_VOL)
    x = pd.DataFrame(
        rng.multivariate_normal(np.zeros(len(FACTOR_NAMES)), factor_covar, size=N_OBS),
        index=dates,
        columns=FACTOR_NAMES,
    )
    names = [f"{group}_{i + 1}" for group in GROUP_NAMES for i in range(GROUP_SIZE)]
    noise = RESIDUAL_VOL * rng.standard_normal((N_OBS, len(names)))
    y = pd.DataFrame(x.to_numpy() @ TRUE_BETA.T + noise, index=dates, columns=names)
    return x, y
```

`x` is a $T \times M$ DataFrame of factor returns and `y` a $T \times N$ DataFrame of response
returns on the same index. `fit` also accepts NumPy arrays and a response panel with missing
values.

## Steps 1 to 3: select the penalty and fit

`LassoModelCV` takes a template `LassoModel`, fits it on every fold and penalty, and refits the
penalty with the highest mean held-out $R^2$ on the full sample. The template here discovers
clusters of responses from their correlation matrix, applies the row-grouped HCGL penalty, and
derives the sign of each loading from univariate slopes pooled within a cluster.

```python
def select_and_fit(x: pd.DataFrame, y: pd.DataFrame) -> fl.LassoModelCV:
    """Select ``reg_lambda`` by out-of-sample R-squared and refit on the full sample."""
    template = fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        auto_sign_constraints=True,        # signs from cluster-pooled univariate slopes
    )
    selector = fl.LassoModelCV(
        lambdas=PENALTY_GRID,
        n_splits=N_SPLITS,
        base_model=template,
        use_lambda_path=True,              # one canonical form per fold, swept over the grid
    )
    return selector.fit(x=x, y=y)
```

The fitted selector and its refitted model expose the results as attributes with a trailing
underscore. The values annotated below are from the committed reference run:

<!-- fragment -->
```python
selector = select_and_fit(x, y)
selector.best_lambda_          # 1e-05
selector.cv_scores_            # penalties by folds, held-out R-squared
model = selector.best_model_   # LassoModel refitted on all 120 months

model.coef_                    # (12, 4) loadings, responses by factors
model.alpha_const_             # (12,) regression intercepts in the units of y
model.clusters_                # cluster label of each response
model.derived_signs_           # (12, 4) sign matrix the solver received
model.predict(x)               # fitted responses, alpha + x @ beta'
model.score(x, y)              # mean R-squared across responses: 0.875
```

[![Three panels. Left and middle: heatmaps of the true and the estimated loadings of twelve funds on four factors, with matching block structure. Right: mean held-out R-squared along the penalty grid, flat near 0.85 for penalties below 1e-4 and falling for larger penalties, with the selected penalty 1e-5 marked.](images/quickstart_workflow.png)](images/quickstart_workflow.png)

**Figure 1.** Synthetic teaching exhibit. Left: the loadings that generated the panel. Middle:
the loadings estimated at the selected penalty; cells below 0.005 in magnitude are left blank.
Right: mean held-out $R^2$ over four expanding-window folds, with the range across folds as a
vertical bar, along the penalty grid; the penalty falls to the right. The score is flat at 0.85
for penalties of $3 \times 10^{-5}$ and below, and the maximum, at $10^{-5}$, is the selected
penalty. Select the image for the
full-resolution view.

| Quantity | Value |
|---|---|
| Selected `reg_lambda` | $10^{-5}$, mean held-out $R^2$ of 0.851 |
| In-sample mean $R^2$ | 0.875 |
| Clusters discovered | 3, equal to the three generating groups |
| Derived signs | One row per cluster; rates is constrained non-positive for equity and real-asset funds and non-negative for bond funds; every other cell is non-negative |
| Loading RMSE against the truth | 0.053, against 0.091 for ordinary least squares |
| Kept loadings | 34 of 48 at the default tolerance, against 20 in the generating matrix |

The figure and table report one CLARABEL run. Small differences in the solver and linear-algebra
libraries across platforms can change the selected penalty along the nearly flat cross-validation
curve, and with it the loading error and number of kept loadings. The executable example checks
that the selected penalty maximises its own score table, the fitted model improves loading RMSE
over ordinary least squares, and the algebraic covariance identity holds on the running platform.

Three observations follow from the table and the figure.

The sign constraints do part of the selection. Credit is 60% correlated with equity, and
unconstrained least squares gives the equity funds negative credit loadings between -0.11 and
-0.22. The pooled univariate slope of an equity fund on credit is positive, so the derived sign is
non-negative and the fitted credit loadings of the equity funds are exactly zero.

The penalty selected by held-out $R^2$ keeps more loadings than the generating model has. This is
a known property of selection by prediction error: small false loadings cost little out of sample.
The fourteen extra loadings here are all below 0.24 in magnitude, and twelve are below 0.1. A
support closer to the truth needs a larger penalty, a stricter sign gate, or a penalty that
selects whole cluster-by-factor blocks; [group penalties](group_penalties_hcgl_fcgl.md) compares
them.

Loadings can be wrong where the data are weak. The fourth real-asset fund has a true equity
loading of 0.20 and an estimate of 0.05, with a false credit loading of 0.23 in its place:
equity and credit are correlated, and the real-asset funds have the largest idiosyncratic
volatility of the panel.

## Step 4: test the residuals

The covariance decomposition assumes that what the factors leave behind is uncorrelated across
series. `diagnose_residuals` tests that on the residual panel, with the degrees of freedom
reduced by the mean number of loadings the fit kept:

```python
def check_residuals(
    model: fl.LassoModel,
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> fl.ResidualDiagnostics:
    """Test the in-sample residuals for a diagonal covariance, charging the loadings kept."""
    residuals = y - model.predict(x)
    sparsity = fl.effective_sparsity(model.coef_)          # the solver returns no exact zeros
    return fl.diagnose_residuals(residuals, n_fitted_per_asset=sparsity.per_asset)
```

The sphericity statistic is 74.7 against a 5% threshold of 86.0 for 66 pairs of series, and the
largest eigenvalue of the residual correlation matrix is 1.61 against a Marchenko-Pastur edge of
1.75. The residuals are consistent with a diagonal covariance.
[Residual diagnostics](residual_diagnostics.md) shows the same test failing when a factor is
withheld.

## Step 5: assemble the covariance

`CurrentFactorCovarData` stores the inputs of $\Sigma_y$ and assembles the matrix on request. The
package does not annualise: the caller chooses the units of the factor covariance and of the
residual variances, and the assembled matrix inherits them. The example multiplies both by 12.

```python
def assemble_covariance(model: fl.LassoModel, x: pd.DataFrame) -> fl.CurrentFactorCovarData:
    """Collect loadings, factor covariance and residual variances, all annualised."""
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
    return fl.CurrentFactorCovarData(
        x_covar=PERIODS_PER_YEAR * x.cov(ddof=0),
        y_betas=model.coef_,
        y_variances=y_variances,
        estimation_date=x.index[-1],
        clusters=model.clusters_,
        derived_signs=model.derived_signs_,
    )
```

<!-- fragment -->
```python
covar_data = assemble_covariance(model, x)
covar_data.get_y_covar()       # (12, 12) annualised covariance, positive definite
covar_data.get_model_vols()    # total, systematic and residual volatility per response
covar_data.get_snapshot()      # loadings, R-squared, alpha and volatilities in one table
```

| Response | Total volatility | Systematic | Residual |
|---|---|---|---|
| `equity_fund_1` | 0.166 | 0.158 | 0.051 |
| `bond_fund_1` | 0.062 | 0.060 | 0.017 |
| `real_asset_fund_1` | 0.217 | 0.200 | 0.084 |

Volatilities are annualised decimals, and the squared total is the sum of the squared systematic
and residual parts. The relative Frobenius error of the assembled matrix against the population
covariance is 0.159, against 0.166 for the sample covariance. With twelve series and 120 months
the two are close, because most of the error is the sampling error of the 4 by 4 factor
covariance, which both share. The difference becomes material when the number of series
approaches the number of observations:
[factor covariance assembly](factor_covariance_assembly.md) measures it for 40 series.

## Run it

The page is backed by the canonical script
[examples/docs/quickstart.py](../examples/docs/quickstart.py).
It needs only the core dependencies, runs offline in a few seconds, and asserts every number
quoted above against a reference computed a different way: predictions and $R^2$ from NumPy, the
selected penalty from the score table, the assembled matrix from a direct matrix product, and
the estimation errors from the generating loadings.

```console
python examples/docs/quickstart.py
```

The numbers on this page were produced with factorlasso 0.20.0.dev2, CVXPY 1.9 and CLARABEL on
Python 3.12. Figure 1 is regenerated from the same script by the documentation analytics runner
described in the [documentation standard](documentation_standard.md); its
[provenance record](images/analytics_manifest.json) holds the configuration, the source identity
and the hash of the image.

## Where next

| To | Read |
|---|---|
| Understand the objective, the intercepts and the units of `reg_lambda` | [Sparse factor model](sparse_factor_model.md) |
| Impose economic signs or shrink toward a prior loading matrix | [Sign constraints and priors](sign_constraints_and_priors.md) |
| Estimate the prior centres from the fitting window | [Prior targets](prior_targets.md) |
| Choose between the LASSO, HCGL, sparse-group and FCGL penalties | [Group penalties](group_penalties_hcgl_fcgl.md) |
| Store, subset and roll the covariance decomposition | [Factor covariance assembly](factor_covariance_assembly.md) |
| Test the diagonal-residual assumption and name a missing factor | [Residual diagnostics](residual_diagnostics.md) |
| Look up shapes, units, EWMA spans and the precedence of signs and priors | [Conventions and glossary](conventions.md) |
| See every exhibit with its sample and producer | [Analytics gallery](analytics_gallery.md) |
| Read a case study from the JSS paper | [Credit attribution in a multi-asset ETF model](app_multi_asset_credit_attribution.md) |
| Find the script and article for a task | [Examples and recipes](task-guides.md) |
| Look up a signature | [API reference](api.rst) |
