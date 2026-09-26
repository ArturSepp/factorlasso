---
myst:
  html_meta:
    description: >-
      Examples and recipes for factorlasso: every runnable script with what it shows and the
      article that explains it, and short checked recipes for constrained fits, automatic prior
      centres, cluster-aware estimation, causal rolling partitions and covariance assembly.
---

# Examples and recipes

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/factorlasso/commit/c89cf6d380358d4744592feeccb6073d989bc004)*

Examples of [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Every script listed here runs offline from a checkout, on synthetic data or on data committed to
the repository, and every one is executed by the test suite. The first table routes from a task to
the article and script that treat it; the recipes below are short, checked starting points.

## Scripts by article

Each article has one canonical script in `examples/docs/`; its code blocks are excerpts of that
script, and the script asserts every number the article quotes.

| Task | Article | Script |
|---|---|---|
| Install and run a first fit | [Installation and first fit](getting-started.md) | `getting_started.py` |
| Go from return panels to a factor covariance | [Quickstart](quickstart.md) | `quickstart.py` |
| Fit sparse loadings along a penalty path | [Sparse multi-output factor model](sparse_factor_model.md) | `sparse_factor_model.py` |
| Weight observations and handle ragged histories | [EWMA weighting](ewma_weighting_and_ragged_histories.md) | `ewma_weighting_and_ragged_histories.py` |
| Impose signs and centre the penalty on priors | [Sign constraints and priors](sign_constraints_and_priors.md) | `sign_constraints_and_priors.py` |
| Derive signs from pooled univariate evidence | [Gated sign derivation](gated_cluster_pooled_signs.md) | `gated_cluster_pooled_signs.py` |
| Set prior centres from OLS fits or an expert map | [Prior targets](prior_targets.md) | `prior_targets.py` |
| Scale penalties by univariate evidence | [Adaptive penalty weights](adaptive_penalty_weights.md) | `adaptive_penalty_weights.py` |
| Choose a group penalty | [Group penalties](group_penalties_hcgl_fcgl.md) | `group_penalties_hcgl_fcgl.py` |
| Favour sign-coherent clusters without imposing signs | [Cooperative LASSO](cooperative_lasso.md) | `cooperative_lasso.py` |
| Fit univariate-guided loadings | [UniLasso](unilasso.md) | `unilasso.py` |
| Select the penalty | [Penalty selection](penalty_selection.md) | `penalty_selection.py` |
| Test residuals for a missing factor | [Residual diagnostics](residual_diagnostics.md) | `residual_diagnostics.py` |
| Discover clusters of responses | [Cluster discovery](cluster_discovery.md) | `cluster_discovery.py` |
| Remove a dominant common mode before clustering | [Common-mode removal](common_mode_removal.md) | `common_mode_removal.py` |
| Stabilise rolling partitions | [Rolling cluster smoothing](rolling_cluster_smoothing.md) | `rolling_cluster_smoothing.py` |
| Weigh evidence by cluster stability | [Cluster stability and pooled scoring](cluster_stability_and_pooled_scoring.md) | `cluster_stability_and_pooled_scoring.py` |
| Give rolling clusters persistent names | [Offline cluster lineage](cluster_lineage.md) | `cluster_lineage.py` |
| Assemble the factor covariance | [Factor covariance assembly](factor_covariance_assembly.md) | `factor_covariance_assembly.py` |
| Add residual correlation to the covariance | [Empirical residual correlation](empirical_residual_correlation.md) | `empirical_residual_correlation.py` |
| Nowcast responses from realised factors | [Residual-alpha nowcasting](residual_alpha_nowcasting.md) | `residual_alpha_nowcasting.py` |
| Keep credit exposure with prior-centred penalties | [Credit attribution case study](app_multi_asset_credit_attribution.md) | `app_multi_asset_credit_attribution.py` |
| Read the yeast eQTL application | [Sign pooling case study](app_sign_pooling_genomics.md) | `app_sign_pooling_genomics.py` |
| Build risk and CMAs from one loading matrix | [Portfolio risk and CMAs case study](app_portfolio_risk_models.md) | `app_portfolio_risk_models.py` |

## Standalone examples

The scripts in `examples/` predate the articles and show complete workflows:

| Script | Shows |
|---|---|
| `examples/finance_factor_model.py` | A multi-asset model with sign constraints, the covariance assembly and its diagnostics. |
| `examples/genomics_factor_model.py` | Gene expression on pathway factors with biological sign constraints. |
| `examples/cv_lambda_selection.py` | Expanding-window selection of `reg_lambda` with `LassoModelCV`. |
| `examples/ols_prior.py` | Automatic OLS prior centres at the loss span, with forced-zero exposures. |
| `examples/alpha_const_vs_intercept.py` | The economic intercept `alpha_const_` against the solver's `intercept_`. |

Run any of them from a checkout, for example:

```console
python examples/docs/task_guides.py
```

## Recipes

The recipes are excerpts of
[`examples/docs/task_guides.py`](../examples/docs/task_guides.py), which checks structural
contracts rather than solver-sensitive decimals.

### Hard signs and a prior centre

In `factors_beta_loading_signs`, 1 constrains a loading to be non-negative, $-1$ non-positive, 0
fixes it at zero and `NaN` leaves it free; `factors_beta_prior` centres the penalty on a supplied
loading. Rows are responses and columns factors in both.

```python
def constrained_fit() -> tuple[fl.LassoModel, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hard signs and a prior centre on three responses, one with a late start."""
    rng = np.random.default_rng(12)
    index = pd.date_range("2020-01-31", periods=80, freq="ME")
    x = pd.DataFrame(rng.normal(size=(80, 2)), index=index, columns=["growth", "rates"])
    beta = np.array([[0.7, -0.2], [-0.4, 0.0], [0.1, 0.6]])
    y = pd.DataFrame(x.to_numpy() @ beta.T + 0.04 * rng.normal(size=(80, 3)),
                     index=index, columns=["asset_a", "asset_b", "asset_c"])
    y.loc[index[:8], "asset_c"] = np.nan
    signs = pd.DataFrame([[1.0, np.nan], [-1.0, 0.0], [np.nan, 1.0]],
                         index=y.columns, columns=x.columns)
    prior = pd.DataFrame(0.0, index=y.columns, columns=x.columns)
    prior.loc["asset_a", "growth"] = 0.5
    model = fl.LassoModel(
        reg_lambda=1e-3,
        span=20,
        warmup_period=12,
        factors_beta_loading_signs=signs,
        factors_beta_prior=prior,
    ).fit(x=x, y=y)
    return model, x, y, signs
```

The fitted loadings respect every constraint, the late-starting response is masked rather than
imputed (`valid_mask_`), and `derived_signs_` holds the sign matrix the solver received. See
[sign constraints and priors](sign_constraints_and_priors.md) and
[EWMA weighting and ragged histories](ewma_weighting_and_ragged_histories.md).

### Automatic prior centres

```python
def automatic_prior_fit(x: pd.DataFrame, y: pd.DataFrame, signs: pd.DataFrame) -> fl.LassoModel:
    """OLS prior centres from the highest-R-squared factor, at the loss span of the fit."""
    return fl.LassoModel(
        span=20, apply_ols_prior=True, prior_selection_type="highest_r2",
        factors_beta_loading_signs=signs,
    ).fit(x=x, y=y, span=16)
```

The centres use the span of the fit, 16 here because `fit(span=...)` overrides the model's span;
`ols_betas_` and `ols_r2_` keep the univariate regressions, and a centre that conflicts with a
hard zero is set to zero in `effective_beta_prior_`. See [prior targets](prior_targets.md).

### A discovered partition

```python
def two_cluster_fit() -> fl.LassoModel:
    """HCGL asked to discover at most two groups of four responses."""
    rng = np.random.default_rng(23)
    x = pd.DataFrame(rng.normal(size=(90, 2)), columns=["market", "style"])
    common_1, common_2 = rng.normal(size=90), rng.normal(size=90)
    y = pd.DataFrame({"a": common_1 + 0.15 * rng.normal(size=90),
                      "b": common_1 + 0.15 * rng.normal(size=90),
                      "c": common_2 + 0.15 * rng.normal(size=90),
                      "d": common_2 + 0.15 * rng.normal(size=90)})
    return fl.LassoModel(
        model_type=fl.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        n_clusters=2,
        reg_lambda=1e-3,
        warmup_period=12,
    ).fit(x=x, y=y)
```

`clusters_`, `linkage_` and `cutoff_` hold the partition and its dendrogram. See
[cluster discovery](cluster_discovery.md) and [group penalties](group_penalties_hcgl_fcgl.md).

### Rolling partitions that ignore later data

`compute_rolling_smoothed_clusters` truncates the panel at every estimation date; changing every
observation after the last date leaves the partitions unchanged, which the recipe checks. See
[causal smoothing of rolling clusters](rolling_cluster_smoothing.md); persistent names across
dates are the subject of the [offline cluster lineage](cluster_lineage.md), which is not causal.

### Covariance assembly

```python
def assembled_covariance() -> tuple[pd.DataFrame, np.ndarray]:
    """B Sigma_F B' + D from a snapshot, and the same matrix by NumPy."""
    factor_covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]],
                                     index=["growth", "rates"], columns=["growth", "rates"])
    loadings = pd.DataFrame([[1.0, 0.2], [0.4, -0.5]],
                            index=["asset_a", "asset_b"], columns=factor_covariance.columns)
    residual_variance = pd.Series([0.02, 0.03], index=loadings.index)
    snapshot = fl.CurrentFactorCovarData(
        x_covar=factor_covariance,
        y_betas=loadings,
        y_variances=pd.DataFrame({fl.VarianceColumns.RESIDUAL_VARS.value: residual_variance}),
    )
    independent = (loadings.to_numpy() @ factor_covariance.to_numpy() @ loadings.to_numpy().T
                   + np.diag(residual_variance.to_numpy()))
    return snapshot.get_y_covar(), independent
```

The container does not convert units: the factor covariance, loadings and residual variances must
share one frequency and scaling. See [factor covariance assembly](factor_covariance_assembly.md)
and [empirical residual correlation](empirical_residual_correlation.md).

## See also

- [Analytics gallery](analytics_gallery.md): every exhibit with its question, sample and script.
- [API reference](api.rst): every public name, grouped by the article that owns it.
- [Conventions and glossary](conventions.md).
