---
myst:
  html_meta:
    description: >-
      Named factors, weighted conditioning, declared loading budgets and experimental
      leave-one-out cluster budgets in factorlasso, with auditable floor contributions.
---

# Expert inputs

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Expert inputs name the exposures that a response should retain and may limit the
competing loadings that can explain its small-regression slope. They determine the
prior, its slack and its lower magnitude bound.

## Overview

The estimator accepts no named factor, one factor, or an ordered pair per response.
Factor identities and budgets serve different roles. A named set specifies the
small regression that estimates the prior. Budgets limit displacement by omitted
competitors. A zero budget asserts no exposure to that competitor.

| Expert provides | Named set and prior | Competitors | Budget $d_j$ |
|---|---|---|---|
| Nothing | Factor with the highest marginal $R^2$, marginal slope | Every $j$ with $(\operatorname{sign}(c)\Gamma_j s_j)_+>0$ | Data rule |
| One named factor | That factor, marginal slope | As above | Data rule |
| One named factor and budgets | That factor, marginal slope | As above | Declared where given (0 asserts no exposure), data rule for the rest |
| Two named factors | Both, joint slopes, if the largest VIF within the pair is at most the threshold. Otherwise the first stays named and the second becomes a competitor | As above, per named factor | Data rule, or declared where given |

## Inputs, notation, and assumptions

The factor panel has $T$ observations and $M$ columns. Responses have the same
ordered index. Returns and frequencies are supplied by the caller. No
annualisation is performed. Budgets are in competitor-loading units, so changing
a factor's units requires changing its budget. The synthetic example below uses
180 dimensionless monthly observations, uniform weights, the default 12-observation
warmup and no missing data. In actual estimation, missing rows retain their original
calendar positions in exponentially weighted moving average (EWMA) weights.

The sign $s_j$ is the solver-facing constraint. An unconstrained sign uses
$\lvert\Gamma_j\rvert$ as its displacement weight. A zero sign contributes zero.

## Methodology

For a named prior slope $c$, the budget slack and magnitude floor are

$$
\kappa = \sum_j (\operatorname{sign}(c)\Gamma_j s_j)_+d_j,
\qquad f = \max(0,\lvert c\rvert-\kappa).
$$

Here $\Gamma_j$ is the coefficient of the named factor in the weighted regression
of competitor $j$ on the named set. With two named factors, each has its own
coefficient and contribution sum. The bounds implement the omitted-variable
identity described by Sepp (2026).

The largest variance inflation factor (VIF) is the largest diagonal entry of
the inverse weighted correlation matrix of the named pair. The calculation uses
the complete rows and original-calendar weights of that response's joint prior.
At the default threshold 2.5, a larger VIF demotes the second factor to a competitor.
Singular pairs have infinite VIF. The first-listed factor retains its marginal
slope. The estimator warns once with the demoted responses.

The two-factor limit and threshold are modelling choices. They constrain the
conditioning of the small regression and keep the specified mandate interpretable.
They do not verify factor identity. A well-conditioned but wrongly named pair can
still impose the wrong exposures.

`n_std` uses a multiple of the heteroskedasticity and autocorrelation consistent
(HAC) standard error when there is no declared row. Within a declared row, finite
cells override the data rule, including zero. Missing cells become zero and warn
as `undeclared_zero` when their displacement weight is positive. An entirely
missing row retains standard-error slack for compatibility.

`cluster_mean` first fits the same configured model with bounds off, OLS priors
off and zero centres. It retains the configured signs, clusters and adaptive
weights. For response $i$ in cluster $C_i$, the data budget is

$$
d_{ij} = \frac{1}{\lvert C_i\rvert-1}
\sum_{m\in C_i,\ m\ne i} \lvert\widehat\beta_{mj}^{(0)}\rvert.
$$

Finite declarations replace individual cells. Clusters below three members fall
back to standard-error slack and report `cluster_too_small`. The multiplier is
`expert_prior_bound_n_std`, or one when that parameter is unset. The rule requires
GROUP_LASSO, HCGL or FCGL with a response partition.

## Worked example

In this synthetic example every response has a competitor loading of 0.4 and the
factor displacement coefficient is 0.35. The independent peer-mean check in
the example recovers a budget of 0.4 and slack of 0.14 to solver tolerance.

```python
    model = LassoModel(
        model_type=LassoModelType.GROUP_LASSO,
        group_data=pd.Series('cluster', index=y.columns),
        apply_ols_prior=True,
        factor_for_prior={name: 'a' for name in y.columns},
        expert_prior_budget_rule='cluster_mean',
        expert_prior_bound_centre='floor',
        reg_lambda=0., loss_normalization='weight_sum',
    ).fit(x=x, y=y)
```

## Implementation in factorlasso

`LassoModel` adds `max_named_factors=2`, `named_set_max_vif=2.5` and
`expert_prior_budget_rule='n_std'`. Existing budget-rule and penalty-centre defaults
remain `n_std` and `prior`. The new conditioning guard deliberately changes pairs
above its threshold. Set `named_set_max_vif=None` for legacy joint-prior behaviour.
More than two explicit factors raises an error even when conditioning is disabled.

`budget_first_pass_coef_` stores the first-pass loadings. `resolved_budgets_` and
`budget_sources_` show each resolved cell and its provenance.
`named_set_diagnostics_` preserves requested and effective sets and the original
largest VIF. `prior_bound_diagnostics_` retains its existing columns.
`prior_bound_contributions_` adds one row per response, named factor and competitor
with slope, sign, budget, contribution, slack, floor, source and binding status.
The binding flag uses the signed distance from the fitted loading to an imposed
floor and a tolerance of $10^{-6}$.

For `n_std`, an implied one-SE budget is reported separately for each permitted
competitor. This is not a unique additive allocation of slack. Contributions are
missing for this source. Declared and cluster budgets have additive contributions.
Cluster budgets depend on the penalty, so each regularisation-path point recomputes
its first pass.

Reproduce with [the canonical offline script](../examples/docs/expert_inputs.py).
The script uses NumPy, pandas and the installed factorlasso package.

## Interpretation and limitations

Cluster means are experimental limits, not upper confidence bounds or guaranteed
true exposures. Their quality depends on the first-pass model and peer grouping.
Declared zeros are substantive assumptions and can tighten a floor sharply.
Standard-error uncertainty describes the selected regression, rather than coverage
of the penalised full-model coefficient. The floor construction does not remove
selection bias or protect against incorrectly named factors.

## See also

- [Prior targets](prior_targets.md)
- [Sign constraints and priors](sign_constraints_and_priors.md)
- [Group penalties](group_penalties_hcgl_fcgl.md)

## References

- Sepp, A. (2026). *Exposure-budget floors for factor loadings under correlated factors*.
  Working paper, link to be added.
- Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica*, 55, 703–708. [doi:10.2307/1913610](https://doi.org/10.2307/1913610).
- Software citation metadata are in [CITATION.cff](../CITATION.cff).
