---
myst:
  html_meta:
    description: >-
      Cell-level sign constraints and prior-centred penalties in factorlasso: a sign matrix with
      non-negative, non-positive, zero and free cells, a penalty that shrinks loadings toward a
      prior matrix instead of zero, and a worked example with two correlated factors.
---

# Sign constraints and prior-centred penalties

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

A sign matrix restricts each loading of the $N \times M$ loading matrix separately: non-negative,
non-positive, exactly zero, or free. A prior matrix moves the centre of the penalty from zero to
a stated loading matrix, so that a strong penalty returns the prior and not an empty model. Both
encode what is known about the responses before the data are seen, and both act inside one
convex programme with the loss and the penalty.

## Overview

Factor loadings of financial series are rarely unknown in sign. A government bond fund has a
non-negative loading on a duration factor and none on equity. An investment-grade credit fund
has a non-negative credit loading. With short histories and correlated factors, an unconstrained
regression does not respect any of this: it moves exposure between correlated factors and
reports loadings that an analyst would reject on sight.

Two inputs carry such knowledge into the estimator.

`factors_beta_loading_signs` is an $N \times M$ matrix $S$ with entries in the set of $1$,
$-1$, $0$ and NaN. It defines a feasible set for $\beta$ and is a hard constraint. scikit-learn
offers one `positive` flag for all coefficients of a model; a separate rule for every cell of a
multi-output loading matrix needs a constrained programme, which is what the package builds.

`factors_beta_prior` is an $N \times M$ matrix $\beta_0$. The penalty becomes
$\lambda \lVert \beta - \beta_0 \rVert_1$. It is a soft target: the data move a loading away from
its prior when the gain in fit exceeds the penalty. The construction is the penalised-regression
counterpart of shrinking toward an informative reference instead of toward zero, in the spirit of
Black and Litterman (1992) for expected returns.

In the worked example, 36 months and an 85% correlated pair of factors, the sign matrix lowers
the mean loading error over 50 simulated panels from 0.117 to 0.101 and removes every violation,
and the prior lowers it further to 0.077, although the prior itself has an error of 0.118.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $\beta$, `coef_` | Loading matrix | $N \times M$, responses by factors |
| $S$, `factors_beta_loading_signs` | Sign matrix | DataFrame indexed by response, columns by factor; entries $1$, $-1$, $0$ or NaN |
| $\beta_0$, `factors_beta_prior` | Prior loading matrix | DataFrame of the same layout, in the units of $\beta$; NaN is read as zero |
| $\lambda$, `reg_lambda` | Penalty strength | Units of `y` squared; see [sparse factor model](sparse_factor_model.md) |
| `nonneg` | One flag for all cells | `True` constrains every loading to be non-negative when no sign matrix is active |
| `derived_signs_` | The sign matrix the solver received | Fitted attribute; equals $S$ restricted to the fitted responses and factors when signs are supplied only explicitly |

| Entry of $S$ | Constraint on $\beta_{ij}$ |
|---|---|
| $1$ | $\beta_{ij} \ge 0$ |
| $-1$ | $\beta_{ij} \le 0$ |
| $0$ | $\beta_{ij} = 0$ |
| NaN | none |

Both matrices must contain every response of `y` as a row label and every factor of `x` as a
column label. Extra rows and columns are ignored, and a missing label raises `KeyError`. The
constraints are exact statements, not beliefs with a confidence: a wrong sign entry forces a
wrong loading. The prior is a belief, and its weight is the penalty.

## Methodology

### The constrained programme

With the notation of the [sparse factor model](sparse_factor_model.md), the estimator solves

$$
\hat\beta = \arg\min_{\beta \in \mathcal{C}(S)}
\frac{1}{T} \lVert W \odot (\tilde X \beta^{\top} - \tilde Y) \rVert_F^2 +
\lambda \lVert \beta - \beta_0 \rVert_1 ,
$$

over the feasible set

$$
\mathcal{C}(S) = \lbrace \beta : \beta_{ij} \ge 0 \text{ if } S_{ij} = 1, \quad
\beta_{ij} \le 0 \text{ if } S_{ij} = -1, \quad
\beta_{ij} = 0 \text{ if } S_{ij} = 0 \rbrace .
$$

$\mathcal{C}(S)$ is a product of half-lines, points and lines, hence convex, and the programme
stays a convex cone programme. The constraints apply in the `LASSO` mode and in the group and
cluster modes `GROUP_LASSO`, `HIERARCHICAL_CLUSTER_GROUP_LASSO` and `FACTOR_CLUSTER_GROUP_LASSO`.

### What a sign constraint does

A binding constraint sets a loading to zero that the unconstrained fit would make negative, or
positive for $S_{ij} = -1$. It is a selection device as well as a restriction: Meinshausen (2013)
and Slawski and Hein (2013) show that non-negative least squares alone, without any penalty, can
recover a sparse non-negative coefficient vector under conditions on the design. At
$\lambda \to 0$ the estimator here is bounded least squares, which the worked example verifies
against `scipy.optimize.lsq_linear`.

A zero entry removes a factor from one response and not from the others. This is the way to
state that a factor is inadmissible for a series, for example a private-equity factor for a
listed bond fund, without dropping the factor from the panel.

### What a prior-centred penalty does

Substituting $\delta = \beta - \beta_0$ gives, without sign constraints,

$$
\hat\delta = \arg\min_{\delta}
\frac{1}{T} \lVert W \odot (\tilde X \delta^{\top} - (\tilde Y - \tilde X \beta_0^{\top})) \rVert_F^2 +
\lambda \lVert \delta \rVert_1 ,
\qquad
\hat\beta = \beta_0 + \hat\delta .
$$

The prior-centred fit is an ordinary LASSO on the part of the responses that the prior does not
explain. Sparsity now applies to the deviations from the prior: a loading equals its prior
exactly unless the data support a departure. Two limits follow.

As $\lambda \to \infty$, $\hat\beta \to \beta_0$ wherever $\beta_0$ is feasible. With a zero
prior the same limit is the empty model. The prior therefore decides what a strong penalty
means, which matters most when the history is short and the penalty has to be strong.

As $\lambda \to 0$, the prior has no effect and the estimate is bounded least squares.

### When the two disagree

The constraint wins. If $\beta_{0,ij} < 0$ and $S_{ij} = 1$, the loading is pulled toward a
point outside the feasible set and stops at the boundary: under a strong penalty the estimate is
zero, not the prior. The package does not warn about a prior that contradicts its sign matrix in
this explicit form; keep the two consistent.

### Relation to derived signs

`auto_sign_constraints=True` derives marginal slopes within response clusters.
The original response and factor masks exclude pre-inception observations and gaps.
The standalone `derive_sign_constraints(..., ewma_span=None)` retains equal observation
weights. A finite span applies decay on the original row grid before masking. The model
can specify `auto_sign_ewma_span`, or set `auto_sign_use_fit_span=True` to follow the
effective span of each fit, path or cross-validation training fold. The latter is useful
when monthly and quarterly fits have different horizons. These options are mutually exclusive.

For predictor $j$, validity indicator $v_{tkj}$ and weights $w_t$, the pooled slope is

$$
D_j = \sum_{t,k} w_t v_{tkj} x_{tj}^{2}, \qquad
\hat b_j = D_j^{-1}\sum_{t,k} w_t v_{tkj}x_{tj}y_{tk}.
$$

The public default remains `variance_estimator="independent"` for compatibility.
Explicit `variance_estimator="date"` (model: `auto_sign_variance="date"`) sums
response scores within each date before squaring:

$$
u_{tj}=w_t x_{tj}\sum_k v_{tkj}(y_{tk}-\hat b_jx_{tj}), \qquad
\widehat{\mathrm{Var}}(\hat b_j)=\frac{n_{\mathrm{eff},j}}{n_{\mathrm{eff},j}-1}
\frac{\sum_t u_{tj}^{2}}{D_j^{2}}.
$$

Here $n_{\mathrm{eff},j}$ is the Kish effective count of dates with a valid predictor
and at least one response; response copies never increase it. The variance follows
the date-clustered score principle described by
[Cameron and Miller (2015)](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf).
Using effective dates in the finite-sample multiplier is an implementation choice for
recency weighting, not a claim of an exact Student distribution. Fewer than two effective
dates cannot pass a positive threshold. The gate allows contemporaneous dependence;
it assumes independence across dates and does not supply HAC inference. A fixed EWMA
span has bounded effective sample size, so increasing the stored history alone gives
no consistency guarantee. Identical response copies leave the date statistic unchanged.

`"independent"` retains the former equal-weight variance for archived replication.
With a finite span it uses independent-cell score variance; it still omits response
covariance. The sign gate is a screening rule that changes the feasible set, not a
multiple-testing-adjusted significance test. Weighting does not make a marginal slope
equal to a conditional multivariate loading.

Explicit hard signs take precedence. On otherwise unrestricted cells, a finite nonzero
prior supplies its sign; automatic detection supplies the remaining signs and zero gates.
The final solver matrix is `derived_signs_`. `detected_signs_`, `sign_slopes_`,
`sign_t_stats_`, `sign_effective_n_` and `sign_valid_counts_` record the original
detection evidence, including cells subsequently overridden. Adaptive weights use those
same detected slopes; `sign_penalty_weights_` and `sign_block_weights_` expose their
effect on the penalty. Prior overrides do not silently replace these magnitudes.

The repair changes numerical results for unbalanced panels even with `ewma_span=None`.
Loss normalization remains unchanged. The public optimiser default remains CLARABEL;
commercial solver selection is an explicit caller setting.

## Worked example

The example uses synthetic data with a fixed seed. Six bond funds are observed for 36 months:
two government bond funds, two investment-grade funds and two high-yield funds. The factors are
rates, credit and equity with monthly volatilities of 0.015, 0.020 and 0.045 in decimal units;
credit and equity have a population correlation of 0.85 and a sample correlation of 0.91 on
this panel. Idiosyncratic volatility is 0.010 per month.

| Fund | True loadings on rates, credit, equity | Sign row | Prior row |
|---|---|---|---|
| `gov_bond_1` | 1.0, 0.0, 0.0 | 1, 0, 0 | 0.8, 0.0, 0.0 |
| `gov_bond_2` | 0.7, 0.0, 0.0 | 1, 0, 0 | 0.8, 0.0, 0.0 |
| `ig_credit_1` | 0.8, 0.4, 0.0 | 1, 1, 0 | 0.8, 0.5, 0.0 |
| `ig_credit_2` | 0.6, 0.6, 0.0 | 1, 1, 0 | 0.8, 0.5, 0.0 |
| `hy_credit_1` | 0.2, 1.0, 0.1 | 1, 1, NaN | 0.3, 1.0, 0.0 |
| `hy_credit_2` | 0.1, 1.2, 0.2 | 1, 1, NaN | 0.3, 1.0, 0.0 |

The prior is a plausible house view by fund type and is deliberately not the truth: its root
mean squared error against the generating loadings is 0.118. The three estimators differ only in
the two inputs:

```python
def fit_signed_with_prior(
    x: pd.DataFrame,
    y: pd.DataFrame,
    reg_lambda: float = REG_LAMBDA,
) -> fl.LassoModel:
    """LASSO under the sign matrix with the penalty ``reg_lambda * ||beta - prior||_1``."""
    model = fl.LassoModel(
        reg_lambda=reg_lambda,
        factors_beta_loading_signs=as_frame(SIGNS),
        factors_beta_prior=as_frame(PRIOR),
    )
    return model.fit(x=x, y=y)
```

`as_frame` labels a NumPy array with the fund and factor names:

```python
def as_frame(values: np.ndarray) -> pd.DataFrame:
    """Label an (N x M) array by asset and factor, the layout ``LassoModel`` expects."""
    return pd.DataFrame(values, index=ASSET_NAMES, columns=FACTOR_NAMES)
```

### One panel

At `reg_lambda` $= 3 \times 10^{-5}$:

| Estimator | Loading RMSE | Share of constrained cells violated |
|---|---|---|
| Free: no signs, zero prior | 0.125 | 0.25 |
| Signs | 0.110 | 0.00 |
| Signs and prior | 0.099 | 0.00 |
| The prior itself, no data | 0.118 | 0.00 |

The free fit violates four of the sixteen constrained cells. It gives `ig_credit_2` an equity
loading of 0.10 and a credit loading of 0.38 against a true 0.0 and 0.6: part of the credit
exposure has moved to the correlated equity factor. Under the sign matrix the equity cell is zero
and the credit loading is 0.57. With the prior added, the rates loading of the same fund moves
from 0.47 to 0.66 against a true 0.6. The combined estimate is better than the prior alone and
better than the data alone.

| Check | Result |
|---|---|
| `derived_signs_` against the supplied matrix | identical |
| Signs at `reg_lambda` $= 10^{-12}$ against `scipy.optimize.lsq_linear` with the same bounds | equal within $3 \times 10^{-5}$ |
| Prior-centred fit without signs against $\beta_0$ plus a zero-centred fit of $Y - X\beta_0^{\top}$ | equal within $10^{-4}$ |
| Signs and prior at `reg_lambda` $= 1$ against the prior | equal within $10^{-5}$ |
| Prior of $-0.5$ on a cell constrained non-negative, `reg_lambda` $= 1$ | loading below $10^{-6}$ in magnitude |

### Where each penalty shrinks to

[![Two panels. Left: loading error along the penalty grid for a penalty centred on zero and one centred on the prior, both under the sign matrix; at large penalties the first rises to the error of all-zero loadings, 0.555, and the second stays at the error of the prior, 0.118. Right: mean and 10th to 90th percentile of the loading error over 50 panels for the free, signed, and signed with prior estimators.](images/sign_constraints_and_priors_error.png)](images/sign_constraints_and_priors_error.png)

**Figure 1.** Synthetic teaching exhibit. Left: root mean squared error of the loadings against
the generating matrix on one panel of 36 months, along a grid of thirteen penalties; the penalty
falls to the right and both fits use the sign matrix. With the penalty centred on zero (orange
squares) a strong penalty returns the empty model, whose error is 0.555. Centred on the prior
(blue circles) it returns the prior, whose error is 0.118. At small penalties the two coincide.
Right: mean (dot) and 10th to 90th percentile (bar) of the same error over 50 redrawn panels at
`reg_lambda` $= 3 \times 10^{-5}$. Select the image for the full-resolution view.

The left panel is the reason to centre the penalty. With a zero-centred penalty the error has a
narrow minimum of 0.112 near $5 \times 10^{-5}$ and rises steeply on its strong side: the error
is 0.186 at a penalty 2.2 times larger and 0.363 at one 4.6 times larger. With the prior-centred penalty the minimum is 0.090 and
the error never exceeds that of the prior, 0.118, on the strong side. A misjudged penalty costs
little.

### Over redrawn panels

| Estimator | Mean loading RMSE | 10th to 90th percentile | Mean share violated |
|---|---|---|---|
| Free | 0.117 | 0.081 to 0.150 | 0.245 |
| Signs | 0.101 | 0.071 to 0.133 | 0.000 |
| Signs and prior | 0.077 | 0.061 to 0.097 | 0.000 |

The signed fit has a smaller error than the free fit in 82% of the 50 panels, and the fit with
the prior a smaller error than the free fit in 90%. The numbers describe this generating process
and this prior. A prior further from the truth helps less and, under a strong penalty, hurts.

## Implementation in factorlasso

The two inputs are constructor arguments of `LassoModel` and keyword arguments of the low-level
solvers; this article owns no separate public symbol. All names are documented in the
[API reference](api.rst).

| Argument or attribute | Role |
|---|---|
| `factors_beta_loading_signs` | DataFrame $S$. Enforced by `LASSO`, `GROUP_LASSO`, `HIERARCHICAL_CLUSTER_GROUP_LASSO` and `FACTOR_CLUSTER_GROUP_LASSO`. Rejected with `ValueError` by `UNILASSO` and the two cooperative modes, whose solvers take no sign constraint. |
| `factors_beta_prior` | DataFrame $\beta_0$. Centres the L1 term and, in the group and cooperative modes, the group norms. Not used by `UNILASSO`. |
| `nonneg` | Global non-negativity, used only when no sign matrix is active. |
| `auto_sign_constraints`, `auto_sign_threshold_t`, `auto_sign_excluded_factors` | Data-derived signs and their overlay with the explicit matrix. |
| `derived_signs_` | Fitted attribute: the final overlaid sign matrix the solver received, or `None` when none was supplied or derived, or when the solver takes none. |
| `plot_signs()` | Heatmap of `derived_signs_`; needs Matplotlib, which is not a runtime dependency. |

A typical construction starts from an all-NaN frame and fills in what is known:

<!-- fragment -->
```python
import numpy as np
import pandas as pd
import factorlasso as fl

signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
signs.loc[["gov_bond_1", "gov_bond_2"], "rates"] = 1       # duration is long
signs.loc[["gov_bond_1", "gov_bond_2"], "equity"] = 0      # no equity in government bonds

prior = pd.DataFrame(0.0, index=y.columns, columns=x.columns)
prior.loc[["gov_bond_1", "gov_bond_2"], "rates"] = 0.8

model = fl.LassoModel(
    reg_lambda=3e-5,
    factors_beta_loading_signs=signs,
    factors_beta_prior=prior,
).fit(x=x, y=y)
model.derived_signs_           # what the solver enforced
```

This fragment assumes factor and response panels `x` and `y`. The runnable version is the
canonical script
[examples/docs/sign_constraints_and_priors.py](../examples/docs/sign_constraints_and_priors.py),
which needs only the core dependencies, runs offline, and asserts every number in the tables
above:

```console
python examples/docs/sign_constraints_and_priors.py
```

The numbers in this article were produced with factorlasso 0.20.0.dev2, CVXPY 1.9 and CLARABEL on
Python 3.12. Figure 1 is regenerated from the same script by the documentation analytics runner
described in the [documentation standard](documentation_standard.md); its
[provenance record](images/analytics_manifest.json) holds the configuration, the source identity
and the hash of the image.

## Interpretation and limitations

- **A sign entry is a hard statement.** A wrong entry forces a wrong loading and no penalty
  repairs it. Leave a cell NaN when the sign is a belief and not knowledge.
- **Three modes take no sign constraint.** The `UNILASSO` solver is a two-stage univariate-guided
  fit and the cooperative solver handles signs softly through the positive and negative parts of
  $\beta$. With these modes a sign matrix or `nonneg=True` raises `ValueError`, and
  `auto_sign_constraints=True` derives signs that are not enforced, so `derived_signs_` stays
  `None`.
- **A zero entry is stronger than a sign.** It removes the factor from that response at every
  penalty. Use it for inadmissible exposures, not for exposures that are merely expected to be
  small.
- **The prior is only as good as its distance from the truth.** Under a strong penalty the
  estimate is the prior. Report the penalty together with the prior, and check the fit at a
  weaker penalty to see what the data say.
- **The prior is in the units of the loadings.** A prior stated for factors in one scaling is
  wrong after the factors are rescaled.
- **Constraints are met to solver tolerance.** A cell constrained to zero is returned below
  $10^{-7}$ in magnitude in the worked example, not as an exact zero.
- **Constrained estimates have no standard errors here.** The sampling distribution of a
  constrained, penalised estimate is not normal near the boundary, and the package reports none.
- **The exhibit is one generating process.** The gains in the tables depend on the factor
  correlation, the history length and the quality of the prior.

## See also

- [Sparse factor model](sparse_factor_model.md) for the loss, the penalty units and the
  intercepts.
- [Group penalties: HCGL and FCGL](group_penalties_hcgl_fcgl.md), which uses zero entries of a
  sign matrix to refit a selected support without shrinkage.
- [Quickstart](quickstart.md) for data-derived signs in a full workflow.
- [Task guides](task-guides.rst) for constrained-regression recipes.
- [API reference](api.rst) for signatures.

## References

- Black, F., and Litterman, R. (1992). Global portfolio optimization. *Financial Analysts
  Journal* 48(5), 28-43. DOI 10.2469/faj.v48.n5.28.
- Meinshausen, N. (2013). Sign-constrained least squares estimation for high-dimensional
  regression. *Electronic Journal of Statistics* 7. DOI 10.1214/13-EJS818.
- Sepp, A., and Kastenholz, M. (2026). Gated cluster-pooled sign constraints for multi-output
  sparse regression. Submitted to *Computational Statistics and Data Analysis*.
- Sepp, A., Ossa, I., and Kastenholz, M. (2026). Robust optimization of strategic and tactical
  asset allocation for multi-asset portfolios. *The Journal of Portfolio Management* 52(4),
  86-120.
- Slawski, M., and Hein, M. (2013). Non-negative least squares for high-dimensional linear
  models: consistency and sparse recovery without regularization. *Electronic Journal of
  Statistics* 7.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal
  Statistical Society: Series B* 58(1), 267-288. DOI 10.1111/j.2517-6161.1996.tb02080.x.
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
