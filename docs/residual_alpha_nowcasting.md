---
myst:
  html_meta:
    description: >-
      Residual-alpha nowcasting in factorlasso: a fitted model's next-period response as the
      factor component of realised factor returns plus the terminal EWMA mean of its residuals,
      the fail-closed rules, how it differs from predict, and the noise of an EWMA alpha.
---

# Residual-alpha nowcasting

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Once the factor returns of a period are known, a fitted factor model can nowcast the responses
before their own returns are observed: the loadings applied to the realised factors, plus the part
of the return that the factors have not explained. `LassoModel.nowcast` returns that sum, with
the unexplained part estimated as the recent mean of the fit's residuals, and refuses to answer
when its inputs cannot support it.

## Overview

A nowcast holds the loadings fixed and adds a statistical alpha, the terminal causal mean of the
residuals $y_t - x_t \beta^{\top}$ in original units, to the factor component of the target
period. The MATF-CMA framework uses the same object for expected returns: the weighted residual
mean of the loading regression is the alpha a committee may admit on top of the factor premia
(Sepp, Hansen and Kastenholz, 2026). The nowcast is an analytic on a fitted model, described here as an
implementation, not an estimator with known optimality.

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| $\beta$ | Fitted loadings, `coef_` | Fixed by the fit |
| $e_{t,i}$ | Residual $y_{t,i} - x_t \beta_i^{\top}$ at a fit date | Original units, not de-meaned |
| $s$ | `alpha_span`, the EWMA span of the alpha | Default: the span of the fit |
| $\hat a_i$ | Statistical alpha, `stat_alpha` | Return per period |
| $x_{T+k}$ | Realised factor returns of target date $T + k$ | Strictly after the fit cutoff $T$ |

## Methodology

### The nowcast

For each response the statistical alpha is the terminal value of an EWMA of its residuals,

$$
m_t = \lambda m_{t-1} + (1 - \lambda) e_t , \qquad \lambda = 1 - \frac{2}{s + 1} ,
$$

started at the response's first observed residual, $m = e$ there, and carried unchanged across
missing residuals; $\hat a_i = m_T$. This is the pandas `ewm(span=s, adjust=False,
ignore_na=True)` mean at the last date. With no span at all, neither `alpha_span` nor a fitted
`span`, $\hat a_i$ is the plain mean of the residuals. The nowcast is

$$
\hat y_{T+k,i} = x_{T+k} \beta_i^{\top} + \hat a_i ,
$$

returned with its factor component, the alpha, copies of the loadings and residuals it used, and
per-response diagnostics.

### What the nowcast is not

`predict` adds the economic intercept `alpha_const_`, reconstructed from the weighted means of the
fit with the weights of the loss; the solver's `intercept_` is a residual of the de-meaned fit and
enters neither. With uniform weights and no alpha span the two constants coincide, since both are
then the mean residual. With an alpha span they differ: the nowcast alpha forgets at its own rate,
which can be shorter than the horizon of the loadings.

### Fail-closed rules

`nowcast` raises instead of guessing when the fit was made with `demean=False`, when any fitted
factor row or the final response row was incomplete, when a loading or the final residual is not
finite, when the target factors are incomplete, differ from the fitted columns in name or order,
or include a date on or before the fit cutoff, and when the dates are not a sorted, unique
`DatetimeIndex` in one time zone. Leading missing responses are allowed: each response's alpha
starts at its own first observation.

### The noise of an EWMA alpha

With residuals of volatility $\sigma$ and no serial correlation, the EWMA mean of span $s$ has
stationary variance $\sigma^2 (1 - \lambda) / (1 + \lambda) = \sigma^2 / s$. A short span adapts
quickly to a change in alpha and pays for it with a standard error of $\sigma / \sqrt{s}$.

## Worked example

The canonical script
[`examples/docs/residual_alpha_nowcasting.py`](../examples/docs/residual_alpha_nowcasting.py)
simulates 240 months of six responses on three factors with 4% monthly volatility and 2% residual
volatility. Two responses carry a constant alpha of 50 basis points per month, two an alpha that
shifts from zero to 100 basis points in month 180, and two none; one response starts in month 60
and one misses five months. Each month from month 120 the loadings are fitted with uniform
weights on the expanding window and the next month is nowcast with a 24-month alpha span:

```python
def fit(x: pd.DataFrame, y: pd.DataFrame) -> fl.LassoModel:
    """Loadings with uniform weights; the nowcast keeps them fixed."""
    return fl.LassoModel(reg_lambda=REG_LAMBDA, span=None).fit(x=x, y=y)


def nowcast_next(x: pd.DataFrame, y: pd.DataFrame, cutoff: int) -> fl.LassoNowcastResult:
    """Fit on months up to the cutoff and nowcast the next month from its realised factors."""
    model = fit(x.iloc[: cutoff + 1], y.iloc[: cutoff + 1])
    return model.nowcast(x.iloc[[cutoff + 1]], alpha_span=ALPHA_SPAN)
```

The script checks the alpha against the pandas EWMA of the residuals, the prediction against the
factor component plus the alpha, and that without an alpha span the alpha equals `alpha_const_`.
It confirms that the nowcast refuses a fit without de-meaning, a target on the cutoff, reordered
factor columns and an incomplete final row; the three monthly fits whose final row has a gap are
therefore skipped in the loop.

Root mean squared error of the alpha estimates against the true alpha of the next month, in basis
points per month:

| Responses | Nowcast alpha, span 24 | Economic intercept | Zero alpha |
|---|---|---|---|
| Constant, 50 | 32 | 12 | 50 |
| Shift, 0 to 100 | 53 | 62 | 72 |
| None | 50 | 17 | 0 |

The EWMA alpha carries the noise the formula predicts, 2% divided by $\sqrt{24}$, about 41 basis
points,
so when alpha is constant or absent the full-sample intercept is closer. After the shift, the
intercept, which averages all months, reaches only a quarter of the new level, and the nowcast
alpha is closer: 67 against 86 basis points over the months after the shift.

![The nowcast alpha and the economic intercept of a response whose alpha shifts, against its true alpha, over expanding fits, and the error of each alpha estimate by response group against the EWMA noise floor](images/nowcast_alpha_tracking.png)

*Synthetic teaching exhibit. Left: for one response whose alpha shifts from 0 to 100 basis points,
the nowcast alpha and the economic intercept of each month's fit against the true alpha of the
next month. Right: root mean squared error of each estimate by response group; the dotted line is
$\sigma / \sqrt{s}$. Produced by `tools/docs_analytics/covariance_residuals.py` from the example
script.*

## Implementation in factorlasso

Verified with factorlasso 0.20.0.

| Name | Role |
|---|---|
| `LassoModel.nowcast(x, alpha_span=None)` | The nowcast of target dates `x` from a fitted model. |
| `LassoNowcastResult` | `prediction`, `factor_component`, `target_factors`, `stat_alpha`, `betas`, `residuals` and `diagnostics`, all copies. |

The fit stores the residual panel for the nowcast in `nowcast_residuals_`, one value per response
and date, with flags for complete factor rows and a complete final row; the snapshot costs one
more array of the size of the response panel. The diagnostics report, per response, the fit
window, the observations used, the effective number of observations of the loss weights, both
spans, the statistical alpha and `alpha_const`, the in-sample fit statistics and the number of
non-zero loadings. To run the example from a checkout:

```console
python examples/docs/residual_alpha_nowcasting.py
```

## Interpretation and limitations

- **The alpha is noisy.** Its standard error is about $\sigma / \sqrt{s}$ per period; choose the
  span for the speed of change you expect in alpha, not for the loadings.
- **The loadings are held fixed.** The nowcast does not refit; a change in loadings after the
  cutoff shows up as residual, and so partly as alpha, only at the next fit.
- **A nowcast, not a forecast.** It needs the realised factor returns of the target period; it
  says nothing about the factors themselves.
- **Residual alpha is model-relative.** It is what the chosen factors do not explain; a missing
  factor moves into it, which the [residual diagnostics](residual_diagnostics.md) test for.

## See also

- [Factor covariance assembly](factor_covariance_assembly.md): the model the nowcast is part of.
- [EWMA weighting](ewma_weighting_and_ragged_histories.md): spans and weights.
- [Sparse multi-output factor model](sparse_factor_model.md): the fit and its intercepts.

## References

- Sepp, A., Hansen, E., and Kastenholz, M. (2026). Capital Market Assumptions and Strategic Asset
  Allocation Using Multi-Asset Tradable Factors. Working paper,
  [SSRN 6785958](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958).
- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
