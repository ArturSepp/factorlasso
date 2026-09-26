---
myst:
  html_meta:
    description: >-
      The research papers behind factorlasso: citation titles taken from the LaTeX sources,
      publication status, the documentation articles each paper supports, and how to reproduce
      the JSS, sign-pooling and fixed-income prior studies.
---

# Research papers and replication

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/factorlasso/commit/c89cf6d380358d4744592feeccb6073d989bc004)*

This page belongs to the documentation of [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Using the package and reproducing a paper are separate workflows. To use factorlasso, install it
and follow the [installation page](getting-started.md) and the [quickstart](quickstart.md); both
run offline on the supported API and need neither a paper checkout nor competitor packages. The
paper trees preserve frozen inputs, seeds, exhibit producers and additional toolchains. A paper's
review status is not a software stability claim, and reproducing a paper is not a prerequisite
for using the package.

## Papers

Each title below is the title in the paper's LaTeX source, and the documentation uses no other.
Methods are cited to these papers; the software itself is cited through
[CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

**Manuscripts with source in this repository**

- Sepp, A. and Kastenholz, M. A. (2026). *factorlasso: Sparse Multi-Output Regression with
  Cluster-Grouped Sign Constraints in Python*. Submitted to the Journal of Statistical Software.
  [Manuscript](../papers/jss_2026/paper/article.pdf) and
  [replication tree](https://github.com/ArturSepp/factorlasso/tree/main/papers/jss_2026).
- Sepp, A. and Kastenholz, M. (2026). *Gated Cluster-Pooled Sign Constraints for Multi-Output
  Sparse Regression*. Submitted to Computational Statistics & Data Analysis.
  [Manuscript](../papers/sign_pooling_2026/paper/article.pdf),
  [replication tree](https://github.com/ArturSepp/factorlasso/tree/main/papers/sign_pooling_2026)
  and archived research bundle [10.5281/zenodo.21000294](https://doi.org/10.5281/zenodo.21000294).
- Sepp, A. (2026). *Selecting Priors for Fixed-Income Factor Models: Validation and Capital Market
  Assumptions*. Working paper, revised 26 September 2026.
  [Manuscript](../papers/prior_targets_2026/paper/article.pdf) and
  editorial source `papers/prior_targets_2026/manuscript.md`. The PDF predates the
  26 September revision of the editorial source, and several tables are still marked for a rerun.

**Published and public working papers**

- Sepp, A., Ossa, I. and Kastenholz, M. (2026). *Robust Optimization of Strategic and Tactical
  Asset Allocation for Multi-Asset Portfolios*.
  [The Journal of Portfolio Management, 52(4), 86–120](https://www.pm-research.com/content/iijpormgmt/52/4/86).
  It introduced hierarchical clustering group LASSO (HCGL) for multi-asset covariance estimation.
- Sepp, A., Hansen, E. and Kastenholz, M. (2026). *Capital Market Assumptions and Strategic Asset
  Allocation Using Multi-Asset Tradable Factors*. Working paper.
  [SSRN 6785958](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958).

**Working papers** (links to be added when the papers are posted)

- Sepp, A. (2026). *Rolling-Ward clustering: noise-calibrated stability for rolling
  correlation-based clusters*. Working paper; link to be added.
- Sepp, A. and Kastenholz, M. A. (2026). *Achievable Sharpe and Universe Selection: A Closed-Form
  Factor Decomposition*. Working paper; link to be added.
- Sepp, A. (2026). *Sparse equity factors: explanation, covariance and portfolio evidence*.
  Working paper; link to be added. This title is taken from the study report, because the study
  has no LaTeX manuscript yet.
- Sepp, A. (2026). *Model-Layer Attribution: Risk, Signal, and Integration Alpha*. Working paper;
  link to be added.

## Which article uses which paper

Articles cite a paper at the claim it supports, by section. A result from a paper is quoted with
its study design and is not restated as a general performance claim.

| Paper | Articles |
|---|---|
| JSS software paper | [Sparse multi-output factor model](sparse_factor_model.md), [EWMA weighting](ewma_weighting_and_ragged_histories.md), [group penalties](group_penalties_hcgl_fcgl.md), [prior targets](prior_targets.md), [credit attribution case study](app_multi_asset_credit_attribution.md), [sign derivation](gated_cluster_pooled_signs.md), [adaptive weights](adaptive_penalty_weights.md), [cooperative LASSO](cooperative_lasso.md), [UniLasso](unilasso.md), [penalty selection](penalty_selection.md), [cluster discovery](cluster_discovery.md) |
| Gated cluster-pooled sign constraints | [Sign constraints and prior-centred penalties](sign_constraints_and_priors.md), [gated cluster-pooled sign derivation](gated_cluster_pooled_signs.md), [cooperative LASSO](cooperative_lasso.md), [UniLasso](unilasso.md), [cluster discovery](cluster_discovery.md), [yeast eQTL case study](app_sign_pooling_genomics.md) |
| Selecting priors for fixed-income factor models | [Prior targets](prior_targets.md), [EWMA weighting and loss normalisation](ewma_weighting_and_ragged_histories.md) |
| Robust optimization of strategic and tactical asset allocation | [Sparse multi-output factor model](sparse_factor_model.md), [sign constraints and prior-centred penalties](sign_constraints_and_priors.md), [group penalties](group_penalties_hcgl_fcgl.md), [factor covariance assembly](factor_covariance_assembly.md), [portfolio risk and CMAs case study](app_portfolio_risk_models.md) |
| Capital market assumptions with multi-asset tradable factors | [Factor covariance assembly](factor_covariance_assembly.md), [credit attribution case study](app_multi_asset_credit_attribution.md), [residual-alpha nowcasting](residual_alpha_nowcasting.md), [portfolio risk and CMAs case study](app_portfolio_risk_models.md) |
| Rolling-Ward clustering | [Causal smoothing of rolling clusters](rolling_cluster_smoothing.md), [offline cluster lineage](cluster_lineage.md) |
| Achievable Sharpe and universe selection | Planned: penalty selection and residual diagnostics |
| Sparse equity factors | Planned: empirical residual correlation |
| Model-layer attribution | Context for the factor-covariance workflow only |

## JSS software-paper replication

The [JSS replication tree](https://github.com/ArturSepp/factorlasso/tree/main/papers/jss_2026)
accompanies the software paper. Its committed 2026-06 ETF and factor panels are the canonical
inputs, so current network data are not part of the reproduced result. From `papers/jss_2026` in
a source checkout:

```console
python -m pip install -r requirements.txt
python replicate.py --log replication_output.txt
```

The default is the quick smoke path. `python replicate.py --full` uses the full manuscript seed
set and is the command for exact manuscript numbers; it takes materially longer. The comparison
stage also needs the compiled or platform-specific packages named in that tree's README. Install
failures for those research dependencies do not imply that the core factorlasso wheel is broken.

## Sign-pooling replication

The
[sign-pooling replication tree](https://github.com/ArturSepp/factorlasso/tree/main/papers/sign_pooling_2026)
includes the public yeast eQTL inputs and cached result tables. From `papers/sign_pooling_2026` in
a source checkout:

```console
python -m pip install -r replication/requirements.txt
make sims
make eqtl
```

`make paper` additionally needs the Elsevier CAS LaTeX template from CTAN. The replication pins
the historical factorlasso version used by that manuscript; do not silently substitute the newest
package and call changed output a reproduction.

## Fixed-income prior replication

The empirical stages of the fixed-income prior study use licensed index histories and the private
portfolio stack, so they cannot be rerun from a public checkout. The known-truth experiments need
only the public package. From the repository root:

```console
python -m papers.prior_targets_2026.replication.fi_validation_synthetic --output-root <new-directory> --examples
```

`--examples` runs the two small offline demonstrations; without it the full Monte Carlo design
runs. The paper README, `papers/prior_targets_2026/paper/README.md`, describes the empirical
orchestration and its evidence records.

## Evidence and provenance

Keep the generated session log with any reproduced exhibit. Record the source commit, package and
solver versions, Python version, platform, command, and whether the quick or full path was used.
Never overwrite committed tables merely to make them agree with a changed environment: report the
difference and establish whether it comes from code, dependency resolution or platform-specific
solver tolerance.

Use the [interoperability page](interoperability.md) and the
[compatibility policy](https://github.com/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md) for
software contracts. Use the paper-specific README, the archived DOI where applicable and the
session log for scientific reproducibility.
