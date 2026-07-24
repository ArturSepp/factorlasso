# AGENTS.md

Guidance for AI coding agents working in the **factorlasso** repository.

## Project overview

`factorlasso` estimates sparse multi-asset factor models with cell-level sign
constraints, prior-centred shrinkage, and hierarchical clustering group LASSO (HCGL),
and assembles the implied factor covariance matrix (Sigma_y = B Sigma_x B' + D). The
API is scikit-learn compatible (`fit` / `predict` / `score`).

It is the estimation engine behind `optimalportfolios` and the companion code to a
paper under review at the *Journal of Statistical Software*. Distribution and import
name `factorlasso`. Licensed **GPL-3.0** (`LICENSE`) — unlike most of the stack, which
is MIT.

## Ecosystem position

This package is one of eight open-source Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Before implementing anything
non-trivial, check whether it already exists in one of these:

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | Performance analytics, factsheets, visualisation |
| `optimalportfolios` | OptimalPortfolios | Portfolio construction and backtesting |
| `factorlasso` | factorlasso | Sparse factor models and factor covariance estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data fetching |
| `trendfollowing` | TrendFollowingSystems | Trend-following systems: closed-form theory and replication |
| `goal-based-allocation` | GoalBasedAllocation | Dynamic MV allocation under regime-switching jump-diffusions |
| `stochvolmodels` | StochVolModels | Stochastic volatility pricing analytics |
| `vanilla-option-pricers` | VanillaOptionPricers | Vanilla option pricers and implied volatility fitters |

Actual package dependencies within the stack: `optimalportfolios` depends on `qis`
and `factorlasso`; `trendfollowing` depends on `qis`; `stochvolmodels` has an
optional `research` extra that pulls in `qis`. The others are independent.

Do not vendor or copy code between these packages. If functionality belongs in a
sibling package, say so rather than reimplementing it here.

## Repository layout

```
factorlasso/
  lasso_estimator.py   main estimator (sklearn-compatible)
  factor_covar.py      factor covariance assembly
  sign_constraints.py  sign-constraint handling
  cluster_utils.py     hierarchical clustering for grouped penalties
  dependence_utils.py  dependence measures for the clustering correlation
  cv.py                cross-validation and lambda paths
  ewm_utils.py         exponentially weighted moment utilities
tests/                 23 test modules (top-level, test_*.py)
benchmarks/            performance benchmarks
examples/              runnable examples
papers/jss_2026/       JSS paper source, replication scripts, simulations
COMPARISON.md          empirical comparison against competing packages
COMPATIBILITY.md       scikit-learn compatibility notes
```

## Commands

```bash
pip install -e ".[dev]"                                   # editable install with dev tools
pytest                                                    # full suite (testpaths = tests)
pytest tests/test_integration.py -v                       # one module
pytest --cov=factorlasso --cov-report=term-missing -q      # as CI runs it
ruff check factorlasso/ tests/                            # lint, as CI runs it
```

Optional extras: `dev`, `docs`, `simulations` (for `papers/jss_2026/simulations/`).
Supported Python is >= 3.10; CI runs 3.11 – 3.14.

## Conventions

- Test files are named `test_*.py` and live in the top-level `tests/` directory.
- Line length 100 (`ruff`, rules `E`, `F`, `W`, `I`).
- The estimator follows scikit-learn conventions: constructor parameters are stored
  unmodified, fitted attributes end with a trailing underscore, and `fit` returns
  `self`. `COMPATIBILITY.md` documents what this guarantees — keep it true.
- Convex problems are expressed with `cvxpy`.
- Dataclasses carry estimator configuration and result containers.
- Runtime dependencies are numpy, pandas, scipy, cvxpy and openpyxl. scikit-learn is a
  **dev/test** dependency only: the package is compatible with sklearn but must not
  import it at runtime.

## Constraints — do not do these

- Do not import scikit-learn in package code. Compatibility is achieved by following
  its conventions, not by depending on it.
- Do not change estimator defaults, penalty scaling, or the sign-constraint logic
  without re-running the replication and comparison material (see below).
- Do not break the sklearn API contract (`get_params`/`set_params`, trailing-underscore
  fitted attributes) — `COMPATIBILITY.md` and downstream `optimalportfolios` rely on it.
- Do not relicense or copy code from MIT-licensed sibling packages into this repository
  without checking direction of licence compatibility.

## Replication contract

`papers/jss_2026/` contains the paper source, replication scripts, and the simulation
harness. Numbers in the paper, in `COMPARISON.md`, and in the JSS submission must
reproduce exactly. Any change to estimator internals, cross-validation, or covariance
assembly requires re-running the replication scripts and diffing the output against the
published tables. Report differences rather than updating the tables to match new
output.

## Release checklist

A release touches three version locations. All three must agree:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the software BibTeX entry in `README.md` (if it pins a version)

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.
