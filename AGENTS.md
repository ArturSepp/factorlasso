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
  residual_diagnostics.py  strict-factor-structure tests on a residual panel
  diagonality.py       LassoModelDiagonalityCV, penalty selection by those tests
  cluster_lineage.py   offline persistent labelling of estimated risk clusters
tests/                 28 test modules (top-level, test_*.py)
benchmarks/            performance benchmarks
examples/              runnable examples
papers/jss_2026/       JSS paper source, replication scripts, simulations
papers/cluster_lineage/ standalone S&P 500 cluster-lineage reproduction
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
  `I` is selected here and nowhere else in the stack — deliberate, because this package's imports
  do not follow the scientific-stack-first grouping the other repositories use. Do not "fix" it in
  either direction.
- **Two invariants are enforced by ruff rather than written down**, both green on the package as
  it stands, so a violation is always something you just introduced:
  - `TID251` fails any import of `qis`, `optimalportfolios`, `trendfollowing`, `privateassets` or
    `stochvolmodels`. `factorlasso` is a leaf: `optimalportfolios` depends on it, never the
    reverse, and the small runtime surface (numpy, pandas, scipy, cvxpy, openpyxl) is a JSS
    submission constraint rather than a preference. If a change appears to need one of these
    imports, the code belongs in the consumer — say so rather than adding it.
  - `TID253` fails a **module-level** import of `sklearn` anywhere in `factorlasso/`. See the
    constraint below for the one deliberate exception and its shape. `tests/`, `benchmarks/` and
    `papers/` are exempt in `per-file-ignores`: scikit-learn is a legitimate dev dependency there.
  - `ICN` pins `import numpy as np` and `import pandas as pd`. Ruff's default alias map is
    replaced rather than extended, so other libraries keep their own aliasing.
- The estimator follows scikit-learn conventions: constructor parameters are stored
  unmodified, fitted attributes end with a trailing underscore, and `fit` returns
  `self`. `COMPATIBILITY.md` documents what this guarantees — keep it true.
- Convex problems are expressed with `cvxpy`.
- Dataclasses carry estimator configuration and result containers.
- **Statistics taken from the literature carry their source in the module docstring.**
  `residual_diagnostics.py` has a `References` section naming Schott (2005) for the sphericity
  statistic, Marchenko-Pastur (1967) and Laloux et al. (1999) for the spectral edge, and
  Gagliardini, Ossola and Scaillet (2019) for reading the largest residual eigenvalue as an
  omitted-factor test and for the "smallest model that passes" selection shape. Keep that section
  true, and state what the package's adaptation does NOT inherit from the source — here, the
  `nu = n - k - 1` charge is heuristic and the calibration does not account for the loadings being
  estimated. A statistic without a source in the header reads as though it was derived here.
- Runtime dependencies are numpy, pandas, scipy, cvxpy and openpyxl. scikit-learn is a
  **dev/test** dependency only: the package is compatible with sklearn but must not
  import it at runtime.

## Constraints — do not do these

- Do not import scikit-learn at module level in package code; `ruff`'s `TID253` fails if you do.
  Compatibility is achieved by following its conventions, not by depending on it — `scikit-learn`
  is in the `dev` extra only, and `import factorlasso` leaves `sklearn` absent from `sys.modules`.
  **One deliberate exception:** `__sklearn_tags__` in `lasso_estimator.py` imports
  `sklearn.utils` inside the method behind a `try/except`, so an older scikit-learn that does not
  call the hook falls through and a missing scikit-learn cannot break an import. It runs only when
  scikit-learn is installed and calling it. Any future exception keeps that shape: inside the
  function, guarded, and reachable only from a scikit-learn code path.
- Do not change estimator defaults, penalty scaling, or the sign-constraint logic
  without re-running the replication and comparison material (see below).
- Do not break the sklearn API contract (`get_params`/`set_params`, trailing-underscore
  fitted attributes) — `COMPATIBILITY.md` and downstream `optimalportfolios` rely on it.
- Do not relicense or copy code from MIT-licensed sibling packages into this repository
  without checking direction of licence compatibility.

<!-- ===== SHARED AGENT CORE (standalone variant) — begin =====
     Generated from SHARED_AGENT_CORE.md in the maintainer's project knowledge. Do not hand-edit
     between these markers — propose the change to the maintainer instead. Variants: builder
     (qis) / consumer / standalone. Last synced 2026-08-08, agent core v1.1. -->

## Domain invariants

- **No look-ahead in any rolling or expanding estimation.** Estimation is point-in-time; a
  full-sample statistic inside a rolling path is forward-looking and wrong even when it runs
  clean.
- Conventions are stated, never implied: return frequency, annualisation, covariance scaling.
  One convention per concept across the stack — if this package and a sibling disagree, that is
  a bug to report, not a difference to accommodate.

## Dependency surface

This package is a leaf: it imports nothing from the stack (see Conventions, `TID251`), and its
small runtime surface is a design constraint, not a preference. Ask before adding any
dependency.

**Never invent a symbol.** If a function, class, or keyword argument is not in the export
surface of this package or of a dependency, it does not exist. Check in one line —
`python -c "import factorlasso; print([n for n in dir(factorlasso) if not n.startswith('_')])"`
— and say a symbol is missing rather than producing code that calls it.

## Verification loop

- Plan → patch → verify. Name the verification command and its result when proposing a patch.
- A second pass is mandatory where a plausible patch can be numerically wrong and still run
  clean: estimation windows, penalty scaling, sign constraints, covariance assembly, anything
  cross-validated or resampled. Verify against a reference computed a different way, and say
  which.
- Prove a new test fails before trusting that it passes: reintroduce the defect, watch it fail,
  restore.

## Escalation and scope

- Stop and propose before proceeding when a change would exceed roughly five files, alter a
  public signature, or touch a numerical path.
- Never change numerical results, random seeds, or computed values unless the change is the
  request.
- A public-signature change carries a `CHANGELOG.md` entry and a version bump in the same
  change. Removing a keyword argument from a function taking `**kwargs` is a silent break — the
  caller's keyword is swallowed and nothing raises. Treat it as breaking.
- Do not refactor beyond the requested scope. Propose the wider change; do not perform it.

## Concurrent sessions

More than one agent or session may work on this checkout at the same time, so a file can change
between your read of it and your write.

- Re-read a file from disk immediately before editing it. Never write a file from an earlier
  read: a whole-file write from a stale copy silently reverts another session's work.
- Prefer minimal anchored edits over whole-file replacement. If the on-disk content is not what
  you expected, stop and reconcile your change onto the current content rather than overwrite.

## Roadmap execution

Feature roadmaps live at the repository root as `ROADMAP_<feature>.md`. An execution request
names the file and the stage. A stage is complete when its stated verification command passes;
its out-of-scope list is binding.

<!-- ===== SHARED AGENT CORE — end ===== -->

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
