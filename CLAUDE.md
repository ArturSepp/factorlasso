# Project Instructions: OSS Quant Stack (v1.3, 2026-07-24)

<!-- v1.3 changes vs v1.2:
     1. Runnable-example convention pinned: dispatcher is run_local_test(local_test) over a
        LocalTest Enum. The predecessor run_unit_test/UnitTests naming is retired; rename
        opportunistically when already in a file, not in a sweep.
     2. Research-workflow entry points (multi-step pipelines in optimalportfolios and the
        rosaa research repo) use a separate run_research(workflow) dispatcher over a
        ResearchWorkflow Enum. Distinct from run_local_test by intent; do not merge.
     3. Licensing: bbg-fetch recorded as MIT (was an unset placeholder).
     Kept from v1.2: Cost of change section; factorlasso mandatory in optimalportfolios;
     GPL-3 implication settled; Python 3.10+; yfinance in the [data] extra. -->

<!-- v1.2 changes vs v1.1:
     1. New section "Cost of change": weigh benefit against churn and time before proposing
        any change; flag risks once instead of converting them into work.
     2. Reverted: factorlasso is a mandatory core dependency of optimalportfolios again.
        The [factorlasso] extra, the lazy-loading routing, and _optional.py are gone.
     3. Licensing: the GPL-3 implication of the mandatory factorlasso dependency is
        recorded as a settled decision. Do not re-raise it.
     Kept from v1.1: Python 3.10+ target; yfinance in the [data] extra with guarded imports. -->

## Scope

This project covers four general-purpose packages with specialisation:

| Package | Role |
|---|---|
| `qis` | base layer: pandas/numpy utilities, performance statistics, plots, `PortfolioData`, factsheets |
| `factorlasso` | sparse factor model estimation, sign-constrained LASSO, HCGL, factor covariance assembly |
| `optimalportfolios` | portfolio optimisation solvers, covariance estimators, rolling backtests |
| `bbg-fetch` | Bloomberg data access (prices, implied vols, fundamentals) |

Out of scope: `StochVolModels`, `VanillaOptionPricers`, `GoalBasedAllocation`. These are subject-based packages and belong elsewhere. If a request concerns them, say so once, then proceed only if I confirm.

Also out of scope: journal manuscripts, referee replies, working papers. Those live in the Publications project under its own rules. The one boundary case is the JSS submission for `factorlasso`: the *paper* is written in the Publications project, but the *package work the submission requires* (sklearn interoperability, `replicate.py`, vignettes, licence compliance) is library work and belongs here.

## Role

I am the author and maintainer of these packages. Work here is library development: API design, module architecture, refactoring, migration, tests, releases, documentation, and code review. Library code is the default standard, not the exception.

## Modes

- **Write mode** (default): produce code in my conventions, ready to run.
- **Design mode** (when asked about architecture or API): argue the trade-off explicitly. State what the design excludes and why that is acceptable. Sign the judgment — "I would do X because Y" — rather than listing options without a recommendation.
- **Review mode** (when asked to review): flag deviations by rule name with the corrected snippet. Do not silently rewrite a whole file. Do not soften criticism.

If the mode is ambiguous, ask once.

## Precedence

When rules conflict, apply this order:

1. **Technical and numerical correctness.** Never change numerical results, random seeds, or computed values when editing code.
2. **The hard invariants** (next section). These are architectural and are not traded away for convenience.
3. The conventions below.
4. An existing repository's established conventions where they conflict with the above. Flag every such override explicitly.

## Cost of change

I maintain this stack alone. Every change costs my time twice — landing it and living with it — so the burden of proof is on the change, not on the status quo.

- **Weigh every proposal before making it**: name the benefit, who receives it, and the full
  cost — files touched, packages released, CI runs, downstream floors, README/CHANGELOG
  updates. If the cost paragraph is longer than the benefit sentence, recommend against your
  own proposal.
- **An observation is not a work item.** Flag a risk or defect once, in one sentence, with a
  severity. Do not design the fix, produce the patch, or restructure code unless I ask for it.
- **Default to the smallest intervention** that solves the stated problem: documentation over
  metadata, metadata over code, code over architecture. A multi-file mechanism where a
  three-line `try/except` suffices is a defect of the proposal, not a feature.
- **Count the revert risk.** A change that is likely to be rolled back on contact with reality
  costs double. Say "this is probably not worth doing" when that is the honest assessment.
- Marginal polish (edge-case ergonomics, hypothetical users, licence hygiene for audiences I
  do not serve) does not justify churn in packaging, public API, or `__init__` wiring. These
  are the highest-blast-radius files in the stack.

# Hard invariants

## Dependency direction

The stack is a DAG. Imports flow one way only:

```
bbg-fetch      (leaf; blpapi only, no stack dependencies)
qis            (base; scientific stack only, no stack dependencies)
factorlasso    (leaf; sklearn-compatible; minimal dependency surface for JSS)
                    ↓
optimalportfolios  (depends on qis and factorlasso)
```

- `qis` must never import `optimalportfolios` or `factorlasso`.
- `factorlasso` must never import `qis`. Its deliberately small dependency surface
  (`numpy`, `pandas`, `scipy`, `cvxpy`, `openpyxl`) is a JSS submission constraint, not a
  preference.
- `factorlasso` is a mandatory core dependency of `optimalportfolios`, imported eagerly.
  Do not propose making it optional, lazy, or extra-gated; this was tried and reverted.
- No lateral or upward imports. If a change appears to require one, that is a signal the code belongs in the lower package. Say so instead of adding the import.

## Public API

- **Public** means exported in the package's `__init__.py`. Everything else is internal and may change freely.
- **Never invent a symbol.** If a function, class, or keyword argument is not in the export lists in project knowledge, do not produce code that calls it. Say that it does not exist, or ask.
- A change to a public signature requires a CHANGELOG entry and a version bump. Provide a deprecation path where it is cheap to do so.
- Cross-package coupling: if a public `qis` signature used by `optimalportfolios` changes, raise the `qis` floor in the `optimalportfolios` `pyproject.toml` in the same change.

## Dependencies and versions

- `pyproject.toml` is the source of truth for version floors. Do not propose code that requires anything below them, and do not propose code that relies on APIs deprecated above them.
- Current targets: NumPy 2.x, pandas 2.2+ with pandas 3.0 compatibility, Python 3.10+. No `inplace=True`, no chained assignment, no `DataFrame.append`, no `np.float_`.
- **Do not add a dependency without asking.** The dependency surface is a first-class design constraint, especially for `factorlasso`.
- Never introduce a competing analytics stack (`quantstats`, `pyfolio`, `empyrical`) as a dependency. We build these primitives.

## Licensing

`qis` and `optimalportfolios` are MIT. `factorlasso` is GPL-3. `bbg-fetch` is MIT.
The GPL-3 implication of `factorlasso` as a mandatory dependency — redistributors of the
combined work take on GPL obligations — is known, was weighed, and is accepted. Do not
re-raise it. If it ever matters, the remedy is dual-licensing `factorlasso`, not restructuring
`optimalportfolios`. Code does not move between packages under different licences without
flagging the implication first.

# Core conventions

- **Module imports, in this order and grouping**, with comment markers:
  ```python
  # packages
  import numpy as np
  import pandas as pd
  from numba import njit
  from enum import Enum
  from dataclasses import dataclass
  from typing import Union, Dict, Tuple, List, Optional
  # qis / project
  import qis as qis
  from qis import TimePeriod, PortfolioData, PerfParams
  ```
  Standard scientific stack first, then project packages under a `# qis` (or project-name) comment.
- **Type-hint every function signature.** Argument types and return type. `Optional[T]` for nullable, `Union[...]` for genuine polymorphism. Annotate containers as `pd.DataFrame` / `pd.Series`, array math as `np.ndarray`.
- **Keyword arguments with defaults, one per line**, vertically aligned, with a trailing inline comment giving units or semantics where not obvious:
  ```python
  def backtest(prices: pd.DataFrame,
               rebalancing_freq: Optional[str] = 'QE',
               rebalancing_costs: Union[float, pd.Series] = None,  # annualised, in bp
               management_fee: float = None,  # annualised on nav
               ) -> PortfolioData:
  ```
  Prefer explicit keyword arguments over positional. Long signatures with many optional flags are acceptable and idiomatic. Do not collapse them into a config object unless there is a natural grouping.
- **Validate inputs at the top of the function** with explicit `raise ValueError(f"...")` carrying the offending value (`got {span!r}`). Check container types and column/index alignment before computing. For shared validation, use small private `_validate_*` helpers as in `factorlasso`.
- **State containers as `@dataclass`**, frozen when the object is an immutable snapshot (`CurrentFactorCovarData`, `SaaTaaUniverseData`). Use `__post_init__` for derived fields and consistency checks.
- **String enums for labels and modes**: `class XColumns(str, Enum)` for DataFrame column-label sets, `class XType(Enum)` for mode switches. Drive behaviour with the enum, not bare string literals.
- **Docstrings**: one-line lower-case summary. For library code add a NumPy-style `Parameters` / `Returns` / `Raises` block. The math a function implements goes in the docstring in symbol form (`Σ_y = β Σ_x β' + D`), not only in prose.
- **Vectorize with pandas/numpy; reserve `@njit` for genuine recursions** (EWM updates, path-dependent backtest loops) that cannot be expressed as vectorized operations — the `ewm_recursion` / `backtest_rebalanced_portfolio` split. Do not hand-loop what pandas does natively.
- **pandas idiom**: business-day frequency via `.asfreq('B', method='ffill')`; rebalancing on calendar anchors (`'QE'`, `'ME'`, `'W-WED'`); log vs arithmetic returns stated explicitly via `qis.to_returns(..., is_log_returns=...)`, never an ad-hoc `.pct_change()` without stating the convention.
- **Time windows are `qis.TimePeriod`**, sliced with `time_period.locate(df)`, not manual date masking.
- **Group/asset-class metadata is a `pd.Series`** indexed by ticker (`group_data`), passed explicitly into reporting.
- **Readability is the priority.** Avoid "super-pythonic" constructions. Transparent names, enough comments, common data types.

## Tests and examples

- Each submodule carries a unit test for its core functions and a localised entry point to them.
- Runnable examples are gated behind a `LocalTest` `Enum` of cases plus a `run_local_test(local_test)` dispatcher, called under `if __name__ == '__main__'`. No top-level execution code scattered through a module. The predecessor `run_unit_test`/`UnitTests` naming is retired; rename to `run_local_test`/`LocalTest` when already editing a file, not in a sweep — these dispatchers are internal, so a mass rename is churn without payoff.
- Research-workflow entry points — multi-step analysis pipelines (`optimalportfolios`, and the `rosaa` research repo), not single-function demos — use a separate `run_research(workflow)` dispatcher over a `ResearchWorkflow` `Enum`. Distinct from `run_local_test`; do not merge them or use `run_research` for a per-function example.
- `examples/` is documentation. Minimal, runnable, top-to-bottom. No class hierarchy, factory, or config layer for a demonstration.
- **`yfinance` is a test and example dependency only.** It never appears in library code and is
  never a core dependency: it lives in the `[data]` extra of `qis` and `optimalportfolios`.
  The two function-local imports in `qis/portfolio/reports/config.py` (the `^IRX` download
  behind `add_rates_data=True`) are guarded with an `ImportError` naming the extra; keep that
  pattern for any future optional-dependency import. The data layer stays outside the
  analytics core.

## Building vs consuming

This project builds the primitives that my analysis code consumes. The polarity is the reverse of the analysis rule.

- **Inside `qis`**: implement performance statistics, drawdowns, annualisation. Here, computing Sharpe by hand is the job.
- **Inside `optimalportfolios`**: reuse `qis` rather than reimplementing it. `PortfolioData`, `TimePeriod`, `PerfParams`, `generate_dates_schedule`, and the factsheet layer are the canonical implementations. Duplicating `qis` functionality inside `optimalportfolios` is a defect, not a convenience.
- **Inside `optimalportfolios`**: covariance estimation with a factor structure goes through `factorlasso`, not bespoke code.
- Solvers keep the three-layer split: mathematical layer (clean inputs → formulated problem → Scipy/CVXPY), then the wrapping layers. Do not collapse it.
- Anything returning a portfolio must satisfy the `qis.PortfolioData` contract.

## Migration and conventions

- Fix deprecated pandas/numpy APIs at the source. Do not suppress warnings.
- Never silently switch return conventions, frequencies, or annualisation factors. State them.
- One convention per concept across the stack. If `qis` and `optimalportfolios` disagree, that is a bug to report, not a difference to accommodate.

## Release discipline

- Every public change carries a `CHANGELOG.md` entry: version, date, and the change classified as added / changed / fixed / removed.
- Bump the version in `pyproject.toml` in the same change.
- Do not propose a release from a state with failing local tests.

# Documentation prose

This covers READMEs, CHANGELOGs, docstrings, release notes, and PyPI descriptions. It does not cover papers.

- **Worked examples over description.** Every claim in a README is a runnable snippet.
- **No marketing adjectives.** Ban *powerful, seamless, comprehensive, robust, state-of-the-art, blazing*. State what the function computes and what it returns.
- Numbers, not descriptors: "125K downloads", "10 solvers", not "widely used" or "a rich solver library".
- Present tense, active voice, second person for instructions to the user ("Install using...", "Pass `weights` as a `Dict`").
- CHANGELOG entries are short, keyed to the public API, and name the symbol that changed.
- No bullet-point padding. If a section has one point, write one sentence.
- Mark every placeholder as `[TODO: ...]`. Never invent a benchmark number, a download count, or a citation.

# Suppress these default habits

- No praise, preamble, or meta-commentary ("Great question", "Let me...", "Here is a polished version").
- **Do not invent API symbols.** This is the failure mode that costs me the most time.
- **Do not convert flags into projects.** Raising a concern and implementing its fix are two
  different actions; the second needs my request.
- Do not over-engineer: no class hierarchy, factory, or configuration layer where a function with keyword arguments suffices.
- Do not refactor beyond the requested scope. Propose the wider change, do not perform it.
- No positional-argument calls on functions with long keyword signatures.
- No `print`-driven debugging left in code. No bare `except:`. No mutable default arguments.

# Maintenance

These instructions are versioned. When we agree on a rule change in conversation, produce an updated full copy of this file with an incremented version number and date.

This file is also the `CLAUDE.md` at each repository root. Keep the two in sync and version it in git.

## Project knowledge hygiene (configuration, not a behavioral rule)

- Canonical knowledge for this project: the `__init__.py` export list of each package, each `pyproject.toml`, each `CHANGELOG.md`, the dependency map above, and this file.
- **Do not mirror package source into the knowledge base.** Install the current release instead; the installed version is fresher and the mirror will drift. Work-in-progress code goes as a chat attachment, which is unambiguous and takes precedence on recency. Whole-file replacements produced in chat must be built from the repository HEAD, never from a released sdist.
- Name knowledge files descriptively and uniquely (`qis_exports_v4.4.py`, not `exports.py`). Retrieval and in-chat references both key off the filename. Replace a superseded file rather than adding the new version alongside it.
- Keep sensitive material — licensing negotiations, compensation, hiring, internal governance — out of this knowledge base. Handle it in a chat or a separate private project.
