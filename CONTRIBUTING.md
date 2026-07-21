# Contributing to factorlasso

Thanks for your interest in `factorlasso`. `factorlasso` accompanies a paper under review at the *Journal of Statistical Software*, so reproducibility of published numbers takes priority over convenience.

## Scope

In scope:

- Bug fixes in the estimator, cross-validation, sign constraints, or covariance assembly
- New penalty structures with a reference for the formulation
- scikit-learn compatibility improvements — see `COMPATIBILITY.md`
- Benchmarks against competing implementations — see `COMPARISON.md`
- Documentation, examples, and tests

Out of scope — these will be declined, so please open an issue to discuss before
writing code:

- Importing scikit-learn in package code. Compatibility is achieved by following its
  conventions, not by depending on it; scikit-learn is a test dependency only
- Breaking the scikit-learn API contract: `get_params`/`set_params`, constructor
  parameters stored unmodified, fitted attributes with a trailing underscore, `fit`
  returning `self`
- Changes to estimator defaults or penalty scaling that alter published results
- Portfolio construction, which belongs in
  [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios)

## Reporting a bug

Open an issue using the bug report template. A report needs the `factorlasso` version, your
Python version, a minimal self-contained reproducer, and the full traceback or the
incorrect numbers. Reproducers that depend on proprietary or licensed data cannot be
run, so please use generated or public data.

## Asking a question

Open an issue and describe what you are trying to do. Questions about methodology are
welcome; where a question is really about the published papers, please say which paper
and section you are reading.

## Development setup

```bash
git clone https://github.com/ArturSepp/factorlasso.git
cd factorlasso
pip install -e ".[dev]"
pytest
ruff check factorlasso/
```

`AGENTS.md` in this repository documents the layout, commands, conventions, and
constraints in more detail — it is written for AI coding agents but is equally useful
to human contributors.

## Pull requests

- One topic per pull request. Unrelated changes in the same PR make review slower and
  are likely to be asked to split.
- Add or update tests for behaviour you change. A bug fix should come with a test that
  fails before the fix.
- Run the test suite and `ruff` before submitting.
- Do not bump the version in `pyproject.toml` or `CITATION.cff`; releases are cut
  separately.
- Do not commit generated output: figures, factsheets, backtest results, or data files.
- Keep the public API stable. If a change alters a public signature or default, say so
  explicitly in the PR description.

## Replication

`papers/jss_2026/` contains the paper source, replication scripts, and the simulation
harness. Numbers in the paper and in `COMPARISON.md` must reproduce exactly. Any change
to estimator internals, cross-validation, or covariance assembly requires re-running the
replication scripts and diffing the output against the published tables. Report a
mismatch in the PR rather than updating the tables to match new output.

## Conduct

Be civil and assume good faith. Technical disagreement is welcome; personal remarks are
not.

## Licence

This project is licensed under the GNU General Public License v3.0, unlike most of the other packages in the stack, which are MIT. By contributing, you agree that your contributions are licensed under
the GPL-3.0 licence of this project.
