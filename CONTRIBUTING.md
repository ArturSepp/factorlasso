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

- Importing scikit-learn at module level in package code. Compatibility follows its
  conventions; scikit-learn is a test dependency only. The existing guarded
  `__sklearn_tags__` hook imports it inside the method only when scikit-learn calls the hook
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
uv sync --locked --group test --group lint --extra docs
uv run --no-sync pytest
uv run --no-sync ruff check src/factorlasso tests examples
uv run --no-sync python tools/check_docs.py --all
```

Development tools live in the `test`, `lint` and `audit` dependency groups; there is no `dev`
extra. The `docs` and `simulations` extras are optional package dependencies.

On the maintainer's Windows host, set `UV_PROJECT_ENVIRONMENT=C:\Python\FactorLasso312` before
any uv project command and use that external environment. Never create or use an environment
inside a OneDrive checkout. The governance launcher in `AGENTS.md` also routes caches and
generated output to C:.

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
- Keep generated factsheets, backtest results and new data files outside Git. Reviewed
  documentation previews under `docs/images/`, their manifest and the explicitly approved
  paper exhibits are the exceptions described in `AGENTS.md`.
- Keep the public API stable. If a change alters a public signature or default, say so
  explicitly in the PR description.

## Replication

The public replication workspace is `papers/sign_pooling_2026/`; its README distinguishes the
current working revision from the archived environment. The JSS manuscript and simulation
harness are retained locally in an ignored workspace and are absent from a fresh clone.
Numbers in the papers and comparison material must reproduce exactly. Any change
to estimator internals, cross-validation, or covariance assembly requires re-running the
replication scripts and diffing the output against the published tables. Report a
mismatch in the PR rather than updating the tables to match new output.

## Conduct

Be civil and assume good faith. Technical disagreement is welcome; personal remarks are
not.

## Licence

This project is licensed under the GNU General Public License v3.0, unlike most of the other packages in the stack, which are MIT. By contributing, you agree that your contributions are licensed under
the GPL-3.0 licence of this project.
