---
myst:
  html_meta:
    description: >-
      Contribution scope, reproducible starter tasks, development checks and review for FactorLasso.
---

# Contributing to factorlasso

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-21](https://github.com/ArturSepp/factorlasso/commit/40dda266fe389ebeeca0b3eb0dd0cf3a25973fac)*

Implemented in [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

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

## Starter tasks

These are proposed, unassigned tasks, not reports of confirmed numerical defects. Check for
existing or completed work and agree a bounded scope before starting. The
[shared contributor guide](https://github.com/ArturSepp/ArturSepp/blob/main/docs/contributing_to_stack.md)
explains setup, review, attribution and validation of automation-assisted work. Reuse this
repository's [feature request form](.github/ISSUE_TEMPLATE/feature_request.yml), or its
[bug report form](.github/ISSUE_TEMPLATE/bug_report.yml) for an observed defect.
No starter label implies that numerical redesign is included.

Commands below use PowerShell's `& $py`, where `$py` is the selected development interpreter.
Run them from the repository source root (a C-local source export on the maintainer host).
Follow [development setup](#development-setup) first; for this Windows/OneDrive checkout,
use the external environment and generated-state helper described in `AGENTS.md` before
running Python or uv. The shared guide gives the exact interpreter table. Each command is
a focused check, not a substitute for the package's documentation and PR gates.

### FL-1: Add an interpretation key for quickstart diagnostics

**Owner and code home:** FactorLasso; [docs/quickstart.md](docs/quickstart.md).
**Read first:** [examples/docs/quickstart.py](examples/docs/quickstart.py).

**Current gap:** The example prints several estimation diagnostics, without a single compact output-to-check interpretation key.

**Reproduce:** Run the canonical offline script and locate the assertions behind selected regularisation, loading RMSE, residual sphericity and covariance relative error.

**Acceptance and scope:** Add a short four-row key linking those printed groups to the existing example assertions and owning methodology pages. Distinguish held-out estimator score from in-sample diagnostics and covariance error against synthetic truth. Do not describe these as portfolio performance or empirical investment evidence. Preserve code excerpts, seeds and displayed numbers.

**Useful verification:**

```powershell
& $py examples/docs/quickstart.py
```

### FL-2: Pin nested Pipeline parameter updates before fitting

**Owner and code home:** FactorLasso; [tests/test_sklearn_interop.py](tests/test_sklearn_interop.py).
**Read first:** [docs/interoperability.md](docs/interoperability.md).

**Current gap:** The existing Pipeline test checks prediction shape; this module does not explicitly exercise a model__reg_lambda update through Pipeline.set_params.

**Reproduce:** Inspect test_pipeline_with_ndarray and run the interoperability module. Search the current suite for the proposed assertion before adding it, to avoid duplicate coverage.

**Acceptance and scope:** Extend the existing Pipeline test or add one narrowly scoped test: set model__reg_lambda, read it back through get_params, and confirm the fitted model retained it. Use the existing synthetic panel and importorskip convention. Verify a deliberately wrong expected value fails, then restore it. Do not change the estimator, add sklearn at package root or assert fragile solver decimals.

**Useful verification:**

```powershell
& $py -m pytest tests/test_sklearn_interop.py -q
```

### FL-3: Explain a skipped interoperability check versus an offline example pass

**Owner and code home:** FactorLasso; [docs/interoperability.md](docs/interoperability.md).
**Read first:** [tests/test_sklearn_interop.py](tests/test_sklearn_interop.py), [tests/test_examples.py](tests/test_examples.py).

**Current gap:** The page describes optional sklearn support but lacks a contributor checklist for interpreting skipped checks and the separate socket-blocked example lane.

**Reproduce:** Read importorskip in the interoperability module and the socket guards in test_examples.py. Run the focused canonical quickstart example test below.

**Acceptance and scope:** Add a short checklist that records installed extras, collected/passed/skipped counts and skip reasons; distinguishes missing optional sklearn from a passing sklearn integration; and points to the existing test dependency setup. Explain that the quickstart example test blocks sockets and rejects output files. Never count a skip as verified interoperability or add sklearn to core dependencies.

**Useful verification:**

```powershell
& $py -m pytest tests/test_examples.py -k quickstart -q
```

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

For the maintainer's Windows/OneDrive checkout, first run the setup helper named in
`AGENTS.md` and select the registered C:\Python interpreter. Set
`UV_PROJECT_ENVIRONMENT` to that external environment before any uv project operation;
never use the unqualified uv examples below against a OneDrive-local environment.
Keep generated output and pure verification source exports on C: as described in the
[shared contributor guide](https://github.com/ArturSepp/ArturSepp/blob/main/docs/contributing_to_stack.md#setup-and-a-first-check).

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

This project is licensed under the [MIT License](LICENSE). By contributing source or
applicable documentation, examples or synthetic fixtures, you agree to distribute
your contribution under that licence and confirm that you have the right to do so.
Preserve existing copyright and permission notices. Dependencies, vendor material
and manuscripts retain their own terms; see [NOTICE](NOTICE).
