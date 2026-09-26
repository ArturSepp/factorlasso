---
myst:
  html_meta:
    description: >-
      Authoring rules for factorlasso documentation: page forms, attribution, portable
      mathematics, the paper ledger, verified references, parameter ownership, executable
      examples, and figure and diagram provenance.
---

# Documentation standard

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-20](https://github.com/ArturSepp/factorlasso/commit/3a91426d6f1c93c1da08a02e70d592d14758078f)*

This standard applies to [factorlasso](https://github.com/ArturSepp/factorlasso).
Software citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

This is the factorlasso supplement to the
[shared OSS documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md).
The shared guide owns the common authoring rules. This page records what is specific to this
repository: formats, docstring convention, reference sources, executable examples, the figure
allowlist, and the verification commands. A general rule change belongs in the shared guide.

## Article structure and formats

Methodology articles are MyST Markdown files in `docs/`. They use the shared
[article structure](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md)
with the H2 heading `Implementation in factorlasso`, and start from the shared
[methodology template](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md)
with `PACKAGE` replaced by `factorlasso` and `REPOSITORY` by `factorlasso`. A new uncommitted page
carries the author-only byline; the linked first-recorded date is added after a commit exists.
Articles whose method comes from a co-authored paper cite that paper under References; the byline
names the author of the article.

Three page forms exist, each recorded in the [page inventory](../tools/docs_inventory.json):

- A **methodology article** has the eight H2 sections of the shared template, in order.
- A **case study** reports a study from one of the papers. Its H2 sections are, in order:
  Overview; Study design and data; Configuration; Results; What the study does and does not show;
  Reproduce; See also; References. Results are paper exhibits captioned with the study design, and
  numbers are quoted from the paper by section, not recomputed. The Python blocks are excerpts of a
  canonical offline script that builds the study's configuration on a synthetic panel and asserts
  the qualitative mechanism, never the paper's numbers.
- A **utility page** (home, installation, conventions, gallery, reference and project pages) has a
  byline, the software-citation line and a metadata description, and no empty sections.

The API entry `api.rst` stays RST; `docs/conf.py` generates its body at build time (see
[Public API and docstrings](#public-api-and-docstrings)). Every other page is Markdown. A page
converted from RST keeps its basename, so its URL does not change, and its byline links the commit that first
recorded the RST page. Do not keep a Markdown and an RST source with the same basename.

## Mathematics and notation

Follow the shared
[portable mathematics rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md).
The MyST `dollarmath` extension is enabled in `docs/conf.py`. Inspect Sphinx HTML, GitHub Markdown
and VS Code separately, and record a viewer as checked only when it was opened.

Inside math, spell every command with letters. GitHub Markdown applies backslash escapes before
the math reaches the renderer, so a thin space written as backslash-comma is displayed as a comma,
a norm bar written as backslash-bar becomes a single bar, and an escaped brace loses its backslash.
Write `\lVert x \rVert`, `\lbrace`, `\rbrace` and `\quad`, or omit fine spacing. GitHub also
renders inline math only when the opening dollar follows a space or an opening parenthesis and the
closing dollar is not followed by a letter or digit, so a hyphenated or quoted symbol stays raw
text there: write "for fixed $p$", not a hyphenated compound. Inside a display block, never start
a line with `+`, `-`, `*`, `>`, `#` or a numbered-list marker: GitHub reads it as a block element
and ends the formula, so break the line after the operator. The source checker rejects all three
patterns. They were established against the GitHub Markdown API on 2026-09-20.

One notation is used across all articles. There are $T$ observations, $M$ factors and $N$
responses. The factor panel $X$ is $T \times M$, the response panel $Y$ is $T \times N$, and the
loading matrix $\beta$ is $N \times M$, indexed by response and then by factor:

$$
Y_t = \alpha + \beta X_t + \varepsilon_t, \qquad
\Sigma_y = \beta \Sigma_x \beta^{\top} + D.
$$

An EWMA span $s$ corresponds to the decay $\lambda = 1 - 2/(s+1)$. It is neither a hard lookback
window nor a half-life. State the return convention, the observation frequency, the span, the
warmup and the missing-data policy wherever data enter an article. Covariance assembly does not
annualise: say whether the inputs are per-period or annualised.

## Papers and references

The package's own papers are cited with one title each, recorded in the `papers` ledger of the
[page inventory](../tools/docs_inventory.json). The title is the title in the paper's LaTeX
source; the ledger also records the authors and the status. Titles used in earlier drafts are
listed as `retired_titles`, and `tools/check_docs.py` fails when one appears on a reader-facing
page, in `CITATION.cff` or in a paper README. The papers fall into three groups:

- manuscripts with source in this repository (`papers/jss_2026`, `papers/sign_pooling_2026`,
  `papers/prior_targets_2026`), which articles cite by section and whose tracked exhibits they
  may display;
- published papers and public working papers, cited with their publisher or SSRN link;
- working papers without a public copy, cited with the status "Working paper; link to be added"
  until a link exists. Their figures are not displayed and their numbers are not quoted.

The [research papers page](scientific-replication.md) lists every paper and the articles that
use it.

A reference to other literature is admissible when it is an entry in
[the JSS bibliography](../papers/jss_2026/paper/refs.bib),
[the sign-pooling bibliography](../papers/sign_pooling_2026/paper/refs.bib) or
[the prior-targets bibliography](../papers/prior_targets_2026/paper/refs.bib), when it is already
cited in the `References` section of a module docstring, or when it has been checked against the
publisher or DOI landing page and that check is recorded in the working audit. Authors, year,
title, venue, volume, pages and DOI are checked against the primary source.

For every statistic or estimator taken from the literature, state what the package's adaptation
does not inherit from the source. The `References` section of `residual_diagnostics.py` is the
model. Separate three kinds of statement: a published method, an implementation choice made in
this package, and an experimental result from one of the papers. A paper result is quoted with
its study design and is not restated as a general performance claim.

## Public API and docstrings

The public surface is `factorlasso.__all__`. The inventory assigns every public name to exactly
one methodology article, and an adopted article names each symbol it owns under
`Implementation in factorlasso`. The inventory also assigns every `LassoModel` configuration
parameter to exactly one article, and an adopted article names each parameter it owns. The
checker reads the dataclass fields from source, so a new constructor parameter without an owner
fails the test suite. Fields that end with an underscore hold fitted state and are not owned.
Verify signatures, keyword names and enum members against the installed package before writing
an example. Names outside `__all__` are labelled internal or omitted.

The API page is generated when the documentation is built: `docs/conf.py` writes
`docs/_generated/api_reference.rst` (git-ignored) from `__all__` and the inventory, with one
section per owning article in the order of the sidebar and a table of the `LassoModel` parameters
by article. The objects stay on `api.html`, so their anchors do not change.

Docstrings use numpydoc, because the package follows scikit-learn conventions and its readers
arrive from that ecosystem. This is a per-package exception within the stack. Do not convert
docstrings and do not mix styles. Documentation work does not edit package source: a mismatch
between a docstring and observed behaviour is reported, not patched in passing.

## Executable examples

Each article has one canonical runnable script, `examples/docs/<article_basename>.py`, named by
the `example` key of the page in `tools/docs_inventory.json`. The article shows the code inline,
so that it reads on GitHub and in an editor as well as on the site, and gives an ordinary source
link to the script. Every fenced block tagged `python` must be a verbatim, contiguous excerpt of
that script, compared after removing common indentation; `tools/check_docs.py` fails the page
otherwise. A block that is not meant to run, such as a listing of fitted attributes, carries the
comment `<!-- fragment -->` on the line before its fence and is exempt. A MyST `literalinclude`
is not used, because GitHub shows the directive instead of the code. A
script is self-contained: it builds its inputs from a fixed seed in a few lines, imports only
numpy, pandas, scipy and factorlasso (Matplotlib where a figure is shown), and needs no network,
no data file and no sibling package. It runs after `pip install factorlasso` without a checkout.

Scripts assert their claims. Structural and algebraic contracts are asserted exactly or at solver
tolerance, and every number quoted in an article is compared inside the script with a reference
computed a different way. Articles quote at most three significant figures and name the solver.
[The example test](../tests/test_examples.py) runs every script under `examples/` with network
access blocked. The recipes of [examples and recipes](task-guides.md) are excerpts of
`examples/docs/task_guides.py` under the same rule. No page carries `testcode` blocks any more,
and `doctest_test_doctest_blocks` is off, so the Sphinx `doctest` builder currently runs no tests;
the step is kept so that a future `testcode` block is executed.

## Figures and analytical provenance

Two classes of exhibit are displayed, and captions label them differently.

A **paper exhibit** is a figure already committed under `papers/*/paper/figures/` with its
producer and frozen inputs in the same tree. It is displayed in place, registered with its
producer script and replication command, and never regenerated by documentation tooling. Its
caption states the study design. A manuscript usually embeds the PDF of a figure; the displayed
file is its PNG twin, un-ignored by name in the paper's `.gitignore`. A twin that the producer
did not write is rasterised from the tracked PDF at the producer's resolution, and the registry
records that.

A **diagram** is Mermaid source in a fenced `mermaid` block. It renders natively on GitHub and,
through `sphinxcontrib-mermaid`, on the site; VS Code needs an extension, so each diagram is
followed by a sentence that states its content in words. A diagram carries no data, is reviewed
as source and is not registered as an image.

A **synthetic teaching exhibit** is generated by a registered producer in
`tools/docs_analytics/` from the canonical script of its article: the producer loads the script,
calls the functions the article shows, and compares the `parameters` recorded for the exhibit in
`registry.json` with the constants of the script before drawing. A producer family lists its
exhibits under `configuration.exhibits`, keyed by PNG basename. Every exhibit also appears in the
[analytics gallery](analytics_gallery.md) with its question, sample, script and producer. The batch command regenerates
the complete bundle into a fresh directory outside the checkout:

```console
python -m tools.docs_analytics.run --list
python -m tools.docs_analytics.run --all --output-root <new-local-directory>
python -m tools.docs_analytics.validate --run-root <that-directory>
python -m tools.docs_analytics.run --verify
```

`--list` checks that every image displayed in `README.md` or `docs/` is registered and uses only
the standard library. `--all` writes `images/`, one CSV per plotted table and
`analytics_manifest.json` with the seed, configuration, conventions, source and dependency
identity, generation time in UTC, and the hash of every output. `validate` reads the bundle back.
`--verify` compares the committed previews with the committed manifest.

The only generated files that may be committed are reviewed PNG previews in `docs/images/` and
`docs/images/analytics_manifest.json`. Publication is a manual copy of a validated bundle after
the images have been inspected at article width and at full resolution. Supporting CSVs, PDFs and
intermediate output stay outside the repository. A successful run is not a visual review, and a
changed generation time does not make an exhibit a new analysis.

## Verification and adoption

Use the interpreter and launcher prescribed in `AGENTS.md`, build from a local source export and
write output outside the checkout.

```console
python tools/check_docs.py
python tools/check_docs.py --files docs/<article>.md
python tools/check_docs.py --all
python -m pytest tests/test_examples.py tests/test_docs_tooling.py tests/test_docs_config.py -q
python -m sphinx -E -W --keep-going -b html docs <output>/html
python -m sphinx -E -W -b doctest docs <output>/doctest
python -m sphinx -E -W -b linkcheck docs <output>/linkcheck
```

| Gate | What a pass establishes |
|---|---|
| `check_docs.py` | Adopted pages meet the source standard, their Python blocks are excerpts of their canonical scripts, and every public name has one owning article. |
| `check_docs.py --all` | No article is planned or pending and every public name is documented. |
| `test_examples.py` | Every example script runs offline and its assertions hold. |
| `docs_analytics.run --list` | Every displayed image is registered with its consumers. |
| `docs_analytics.run --verify` | Committed previews match the committed manifest. |
| Strict Sphinx HTML, doctest, linkcheck | The site builds without warnings, any `testcode` block passes, and external links resolve. |

These gates answer different questions. None of them establishes mathematical correctness,
bibliographic accuracy or rendered layout. Mark a page `adopted` in the inventory only after its
example passes, its references are verified, and its mathematics and figures have been inspected
in the viewers named in the working audit. Working audits and roadmaps live in the git-ignored
`agents/` directory.

## See also

- [Documentation home](index.md)
- [Examples and recipes](task-guides.md)
- [Research papers and replication](scientific-replication.md)
- [Contributor guidance](https://github.com/ArturSepp/factorlasso/blob/main/AGENTS.md)

## References

- [factorlasso software citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
- [Shared OSS documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md).
- [MyST: math and equations](https://myst-parser.readthedocs.io/en/latest/syntax/math.html).
- [GitHub: writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
