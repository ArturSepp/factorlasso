# Gated Cluster-Pooled Sign Constraints for Multi-Output Sparse Regression

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Software: [factorlasso](https://github.com/ArturSepp/factorlasso).
Citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Companion workspace for Sepp and Kastenholz, *Gated Cluster-Pooled Sign Constraints
for Multi-Output Sparse Regression*, submitted to *Computational Statistics &
Data Analysis*. This is FL's only currently tracked paper workspace. The exact
approved current manuscript, figures, table fragments and static inputs are
listed in `.gitignore`; see the [paper contract](../AGENTS.md).

Archived release: [10.5281/zenodo.21000294](https://doi.org/10.5281/zenodo.21000294).

## Working revision and archived reproduction

The working revision of 25 September 2026 adds masked EWMA sign slopes and a
date-score sandwich gate, with a separate 6,000-design validation. It requires
FactorLasso 0.20.0.dev5. The original simulation and eQTL exhibits remain
identified as independent-gate results. Research fits explicitly select MOSEK;
the package default remains CLARABEL.

The Zenodo bundle and its `replication/requirements.txt` pin to 0.7.2 are
unchanged. Use that archived bundle in a separate external environment for exact
historical reproduction. Do not install its requirements over the development
environment. Current working code explicitly requests the independent gate for
the legacy experiments; a smoke run is not proof of identical archived results.

## Layout

```text
sign_pooling_2026/
  paper/                 current article.tex, article.pdf, refs.bib, tables, figures/
  drafts/                old versions and bundles; local and ignored
  presentations/         local by default; no approved presentation currently
  private/               correspondence and permissions; local and ignored
  agents/                paper-specific working records; local and ignored
  replication/
    data/                yeast inputs and SHA-256 inventory
      reference/         frozen CSV/NPZ inputs formerly in results/
      local/             restricted inputs, if needed; ignored
    tests/               offline resource and output-path checks
    *.py                 simulation, eQTL, gate validation and exhibit producers
```

Local sections may be absent in a fresh clone. Frozen reference results were
moved without numerical changes; their presence does not certify that the
current environment reproduces them. The JSS and prior-targets workspaces are
local-only and are not dependencies of this public workspace.

## Runtime and commands

On the maintainer's Windows host, configure the runtime from root `AGENTS.md`
and use `C:\Python\FactorLasso312\Scripts\python.exe`. Never create an environment
under OneDrive. New outputs default to
`AGENT_LOCAL_ROOT/outputs/sign_pooling_2026`; otherwise they use the platform's
local application-data or temporary directory. Set an absolute
`FACTORLASSO_PAPER_OUTPUT_DIR` outside the checkout and OneDrive for a distinct run.

From the repository root in that configured environment:

```powershell
python -m unittest discover -s papers/sign_pooling_2026/replication/tests -p 'test_*.py'
python -m papers.sign_pooling_2026.replication.exhibits
```

The second command regenerates simulation exhibits using frozen reference tables
and writes them to the external `paper/` output directory. It does not replace
the approved manuscript assets. To generate new simulation results and use only
those new results:

```powershell
python -m papers.sign_pooling_2026.replication.sign_pooling_simulation
python -m papers.sign_pooling_2026.replication.sign_pooling_robustness
python -m papers.sign_pooling_2026.replication.exhibits --from-run
```

These full experiments can be expensive. The simulation accepts an integer
replication count, for example `50`, for a shorter run. Seed conventions and
numerical defaults are unchanged. Research eQTL fits require the optional `rdata`
package and a working MOSEK installation/license:

```powershell
python -m papers.sign_pooling_2026.replication.eqtl_pipeline
python -m papers.sign_pooling_2026.replication.eqtl_exhibits
```

Both the pipeline and eQTL exhibit producer perform fits. New tables and fitted
arrays go to external `results/`; generated figures and LaTeX tables go to
external `paper/`. They never overwrite `replication/data/reference/` or the
tracked manuscript. For the revised weighted-gate experiment use
`weighted_gate_validation.py --out <absolute-C-local-output>` and
`weighted_gate_exhibits.py --root <that-output> --out <absolute-C-local-exhibits>`.

## Build the current manuscript

Install the Elsevier CAS template through the LaTeX distribution, or retain its
class/style files locally in `paper/`; those files are not redistributed here.
From the repository root:

```powershell
python -m papers.sign_pooling_2026.replication.build_paper
```

The helper copies the approved assets into external `latex-build/` and runs
pdflatex, bibtex and two pdflatex passes there. It builds the current approved
assets, not unreviewed outputs from a fresh run. Review regenerated results
before explicitly promoting any into `paper/`. The existing reading PDF is
retained; this folder migration does not imply that it was rebuilt.

With GNU Make, the equivalent targets from this directory are `make paper`,
`make sims`, `make eqtl`, and `make cached-figures`; set `PY` to the configured
external interpreter. Full figure regeneration and current-manuscript compilation
are separate targets.

## Inputs and provenance

See [data provenance](replication/data/README.md),
[frozen result notes](replication/data/reference/README.md), and the
[SHA-256 inventory](replication/data/sha256.json). Record the source revision,
Python/package/solver versions, command, seed design and input hashes with each
new run. Compare results before changing frozen references; do not silently
substitute live data or claim a new run is the historical archive.

## License

Code under GPL-3, matching `factorlasso`. Manuscript copyright remains with the
authors. This reorganization grants no additional publication rights.
