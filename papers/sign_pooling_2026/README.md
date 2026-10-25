# Gated Cluster-Pooled Sign Constraints for Multi-Output Sparse Regression

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-06-22](https://github.com/ArturSepp/factorlasso/commit/725d3f5a6211a38e070ea7b794cd43bab20290f8)*

Software: [factorlasso](https://github.com/ArturSepp/factorlasso).
Citation: [CITATION.cff](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).

Companion workspace for Sepp and Kastenholz, *Gated Cluster-Pooled Sign Constraints
for Multi-Output Sparse Regression*, originally submitted to *Computational
Statistics & Data Analysis*. The current revision is prepared for
*Computational Statistics*. The [main manuscript](paper/article.pdf),
[main source](paper/article.tex), [Online Resource 1](paper/supplement.pdf) and
[supplement source](paper/supplement.tex) share the verified results described
below. Preparation does not imply that a new submission has occurred.
This is FactorLasso's only currently tracked paper workspace. Exact approved
manuscript assets and static inputs are listed in [.gitignore](.gitignore);
see the [paper contract](../AGENTS.md).

Archived release: [10.5281/zenodo.21000294](https://doi.org/10.5281/zenodo.21000294).

## Working revision of 25 October 2026

The article follows the [Computational Statistics author instructions](https://link.springer.com/journal/180/submission-guidelines).
Detailed proofs, complete study protocols, extended results and historical
corrections are in Online Resource 1. Online Resource 2 in the prepared
submission package contains the preserved clean-reproduction baseline and a
versioned overlay for presentation, build/audit tooling, the documented
infinite-threshold edge-case repair and the later variance-sensitivity evidence. Its top-level
index distinguishes those versions; it does not replace the historical DOI.
The author list uses Artur Sepp and Mika A. Kastenholz. The separate software
companion is unpublished and remains under review; no acceptance is asserted.

The revision separates the sign detector from the fitted estimator. Its theorem
establishes correct gate signs on nonzero coefficients for fixed numbers of
predictors and responses, complete observations, equal weights and independent
identically distributed dates, under explicit cluster and marginal-sign
assumptions. A population Ward separation/cut-margin corollary and an explicit
Gaussian block model supply sufficient conditions for consistent learned groups.
These results do not establish coefficient-support recovery, an oracle
distribution, or the assumptions for the yeast application. If response scales
are estimated, the gate proof uses a positive score-scale limit without claiming
that the score sandwich estimates the full sampling variance of the normalized slope.

The date-score gate accounts for contemporaneous response dependence. A failed
gate normally fixes a coefficient to zero; releasing failed gates and removing
all constraints are separate ablations. The benchmark distinguishes these
actions and evaluates fitted coefficients and independent test predictions.

| Experiment | Current scope |
|---|---|
| [Complete-estimator benchmark](replication/estimator_benchmark.py) | Six synthetic scenarios and nine methods, including HCGL/FCGL, ordinary LASSO, cooperative LASSO and full UniLasso. Training-only validation selects penalties and applicable gate thresholds. The known-partition method is an explicitly identified oracle diagnostic. Primary fits use zero targets and no adaptive weights. |
| [Nested eQTL validation](replication/eqtl_validation.py) | Six initial methods, extended to eight by the matched controls below, with LD thinning, outcome screening, scaling, clustering and gating repeated inside every training fold. Gate and cluster settings are fixed references for matched comparisons; only penalty fractions are tuned. Repeated splits of one yeast cross measure split variation, not independent biological replication. |
| [Learned-pooling validation](replication/learned_pooling_validation.py) and [expanded tuning](replication/learned_pooling_expanded.py) | Five identifiable-group designs, including mixed directions, partial supports, and separate predictor/error correlations. Forty paired seeds per design; eight fitted methods; training-only Cartesian cut/mixing/gate tuning and paired fixed-setting sensitivity. Frequent upper mixing-boundary choices triggered a separately recorded extension to alpha = 1. Population signal share is 0.65, a diagnostic grouped regime rather than a model of the yeast data. |
| [eQTL sensitivity](replication/eqtl_sensitivity.py) and [stability](replication/eqtl_stability.py) | Ten one-factor neighboring configurations on the original nine outer splits, plus matched singleton-gated HCGL and unconstrained FCGL. Partition, marker and fitted-cell stability distinguish absence, fitted zero and coefficient direction. Outer results do not select a preferred configuration. |
| [Variance-convention sensitivity](../../benchmarks/compare_sign_variance.py) | 35 paired cases switch only independent-cell versus date-score variance. Synthetic applications and nine yeast outer splits retain common penalties and training-only preprocessing. This fixed-setting comparison does not retune each variance mode or replace the nested study. |
| [Computational validation](replication/computational_validation.py) | Fresh-process seven-penalty path timing and peak working set across four dimensions; independent small-design objective checks using a tighter, differently formulated CLARABEL solve. These are path timings, not nested-validation costs or high-dimensional statistical guarantees. |
| [Date-score gate validation](replication/weighted_gate_validation.py) | The September experiment contains 6,000 paired scenario--weighting evaluations with reused seeds, not 6,000 independent panels. It compares screening statistics, including missing histories and recency weighting; those extensions are outside the complete-data theorem. |
| [Corrected cooperative diagnostic](replication/sign_pooling_robustness.py) | The historical known-cluster comparison is rerun with a common coefficient-zero tolerance. Tuning still uses true coefficient MSE, so this is distinct from the feasible complete-estimator benchmark. |
| Archived sign-only and eQTL exhibits | Frozen reference results retain their original independent-cell gate. The earlier globally screened, fixed-penalty eQTL prediction scores are superseded by nested validation. |

The [boundary extension](replication/tuning_boundary_validation.py) revisits both
synthetic studies after inspecting tuning-boundary counts. It reaches zero and
unit mixing, singleton and fully pooled cuts, zero penalty, and an infinite
gate threshold interpreted as complete gate failure. The original benchmark
retains its fixed mixing and cut; its penalty and threshold grids are extended.
All applicable settings in the learned-group study remain jointly selected
inside training. These extensions were not part of the initial protocol.

The final submission audit made the research wrapper's positive-infinite
threshold an explicit all-failed-gate endpoint, including exactly noiseless
nonzero slopes. A regression test fails with the previous implementation and
passes with the repair. Comparing every grid threshold over 320 synthetic
panels, 1,280 full/inner training splits and 4,659 gate partitions found no
changed signs or slopes in 4,184,640 coefficient-cell evaluations. The saved
fit inputs and objectives are therefore unchanged. A later package repair adopts
the same endpoint convention in both pooled and per-response paths, preserving
raw slopes, diagnostics and explicit override precedence. Replaying 94 selected
fits reproduces archived coefficients and predictions exactly; the full tuning
grid was not refitted. The 12,000-evaluation weighted-gate experiment also
reproduces its saved tables byte-for-byte. Online Resource 2 preserves the original
source snapshots and identifies these later source changes and checks separately.

The variance comparison finds unchanged estimates in 12 synthetic credit
configurations and two production teaching configurations (the latter have
singleton clusters), a 0.29% relative loading-matrix change in the quickstart
holdout, and a 19.96% change in a low-signal teaching panel. In yeast, switching
from independent-cell to date-score variance newly excludes 32.71% of all
coefficient cells. Median loading-matrix changes are 53.62% for HCGL and 18.10%
for FCGL; standardized held-out MSE increases by 1.39% and 0.56%, respectively.
The supplement defines the aggregation and reports the full protocol. These
results neither establish variance-mode superiority nor calibrate the gate.
The defaults remain automatic signs off, threshold 0.75 and independent-cell variance.

The supplementary gate figure now uses distinct line patterns and markers,
labelled panels and lettering sized for the final page. All 144 plotted
coordinates and both generated table files are unchanged. Its updated producer
is included in the same versioned overlay.

The installed CVXPY 1.9.2 has a documented
[uninitialized-memory shape-inference bug](https://github.com/cvxpy/cvxpy/issues/3235).
The [scoped compatibility wrapper](replication/cvxpy_shape_compat.py) uses initialized
integer zeros for the shape probe and restores the original method on exit.
It changes no installed files, numeric evaluation or solver configuration.
Deterministic memory poisoning reproduces the original invalid-reduction warning;
162 matched penalty solves are bit-identical with the repair. The endpoint runner
uses the wrapper and raises on RuntimeWarnings and NumPy invalid/divide/overflow
conditions. The four historical messages lacked stacks and cannot be individually
attributed with certainty; their matching upstream defect is now covered by a
regression test.

The final endpoint studies complete all 120 original and 200 learned-group
scenario-seeds, with 2,680 primary fits and 16,800 fixed-setting sensitivity fits.
No new RuntimeWarning occurred. Independent reconstruction from saved
coefficients checks prediction, coefficient, support and gate metrics, and
verifies every selection against its saved training-only CV surface. The added
pure-group and zero-penalty endpoints also pass 108 independent scalar-conic
objective checks. These checks preserve the distinction between objective
accuracy and tolerance-dependent coefficient-support classification.

Pooling still improves the coherent-group comparison against unconstrained
HCGL, but its predictor-correlation interval includes zero and established
competitors remain competitive. The eQTL penalty selections are interior, so
its nested validation and stability findings are unchanged. Natural grid
endpoints are now covered; finite interior spacing and CV uncertainty remain.

The displayed synthetic support tables use a standardized coefficient-zero
tolerance of `1e-4`, selected after the independent solver audit exposed
sensitivity at `1e-6`. Both thresholds and the original outputs remain in the
replication bundle. Prediction and unthresholded coefficient error are unchanged.
The eQTL analysis retains its prespecified `1e-6` threshold and reports sensitivity
to `1e-4`; high direction agreement is conditional on selection in both fits.

Use the exact revision source and the observed dependency versions in
[requirements-revision.txt](replication/requirements-revision.txt) for the new
protocols. The local revision bundle includes source/result hashes and dependency
metadata; it is distinct from the historical DOI and has not been published.
The clean Windows CPython 3.12.7 reproduction pins the complete runtime dependency
closure in [requirements-reproduction.txt](replication/requirements-reproduction.txt).
Install the matching source wheel; this avoids the development environment's
historical mismatch between editable source and installed distribution metadata.
Research fits select MOSEK explicitly; the package default remains
CLARABEL. Run receipts identify the actual source, installed distribution,
configuration and dependencies rather than treating a package version alone as
proof of reproduction.

## Archived reproduction

The Zenodo bundle and the historical `replication/requirements.txt` pins refer
to 0.7.2. Use that bundle in a separate external environment for historical
reproduction. Do not install its requirements over the development environment.
Current working code explicitly requests the independent-cell gate for legacy
experiments; a smoke run is not proof of identical archived results.

Frozen inputs in `replication/data/reference/` are not overwritten by the new
protocols. A corrected table produced from a new run must be identified as such,
even when its design matches an archived experiment.

## Layout

~~~text
sign_pooling_2026/
  paper/                 current article.tex, reading PDF, bibliography, tables and figures/
  drafts/                earlier versions and bundles; local and ignored
  presentations/         local by default
  private/               correspondence and permissions; local and ignored
  agents/                paper-specific working records; local and ignored
  replication/
    data/                yeast inputs and SHA-256 inventory
      reference/         frozen CSV/NPZ inputs
      local/             restricted inputs, if needed; ignored
    tests/               resource, leakage, objective and metric checks
    *.py                 validation, simulation and exhibit producers
~~~

Local sections may be absent in a fresh clone. Frozen reference results were
moved without numerical changes; their presence does not certify reproduction
in the current environment. Local JSS and prior-targets workspaces are not
dependencies of this public workspace.

## Runtime and revised commands

On the maintainer's Windows host, use
`C:\Python\FactorLasso312\Scripts\python.exe` as required by the
[repository instructions](../../AGENTS.md). Never create an environment under
OneDrive. From the repository root, choose a fresh output directory outside
both OneDrive and the checkout:

~~~powershell
$paperPython = 'C:\Python\FactorLasso312\Scripts\python.exe'
$paperRunRoot = Join-Path $env:LOCALAPPDATA 'FactorLasso\sign_pooling_2026\revision_20261024_reproduction'
$env:FACTORLASSO_PAPER_OUTPUT_DIR = $paperRunRoot
~~~

The variables above are used in the following commands. Choose another fresh
`$paperRunRoot` for a separate run; completed result files must not be replaced.
Without an explicit override, shared output helpers use
`AGENT_LOCAL_ROOT/outputs/sign_pooling_2026` or the platform's local application
data or temporary directory.

For a clean reproduction, create a separate environment under `C:\Python`, install
`requirements-reproduction.txt`, build the exact source with `build==1.6.1`,
`setuptools==84.2.1` and `wheel==0.48.0`, and install the resulting wheel. Run the paper modules from the
source export outside OneDrive. Set `$paperPython` to the new environment's
`Scripts\python.exe` before running the following commands.
The installed `factorlasso.__file__` must point into the new
environment's `site-packages`; paper sources and frozen inputs remain in the
export. A Git directory is unnecessary for the final-grid and eQTL commands;
eQTL receipts explicitly record unavailable Git metadata and retain source hashes.

The final synthetic grids can be evaluated without any previous result cache:

~~~powershell
$env:PYTHONWARNINGS = 'error::RuntimeWarning'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $paperPython -m papers.sign_pooling_2026.replication.tuning_boundary_validation --study learned --fresh-cv --out "$paperRunRoot\learned_endpoint_grid" --workers 6
& $paperPython -m papers.sign_pooling_2026.replication.tuning_boundary_validation --study original --fresh-cv --out "$paperRunRoot\original_endpoint_grid" --workers 2
& $paperPython -m papers.sign_pooling_2026.replication.tuning_boundary_sensitivity --root "$paperRunRoot\learned_endpoint_grid" --out "$paperRunRoot\endpoint_sensitivity" --workers 2
~~~

Here `--fresh-cv` recomputes every training loss. It is mutually exclusive with
`--prior`. Candidate order, grids and first-minimum selection remain explicit.
Mathematically identical paths within each training fold share a computation.
Equivalent cuts can represent the same fit at pure-L1 mixing; their selected
labels are not evidence that the partition is uniquely identified.

After completing these runs, check the manuscript's synthetic tables directly
against their raw draws:

~~~powershell
& $paperPython -m papers.sign_pooling_2026.replication.verify_revision_tables --paper papers\sign_pooling_2026\paper\article.tex --supplement papers\sign_pooling_2026\paper\supplement.tex --original "$paperRunRoot\original_endpoint_grid\estimator_draws.csv" --learned "$paperRunRoot\learned_endpoint_grid\estimator_draws.csv" --sensitivity "$paperRunRoot\endpoint_sensitivity\estimator_draws.csv" --out "$paperRunRoot\table_verification.json"
~~~

The commands below also retain the earlier staged-grid workflow for reproducing
its separately archived results. Use separate output directories for that workflow
and the direct final-grid run.

Run the paper checks and the complete default experiments:

~~~powershell
& $paperPython -m unittest discover -s papers/sign_pooling_2026/replication/tests -p 'test_*.py'
& $paperPython -m papers.sign_pooling_2026.replication.estimator_benchmark --out "$paperRunRoot\estimators" --reps 20
& $paperPython -m papers.sign_pooling_2026.replication.eqtl_validation --out "$paperRunRoot\eqtl" --repeats 3 --outer-folds 3 --inner-folds 3
~~~

The benchmark defaults run all six scenarios and all nine methods, with
20 replications per scenario. It reports coefficient error, support and sign
metrics, test prediction, timing and coefficient-tolerance sensitivity. The
nested eQTL command runs three repeats of three outer and three inner folds,
saving fold manifests, selected markers and penalties, coefficients,
out-of-fold predictions and held-out diagnostics. These fits can be expensive.
The eQTL loader also requires `rdata`; default conic fits require a working
MOSEK installation and license. Both commands accept `--solver CLARABEL` for a
separately identified solver run.

Run the follow-up experiments after producing the default nested eQTL run:

~~~powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $paperPython -m papers.sign_pooling_2026.replication.learned_pooling_validation --out "$paperRunRoot\learned_pooling\main_40" --reps 40 --first-seed 1000 --workers 6
& $paperPython -m papers.sign_pooling_2026.replication.learned_pooling_expanded --prior "$paperRunRoot\learned_pooling\main_40" --out "$paperRunRoot\learned_pooling\expanded_alpha1" --workers 6
& $paperPython -m papers.sign_pooling_2026.replication.tuning_boundary_validation --study learned --prior "$paperRunRoot\learned_pooling" --out "$paperRunRoot\learned_endpoint_grid" --workers 6
& $paperPython -m papers.sign_pooling_2026.replication.tuning_boundary_validation --study original --prior "$paperRunRoot\learned_pooling" --out "$paperRunRoot\original_endpoint_grid" --workers 2
& $paperPython -m papers.sign_pooling_2026.replication.tuning_boundary_sensitivity --root "$paperRunRoot\learned_endpoint_grid" --out "$paperRunRoot\endpoint_sensitivity" --workers 2
& $paperPython -m papers.sign_pooling_2026.replication.eqtl_sensitivity --prior "$paperRunRoot\eqtl" --out "$paperRunRoot\eqtl_sensitivity" --workers 2
& $paperPython -m papers.sign_pooling_2026.replication.computational_validation --out "$paperRunRoot\computation" --repeats 3
~~~

After the eQTL follow-up completes, check its five printed tables against the
saved repeat scores and fold-pair observations:

~~~powershell
& $paperPython -m papers.sign_pooling_2026.replication.verify_eqtl_tables --paper papers\sign_pooling_2026\paper\article.tex --supplement papers\sign_pooling_2026\paper\supplement.tex --root "$paperRunRoot\eqtl_sensitivity" --out "$paperRunRoot\eqtl_table_verification.json"
~~~

Each producer declares its protocol before evaluating results. The eQTL follow-up
reuses a prior fit only where the changed setting cannot alter that estimator;
the method-by-setting ledger identifies every reuse. Its settings are sensitivity
experiments, not an outer-test search for a new optimum. Gate equal-null curves
in the synthetic study use truth for a labeled diagnostic and report how many
seeds bracket each target; they do not define a practical error-control procedure.
The expanded synthetic command validates and reuses the original candidate losses,
adds the pure-L1 mixing endpoint, and selects on the same inner folds. Add
`--fresh-cv` to recompute and verify the original losses as well. Original results
remain separate from this extension, which was prompted by its tuning-boundary
diagnostic rather than prespecified with the original grid.
The final boundary command checks the saved source/input/CV hashes before reusing
old candidate losses and evaluates every added cell inside the same training
folds. Pure-group penalty scales use the appropriate dual norm. Zero penalty
uses bounded least squares with pinned-zero columns removed; UniLasso retains
its nonnegative second-stage endpoint. The upper fraction is a sufficient
zero-solution bound, so larger fractions add no distinct fit. Natural endpoint
selection is distinguished from truncation at an artificial grid boundary.
Finite spacing and finite Monte Carlo replication still limit conclusions.
The sensitivity command refits the reference and each one-factor setting using
that same extended penalty grid and its saved inner-CV losses. Its test outcomes
do not select the penalty or which setting to report. The earlier sensitivity
outputs are preserved separately.
Computational wall times depend on other activity on the host. Peak working set
includes the interpreter and imported numerical libraries. The manuscript summarizes
historical corrections in Online Resource 1; the old plots and frozen inputs remain
preserved in the archive and source workspace. Additional coefficient and sensitivity
tables accompany the new fitted studies.

The independent optimization audit uses CLARABEL reference tolerances of `1e-9`.
The original stricter `1e-10` attempt produced six `optimal_inaccurate` reference
statuses; those attempts are preserved in the run provenance. To rerun only the
45-case accuracy audit without repeating the scaling paths:

~~~powershell
& $paperPython -m papers.sign_pooling_2026.replication.computational_validation --accuracy-only --reference-tolerance 1e-9 --out "$paperRunRoot\accuracy"
~~~

Objective agreement does not imply identical selected supports; the audit also
reports zero/nonzero and sign-category differences at three coefficient thresholds.

To rerun the earlier gate validation and render its exhibits:

~~~powershell
& $paperPython -m papers.sign_pooling_2026.replication.weighted_gate_validation --out "$paperRunRoot\gate"
& $paperPython -m papers.sign_pooling_2026.replication.weighted_gate_exhibits --root "$paperRunRoot\gate" --out "$paperRunRoot\gate_exhibits"
~~~

## Historical exhibits and the tolerance correction

The default exhibit command reads frozen reference tables and writes externally:

~~~powershell
& $paperPython -m papers.sign_pooling_2026.replication.exhibits
~~~

Its corrected cooperative table is an explicit opt-in. The following first
writes a new `results/coop_benchmark_tolerance.csv`, then renders that table at
coefficient tolerance `1e-6` while leaving the other exhibits on their archived
inputs:

~~~powershell
& $paperPython -c 'from papers.sign_pooling_2026.replication.sign_pooling_robustness import study_cooperative; study_cooperative(reps=40)'
& $paperPython -m papers.sign_pooling_2026.replication.exhibits --corrected-coop "$paperRunRoot\results\coop_benchmark_tolerance.csv"
~~~

The unmodified default still uses the frozen historical cooperative results.
The correction changes numerical zero classification and remains oracle-tuned;
it is not a replacement for the new prediction-tuned benchmark.

For fresh legacy-design simulation results, run the original studies and tell
the exhibit producer to use those outputs explicitly:

~~~powershell
& $paperPython -m papers.sign_pooling_2026.replication.sign_pooling_simulation
& $paperPython -m papers.sign_pooling_2026.replication.sign_pooling_robustness
& $paperPython -m papers.sign_pooling_2026.replication.exhibits --from-run --corrected-coop "$paperRunRoot\results\coop_benchmark_tolerance.csv"
~~~

`--from-run` requires the selected external result files and does not silently
fall back to frozen caches. The simulation script accepts an integer
replication count, such as `50`, for a shorter protocol check.

The archived eQTL figures can be rebuilt from frozen coefficients, signs,
correlations and marker positions without fitting:

~~~powershell
& $paperPython -m papers.sign_pooling_2026.replication.eqtl_exhibits --cached-figures
~~~

This mode verifies gene/marker alignment and association counts, and generates
only the three figures. Their revised labels preserve the plotted values.
Without that flag, the older `eqtl_pipeline` and `eqtl_exhibits` modules reproduce
descriptive historical material through fresh fits. Their prediction path is explicitly
marked as superseded: use `eqtl_validation` for assessing the complete pipeline.
Fresh legacy fits are not an exact replay of the frozen coefficients: in the
clean reproduction, marker counts at the historical `1e-8` cutoff differed.
Use `--cached-figures` to regenerate the archived figures, and keep any fresh
legacy refit separate from the frozen descriptions and the nested validation.
These shared producers write results under `$paperRunRoot\results` and figures
and tables under `$paperRunRoot\paper`, leaving manuscript assets and frozen
reference files unchanged.

## Compare independent-cell and date-score variance

From the repository root, the paired runner reuses the existing synthetic
examples and the revised yeast preprocessing. Use a fresh external directory:

~~~powershell
& $paperPython -B -m benchmarks.compare_sign_variance --case synthetic --out "$paperRunRoot\variance\pilot_synthetic"
& $paperPython -B -m benchmarks.compare_sign_variance --case yeast --out "$paperRunRoot\variance\yeast_comparison"
& $paperPython -B -m papers.sign_pooling_2026.replication.verify_variance_sensitivity --root "$paperRunRoot\variance" --supplement papers\sign_pooling_2026\paper\supplement.tex --out "$paperRunRoot\variance_table_verification.json"
~~~

The yeast comparison uses MOSEK by default. Each pair shares the same numerical
penalty, computed at 0.3 times the unconstrained training KKT scale; the two
variance modes are not separately tuned. Quickstart selects its penalty using
the training sample only. Production comparisons concern point estimates, not
bootstrap intervals. The verifier independently reconstructs printed results
from saved coefficients, gates and predictions; it does not refit models.

Use the source identified by the variance overlay for these commands. Its
unchanged declared package version is 1.0.5, but the gate-endpoint repair is
later than the original clean-reproduction wheel. The preserved run receipts
identify actual imported source and dependency versions; the executable
comparison and table verification do not depend on having Git metadata.

## Build the current manuscript and supplement

Download the [Springer Nature LaTeX template](https://www.springernature.com/gp/authors/campaigns/latex-author-support)
and unpack it outside OneDrive. The build uses `sn-jnl.cls` and
`bst/sn-basic.bst` from that directory, without vendoring the template
into the repository. Install `pdflatex` and `bibtex` through the TeX distribution.
With the runtime variables above, run:

~~~powershell
& $paperPython -m papers.sign_pooling_2026.replication.build_paper --template-dir C:\Python\latex-templates\sn-article-template --flat-source
~~~

Replace the example template path with the external directory where you unpacked
the official template. The helper copies the explicitly approved assets into
`$paperRunRoot\latex-build`, runs pdflatex and bibtex, and repeats paired passes
until the main article and supplement cross-references stabilize. Outputs are
`article.pdf` and `supplement.pdf` in that directory. It compiles the current
manuscript assets; it does not integrate new experimental results.

With `--flat-source`, it also creates and independently compiles
`$paperRunRoot\submission-source`. This export contains editable sources,
bibliography, the official class and style, and the scientific figure, with
external references frozen to `main.pdf` and `ESM_1.pdf`. Its source manifest
identifies the exact files for a flat submission source ZIP; build logs and
auxiliary files are excluded. Online Resource 2 preserves the complete clean
numerical reproduction and adds a separately hashed current-presentation layer.

The October revision's reading PDF was rebuilt and visually reviewed after
integrating the new results. Later source changes require another successful
compilation and visual review before promoting the reading PDF into `paper/`.
Full experiment execution, exhibit promotion and manuscript compilation are
separate steps.

## Inputs and provenance

See [data provenance](replication/data/README.md),
[frozen result notes](replication/data/reference/README.md), and the
[SHA-256 inventory](replication/data/sha256.json). Preserve the source revision,
source hashes, actual Python/package/solver versions, command, seed design,
configuration and input hashes with each new run. Keep generated results in
their external run directory and promote only reviewed, explicitly approved
manuscript assets.

The available-history first-recorded date above was verified against
commit `725d3f5a6211a38e070ea7b794cd43bab20290f8`, which contains this README at
`papers/sign_pooling_2026/README.md`. It identifies repository inclusion, not a
publication or submission date.

## License

The replication code in this workspace retains its recorded GPL-3 terms.
FactorLasso package source starting with the 1.0.4.post1 MIT transition uses the
MIT License; this does not change the replication workspace's licence.
Manuscript copyright remains with the authors, and data retain their documented
source terms. This transition grants no additional publication rights.
