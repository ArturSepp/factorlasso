# Practical fixed-income prior validation note

[Artur Sepp](https://artursepp.com). Revised 26 September 2026. Internal research draft.

The 26 September revision changed the manuscript and the replication code. The PDF and
source ZIP in this folder predate it; rebuild them after the reruns below. The manuscript
still has 17 tables and 11 figures. The matched penalty controls moved from the appendix
to Section 5 as Table 8. Placeholders marked `[TODO]` (six tagged numbers, the new
bootstrap rows and stability column, and the two Monte Carlo tables) are filled by `prepare`.

## Current calibration

The study uses per-response valid EWMA weight normalization and a fixed monthly penalty
of 0.000104264890398061. Sign analytics preserve observation masks, use the native fit
horizon and aggregate scores by date. MOSEK is explicit in private computations; public
FactorLasso solver and sign defaults remain unchanged. Monthly loadings use span 60,
with span 36 as a sensitivity; annual factor covariance uses weekly returns and span 260.
Historical quarterly splits apply the current calibrated penalty retrospectively. They
are fixed-vintage conditional return reconstruction, not expected-return forecast validation.

The June cutoff, 30-index/12-fund roster, observed return panels and random seeds are
unchanged. Current blended IG/HY/EM factor premia are 1.6831%/2.3669%/1.9829%.
The broader MATF-CMA baseline retains PE 50% and ILS 100%; the FI study indices admit
zero alpha. No new fund-alpha policy is introduced.

## Results and qualifications

Conditional priors retain the highest average held-out R-squared among the four prior
policies at both horizons and in both panels. At span 60 their monthly RMSE is 1.340%
for indices and 1.478% for funds. A sign-constrained fit without penalty using the detected
signs scores 1.350% and 1.453%, and the Zero prior at the fixed penalty scores 1.527% and
1.660%. Most of the gap to Zero prior is therefore the penalty's cost at a zero centre.
The prior-informed sign set is the component that improves held-out fit (1.334% and 1.437%
at zero penalty). The paper now rests the case for the policy on attribution, signs and
coefficient stability.

The Monte Carlo sets residual noise per asset to a population R-squared of 0.80, adds a
zero-penalty arm, and reports loading error before CMA error. In the last run the wrong
IG-to-EM mapping increased loading error but reduced CMA error against automatic
selection under the narrower current premium gap. The paper reports both; small CMA error
is not independent validation of economic attribution. Duration diagnostics, mapping failures
and fixed-vintage limitations remain explicit.

## Sources and rebuilding

The editorial source is [../manuscript.md](../manuscript.md). All research/build code and
protocols live in [../replication/](../replication/). `paper_sources.json` identifies the
active C-local evidence: `FactorLasso/analyses/fi_current_refresh_20260925/evidence` under
the configured AgentWork root. Earlier evidence and the previous paper remain archived.

Use `C:/Python/FactorLasso312/Scripts/python.exe` after dot-sourcing `Enter-AgentRepo.ps1`.
Private empirical runs additionally need the existing ROSAA analysis dependencies and the
current FactorLasso, QIS, OptimalPortfolios and ROSAA source trees on PYTHONPATH. Do not
create an environment or generated output below OneDrive.

`refresh_current.py` is the current orchestration entry point. Start `init` in a new
C-local evidence directory with explicit `--previous` and `--cma-root` paths, then run
`endpoint`, `oos`, `sign`, `mc`, `checks`, and `prepare`, all with the same `--root`.
The 26 September revision needs a fresh root: `oos` now fits and saves the two zero-penalty
controls, and `mc` uses the recalibrated design. After `prepare`, point `paper_sources.json`
at the new root (prepare does this) and review `editorial_review_flags.json` before building.
The endpoint must finish before the sign replay. The checks/prepare stages require every
calculation to be complete. The approved source files are frozen in `owner_inputs/`;
originals are not modified. Earlier runners are retained as historical research code.

Build the verified editorial source from the repository root with:

```powershell
& C:/Python/FactorLasso312/Scripts/python.exe -m papers.prior_targets_2026.replication.build_latex --output-root C:/path/to/new/build
```

The builder verifies all 17 evidence blocks and every `<!-- value: key -->` tag in the prose,
rejects deliberate corruption of each block and each tag, regenerates figures and requires
converged references. `prepare` fills blocks and value tags only and never rewrites prose;
it writes `editorial_review_flags.json` listing any directional claim whose sign changed. Its only known layout diagnostic is
the CAS title-keyword zero-width box; all pages have been visually reviewed.

The local source ZIP contains standalone LaTeX, every replication script/protocol, derived
loading/CMA/score evidence and verification receipts. Compile `latex/article.tex` with
pdflatex, BibTeX and repeated pdflatex. Original workbooks, licensed histories and held-out
files containing actual returns are excluded. Empirical re-estimation needs the private
inputs; the standalone LaTeX build does not. This is a local review bundle, not a release.

The two synthetic examples run using public FactorLasso alone. The portable CLI default
is CLARABEL; pass `--solver MOSEK` for the private configuration used here. Their coefficients
match the full private run without importing ROSAA, QIS or OptimalPortfolios.

[FactorLasso](https://github.com/ArturSepp/FactorLasso) ·
[Software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
