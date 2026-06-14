# JSS 2026 — multi-asset ETF application study

The application of Section~\ref{sec:application-etf}: factor-loading estimation
on a multi-asset universe of exchange-traded funds against the nine MATF
factors, demonstrating the `factorlasso` methodology on a publicly reproducible
cross-section. Mirrors the calibrated-simulation findings of the simulation
study on real market data.

## Scripts

1. **`fetch_etf_panel.py`** — builds the ETF return panel from the public ticker
   list `etf_universe.csv` via `yfinance`, writing the excess-return panel to
   `data/`. Requires a plain internet connection. The committed
   `data/etf_excess_logreturns.csv` is the output of this step, so the panel
   need not be refetched to reproduce the paper.

2. **`etf_simulation_study.py`** — the calibrated multi-asset benchmark
   (Table~\ref{tab:competitor}). Monte Carlo over seeds at
   `T \in {60, 112, 240}` and the oracle/BIC selectors, across the competitor
   packages and the `factorlasso` configurations including the HCGL and FCGL
   penalty modes. Writes `etf_competitor_study_raw.csv` and
   `etf_competitor_study_summary.csv`.

3. **`plot_competitor_study.py`** — draws the benchmark figures
   (`etf_study_selector_contrast`, and the oracle-metrics and t-robustness
   exhibits) from the study output.

4. **`run_etf_study.py`** — the empirical credit-attribution study
   (Table~\ref{tab:etf-credit} and the credit-beta-vs-lambda figure): OLS,
   HCGL shrink-to-zero, HCGL prior, and FCGL prior loadings on the real ETF
   panel, plus the calibrated simulation diagnostic. Writes
   `etf_empirical_credit_betas.csv`.

5. **`prior_sensitivity_study.py`** — the prior-misspecification sweep
   (Table~\ref{tab:prior-sensitivity}).

6. **`recalibrate_dgp_sweep.py`** — the within-cluster heterogeneity and anchor
   sweep that backs the no-dominance characterisation of the two penalty modes.
   Sweeps the ground-truth scatter and the per-sub-class centre
   (`--anchor {ols,hcgl}`), with an optional `--with-prior` flag.

7. **`compare_modes_on_tables.py`** — runs the HCGL and FCGL penalty modes
   through the calibrated benchmark on the same metrics as
   Table~\ref{tab:competitor}, for the direct mode comparison.

## Running

The recommended entry point is the staged replication script
`papers/jss_2026/replicate.py`, which runs the simulation and application
stages in sequence (see the repository README and
Appendix~\ref{app:reproduction} of the paper). To run the application stages
individually, from the **repository root**:

```bash
# Calibrated benchmark (Table 5), T=112 oracle slice
python papers/jss_2026/applications/etf_simulation_study.py \
    --seeds 15 --seed-start 101 --sample-sizes 112 --selectors oracle

# Empirical credit attribution (Table etf-credit + figure)
python papers/jss_2026/applications/run_etf_study.py --seeds 15

# Penalty-mode anchor sweep
python papers/jss_2026/applications/recalibrate_dgp_sweep.py \
    --anchor ols --sigmas 0.0 0.15 0.30 0.45 --seeds 15 --seed-start 101
```

## Data inputs (`data/`)

| File | Source | Role |
|---|---|---|
| `etf_universe.csv` | public ETF ticker list (in `applications/`) | universe definition for the panel build |
| `etf_excess_logreturns.csv` | from `fetch_etf_panel.py` | ETF excess monthly log returns (reproduction input) |
| `futures_risk_factors.csv` | production MATF factor NAV series | factor returns for the study and application |
| `riskfree_monthly.csv` | risk-free series | excess-return construction |

The ETF panel is rebuilt by `fetch_etf_panel.py` from `etf_universe.csv`; the
committed excess-return CSV lets the paper reproduce without a refetch.

## Outputs

The study scripts write raw and summary CSVs to
`papers/jss_2026/simulations/results_calibrated/` and the empirical table to the
same location; `plot_competitor_study.py` writes figures into
`papers/jss_2026/paper/figures/` for the LaTeX build.

## Dependencies

In addition to the published `factorlasso` wheel and its `[simulations]` extra:

- `yfinance` — only for `fetch_etf_panel.py`
- `pandas`, `numpy`, `matplotlib`, `scipy` — already in `factorlasso` deps
