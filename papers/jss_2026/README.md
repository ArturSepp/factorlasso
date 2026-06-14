# factorlasso — JSS 2026 replication materials

Replication for *"factorlasso: Hierarchical Clustering Group LASSO (HCGL) with
Cluster-Pooled Sign Derivation for Multi-Asset Factor Models in Python."*

## Install
The package is on PyPI:

    pip install factorlasso

(or install the accompanying software-source archive with `pip install .`).
The replication also needs the plotting stack and the **four comparison
packages benchmarked in Table tab:competitor**:

    pip install matplotlib joblib scikit-learn skglm asgl adelie

All four of scikit-learn, skglm, asgl, and adelie are REQUIRED: without asgl or
adelie, their rows in the competitor table are produced as blank ("--").
Platform note (JSS): asgl and adelie are compiled packages. If a prebuilt wheel
is unavailable on your platform, install from source or use a platform that has
wheels; this is the only external platform dependency.

## Reproduce
A single commented standalone script regenerates every computed exhibit and
writes a log with a session-information footer (the Python analogue of
`code.html` / `sessionInfo()`):

    python replicate.py             # quick: 1 sim seed, 2 ETF seeds (~minutes)
    python replicate.py --full      # EXACT manuscript numbers (15 ETF seeds); longer
    python replicate.py --skip-sim  # reuse the committed simulation parquet
    python replicate.py --log replication_output.txt

`--full` reproduces every number in the manuscript: the synthetic study at its
full seed set and the ETF competitor study at the paper's 15 seeds (101-115)
over the full sample ladder T in {60, 112, 240}. `--quick` is for a fast smoke
check; its tables carry wider Monte Carlo error and a single sample size.

## Stages → manuscript exhibits
    1  simulations.run                     -> simulations/results/*.parquet
    2  paper/analysis.py                    -> Table 2; Figures 1-4
    3  applications/etf_simulation_study.py -> Table (tab:competitor)
    4  applications/plot_competitor_study.py-> Figures etf_study_selector_contrast,
                                               etf_study_t_robustness
    5  applications/run_etf_study.py        -> Table (tab:etf-credit);
                                               Figure etf_credit_beta_vs_lambda

## Data (frozen)
`applications/data/` ships the frozen 2026-06 inputs:

    futures_risk_factors.csv     factor NAV series (9 MATF factors)
    etf_excess_logreturns.csv    102-fund ETF excess monthly log returns (2017-02..2026-06)

(plus prices / log-returns / risk-free / meta as provenance).
`applications/fetch_etf_panel.py` documents how the public ETF panel was built
from `etf_universe.csv` via yfinance. The committed CSV is the canonical input,
so the results reproduce regardless of any later data vintage.

## Expected output
`replication_output.txt` is a captured run with package versions and platform.
The deterministic empirical anchors (OLS mean credit 0.82, prior 0.29, true
credit 0.36, credit-equity correlation 0.84) reproduce exactly on any platform.
All figures and tables are deterministic given the committed data and the fixed
seeds 101-115.
