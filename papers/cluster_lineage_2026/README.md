# S&P 500 cluster-lineage reproduction

This folder reproduces the cluster-lineage stability experiment using FactorLasso primitives
only. It deliberately does not import OptimalPortfolios or qis. The expanding-window fits use
the same weekly S&P 500 input CSV and estimator configuration as the original pipeline study:

- W-WED log asset returns, 156-observation factor-lasso span;
- FCGL with `reg_lambda=1e-5`, Ward clustering and the calibrated sign settings;
- a W-WED span-52 demeaned EWMA market-factor covariance, annualised by 52;
- 60 month-end evaluation dates from 2021-08-31 through 2026-07-31;
- baseline unsmoothed partitions and M1 partition-distance bonus with delta 0.05.

The only assembly difference is architectural: this script constructs the equal-weight market
factor return and the `CurrentFactorCovarData` diagnostics directly, instead of calling the
consumer package's estimator wrapper and frequency/annualisation helpers. Return convention,
spans, annualisation, clustering, penalty geometry, and expanding-window dates remain the same.
The output table reports exact deltas against the original pipeline values so any resulting
difference is visible.

## Inputs and outputs

By default, inputs are read from:

```text
~/OneDrive/analytics/outputs/factorlasso_returns/
```

Set `FACTORLASSO_SP500_DATA_DIR` to override that folder. It must contain
`sp500_adjusted_close_2005_to_current.csv`. Fits and CSV tables are cached outside the repository
under `FACTORLASSO_LINEAGE_OUTPUT_DIR` or, by default,
`~/OneDrive/analytics/outputs/factorlasso_cluster_lineage/`.

## Run

```bash
python papers/cluster_lineage_2026/reproduce_sp500.py
```

`FACTORLASSO_LINEAGE_WORKERS` controls parallel fitting and defaults to 2. The final assertions
require baseline lineage churn within 2% of 3.211469 and at least 75% M1 churn reduction.

The script contains an optional guarded `fetch_prices_with_yfinance` helper for external users.
It is never called by the reproduction and yfinance is not a FactorLasso dependency.
