# JSS 2026 — Simulation study

Simulation harness and study config driving the §5 ablation in the
JSS submission paper. Not part of the published `factorlasso` wheel.

## Specification

- **Regimes:** 19 cells across (sparsity, sign_mix, snr, T, N, K, factor_cov, residual_cov)
- **Seeds:** 5 per cell (42–46)
- **λ grid:** 5 values, log-spaced 1e-5 to 1e-1
- **Estimators:** 6 factorlasso configurations (LASSO baseline through Sparse Group LASSO + sign + adaptive)
- **Total fits:** 19 × 5 × 6 × 5 = 2,850

External competitor estimators (`sklearn_lasso`, `sparsegl_sgl`, `adelie_grp`) are
commented out in `study.yaml`. Uncomment and wire in `estimators.py` to include.

## Modules

| Module | Role |
|---|---|
| `dgp.py` | Data-generating processes (synthetic factor panels with controllable cluster / sign / SNR structure) |
| `estimators.py` | Unified `fit(X, y, λ, …) → EstimatorResult` wrappers around each estimator |
| `metrics.py` | Pure metric functions on `(β_true, β_hat, …)` |
| `run.py` | Orchestration CLI; expands `study.yaml` into the full grid, runs in parallel, saves long-form results |

## Running

From the **repository root**:

```bash
# Full study
python -m papers.jss_2026.simulations.run \
    --config papers/jss_2026/simulations/study.yaml \
    --output papers/jss_2026/simulations/results

# Smoke test (1 seed, ~2 min)
python -m papers.jss_2026.simulations.run \
    --config papers/jss_2026/simulations/study.yaml \
    --output /tmp/smoke --seeds-limit 1

# Dry run (config validation, no fits)
python -m papers.jss_2026.simulations.run \
    --config papers/jss_2026/simulations/study.yaml \
    --output /tmp/dry --dry-run
```

Expected wall-clock at default 8-core parallelism: 10–15 minutes for the full run.
The `moderate_clean_midsnr_N200` cell dominates (~3–5 minutes alone) because N=200;
drop it from `study.yaml` for fast iteration and add it back for the final paper-table run.

## Outputs

After a successful run, `results/` contains:

| File | Contents |
|---|---|
| `results_long.parquet` | One row per (regime, seed, estimator, λ). Source for any reanalysis. |
| `results_oracle_lambda.parquet` | For each (regime, seed, estimator), the row at the λ minimising `beta_mse_norm`. **Source for publication tables.** |
| `manifest.json` | Reproducibility receipt — config snapshot, software versions, wall-clock, timestamps. |

## Result schema

Each row in `results_long.parquet` carries:

- Identification: `regime_id`, `seed`, `estimator`, `reg_lambda`
- Status: `status` ∈ {`ok`, `failed`, `not_implemented`}, `error` (when applicable)
- Timing: `runtime` (seconds, wall-clock per fit)
- Metrics: `support_f1`, `sign_rate`, `beta_mse_norm`, `cluster_coherence_hat`, `factor_rp_rmse`, `oos_r2`, `cluster_ari`
- Regime parameters: `T`, `N`, `M`, `K`, `rho_beta`, `sign_mix`, `sparsity`, `snr`, `factor_cov`, `residual_cov`

`oos_r2` is NaN unless a held-out window is supplied (currently not in the JSS 2026 study).

## Regime axes — what each cell tests

| Group | Cells | Tests |
|---|---|---|
| Core 2×2×2 | `*_clean_lowsnr`, `*_clean_highsnr`, `*_mixed_lowsnr`, `*_mixed_highsnr` (4×2 sparsity) | SNR × sparsity × sign_mix interaction |
| Centerpoints | `moderate_*_midsnr` | Default reference point |
| Idiosyncratic | `*_idio_midsnr` | Method should not hurt on neutral data |
| T variation | `sparse_clean_lowsnr_T{60,240}` | Small-T benefit of cluster pooling |
| N variation | `moderate_clean_midsnr_N200` | Scaling to larger universes |
| K variation | `moderate_clean_midsnr_K{3,12}` | Sensitivity to true cluster count |
| Factor cov | `orthogonal_factors` | Robustness to factor structure |
| Residual cov | `rank1_residuals` | Robustness to unmodelled residual factor |
