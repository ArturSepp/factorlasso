"""Summarise verified base-grid evidence; does not complete the later stress studies."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import read_json, save_json, validate_root, verify


def verify_selected_package_fits(root, raw):
    """Recheck difficult selected simulation fits against the package solver."""
    import sys
    sys.path.insert(0, str(root / "source_snapshot/src"))
    from factorlasso import solve_lasso_cvx_problem
    from papers.prior_targets_2026.replication.estimators import prepare

    subset = raw[(raw.rho == .99) & (raw.n == 60) & (raw.snr == 1)
                 & (raw.seed == 920000) & (raw.method != "M5_post_lasso")]
    errors = []
    for _, row in subset.iterrows():
        sigma = np.array([[1., row.rho], [row.rho, 1.]])
        beta = np.array([1., row.secondary])
        noise_var = float(beta @ sigma @ beta / row.snr)
        rng = np.random.default_rng(np.random.SeedSequence(
            [int(row.seed), int(row.cell_id), 0]))
        x = rng.normal(size=(int(row.n), 2)) @ np.linalg.cholesky(sigma).T
        y = x @ beta + np.sqrt(noise_var) * rng.normal(size=int(row.n))
        z, v, _, _, xs, ys = prepare(x, y)
        target = np.array([row.target_dominant, row.target_secondary]) * xs / ys
        weights = np.ones((1, 2))
        if row.method == "M4_free_winner":
            weights[0, int(row.winner)] = 0.
        reference = solve_lasso_cvx_problem(
            z, v[:, None], reg_lambda=float(row["lambda"]),
            factors_beta_prior=target[None], penalty_weights=weights,
            solver="CLARABEL",
        ).estimated_beta[0] * ys / xs
        expected = np.array([row.beta_dominant, row.beta_secondary])
        errors.append(float(np.max(np.abs(reference - expected))))
    assert len(errors) == 28
    assert max(errors) < 3e-4, max(errors)
    return {"selected_fits_checked": len(errors), "maximum_beta_difference": max(errors)}



def main():
    """Write a bounded confirmation review and a figure in governed output storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    config = read_json(root / "source_snapshot/protocol.json")
    verification = verify(root, root / "E2_confirmation", "E2", config)
    destination = root / "confirmation_review_v2"
    destination.mkdir(exist_ok=False)
    table = pd.read_csv(root / "E2_confirmation/paired_summary.csv")
    raw = pd.read_csv(root / "E2_confirmation/selected_results.csv")
    package_checks = verify_selected_package_fits(root, raw)
    chosen = table[(table.rho.isin([.95, .99])) & (table.secondary == .3)
                   & (table.n == 60)]
    columns = ["rho", "secondary", "n", "snr", "method", "secondary_mse",
               "secondary_mse_delta", "secondary_mse_delta_low",
               "secondary_mse_delta_high", "prediction_mse_delta", "secondary_zero_rate"]
    chosen[columns].to_csv(destination / "high_correlation_cells.csv", index=False)
    counts = []
    for method, part in table.groupby("method"):
        counts.append({
            "method": method, "cells": len(part),
            "secondary_error_lower": int((part.secondary_mse_delta < 0).sum()),
            "secondary_error_higher": int((part.secondary_mse_delta > 0).sum()),
            "pointwise_interval_below_zero": int((part.secondary_mse_delta_high < 0).sum()),
            "pointwise_interval_above_zero": int((part.secondary_mse_delta_low > 0).sum()),
            "disclaimer": "Descriptive cell counts; intervals are not multiplicity adjusted",
        })
    pd.DataFrame(counts).to_csv(destination / "cell_counts.csv", index=False)
    edges = raw.groupby("method").grid_edge.mean()
    edges.rename("grid_edge_rate").to_csv(destination / "grid_edges.csv")
    endpoints = raw.assign(ols_end=raw["lambda"] == 0.,
                           target_end=raw["lambda"] == 10.).groupby("method")
    endpoints[["ols_end", "target_end"]].mean().to_csv(destination / "endpoints.csv")
    # The last path point is exact OLS; the first is a target-dominated solution.
    from matplotlib import pyplot as plt
    plt.switch_backend("Agg")
    methods = ["M2_auto", "M4_free_winner", "M3_noisy"]
    colours = ["#cc5533", "#4477aa", "#228833"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, secondary in enumerate([0., .3]):
        for column, snr in enumerate([1., 4.]):
            ax = axes[row, column]
            subset = table[(table.n == 60) & (table.secondary == secondary)
                           & (table.snr == snr)]
            for method, colour in zip(methods, colours):
                points = subset[subset.method == method].sort_values("rho")
                ax.plot(points.rho, points.secondary_mse_delta,
                        color=colour, marker="o", linewidth=1.5, label=method)
                ax.fill_between(points.rho, points.secondary_mse_delta_low,
                                points.secondary_mse_delta_high, color=colour, alpha=.13)
            ax.axhline(0, color="grey", linewidth=1)
            ax.set(title=f"True secondary beta={secondary:g}, SNR={snr:g}",
                   xlabel="Factor correlation",
                   ylabel="MSE difference vs zero target")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fresh-seed confirmation: n=60; 200 paired seeds per cell\n"
                 "Bands: pointwise 95% Monte Carlo intervals; noisy target adds information")
    fig.savefig(destination / "confirmation_secondary_error.png", dpi=170)
    plt.close(fig)
    # Recompute a primary cell's paired mean and SE directly from selected coefficients.
    check = raw[(raw.rho == .95) & (raw.secondary == .3)
                & (raw.n == 60) & (raw.snr == 1)]
    pivot = check.pivot(index="seed", columns="method", values="beta_secondary")
    differences = ((pivot.M2_auto - .3) ** 2 - (pivot.M1_zero - .3) ** 2)
    reference = table[(table.rho == .95) & (table.secondary == .3) & (table.n == 60)
                      & (table.snr == 1) & (table.method == "M2_auto")].iloc[0]
    np.testing.assert_allclose(differences.mean(), reference.secondary_mse_delta, atol=1e-12)
    np.testing.assert_allclose(differences.std(ddof=1) / np.sqrt(len(differences)),
                               reference.secondary_mse_delta_se, atol=1e-12)
    save_json(destination / "review_checks.json", {
        "stage_verification": verification, "independent_paired_aggregation": True,
        "raw_rows": len(raw), "cells": 120, "seeds_per_cell": 200,
        "pending_stresses": True, "package_checks": package_checks,
    })
    print(chosen[chosen.method.isin(["M1_zero", "M2_auto", "M4_free_winner",
                                    "M3_noisy"])][columns].to_string(index=False))
    print(pd.DataFrame(counts).to_string(index=False))


if __name__ == "__main__":
    main()



