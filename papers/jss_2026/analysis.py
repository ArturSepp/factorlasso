"""
analysis.py — regenerate the §5 simulation tables and figures.

Reads the oracle-lambda parquet produced by
``papers.jss_2026.simulations.run`` and writes:

  * Table 2 (headline ablation) as ``table2_headline_ablation.csv`` and a
    LaTeX-ready ``table2_headline_ablation.tex``;
  * Figure 1 ``fig1_cluster_coherence_by_sign_mix.png``;
  * Figure 2 ``fig2_beta_mse_core_grid.png``;
  * Figure 3 ``fig3_T_effect.png``;
  * Figure 4 ``fig4_lambda_selection.png``.

Usage
-----
    python papers/jss_2026/paper/analysis.py \
        --results papers/jss_2026/simulations/results \
        --output  papers/jss_2026/paper/figures

The script consumes ``results_oracle_lambda.parquet`` for the tables and
Figures 1-3 (which report metrics at the oracle-selected lambda) and
``results_long.parquet`` for Figure 4 (the distribution of the
oracle-selected lambda, which needs the full grid).

The figures are written at 150 dpi, matching the source-of-truth artefacts
the manuscript embeds via \\includegraphics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------- #
# Estimator label map and display order (paper §5 Table 2 / figures).
# --------------------------------------------------------------------- #
ESTIMATOR_LABELS = {
    "factorlasso_lasso": "LASSO",
    "factorlasso_grp_oracle": "GRP-ORACLE",
    "factorlasso_grp_hcgl": "GRP-HCGL",
    "factorlasso_grp_hcgl_sign": "+SIGN",
    "factorlasso_grp_hcgl_sign_adapt": "+SIGN+ADAPT",
    "factorlasso_sgl_hcgl_sign_adapt": "SGL+SIGN+ADAPT",
}
ESTIMATOR_ORDER = [
    "LASSO",
    "GRP-ORACLE",
    "GRP-HCGL",
    "+SIGN",
    "+SIGN+ADAPT",
    "SGL+SIGN+ADAPT",
]
# The three sign-derivation rows vs the three unconstrained rows.
SIGN_ROWS = {"+SIGN", "+SIGN+ADAPT", "SGL+SIGN+ADAPT"}
UNCONSTRAINED_ROWS = {"LASSO", "GRP-ORACLE", "GRP-HCGL"}

# Metric column → display name and "higher is better" flag.
METRICS = [
    ("support_f1", "Support $F_1$", True),
    ("sign_rate", "Sign rate", True),
    ("beta_mse_norm", r"$\beta$-MSE", False),
    ("cluster_coherence_hat", "Coherence", True),
    ("factor_rp_rmse", "Risk-premium RMSE", False),
]


def _label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["est"] = df["estimator"].map(ESTIMATOR_LABELS)
    return df


def load_oracle(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "results_oracle_lambda.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the study first:\n"
            f"  python -m papers.jss_2026.simulations.run "
            f"--config papers/jss_2026/simulations/study.yaml "
            f"--output {results_dir}"
        )
    return _label(pd.read_parquet(path))


def load_long(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "results_long.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the study first.")
    return _label(pd.read_parquet(path))


# --------------------------------------------------------------------- #
# Table 2 — headline ablation (mean of each metric across all regimes
# and seeds at the oracle-selected lambda).
# --------------------------------------------------------------------- #
def table2_headline(oracle: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [m[0] for m in METRICS]
    agg = (
        oracle.groupby("est")[cols].mean().reindex(ESTIMATOR_ORDER)
    )
    agg.columns = [m[1] for m in METRICS]

    csv_path = out_dir / "table2_headline_ablation.csv"
    agg.round(3).to_csv(csv_path)

    # LaTeX body matching the manuscript tabular.
    tex_path = out_dir / "table2_headline_ablation.tex"
    lines = []
    for est in ESTIMATOR_ORDER:
        row = agg.loc[est]
        cells = " & ".join(f"{row[c]:.3f}" for c in agg.columns)
        lines.append(rf"\code{{{est}}} & {cells} \\")
    tex_path.write_text("\n".join(lines) + "\n")

    print(f"[table2] wrote {csv_path.name} and {tex_path.name}")
    print(agg.round(3).to_string())
    return agg


# --------------------------------------------------------------------- #
# Figure 1 — cluster sign coherence by sign-mix regime.
# --------------------------------------------------------------------- #
def fig1_coherence_by_sign_mix(oracle: pd.DataFrame, out_dir: Path) -> None:
    sign_mixes = ["clean", "mixed", "idiosyncratic"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ax, mix in zip(axes, sign_mixes):
        sub = oracle[oracle["sign_mix"] == mix]
        means = sub.groupby("est")["cluster_coherence_hat"].mean()
        stds = sub.groupby("est")["cluster_coherence_hat"].std()
        means = means.reindex(ESTIMATOR_ORDER)
        stds = stds.reindex(ESTIMATOR_ORDER)
        colors = [
            "#4C72B0" if e in SIGN_ROWS else "#999999"
            for e in ESTIMATOR_ORDER
        ]
        ax.bar(
            range(len(ESTIMATOR_ORDER)),
            means.values,
            yerr=stds.values,
            color=colors,
            capsize=3,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(range(len(ESTIMATOR_ORDER)))
        ax.set_xticklabels(ESTIMATOR_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{mix} sign-mix")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Cluster sign coherence")
    fig.suptitle(
        "Cluster sign coherence by sign-mix regime "
        "(grey: unconstrained, blue: sign-derivation)"
    )
    fig.tight_layout()
    path = out_dir / "fig1_cluster_coherence_by_sign_mix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] wrote {path.name}")


# --------------------------------------------------------------------- #
# Figure 2 — normalised beta-MSE across the core 2x2x2 ablation.
# --------------------------------------------------------------------- #
def fig2_core_grid(oracle: pd.DataFrame, out_dir: Path) -> None:
    # Core 2x2x2: sparsity {sparse,dense} x snr {0.10,0.50} x sign_mix {clean,mixed}.
    core = oracle[
        oracle["regime_id"].isin(
            [
                "sparse_clean_lowsnr",
                "sparse_clean_highsnr",
                "sparse_mixed_lowsnr",
                "sparse_mixed_highsnr",
                "dense_clean_lowsnr",
                "dense_clean_highsnr",
                "dense_mixed_lowsnr",
                "dense_mixed_highsnr",
            ]
        )
    ]
    panels = [
        ("sparse", 0.10, "sparse, low-SNR"),
        ("sparse", 0.50, "sparse, high-SNR"),
        ("dense", 0.10, "dense, low-SNR"),
        ("dense", 0.50, "dense, high-SNR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, (sp, snr, title) in zip(axes.ravel(), panels):
        sub = core[(core["sparsity"] == sp) & (np.isclose(core["snr"], snr))]
        width = 0.38
        xs = np.arange(len(ESTIMATOR_ORDER))
        for off, mix, color in [
            (-width / 2, "clean", "#55A868"),
            (+width / 2, "mixed", "#DD8452"),
        ]:
            m = (
                sub[sub["sign_mix"] == mix]
                .groupby("est")["beta_mse_norm"]
                .mean()
                .reindex(ESTIMATOR_ORDER)
            )
            ax.bar(
                xs + off, m.values, width, label=mix, color=color,
                edgecolor="black", linewidth=0.4,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(ESTIMATOR_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(r"Normalised $\beta$-MSE")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(r"Normalised $\beta$-MSE across the core $2\times2\times2$ ablation")
    fig.tight_layout()
    path = out_dir / "fig2_beta_mse_core_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] wrote {path.name}")


# --------------------------------------------------------------------- #
# Figure 3 — sample-length sensitivity (sparse, clean, low-SNR).
# --------------------------------------------------------------------- #
def fig3_T_effect(oracle: pd.DataFrame, out_dir: Path) -> None:
    # The three T regimes on (sparse, clean, low-SNR).
    t_regimes = {
        60: "sparse_clean_lowsnr_T60",
        120: "sparse_clean_lowsnr",
        240: "sparse_clean_lowsnr_T240",
    }
    Ts = [60, 120, 240]

    fig, (ax_mse, ax_coh) = plt.subplots(1, 2, figsize=(12, 4.5))
    for est in ESTIMATOR_ORDER:
        mse = []
        coh = []
        for T in Ts:
            sub = oracle[
                (oracle["regime_id"] == t_regimes[T]) & (oracle["est"] == est)
            ]
            mse.append(sub["beta_mse_norm"].mean())
            coh.append(sub["cluster_coherence_hat"].mean())
        style = dict(marker="o", linestyle="-") if est in SIGN_ROWS else dict(
            marker="s", linestyle="--"
        )
        ax_mse.plot(Ts, mse, label=est, **style)
        ax_coh.plot(Ts, coh, label=est, **style)

    ax_mse.set_xlabel("Sample length $T$ (months)")
    ax_mse.set_ylabel(r"Normalised $\beta$-MSE")
    ax_mse.set_title(r"$\beta$-MSE vs $T$ (lower better)")
    ax_mse.set_xticks(Ts)
    ax_mse.grid(alpha=0.3)
    ax_coh.set_xlabel("Sample length $T$ (months)")
    ax_coh.set_ylabel("Cluster sign coherence")
    ax_coh.set_title("Coherence vs $T$ (higher better)")
    ax_coh.set_xticks(Ts)
    ax_coh.grid(alpha=0.3)
    ax_coh.legend(fontsize=8, loc="center right")
    fig.suptitle("Effect of sample length on sparse + clean + low-SNR regime")
    fig.tight_layout()
    path = out_dir / "fig3_T_effect.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] wrote {path.name}")


# --------------------------------------------------------------------- #
# Figure 4 — distribution of the oracle-selected lambda per estimator.
# Needs the full long table (the oracle table keeps only the winner).
# --------------------------------------------------------------------- #
def fig4_lambda_selection(long: pd.DataFrame, out_dir: Path) -> None:
    ok = long[long["status"] == "ok"].copy()
    # Per (regime, seed, estimator): the lambda minimising beta_mse_norm.
    idx = (
        ok.groupby(["regime_id", "seed", "estimator"])["beta_mse_norm"]
        .idxmin()
        .dropna()
    )
    sel = ok.loc[idx]
    sel = _label(sel)
    lambdas = sorted(ok["reg_lambda"].unique())
    lambda_labels = [f"$10^{{{int(round(np.log10(lam)))}}}$" for lam in lambdas]

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(lambdas))
    width = 0.13
    for i, est in enumerate(ESTIMATOR_ORDER):
        sub = sel[sel["est"] == est]
        counts = [
            (np.isclose(sub["reg_lambda"], lam)).mean() for lam in lambdas
        ]
        ax.bar(xs + (i - 2.5) * width, counts, width, label=est)
    ax.set_xticks(xs)
    ax.set_xticklabels(lambda_labels)
    ax.set_xlabel(r"Oracle-selected $\lambda$")
    ax.set_ylabel("Fraction of (regime, seed) pairs")
    ax.set_title(r"Distribution of oracle-selected $\lambda$ per estimator")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig4_lambda_selection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] wrote {path.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python papers/jss_2026/paper/analysis.py",
        description=(
            "Regenerate the §5 headline ablation table and Figures 1-4 from "
            "the simulation study parquet outputs."
        ),
    )
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--results",
        type=Path,
        default=here / "simulations" / "results",
        help="Directory holding results_oracle_lambda.parquet and "
        "results_long.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "paper" / "figures",
        help="Directory to write the table and figures into.",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    oracle = load_oracle(args.results)
    table2_headline(oracle, args.output)
    fig1_coherence_by_sign_mix(oracle, args.output)
    fig2_core_grid(oracle, args.output)
    fig3_T_effect(oracle, args.output)

    long = load_long(args.results)
    fig4_lambda_selection(long, args.output)

    print(f"\nDone. Artefacts written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
