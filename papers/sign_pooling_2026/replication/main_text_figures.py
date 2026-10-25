"""Main-text figures from saved results: gate tradeoff curves and eQTL per-gene R-squared.

Fig2_gate_tradeoff. For each of the five learned-group designs, the saved per-seed
gate curves (``learned_gate_curves/<design>_<seed>_gates.csv``, singleton and pooled
date-score gates at cut 0.5 over thresholds 0 to 8 in steps of 0.5) give, per seed and
threshold, the proportion of correct active signs and the null-cell retention. Panels
(a)-(e) draw every seed's curve faintly and the across-seed mean curve, parametrised by
the threshold, with the threshold-2 point marked. Panel (f) draws the paired pooled
minus singleton difference in correct active signs at null retention 0.10 from the
saved ``*_equal_null.csv`` oracle interpolations, with paired t intervals, and should
reproduce the article's equal-null table.

Fig3_eqtl_gene_r2. Per-gene R-squared from the nested yeast validation
(``eqtl/genes.csv`` for six methods, ``eqtl_sensitivity/runs/baseline_additions/genes.csv``
for the two added baselines): 64 genes by three repeats per method. Panel (a) shows the
distribution per method with the article's median-gene statistic (mean over repeats of
the per-repeat median) marked. Panel (b) shows, per gene and averaged over repeats, the
paired difference of pooled exclusion and pooled release against unconstrained HCGL.

Units. Proportions in [0, 1]; R-squared in original expression units as saved. Nothing
is refitted; the script only reads archived outputs. Printed summary numbers are for
cross-checking the article text.
"""
# packages
from __future__ import annotations
import argparse
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# project
if __package__:
    from .paths import OUTPUT_DIR, PAPER_DIR
else:
    from paths import OUTPUT_DIR, PAPER_DIR


DESIGNS: Dict[str, str] = {
    "coherent": "Coherent",
    "mixed_directions": "Mixed directions",
    "partial_support": "Partial support",
    "predictor_correlation": "Predictor correlation",
    "residual_correlation": "Residual correlation",
}
GATE_STYLE = {"singleton": dict(color="#aa634d", marker="s", ls="--"),
              "pooled": dict(color="#245875", marker="o", ls="-")}
EQTL_METHODS: List[Tuple[str, str]] = [
    ("lasso", "Lasso"),
    ("singleton_exclude", "Singleton-gated Lasso"),
    ("hcgl_unconstrained", "HCGL, unconstrained"),
    ("hcgl_singleton_exclude", "HCGL, singleton/exclude"),
    ("hcgl_exclude", "HCGL, pooled/exclude"),
    ("hcgl_unrestricted", "HCGL, pooled/release"),
    ("fcgl_unconstrained", "FCGL, unconstrained"),
    ("fcgl_exclude", "FCGL, pooled/exclude"),
]


def load_gate_curves(curves_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate per-seed gate curves and equal-null interpolations."""
    if not curves_dir.is_dir():
        raise FileNotFoundError(f"gate-curve directory not found: {curves_dir}")
    gates = pd.concat([pd.read_csv(p) for p in sorted(curves_dir.glob("*_gates.csv"))],
                      ignore_index=True)
    equal = pd.concat([pd.read_csv(p) for p in sorted(curves_dir.glob("*_equal_null.csv"))],
                      ignore_index=True)
    for name, frame in (("gates", gates), ("equal_null", equal)):
        if frame.empty:
            raise ValueError(f"no rows read for {name} in {curves_dir}")
    return gates, equal


def paired_interval(values: np.ndarray, level: float = 0.95) -> Tuple[float, float, float]:
    """Mean and paired t interval (Student quantile from scipy if present, else normal)."""
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    try:
        from scipy.stats import t as student_t
        q = float(student_t.ppf(0.5 + level / 2.0, df=n - 1))
    except ImportError:  # pragma: no cover
        q = 1.96
    return mean, mean - q * se, mean + q * se


def plot_gate_tradeoff(gates: pd.DataFrame,
                       equal: pd.DataFrame,
                       figure_path: Path,
                       target_null: float = 0.10,
                       mark_tau: float = 2.0
                       ) -> pd.DataFrame:
    """Five tradeoff panels plus the equal-null difference panel."""
    fig, axes = plt.subplots(2, 3, figsize=(5.17, 4.2), layout="constrained")
    summary_rows = []
    short = {"Predictor correlation": "Predictor corr.", "Residual correlation": "Residual corr."}
    for panel, (ax, (design, title)) in enumerate(zip(axes.flat[:5], DESIGNS.items())):
        title = short.get(title, title)
        d = gates[gates["scenario"] == design]
        for gate, style in GATE_STYLE.items():
            g = d[d["gate"] == gate]
            for _, s in g.groupby("seed"):
                s = s.sort_values("tau")
                ax.plot(s["null_retention"], s["active_correct"], color=style["color"],
                        lw=0.4, alpha=0.12, ls=style["ls"])
            mean = g.groupby("tau")[["null_retention", "active_correct"]].mean().sort_index()
            ax.plot(mean["null_retention"], mean["active_correct"], color=style["color"],
                    lw=1.3, ls=style["ls"], label=gate)
            if mark_tau in mean.index:
                ax.plot(mean.loc[mark_tau, "null_retention"], mean.loc[mark_tau, "active_correct"],
                        marker=style["marker"], color=style["color"], ms=5, ls="none",
                        markeredgecolor="white", markeredgewidth=0.6)
        ax.axvline(target_null, color="0.5", lw=0.7, ls=":")
        ax.set_title(f"({chr(97 + panel)}) {title}", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[3:5]:
        ax.set_xlabel("null-cell retention", fontsize=8.5)
    axes[0, 0].set_ylabel("correct active signs", fontsize=8.5)
    axes[1, 0].set_ylabel("correct active signs", fontsize=8.5)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    # Panel (f): paired difference at the target null retention.
    ax = axes[1, 2]
    rows = []
    for k, (design, title) in enumerate(DESIGNS.items()):
        e = equal[(equal["scenario"] == design)
                  & (np.isclose(equal["target_null_retention"], target_null))]
        mean, lo, hi = paired_interval(e["active_correct_difference"].to_numpy())
        rows.append((title, mean, lo, hi, len(e)))
        ax.errorbar(mean, k, xerr=[[mean - lo], [hi - mean]], fmt="o", color="#1b1b1b",
                    ms=4, capsize=2, lw=0.9)
    ax.axvline(0.0, color="0.5", lw=0.7)
    ax.set_yticks(range(len(DESIGNS)))
    ax.set_yticklabels([t for t in DESIGNS.values()], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("pooled minus singleton", fontsize=8.5)
    ax.set_title(f"(f) null retention {target_null:.2f}", fontsize=8.5)
    ax.tick_params(labelsize=8)
    fig.savefig(figure_path)
    fig.savefig(figure_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    summary = pd.DataFrame(rows, columns=["design", "difference", "ci_low", "ci_high", "seeds"])
    return summary


def load_eqtl_genes(eqtl_dir: Path, additions_dir: Path) -> pd.DataFrame:
    frames = []
    for directory in (eqtl_dir, additions_dir):
        path = directory / "genes.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        frames.append(pd.read_csv(path))
    genes = pd.concat(frames, ignore_index=True)
    missing = {m for m, _ in EQTL_METHODS} - set(genes["method"])
    if missing:
        raise ValueError(f"methods absent from saved gene results: {sorted(missing)}")
    return genes


def plot_eqtl_genes(genes: pd.DataFrame, figure_path: Path) -> pd.DataFrame:
    """Distribution of per-gene R-squared by method and paired per-gene differences."""
    fig, axes = plt.subplots(1, 2, figsize=(5.17, 3.3), layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    data, labels, medians = [], [], []
    for key, label in EQTL_METHODS:
        g = genes[genes["method"] == key]
        data.append(g["r2"].to_numpy())
        labels.append(label)
        medians.append(g.groupby("repeat")["r2"].median().mean())
    positions = np.arange(len(EQTL_METHODS))
    ax.boxplot(data, positions=positions, vert=False, widths=0.6, showfliers=False,
               medianprops=dict(color="0.3", lw=0.8),
               boxprops=dict(color="0.3", lw=0.8), whiskerprops=dict(color="0.3", lw=0.8),
               capprops=dict(color="0.3", lw=0.8))
    for pos, values in zip(positions, data):
        jitter = (np.random.default_rng(pos).uniform(-0.18, 0.18, len(values)))
        ax.plot(values, pos + jitter, ".", color="#245875", ms=2.2, alpha=0.35)
    ax.plot(medians, positions, marker="D", color="#aa634d", ms=4.5, ls="none")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("per-gene held-out $R^2$ (64 genes x 3 repeats)", fontsize=8.5)
    ax.set_title("(a) prediction by method", fontsize=8.5)
    ax.tick_params(labelsize=8)
    # Panel (b): paired per-gene differences against unconstrained HCGL.
    ax = axes[1]
    wide = genes.pivot_table(index=["gene", "repeat"], columns="method", values="r2")
    per_gene = wide.groupby(level="gene").mean()
    d_excl = (per_gene["hcgl_exclude"] - per_gene["hcgl_unconstrained"]).sort_values()
    d_rel = (per_gene["hcgl_unrestricted"] - per_gene["hcgl_unconstrained"]).loc[d_excl.index]
    rank = np.arange(1, len(d_excl) + 1)
    ax.plot(rank, d_excl.to_numpy(), "o", color="#aa634d", ms=3, label="pooled/exclude")
    ax.plot(rank, d_rel.to_numpy(), "s", color="#245875", ms=3, label="pooled/release")
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set_xlabel("gene rank", fontsize=8.5)
    ax.set_ylabel("$R^2$ minus unconstrained HCGL", fontsize=8.5)
    ax.set_title("(b) difference per gene", fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.tick_params(labelsize=8)
    fig.savefig(figure_path)
    fig.savefig(figure_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    summary = pd.DataFrame({
        "genes": [len(d_excl)],
        "exclude_negative": [int((d_excl < 0).sum())],
        "release_negative": [int((d_rel < 0).sum())],
        "exclude_mean_diff": [float(d_excl.mean())],
        "release_mean_diff": [float(d_rel.mean())],
        "exclude_median_diff": [float(d_excl.median())],
        "release_median_diff": [float(d_rel.median())],
    })
    return summary


class LocalTest(Enum):
    GATE_TRADEOFF = 1
    EQTL_GENES = 2
    ALL = 3


def run_local_test(local_test: LocalTest,
                   curves_dir: Path,
                   eqtl_dir: Path,
                   additions_dir: Path,
                   figure_dir: Path,
                   results_dir: Path
                   ) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if local_test in (LocalTest.GATE_TRADEOFF, LocalTest.ALL):
        gates, equal = load_gate_curves(curves_dir)
        summary = plot_gate_tradeoff(gates=gates, equal=equal,
                                     figure_path=figure_dir / "Fig2_gate_tradeoff.pdf")
        summary.to_csv(results_dir / "fig1_equal_null_summary.csv", index=False)
        print(summary.round(4).to_string(index=False))
    if local_test in (LocalTest.EQTL_GENES, LocalTest.ALL):
        genes = load_eqtl_genes(eqtl_dir=eqtl_dir, additions_dir=additions_dir)
        summary = plot_eqtl_genes(genes=genes, figure_path=figure_dir / "Fig3_eqtl_gene_r2.pdf")
        summary.to_csv(results_dir / "fig3_eqtl_gene_summary.csv", index=False)
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--curves-dir", type=Path, default=OUTPUT_DIR / "learned_gate_curves")
    parser.add_argument("--eqtl-dir", type=Path, default=OUTPUT_DIR / "eqtl")
    parser.add_argument("--additions-dir", type=Path,
                        default=OUTPUT_DIR / "eqtl_sensitivity" / "runs" / "baseline_additions")
    parser.add_argument("--figure-dir", type=Path, default=PAPER_DIR / "paper" / "figures")
    parser.add_argument("--results-dir", type=Path, default=OUTPUT_DIR / "results")
    parser.add_argument("--only", choices=[t.name for t in LocalTest], default="ALL")
    args = parser.parse_args()
    run_local_test(local_test=LocalTest[args.only],
                   curves_dir=args.curves_dir,
                   eqtl_dir=args.eqtl_dir,
                   additions_dir=args.additions_dir,
                   figure_dir=args.figure_dir,
                   results_dir=args.results_dir)
