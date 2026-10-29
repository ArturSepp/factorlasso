"""Tables, paired contrasts and the per-gene figure for the jointly tuned eQTL run.

Reads the outputs of ``eqtl_joint_tuning.py`` (both arms) and the archived MOSEK
fixed-setting outputs (``eqtl/`` for six methods and ``baseline_additions/`` for the
two added baselines). Writes:

  table_joint_prediction.tex    joint arm: median, mean and pooled held-out R2 by method,
                                means over the three repeats with split SD in parentheses
  table_joint_selections.tex    joint arm: selected cut, threshold and fraction counts
                                over the nine outer fits, by method
  table_reference_check.csv     reference arm under this run's solver against the archived
                                MOSEK values, with the largest absolute deviation
  contrasts.csv                 paired median-gene R2 differences by repeat for both arms
  key_numbers.json              the figures quoted in the manuscript text
  Fig3_eqtl_gene_r2.pdf/.png    per-gene distributions and paired differences, fixed settings
                                (archived MOSEK run) above and joint tuning below

Units: held-out R2 in original expression units as saved. Nothing is refitted.
"""
# packages
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
# project
if __package__:
    from .main_text_figures import EQTL_METHODS, plot_eqtl_genes, plot_eqtl_genes_two_arms
else:
    from main_text_figures import EQTL_METHODS, plot_eqtl_genes, plot_eqtl_genes_two_arms


LABELS: Dict[str, str] = dict(EQTL_METHODS)
CONTRASTS: List[Tuple[str, str, str]] = [
    ("pooled exclusion minus unconstrained HCGL", "hcgl_exclude", "hcgl_unconstrained"),
    ("pooled exclusion minus singleton exclusion HCGL", "hcgl_exclude", "hcgl_singleton_exclude"),
    ("pooled release minus unconstrained HCGL", "hcgl_unrestricted", "hcgl_unconstrained"),
    ("FCGL exclusion minus unconstrained FCGL", "fcgl_exclude", "fcgl_unconstrained"),
]


def fmt(mean: float, sd: float) -> str:
    return f"{mean:.3f} ({sd:.3f})"


def prediction_table(summary: pd.DataFrame, arm: str, label: str) -> str:
    rows = []
    s = summary[summary["arm"] == arm].set_index("method")
    for key, name in EQTL_METHODS:
        r = s.loc[key]
        rows.append(f"{name} & {fmt(r.r2_median_mean, r.r2_median_split_sd)} & "
                    f"{fmt(r.r2_mean, r.r2_mean_split_sd)} & "
                    f"{fmt(r.r2_pooled, r.r2_pooled_split_sd)} \\\\")
    return (r"\begin{tabular}{lccc}\toprule" "\n"
            r"Method & Median gene $R^2$ & Mean gene $R^2$ & Pooled $R^2$\\\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}" "\n")


def selection_table(folds: pd.DataFrame, arm: str) -> str:
    f = folds[folds["arm"] == arm]
    rows = []
    for key, name in EQTL_METHODS:
        g = f[f["method"] == key]
        def counts(col: str) -> str:
            if g[col].isna().all():
                return "--"
            c = g[col].value_counts().sort_index()
            return ", ".join(f"{v:g}: {n}" for v, n in c.items())
        rows.append(f"{name} & {counts('cut')} & {counts('tau')} & {counts('fraction')} \\\\")
    return (r"\begin{tabular}{llll}\toprule" "\n"
            r"Method & Cut $\kappa$ & Threshold $\tau$ & Fraction of $\lambda_{\max}$\\\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}" "\n")


def reference_check(summary: pd.DataFrame, archived: pd.DataFrame) -> pd.DataFrame:
    ref = summary[summary["arm"] == "reference"].set_index("method")
    rows = []
    for key, name in EQTL_METHODS:
        for col in ("r2_median_mean", "r2_mean", "r2_pooled"):
            rows.append(dict(method=key, label=name, metric=col,
                             clarabel=float(ref.loc[key, col]),
                             mosek=float(archived.set_index("method").loc[key, col])))
    out = pd.DataFrame(rows)
    out["abs_diff"] = (out["clarabel"] - out["mosek"]).abs()
    return out


def contrasts(repeats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, frame in repeats.groupby("arm", sort=False):
        wide = frame.pivot(index="repeat", columns="method", values="r2_gene_median")
        for label, a, b in CONTRASTS:
            d = wide[a] - wide[b]
            rows.append(dict(arm=arm, contrast=label, mean=float(d.mean()),
                             split_sd=float(d.std(ddof=1)), negative_repeats=int((d < 0).sum()),
                             repeats=int(len(d)), by_repeat=[round(float(v), 4) for v in d]))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archived-eqtl", type=Path, required=True)
    parser.add_argument("--archived-additions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.run / "summary.csv")
    folds = pd.read_csv(args.run / "folds.csv")
    repeats = pd.read_csv(args.run / "repeats.csv")
    genes = pd.read_csv(args.run / "genes.csv")
    archived = pd.concat([pd.read_csv(args.archived_eqtl / "summary.csv"),
                          pd.read_csv(args.archived_additions / "summary.csv")], ignore_index=True)
    (args.out / "table_joint_prediction.tex").write_text(
        prediction_table(summary, "joint", "joint"), encoding="utf-8")
    (args.out / "table_reference_prediction.tex").write_text(
        prediction_table(summary, "reference", "reference"), encoding="utf-8")
    (args.out / "table_joint_selections.tex").write_text(
        selection_table(folds, "joint"), encoding="utf-8")
    check = reference_check(summary, archived)
    check.to_csv(args.out / "table_reference_check.csv", index=False)
    con = contrasts(repeats)
    con.to_csv(args.out / "contrasts.csv", index=False)
    joint_genes = genes[genes["arm"] == "joint"].drop(columns="arm")
    fixed_genes = pd.concat([pd.read_csv(args.archived_eqtl / "genes.csv"),
                             pd.read_csv(args.archived_additions / "genes.csv")], ignore_index=True)
    gene_summary = plot_eqtl_genes_two_arms(fixed_genes, joint_genes,
                                            args.figure_dir / "Fig3_eqtl_gene_r2.pdf")
    gene_summary.to_csv(args.out / "fig3_gene_summary.csv", index=False)
    plot_eqtl_genes(joint_genes, args.figure_dir / "FigS_eqtl_gene_r2_joint.pdf")
    js = summary[summary["arm"] == "joint"].set_index("method")
    key = dict(
        reference_check_max_abs_diff=float(check["abs_diff"].max()),
        reference_check_max_abs_diff_metric=check.loc[check["abs_diff"].idxmax(), ["label", "metric"]].to_dict(),
        joint_median_r2={LABELS[m]: round(float(js.loc[m, "r2_median_mean"]), 3) for m, _ in EQTL_METHODS},
        joint_contrasts=con[con["arm"] == "joint"].drop(columns="arm").to_dict("records"),
        reference_contrasts=con[con["arm"] == "reference"].drop(columns="arm").to_dict("records"),
        joint_selections=folds[folds["arm"] == "joint"].groupby("method")[["cut", "tau", "fraction"]]
            .agg(lambda s: s.value_counts(dropna=False).to_dict()).to_dict("index"),
        gene_summary=gene_summary.to_dict("records"),
    )
    (args.out / "key_numbers.json").write_text(json.dumps(key, indent=2, default=str), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("=== reference arm (this solver) vs archived MOSEK ===")
        print(check.round(4).to_string(index=False))
        print("=== joint arm summary ===")
        print(summary[summary["arm"] == "joint"].round(4).to_string(index=False))
        print("=== contrasts ===")
        print(con.to_string(index=False))
        print("=== selections (joint) ===")
        print(selection_table(folds, "joint"))


if __name__ == "__main__":
    main()
