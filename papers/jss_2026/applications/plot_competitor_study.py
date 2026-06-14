"""
Figures for the ETF-calibrated competitor study (JSS 2026).

Consumes ``etf_competitor_study_raw.csv`` (one row per method x seed x T x
selector, written by ``etf_simulation_study.py``) and renders three paper
figures into ``--figdir``:

1. ``etf_study_selector_contrast`` — the headline. β-MSE and credit recovery
   per method, oracle-λ vs feasible BIC, at T=112. Shows the unstructured
   methods reverting to OLS under BIC while the structured factorlasso configs
   stay stable and keep recovering credit.
2. ``etf_study_oracle_metrics`` — small multiples of the discriminating metrics
   (β-MSE, support F1, OOS R², credit recovery, credit risk-share error,
   systematic-covariance error) by method at oracle-λ, T=112; bars coloured by
   method tier.
3. ``etf_study_t_robustness`` — credit recovery and β-MSE vs sample size
   T in {60,112,240} for representative methods (oracle-λ), with ±1 SE bands.

Usage (from repository root)::

    python -m papers.jss_2026.applications.plot_competitor_study \
        --raw papers/jss_2026/simulations/results_calibrated/etf_competitor_study_raw.csv
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[3]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from papers.jss_2026.applications.etf_simulation_study import ROSTER

LABEL = dict(ROSTER)
TIER = {
    "competitor": ["OLS", "sklearn_lasso", "skglm_grouplasso", "asgl_sgl"],
    "fl_method": ["factorlasso_lasso", "factorlasso_grp_hcgl", "factorlasso_grp_hcgl_sign",
                  "factorlasso_grp_hcgl_sign_adapt", "factorlasso_fcgl_sign_adapt",
                  "factorlasso_sgl_hcgl_sign_adapt",
                  "factorlasso_grp_oracle", "factorlasso_grp_oracle_sign_adapt"],
    "fl_econ": ["factorlasso_econ_prior", "factorlasso_econ_sign_prior",
                "factorlasso_fcgl_sign_prior",
                "factorlasso_econ_oracle_sign_prior"],
}
TIER_OF = {m: t for t, ms in TIER.items() for m in ms}
TIER_COLOR = {"competitor": "#8c8c8c", "fl_method": "#2471a3", "fl_econ": "#1e8449"}
TIER_NAME = {"competitor": "competitor packages", "fl_method": "factorlasso (structural)",
             "fl_econ": "factorlasso + economic prior"}
REP_METHODS = ["skglm_grouplasso", "factorlasso_grp_hcgl",
               "factorlasso_sgl_hcgl_sign_adapt", "factorlasso_econ_sign_prior"]
DPI = 150


def _agg(raw):
    g = raw.groupby(["T", "selector", "method"])
    mean = g.mean(numeric_only=True)
    sem = g.sem(numeric_only=True)
    return mean, sem


def _order(present):
    return [m for m, _ in ROSTER if m in present]


def _bar_colors(methods):
    return [TIER_COLOR[TIER_OF[m]] for m in methods]


def fig_selector_contrast(mean, T, figdir, true_credit):
    if (T, "oracle") not in mean.index.droplevel(2).unique() or \
       (T, "bic") not in mean.index.droplevel(2).unique():
        return
    methods = _order(mean.loc[(T, "oracle")].index)
    x = np.arange(len(methods)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, metric, title in [(axes[0], "beta_mse", "β-MSE  (lower better)"),
                              (axes[1], "credit_recovery", "credit recovery  (toward truth)")]:
        o = mean.loc[(T, "oracle")].reindex(methods)[metric].values
        b = mean.loc[(T, "bic")].reindex(methods)[metric].values
        cols = _bar_colors(methods)
        ax.bar(x - w / 2, o, w, color=cols, edgecolor="white", linewidth=0.5, label="oracle-λ")
        ax.bar(x + w / 2, b, w, color=cols, edgecolor="0.25", linewidth=0.6, hatch="////", alpha=0.85, label="BIC-λ")
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in methods], rotation=55, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=0.25)
        if metric == "beta_mse" and "OLS" in methods:
            ax.axhline(mean.loc[(T, "oracle")].loc["OLS", "beta_mse"], ls=":", lw=1.1, color="0.4")
        if metric == "credit_recovery" and true_credit is not None:
            ax.axhline(true_credit, ls="--", lw=1.2, color="0.3")
            ax.text(0.02, true_credit, f" true β={true_credit:.2f}", va="bottom", fontsize=8, color="0.3", transform=ax.get_yaxis_transform())
            ax.axhline(0, ls=":", lw=0.8, color="0.7")
    sel_handles = [Patch(facecolor="0.6", label="oracle-λ"),
                   Patch(facecolor="0.6", hatch="////", label="BIC-λ (feasible)")]
    tier_handles = [Patch(facecolor=TIER_COLOR[t], label=TIER_NAME[t]) for t in ["competitor", "fl_method", "fl_econ"]]
    axes[0].legend(handles=sel_handles, fontsize=8, loc="upper right")
    axes[1].legend(handles=tier_handles, fontsize=8, loc="upper left")
    fig.suptitle(f"λ-selector contrast (T={T}): under feasible BIC the unstructured methods revert to OLS;\n"
                 "factorlasso's structure stays stable and recovers credit", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"etf_study_selector_contrast.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_oracle_metrics(mean, sem, T, figdir):
    if (T, "oracle") not in mean.index.droplevel(2).unique():
        return
    methods = _order(mean.loc[(T, "oracle")].index)
    panels = [("beta_mse", "β-MSE ↓"), ("support_f1", "support F1 ↑"), ("oos_r2", "OOS R² ↑"),
              ("credit_recovery", "credit recovery ↑"), ("risk_share_err", "credit risk-share err ↓"),
              ("cov_err", "systematic-cov err ↓")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(methods)); cols = _bar_colors(methods)
    for ax, (metric, title) in zip(axes.ravel(), panels):
        m = mean.loc[(T, "oracle")].reindex(methods)[metric].values
        e = sem.loc[(T, "oracle")].reindex(methods)[metric].values
        ax.bar(x, m, yerr=e, color=cols, edgecolor="white", linewidth=0.5, error_kw=dict(lw=0.8, ecolor="0.4"))
        ax.set_xticks(x); ax.set_xticklabels([LABEL[mm] for mm in methods], rotation=60, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.25)
    handles = [Patch(facecolor=TIER_COLOR[t], label=TIER_NAME[t]) for t in ["competitor", "fl_method", "fl_econ"]]
    fig.legend(handles=handles, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Estimator comparison at oracle-λ (T={T}, ±1 SE)", y=0.95, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"etf_study_oracle_metrics.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_t_robustness(mean, sem, figdir, true_credit):
    Ts = sorted({t for (t, s) in mean.index.droplevel(2).unique() if s == "oracle"})
    if len(Ts) < 2:
        return
    present = set(mean.loc[(Ts[0], "oracle")].index)
    methods = [m for m in REP_METHODS if m in present]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, metric, title in [(axes[0], "credit_recovery", "credit recovery vs sample size"),
                              (axes[1], "beta_mse", "β-MSE vs sample size")]:
        for m in methods:
            ys = [mean.loc[(t, "oracle")].loc[m, metric] if m in mean.loc[(t, "oracle")].index else np.nan for t in Ts]
            es = [sem.loc[(t, "oracle")].loc[m, metric] if m in sem.loc[(t, "oracle")].index else np.nan for t in Ts]
            ys, es = np.array(ys), np.array(es)
            ax.plot(Ts, ys, marker="o", ms=4, lw=1.8, color=TIER_COLOR[TIER_OF[m]], label=LABEL[m])
            ax.fill_between(Ts, ys - es, ys + es, color=TIER_COLOR[TIER_OF[m]], alpha=0.15)
        ax.set_xticks(Ts); ax.set_xlabel("training months T"); ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        if metric == "credit_recovery" and true_credit is not None:
            ax.axhline(true_credit, ls="--", lw=1.2, color="0.3", label=f"true β={true_credit:.2f}")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Small-sample behaviour (oracle-λ): the economic prior's credit-recovery edge widens as T shrinks", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"etf_study_t_robustness.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    default_raw = here.parents[0] / "simulations" / "results_calibrated" / "etf_competitor_study_raw.csv"
    ap.add_argument("--raw", type=Path, default=default_raw)
    ap.add_argument("--figdir", type=Path, default=here.parents[0] / "paper" / "figures")
    ap.add_argument("--headline-T", type=int, default=112)
    ap.add_argument("--true-credit", type=float, default=0.36, help="DGP true mean credit β for reference lines")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    mean, sem = _agg(raw)
    args.figdir.mkdir(parents=True, exist_ok=True)
    T = args.headline_T if args.headline_T in raw["T"].unique() else int(raw["T"].max())

    fig_selector_contrast(mean, T, args.figdir, args.true_credit)
    fig_oracle_metrics(mean, sem, T, args.figdir)
    fig_t_robustness(mean, sem, args.figdir, args.true_credit)
    print(f"headline T={T}; sample sizes {sorted(raw['T'].unique())}; selectors {sorted(raw['selector'].unique())}")
    print("wrote: etf_study_selector_contrast, etf_study_oracle_metrics, etf_study_t_robustness ({pdf,png})")
    print(f"  -> {args.figdir}")


if __name__ == "__main__":
    main()
