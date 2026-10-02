"""Human-readable summary and the optional sign-matrix plot of a fitted model."""

from __future__ import annotations

import numpy as np


def summary(model) -> str:
    """Problem dimensions, mode, clusters, active coefficients and mean R-squared."""
    if model.coef_ is None:
        raise RuntimeError("Model not fitted. Call fit() first.")
    n, m = model.coef_.shape
    k_active = int((model.coef_.abs() > 1e-8).sum().sum())
    lines = [
        "LassoModel summary",
        "-" * 40,
        f"model_type        : {model.model_type.name}",
        f"responses (N)     : {n}",
        f"factors (M)       : {m}",
        f"reg_lambda        : {model.reg_lambda:g}",
        f"l1_weight (alpha) : {model.l1_weight:g}",
        f"active coefs      : {k_active} / {n * m} "
        f"({100.0 * k_active / (n * m):.1f}%)",
    ]
    if model.clusters_ is not None:
        lines.append(f"clusters (HCGL)   : {int(model.clusters_.nunique())}")
    if model.auto_sign_constraints and model.derived_signs_ is not None:
        gated = int((model.derived_signs_ == 0).sum().sum())
        lines.append(f"sign-gated cells  : {gated}")
    if model.estimation_result_ is not None and getattr(
        model.estimation_result_, "r2", None
    ) is not None:
        try:
            lines.append(
                f"mean R^2          : {float(np.nanmean(model.estimation_result_.r2)):.4f}"
            )
        except Exception:  # pragma: no cover
            pass
    return "\n".join(lines)


def plot_signs(model, ax=None):
    """Heatmap of ``derived_signs_``; matplotlib is imported only when called."""
    if model.derived_signs_ is None:
        raise RuntimeError(
            "No derived_signs_ to plot. Fit with auto_sign_constraints=True."
        )
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_signs requires matplotlib.") from exc
    s = model.derived_signs_.values.astype(float)
    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(4, s.shape[1]), max(3, s.shape[0] * 0.12))
        )
    ax.imshow(s, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(s.shape[1]))
    ax.set_xticklabels(list(model.derived_signs_.columns), rotation=45,
                       ha="right", fontsize=8)
    ax.set_ylabel("assets")
    ax.set_title("Derived sign matrix (+1 / 0 / -1)")
    return ax
