#!/usr/bin/env python
"""
usage_example.py -- reproduce the manuscript's minimal usage example.

Runs the exact code shown in the CodeChunk blocks of the manuscript
subsection "A minimal usage example" and prints the outputs shown in its
CodeOutput blocks: the fitted-attribute shapes, the discovered cluster
count, the ``summary()`` report, and the ``GridSearchCV`` selection of
``reg_lambda``. Writes the gated sign-matrix heatmap embedded by the
manuscript as ``figures/usage_signs_heatmap.{png,pdf}``.

The run is deterministic (``numpy.random.default_rng(0)``, no external
data), so every printed value must match the manuscript exactly.

Usage
-----
    python papers/jss_2026/paper/usage_example.py \
        --output papers/jss_2026/paper/figures
"""
from __future__ import annotations

# packages
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.model_selection import GridSearchCV
# factorlasso
from factorlasso import LassoModel, LassoModelType

HERE = Path(__file__).resolve().parent


def run_usage_example(output_dir: Path) -> None:
    """fit the manuscript's HCGL example, print its outputs, save the heatmap"""
    # -- data generation and fit, verbatim from the manuscript ------------
    rng = np.random.default_rng(0)
    T, N, M = 120, 50, 9
    X = pd.DataFrame(rng.standard_normal((T, M)),
                     columns=[f"F{j}" for j in range(M)])
    beta_true = rng.standard_normal((N, M)) * 0.4
    Y = pd.DataFrame(X.values @ beta_true.T +
                     0.3 * rng.standard_normal((T, N)),
                     columns=[f"A{k}" for k in range(N)])

    model = LassoModel(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-3,
        cutoff_fraction=0.5,
        auto_sign_constraints=True,
        auto_sign_threshold_t=0.75,
        auto_sign_adaptive_weights=True,
    ).fit(x=X, y=Y)

    # -- fitted attributes (manuscript CodeOutput: (50, 9) / (50, 9) / 12)
    print(model.coef_.shape)
    print(model.derived_signs_.shape)
    print(int(model.clusters_.nunique()))
    print()

    # -- summary() report --------------------------------------------------
    print(model.summary())
    print()

    # -- GridSearchCV selection (manuscript CodeOutput: 0.01) --------------
    search = GridSearchCV(
        LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO),
        param_grid={"reg_lambda": [1e-4, 1e-3, 1e-2]},
        cv=3,
    ).fit(X, Y)
    print(search.best_params_["reg_lambda"])
    print()

    # -- gated sign-matrix heatmap (manuscript figure usage_signs_heatmap) -
    fig, ax = plt.subplots(figsize=(5, 4))
    model.plot_signs(ax=ax)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("png", {"dpi": 150}), ("pdf", {})):
        out = output_dir / f"usage_signs_heatmap.{ext}"
        fig.savefig(out, bbox_inches="tight", **kwargs)
        print(f"wrote {out}")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the manuscript's minimal usage example."
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "paper" / "figures",
        help="Directory for usage_signs_heatmap.{png,pdf} "
             "(default: papers/jss_2026/paper/figures).",
    )
    args = parser.parse_args(argv)
    run_usage_example(output_dir=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
