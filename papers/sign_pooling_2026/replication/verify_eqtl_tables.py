"""Verify the five eQTL tables against repeat and fold-pair observations."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


METHODS = {
    "Unconstrained Lasso": "lasso",
    "Singleton-gated Lasso, exclude": "singleton_exclude",
    "HCGL, unconstrained": "hcgl_unconstrained",
    "HCGL, singleton/exclude": "hcgl_singleton_exclude",
    "HCGL, pooled/exclude": "hcgl_exclude",
    "HCGL, pooled/release": "hcgl_unrestricted",
    "FCGL, unconstrained": "fcgl_unconstrained",
    "FCGL, pooled/exclude": "fcgl_exclude",
}
SETTINGS = {
    "Baseline": "baseline",
    "LD $0.85$": "ld_085",
    "LD $0.98$": "ld_098",
    "Screen $p=0.005$": "screen_0005",
    "Screen $p=0.02$": "screen_002",
    "Cut $0.5$": "cut_05",
    "Cut $0.9$": "cut_09",
    r"$\tau=1$": "tau_1",
    r"$\tau=3$": "tau_3",
    r"$\alpha=0.2$": "alpha_02",
    r"$\alpha=0.8$": "alpha_08",
}


def verify(paper, root, out, supplement=None):
    sources = [paper] + ([supplement] if supplement is not None else [])
    text = "\n".join(source.read_text(encoding="utf-8") for source in sources)
    repeats = pd.read_csv(root / "prediction_repeats.csv")
    pairs = pd.read_csv(root / "stability/baseline/coefficient_pairs.csv.gz")
    setting_pairs = pd.read_csv(root / "same_fold_coefficient_sensitivity.csv")
    partition_pairs = pd.read_csv(root / "same_fold_partition_marker_sensitivity.csv")
    count = 0

    def rows(label, names):
        anchor = text.index(r"\label{" + label + "}")
        table = text[text.rfind(r"\begin{table}", 0, anchor) : text.index(r"\end{table}", anchor)]
        found = {}
        for line in table.splitlines():
            cells = line.split("&")
            if cells[0].strip() in names:
                key = names[cells[0].strip()]
                assert key not in found
                found[key] = [list(map(float, re.findall(r"-?\d+\.\d+", c))) for c in cells[1:]]
        assert set(found) == set(names.values()), (label, found.keys())
        return found

    def check(printed, expected, digits=3):
        nonlocal count
        assert len(printed) == len(expected)
        for a, b in zip(printed, expected):
            assert np.isfinite(b) and abs(a - b) <= 0.5 * 10 ** (-digits) + 1e-12, (a, b, digits)
            count += 1

    def mean_sd(v):
        v = np.asarray(v)
        assert len(v) == 3
        mean = sum(v) / 3
        return [mean, np.sqrt(sum((x - mean) ** 2 for x in v) / 2)]

    for method, cells in rows("tab:eqtl-nested", METHODS).items():
        values = repeats[(repeats.configuration == "baseline") & (repeats.method == method)]
        assert len(cells) == 3
        for cell, metric in zip(cells, ("r2_gene_median", "r2_gene_mean", "r2_pooled")):
            check(cell, mean_sd(values[metric]))
    for method, cells in rows("tab:eqtl-stability", METHODS).items():
        values = pairs[pairs.method == method]
        assert len(values) == 36 and len(cells) == 4
        for i, (cell, metric) in enumerate(
            zip(
                cells,
                (
                    "selection_jaccard",
                    "selection_jaccard_common_markers",
                    "direction_agreement_selected_both",
                    "selected_both",
                ),
            )
        ):
            check(cell, [sum(values[metric]) / 36], 1 if i == 3 else 3)
    methods = ("lasso", "hcgl_unconstrained", "hcgl_exclude", "hcgl_unrestricted")
    for setting, cells in rows("tab:eqtl-sensitivity", SETTINGS).items():
        values = repeats[repeats.configuration == setting]
        assert len(cells) == 5
        for cell, method, metric in zip(
            cells, (*methods, "hcgl_exclude"), (*("r2_gene_median",) * 4, "r2_pooled")
        ):
            check(cell, [mean_sd(values.loc[values.method == method, metric])[0]])
    for setting, cells in rows("tab:eqtl-sensitivity-paired", SETTINGS).items():
        values = repeats[repeats.configuration == setting].pivot(
            index="repeat", columns="method", values="r2_gene_median"
        )
        assert len(cells) == 2
        for cell, other in zip(cells, ("hcgl_unconstrained", "hcgl_unrestricted")):
            check(cell, mean_sd(values.hcgl_exclude - values[other]))
    for setting, cells in rows("tab:eqtl-configuration-stability", SETTINGS).items():
        assert len(cells) == 6
        if setting == "baseline":
            for cell in cells:
                check(cell, [1.0])
            continue
        values = partition_pairs[partition_pairs.configuration == setting]
        assert len(values) == 9
        for cell, metric in zip(
            cells[:2], ("ari_to_baseline", "selected_marker_jaccard_to_baseline")
        ):
            check(cell, [sum(values[metric]) / 9])
        for cell, method in zip(cells[2:], methods):
            values = setting_pairs[
                (setting_pairs.configuration == setting) & (setting_pairs.method == method)
            ]
            assert len(values) == 9
            check(cell, [sum(values.selection_jaccard) / 9])
    receipt = dict(
        source_sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
        source_sha256_by_file={
            source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources
        },
        tables_checked=5,
        numeric_values_checked=count,
        mismatches=0,
        method=(
            "Direct repeat means/SD, paired differences and fold-pair averages; "
            "declared rounding intervals"
        ),
    )
    out.write_text(json.dumps(receipt, indent=2))
    print(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("paper", "root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--supplement", type=Path,
                        help="Supplement source containing tables moved from the main article")
    args = parser.parse_args()
    verify(args.paper, args.root, args.out, args.supplement)


if __name__ == "__main__":
    main()
