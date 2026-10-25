"""Check the rendered source numbers independently against raw paired draws."""

from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import t

NAMES = {
    "Lasso": "lasso",
    "HCGL, unconstrained": "hcgl_unconstrained",
    "HCGL, singleton/exclude": "hcgl_per_response",
    "HCGL, pooled/exclude": "hcgl_pooled_zero",
    "HCGL, pooled/release": "hcgl_pooled_free",
    "FCGL, unconstrained": "fcgl_unconstrained",
    "Cooperative Lasso": "cooperative",
    "UniLasso": "unilasso",
    "HCGL, known/exclude": "hcgl_known_zero",
}
DESIGNS = {
    "Coherent": "coherent",
    "Mixed directions": "mixed_directions",
    "Partial support": "partial_support",
    "Predictor correlation": "predictor_correlation",
    "Residual correlation": "residual_correlation",
}
NUMBER = r"-?\d+\.\d+"


def verify(paper, original_path, learned_path, sensitivity_path, output, supplement=None):
    """Compare printed values with independently summed raw draws and paired errors."""
    sources = [paper] + ([supplement] if supplement is not None else [])
    text = "\n".join(source.read_text(encoding="utf-8") for source in sources)
    original = pd.read_csv(original_path)
    learned = pd.read_csv(learned_path)
    sensitivity = pd.read_csv(sensitivity_path)
    original = original[original.zero_tolerance == 1e-4]
    learned = learned[learned.zero_tolerance == 1e-4]
    sensitivity = sensitivity[sensitivity.zero_tolerance == 1e-4]
    count = 0
    tables = 0

    def block(anchor):
        at = text.index(anchor)
        return text[text.rfind(r"\begin{table}", 0, at) : text.index(r"\end{table}", at)]

    def compare(actual, expected, digits):
        nonlocal count
        # Independent summation can straddle an exact half-rounding boundary.
        assert abs(actual - expected) <= 0.5 * 10 ** (-digits) + 1e-12, (actual, expected, digits)
        count += 1

    def check_means(anchor, data, columns):
        nonlocal tables
        tables += 1
        rows = [
            line.split("&")
            for line in block(anchor).splitlines()
            if line.split("&")[0].strip() in NAMES
        ]
        assert len(rows) == data.method.nunique()
        for row in rows:
            method = NAMES[row[0].strip()]
            assert len(row) - 1 == len(columns)
            for printed, (scenario, metric, digits, scale) in zip(row[1:], columns):
                values = (
                    data[(data.method == method) & (data.scenario == scenario)][metric] * scale
                ).to_numpy()
                n = len(values)
                mean = sum(values) / n
                se = np.sqrt(sum((v - mean) ** 2 for v in values) / (n * (n - 1)))
                numbers = list(map(float, re.findall(NUMBER, printed)))
                assert len(numbers) == 2
                compare(numbers[0], mean, digits)
                compare(numbers[1], se, digits)

    old_order = [
        "baseline",
        "weak_signal",
        "partial_support",
        "mixed_signs",
        "correlated",
        "unequal",
    ]
    check_means(
        r"\label{tab:fitted-prediction}",
        original,
        [(s, "prediction_mse_standardized", 3, 1) for s in old_order],
    )
    check_means(
        r"\label{tab:fitted-support}",
        original,
        [
            ("baseline", m, d, 1)
            for m, d in [
                ("beta_mse_standardized", 4),
                ("support_precision", 3),
                ("active_sign_recovery", 3),
                ("null_selection", 3),
            ]
        ],
    )
    for scenario, caption in zip(
        old_order[1:],
        [
            "Weak signal",
            "Partial supports",
            "Mixed signs",
            "Correlated predictors and errors",
            "Unequal groups and variances",
        ],
    ):
        check_means(
            r"\caption{" + caption + ": fitted-coefficient results",
            original,
            [
                (scenario, m, 4, 1)
                for m in [
                    "beta_mse_standardized",
                    "active_sign_recovery",
                    "active_sign_flip",
                    "support_fdp",
                ]
            ],
        )
    check_means(
        r"\label{tab:learned-prediction}",
        learned,
        [(s, "prediction_r2", 3, 1) for s in DESIGNS.values()],
    )
    for scenario in DESIGNS.values():
        check_means(
            r"\label{tab:learned-metrics-" + scenario.replace("_", "-") + "}",
            learned,
            [
                (scenario, m, 2, scale)
                for m, scale in [
                    ("beta_mse_standardized", 1000),
                    ("support_fdp", 100),
                    ("active_sign_recovery", 100),
                    ("active_sign_flip", 100),
                    ("active_abstention", 100),
                ]
            ],
        )

    def check_pair(printed, a, b):
        difference = (
            a.set_index("seed").prediction_mse_standardized
            - b.set_index("seed").prediction_mse_standardized
        ).to_numpy()
        assert len(difference) == 40 and np.isfinite(difference).all()
        mean = sum(difference) / 40
        se = np.sqrt(sum((v - mean) ** 2 for v in difference) / (40 * 39))
        radius = t.ppf(0.975, 39) * se
        numbers = list(map(float, re.findall(NUMBER, printed)))
        assert len(numbers) == 3
        for value, expected in zip(numbers, [mean, mean - radius, mean + radius]):
            compare(value, expected, 4)

    for line in block(r"\label{tab:learned-groups-paired}").splitlines():
        cells = line.split("&")
        if cells[0].strip() in DESIGNS:
            scenario = DESIGNS[cells[0].strip()]
            frame = learned[learned.scenario == scenario]
            check_pair(
                cells[-1],
                frame[frame.method == "hcgl_pooled_zero"],
                frame[frame.method == "hcgl_unconstrained"],
            )
    tables += 1
    variants = {
        "Cut 0.35": "cut_0.35",
        "Cut 0.70": "cut_0.7",
        r"$\alpha=0.25$": "alpha_0.25",
        r"$\alpha=0.75$": "alpha_0.75",
        r"$\tau=0$": "tau_0",
        r"$\tau=3$": "tau_3",
        "Joint CV": "joint_cv",
    }
    scenario = None
    for line in block(r"\label{tab:learned-tuning-sensitivity}").splitlines():
        cells = line.split("&")
        if len(cells) != 3 or cells[1].strip() not in variants:
            continue
        if cells[0].strip():
            scenario = DESIGNS[cells[0].strip()]
        variant = variants[cells[1].strip()]
        data = learned if variant == "joint_cv" else sensitivity
        a = data[
            (data.scenario == scenario)
            & (data.method == "hcgl_pooled_zero")
            & (data.variant == variant)
        ]
        b = sensitivity[
            (sensitivity.scenario == scenario)
            & (sensitivity.method == "hcgl_pooled_zero")
            & (sensitivity.variant == "reference")
        ]
        check_pair(cells[-1], a, b)
    tables += 1
    param = None
    values = {
        "cut": [0, 0.175, 0.35, 0.5, 0.7, 0.85, 1],
        "alpha": [0, 0.1, 0.25, 0.5, 0.75, 1],
        "tau": [0, 0.75, 1.5, 3, 6, np.inf],
    }
    for line in block(r"\label{tab:learned-tuning-selections}").splitlines():
        cells = line.split("&")
        label = cells[0].strip()
        if label.startswith("Cut "):
            param = "cut"
        if label.startswith("Mixing "):
            param = "alpha"
        if label.startswith("Gate "):
            param = "tau"
        if label not in DESIGNS:
            continue
        data = learned[
            (learned.scenario == DESIGNS[label]) & (learned.method == "hcgl_pooled_zero")
        ]
        printed = [int(re.sub(r"\\.*", "", s).strip()) for s in cells[1 : 1 + len(values[param])]]
        expected = [int(np.count_nonzero(data[param].to_numpy() == v)) for v in values[param]]
        assert printed == expected and sum(printed) == 40
        count += len(printed)
    tables += 1
    receipt = dict(
        source_sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
        source_sha256_by_file={
            source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources
        },
        tables_checked=tables,
        numeric_values_checked=count,
        mismatches=0,
        method=("Direct sums and paired differences from raw draws; parsed manuscript source; "
                "declared-rounding intervals"),
    )
    output.write_text(json.dumps(receipt, indent=2))
    print(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("paper", "original", "learned", "sensitivity", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--supplement", type=Path,
                        help="Supplement source containing tables moved from the main article")
    args = parser.parse_args()
    verify(args.paper, args.original, args.learned, args.sensitivity, args.out, args.supplement)


if __name__ == "__main__":
    main()
