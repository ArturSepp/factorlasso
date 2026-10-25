"""Check variance-sensitivity tables/prose directly from saved fit arrays.

No models are fitted and no frozen outputs are replaced. ``--root`` must contain
``pilot_synthetic`` and ``yeast_comparison`` from the paired-application run.
The checker validates their original receipts, recomputes every reported metric
from coefficient, gate and prediction CSVs, and checks rounding in the supplied
supplement and optional main article.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


VARIANCES = ("independent", "date")
NUMBER = r"[-+]?\d+(?:\.\d+)?"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frame(path):
    return pd.read_csv(path, index_col=0, float_precision="round_trip")


def load_pairs(folder):
    manifest_path = folder / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not manifest["sources_changed_during_run"]
    for name, expected in manifest["output_hashes"].items():
        path = folder / name.replace("\\", "/")
        assert digest(path) == expected, f"Changed saved output: {path}"
    pairs = {}
    for record in manifest["cases"]:
        case = record["case"]
        saved = folder / case
        config = json.loads((saved / "configuration.json").read_text(encoding="utf-8"))
        target = saved / "test_y.csv"
        held_out = target.is_file()
        actual = read_frame(target if held_out else saved / "train_y.csv")
        training = read_frame(saved / "train_y.csv")
        prep = record["metadata"].get("preprocessing")
        baseline = (np.array([prep["y_mean"][c] for c in actual.columns]) if prep
                    else training.mean().to_numpy())
        scale = (np.array([prep["y_sd"][c] for c in actual.columns]) if prep
                 else np.ones(actual.shape[1]))
        baseline_errors = np.where(actual.notna(), (actual - baseline) ** 2, 0.)
        beta, gates, errors, scaled_errors, r2 = {}, {}, {}, {}, {}
        for variance in VARIANCES:
            beta[variance] = read_frame(saved / f"coef_{variance}.csv").to_numpy()
            gates[variance] = read_frame(saved / f"detected_signs_{variance}.csv").to_numpy()
            final = read_frame(saved / f"final_signs_{variance}.csv").to_numpy()
            assert np.isfinite(gates[variance]).all() and np.isfinite(final).all()
            prediction = read_frame(saved / f"predictions_{variance}.csv").to_numpy()
            residual = actual.to_numpy() - prediction
            errors[variance] = np.where(actual.notna(), residual ** 2, 0.)
            scaled_errors[variance] = np.where(actual.notna(), (residual / scale) ** 2, 0.)
            r2[variance] = float(1 - errors[variance].sum() / baseline_errors.sum())
        np.testing.assert_array_equal(read_frame(saved / "slopes_independent.csv"),
                                      read_frame(saved / "slopes_date.csv"))
        changed = gates["independent"] != gates["date"]
        delta = beta["date"] - beta["independent"]
        d_beta = np.sqrt(np.square(delta).sum() / np.square(beta["independent"]).sum())
        pair = dict(config=config, metadata=record["metadata"], held_out=held_out,
                    beta=beta, changed=int(changed.sum()), cells=int(changed.size),
                    d_beta=float(d_beta), r2=r2, errors=errors, scaled_errors=scaled_errors,
                    baseline_errors=baseline_errors, n_train=len(training), n_test=len(actual),
                    newly_excluded=int(((gates["independent"] != 0)
                                        & (gates["date"] == 0)).sum()),
                    newly_retained=int(((gates["independent"] == 0)
                                        & (gates["date"] != 0)).sum()))
        if (saved / "covariance_independent.csv").is_file():
            covariances = [np.loadtxt(saved / f"covariance_{v}.csv", delimiter=",")
                           for v in VARIANCES]
            pair["d_covariance"] = float(np.sqrt(
                np.square(covariances[1] - covariances[0]).sum()
                / np.square(covariances[0]).sum()))
        pairs[case] = pair
    return pairs, dict(manifest_sha256=digest(manifest_path),
                       recorded_output_hashes_checked=len(manifest["output_hashes"]))


def table_rows(text, label):
    anchor = text.index(r"\label{" + label + "}")
    table = text[text.rfind(r"\begin{table}", 0, anchor):text.index(r"\end{table}", anchor)]
    return {cells[0].strip(): [float(n) for n in re.findall(NUMBER, " ".join(cells[1:]))]
            for line in table.splitlines() if "&" in line
            for cells in [line.split("&")]}


def verify(root, supplement, out, paper=None):
    synthetic, synthetic_receipt = load_pairs(root / "pilot_synthetic")
    yeast, yeast_receipt = load_pairs(root / "yeast_comparison")
    assert len(synthetic) == 17 and len(yeast) == 18
    text = supplement.read_text(encoding="utf-8")
    start = text.index(r"\label{app:variance-sensitivity}")
    section = text[start:text.index(r"\section{", start)]
    checked = 0

    def check(printed, expected, digits):
        nonlocal checked
        assert len(printed) == len(expected), (printed, expected)
        precision = [digits] * len(expected) if isinstance(digits, int) else digits
        for actual, reference, places in zip(printed, expected, precision):
            assert np.isfinite(reference)
            assert abs(actual - reference) <= 0.5 * 10 ** (-places) + 1e-12, (
                actual, reference, places)
            checked += 1

    def check_prose(pattern, expected, digits):
        match = re.search(pattern, section, re.DOTALL)
        assert match is not None, pattern
        check([float(n) for n in re.findall(NUMBER, match.group(1))], expected, digits)

    synthetic_rows = table_rows(text, "tab:variance-synthetic")
    groups = {
        "Credit": [name for name in synthetic if name.startswith("credit_")],
        "Quickstart full": ["quickstart_full"],
        "Quickstart holdout": ["quickstart_holdout"],
        "Production": ["production_full", "production_holdout"],
        "Known-group teaching": ["gate_teaching_panel"],
    }
    for label, names in groups.items():
        reference = synthetic[names[0]]
        expected = [len(names), reference["changed"], reference["cells"],
                    100 * reference["d_beta"], reference["r2"]["date"]
                    - reference["r2"]["independent"]]
        check(synthetic_rows[label], expected, [0, 0, 0, 3, 6])
        for name in names[1:]:
            other = synthetic[name]
            check([other["changed"], other["cells"], other["d_beta"],
                   other["r2"]["date"] - other["r2"]["independent"]],
                  [reference["changed"], reference["cells"], reference["d_beta"],
                   reference["r2"]["date"] - reference["r2"]["independent"]], 12)
    for name, pair in synthetic.items():
        config = pair["config"]
        assert config["auto_sign_ewma_span"] is None and not config["auto_sign_use_fit_span"]
        assert config["l1_weight"] == 0 and config["solver"] == "CLARABEL"
        assert config["auto_sign_threshold_t"] == (1 if name.startswith("credit_") else .75)
        if name.startswith("credit_"):
            assert config["reg_lambda"] in (1e-7, 1e-5, 1e-4, 1e-3)
            assert config["auto_sign_adaptive_floor"] == .5
        if name.startswith(("quickstart_", "production_")):
            assert config["reg_lambda"] == 1e-5
        if name.startswith("production_"):
            assert config["span"] == 36 and config["loss_normalization"] == "weight_sum"
            assert config["auto_sign_adaptive_floor"] == 1e-3
            assert config["apply_ols_prior"]
        else:
            assert config["span"] is None and config["loss_normalization"] == "sample"
        adaptive = name.startswith(("credit_", "production_"))
        assert config["auto_sign_adaptive_weights"] == adaptive
        if adaptive:
            assert config["auto_sign_adaptive_gamma"] == 1 and pair["changed"] == 0
    assert synthetic["quickstart_holdout"]["newly_excluded"] == 4
    assert synthetic["gate_teaching_panel"]["newly_retained"] == 6
    check_prose(r"covariance changes by (.*?) in relative Frobenius norm",
                [100 * synthetic["quickstart_holdout"]["d_covariance"]], 3)

    summaries = {}
    yeast_rows = table_rows(text, "tab:variance-yeast")
    for method in ("hcgl", "fcgl"):
        pairs = [p for name, p in yeast.items() if name.startswith("yeast_" + method)]
        assert len(pairs) == 9
        changes = np.array([p["d_beta"] for p in pairs])
        gate_fraction = sum(p["changed"] for p in pairs) / sum(p["cells"] for p in pairs)
        mse = {v: float(np.concatenate([p["scaled_errors"][v] for p in pairs]).mean())
               for v in VARIANCES}
        base = np.concatenate([p["baseline_errors"] for p in pairs])
        r2 = {v: float(1 - np.concatenate([p["errors"][v] for p in pairs]).sum() / base.sum())
              for v in VARIANCES}
        increase = 100 * (mse["date"] / mse["independent"] - 1)
        expected = [100 * gate_fraction, 100 * np.median(changes), 100 * changes.min(),
                    100 * changes.max(), mse["independent"], mse["date"], increase]
        check(yeast_rows[method.upper()], expected, [2, 2, 2, 2, 6, 6, 2])
        support = {str(tol): {v: float(np.mean([(abs(p["beta"][v]) > tol).sum()
                                              for p in pairs])) for v in VARIANCES}
                   for tol in (1e-4, 1e-6)}
        summaries[method] = dict(gate_fraction=gate_fraction,
            gate_range=[min(p["changed"] / p["cells"] for p in pairs),
                        max(p["changed"] / p["cells"] for p in pairs)],
            coefficient_median=float(np.median(changes)), standardized_mse=mse,
            mse_increase_percent=increase, support=support, raw_pooled_r2=r2)
        for pair in pairs:
            config, meta = pair["config"], pair["metadata"]
            protocol = meta["protocol"]
            assert config["auto_sign_threshold_t"] == 2 and config["l1_weight"] == .5
            assert config["solver"] == "MOSEK" and config["span"] is None
            assert config["loss_normalization"] == "sample"
            assert config["group_penalty"] == "normalized"
            assert not config["auto_sign_adaptive_weights"] and not config["apply_ols_prior"]
            assert config["factors_beta_prior"] is None
            assert config["auto_sign_ewma_span"] is None and not config["auto_sign_use_fit_span"]
            assert protocol["cluster_cut"] == .7 and protocol["ld_correlation"] == .92
            assert protocol["screen_p"] == .01 and protocol["screen_min_genes"] == 2
            assert protocol["repeats"] == 3 and protocol["outer_folds"] == 3
            assert meta["penalty_fraction"] == .3
            assert config["reg_lambda"] == .3 * meta["lambda_max_unconstrained"]
            assert pair["newly_retained"] == 0 and pair["changed"] == pair["newly_excluded"]
        assert sum(p["n_test"] for p in pairs) == 3 * 112

    check_prose(r"gate-change fraction ranges from (.*?) across splits",
                [100 * value for value in summaries["hcgl"]["gate_range"]], 2)
    check_prose(r"selected-cell counts at.*?fall from (.*?) for FCGL\.",
                [summaries[m]["support"]["0.0001"][v]
                 for m in ("hcgl", "fcgl") for v in VARIANCES], 2)
    check_prose(r"At \$10\^\{-6\}\$ the corresponding counts are\s*(.*?), showing",
                [summaries[m]["support"]["1e-06"][v]
                 for m in ("hcgl", "fcgl") for v in VARIANCES], 2)
    check_prose(r"Original-expression pooled.*?changes from\s*(.*?) for FCGL\.",
                [summaries[m]["raw_pooled_r2"][v]
                 for m in ("hcgl", "fcgl") for v in VARIANCES], 6)
    sources = [supplement]
    if paper is not None:
        main = paper.read_text(encoding="utf-8")
        paragraph = re.search(r"Switching to date scores(.*?)respectively\.", main, re.DOTALL)
        assert paragraph is not None
        check([float(n) for n in re.findall(NUMBER, paragraph.group(1))],
              [100 * summaries["hcgl"]["gate_fraction"],
               100 * summaries["hcgl"]["coefficient_median"],
               100 * summaries["fcgl"]["coefficient_median"],
               summaries["hcgl"]["mse_increase_percent"],
               summaries["fcgl"]["mse_increase_percent"]], 2)
        sources.append(paper)
    receipt = dict(status="passed", tables_checked=2, numeric_values_checked=checked,
                   paired_cases_checked=len(synthetic) + len(yeast), mismatches=0,
                   source_sha256_by_file={p.name: digest(p) for p in sources},
                   verifier_sha256=digest(Path(__file__)),
                   input_receipts=dict(synthetic=synthetic_receipt, yeast=yeast_receipt),
                   yeast_recomputed=summaries,
                   method="Direct coefficient/gate/prediction arrays, original output hashes, "
                          "configuration checks and printed rounding intervals; no model refits")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--supplement", required=True, type=Path)
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    verify(args.root, args.supplement, args.out, args.paper)
