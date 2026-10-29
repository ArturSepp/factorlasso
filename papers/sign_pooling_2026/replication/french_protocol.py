"""Frozen design and chronological calendars for the French portfolio study.

This paper-local module defines a study, not estimator defaults. No model is
fitted here. Candidate counts are logical upper bounds before exact deduplication.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
PORTFOLIO_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "100_Portfolios_10x10_Daily_CSV.zip"
)
INPUT_HASHES = {
    "factor_daily.csv": "8039ab3c4097bd099991eb108947c773b621b8173d3832b4541fcbd84e33ab4e",
    "factor_weekly.csv": "3e1511b72730a05cf9d2e35adb7e02f9ddc33c1a43fcf647f1aa19b32d2383ae",
    "factor_producer.py": "6b3c2bfae3c14525199c749cc104791ae905e943e90266dc6d7bd58535cec417",
}
METHOD_AXES = {
    "lasso": (False, False),
    "singleton_exclude": (False, True),
    "hcgl_unconstrained": (True, False),
    "hcgl_singleton_exclude": (True, True),
    "hcgl_exclude": (True, True),
    "hcgl_unrestricted": (True, True),
    "fcgl_unconstrained": (True, False),
    "fcgl_exclude": (True, True),
    "ols": (False, False),
    "ridge": (False, False),
    "cooperative": (True, False),
    "unilasso": (False, False),
}


def sha256(path: Path) -> str:
    """Hash exact bytes without newline normalization."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: object) -> bytes:
    """Use portable strict JSON, with infinity represented by the string 'inf'."""
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def external_path(path: Path) -> Path:
    """Reject output paths inside the checkout or any OneDrive tree."""
    if not path.is_absolute():
        raise ValueError("Output path must be absolute")
    result = path.resolve()
    if result == REPOSITORY or REPOSITORY in result.parents:
        raise ValueError("Generated outputs must be outside the checkout")
    if any(part.casefold().startswith("onedrive") for part in result.parts):
        raise ValueError("Generated outputs must be outside OneDrive")
    return result


def study_protocol() -> dict:
    """Return the prespecified design without machine-dependent paths or timestamps."""
    return {
        "schema_version": 1,
        "study_id": "ff98_ff6b_logrelative_v1",
        "protocol_version": "2026-10-26.2",
        "amendment": {
            "authority": "User approved 98 complete portfolios on 2026-10-26",
            "initial_protocol_sha256":
                "c9dd26790f12c6bf9b4563e679f32d4014c3a26faf40cea3bf102df35a198706",
            "reason": "732 source missing cells in two portfolios; no estimator results inspected",
            "original_design": "100 complete portfolios; rejected at data-intake validation",
        },
        "scope": "Stages 0-1 only; later stages require this frozen design",
        "estimand": "held-out contemporaneous conditional return reconstruction",
        "observed_factor_timing": "realized test-week factors; no return-forecast claim",
        "responses": 98,
        "response_selection": {
            "source_portfolios": 100,
            "exclude": ["S09_BM10", "S10_BM10"],
            "original_labels": ["ME9 BM10", "BIG HiBM"],
            "reason": "incomplete daily coverage over the agreed 2006-2026 history",
            "no_imputation": True,
            "selection_scope": "retrospective full-period availability, not model performance",
        },
        "predictors": ["MKT_LOGREL", "SMB", "HML", "RMW", "CMA", "MOM"],
        "portfolio_url": PORTFOLIO_URL,
        "portfolio_section": "Average Value Weighted Returns -- Daily",
        "input_hashes": INPUT_HASHES.copy(),
        "returns": {
            "frequency": "W-WED", "closed": "right", "label": "right",
            "portfolio": "sum_d(log1p(R_portfolio_d)-log1p(RF_3M_d))",
            "market": "sum_d(log1p(MKT_RF_3M_d+RF_3M_d)-log1p(RF_3M_d))",
            "style": "sum_d(log1p(style_d)); RF is not subtracted again",
            "units": "weekly decimal log units; no annualization",
            "cash": "supplied lagged DGS3MO ACT/365 yield accrual proxy",
            "styles_are": "transformed long-short spreads, not cash-relative wealth",
            "boundary_policy": "drop initial partial and final incomplete W-WED week",
            "invalid_policy": "fail; no fill, clipping, zero substitution or winsorization",
        },
        "calendar": {
            "daily_start": "2006-08-01", "daily_end": "2026-06-30", "daily_rows": 5009,
            "weekly_start": "2006-08-09", "weekly_end": "2026-06-24", "weekly_rows": 1038,
            "lookbacks": [52, 104], "primary_lookback": 52, "outer_burn_in": 104,
            "outer_horizon": 13, "terminal_policy": "include remaining 11 weeks",
            "origins": 72, "test_weeks": 934, "first_test": "2008-08-06",
            "inner_validation_length": 8,
            "inner_train_ends": {"52": [28, 36, 44], "104": [80, 88, 96]},
            "inner_validation_ends": {"52": [36, 44, 52], "104": [88, 96, 104]},
            "no_shuffle": True, "same_validation_dates_across_lookbacks": True,
        },
        "methods": list(METHOD_AXES),
        "secondary_methods": ["cooperative", "unilasso"],
        "objective": {
            "loss": "SSE/n in training-standardized units", "mixing_weight": 0.5,
            "fit_intercept": True, "sd_ddof": 0, "prior_target": 0,
            "adaptive_weights": False, "observation_weights": "equal",
            "group_weights": "sqrt(cluster_size / number_of_clusters)",
            "penalty_scale": "method/candidate-specific training-only KKT scale",
            "zero_endpoint": "exact objective-specific constrained least squares",
            "unilasso": "LOO-feature nonnegative second stage; own KKT scale and NNLS endpoint",
            "ridge": "SSE/n + lambda*sum(beta**2); direct solve; no KKT all-zero threshold",
            "preprocessing": "refit means/scales/groups/gates/KKT in every inner training subset",
        },
        "clustering": {
            "method": "Ward on Euclidean standardized response vectors / sqrt(n)",
            "partitions": ["singleton", 0.35, 0.5, 0.7, 0.9, "one_cluster"],
            "cut_interpretation": "fraction of final Ward merge height",
            "no_posthoc_residual_clustering_or_portfolio_subsets": True,
        },
        "gate": {
            "constructor": "derive_sign_constraints",
            "variance_estimator": "date", "ewma_span": None,
            "thresholds": [0, 0.75, 1, 2, 3, "inf"],
            "automatic_gating_when_constraints_supplied": False,
            "calibration": "heuristic; date variance is not serial-correlation robust",
            "score_acf_lags": [1, 4, 13],
        },
        "grid": {
            "fractions": [1, 0.5, 0.3, 0.17, 0.1, 0.05, 0.03, 0],
            "ridge_lambdas": ["inf", 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0],
            "refinement": {
                "maximum_passes": 1,
                "trigger": "any inner minimizer within tie tolerance at applicable finite edge",
                "fractions_if_smallest_positive_wins": [0.01, 0.003, 0.001],
                "cuts_if_lowest_or_highest_finite_wins": [0.175, 0.25, 0.95],
                "thresholds_if_highest_finite_wins": [4, 6],
                "ridge": "fixed grid; report boundary flags without L1 refinement",
                "no_outer_loss_trigger": True,
            },
            "deduplication": "exact partition/constraints/objective identity; preserve aliases",
        },
        "selection": {
            "score": "pooled inner SSE / training-response variance / validation-cell count",
            "tie_atol": 1e-10, "tie_rtol": 1e-8,
            "tie_definition": "score <= minimum + atol + rtol*abs(minimum)",
            "tie_order": "descending fraction, listed partition/threshold; ridge descending lambda",
        },
        "solver": {
            "conic": "CLARABEL", "options": {
                "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8, "max_iter": 500,
            },
            "retry": {"count": 1, "same_solver": True, "options": {
                "tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9, "max_iter": 2000,
            }},
            "selected_fit_audit": "independent objective, feasibility and stationarity/duality",
            "unresolved_failure": "block method/window; do not discard or silently switch solver",
            "max_constraint_violation": 1e-7,
            "workers": 2, "blas_threads_per_worker": 1,
            "direct_references": ["OLS", "ridge", "zero-penalty and intercept-only endpoints"],
        },
        "reporting": {
            "primary_contrast": ["hcgl_exclude", "hcgl_unconstrained"],
            "other_contrasts": [["hcgl_unrestricted", "hcgl_unconstrained"],
                                ["hcgl_exclude", "hcgl_singleton_exclude"],
                                ["fcgl_exclude", "fcgl_unconstrained"]],
            "primary_metrics": ["mean weekly standardized MSE", "sqrt(pooled standardized MSE)"],
            "scale": "outer-training population response SD; equal portfolio/week weights",
            "secondary": ["raw RMSE", "per-portfolio and pooled R2 vs training mean",
                          "direction/coverage by category", "ARI vs size and BM separately",
                          "raw-unit loading changes", "gate/support transitions"],
            "support_tolerances_standardized": [1e-6, 1e-4],
            "bootstrap": {"type": "stationary bootstrap of paired ordered outer-block sums/counts",
                          "resamples": 5000, "seed": 20261026,
                          "mean_blocks": {"52": 4, "104": 8},
                          "length_sensitivity_multipliers": [0.5, 2],
                          "scope": "conditional historical evaluation; no selection coverage"},
            "matched_diagnostic": {"geometry": "HCGL", "tau": 2,
                                   "partition_and_numeric_lambda": "selected unconstrained HCGL",
                                   "variants": ["singleton_exclude", "pooled_exclude",
                                                "pooled_release"]},
        },
        "rights_and_provenance": {
            "raw_and_derived_distribution": "not established; keep local pending separate review",
            "factors": "supplied frozen bytes; original retrieval receipt unavailable",
            "matched_release_vintage": "not verified",
            "data_vintage_backtest": False,
        },
        "package_defaults": {"auto_sign_constraints": False, "auto_sign_threshold_t": 0.75,
                             "auto_sign_variance": "independent"},
        "outcome_rule": "report FF result regardless of outcome; preserve negative yeast evidence",
    }


def candidate_inventory(protocol: dict, *, extended: bool = False) -> dict:
    """Count and identify candidates without touching response values or fitting."""
    partitions = list(protocol["clustering"]["partitions"])
    thresholds = list(protocol["gate"]["thresholds"])
    fractions = list(protocol["grid"]["fractions"])
    if extended:
        refinement = protocol["grid"]["refinement"]
        partitions += refinement["cuts_if_lowest_or_highest_finite_wins"]
        thresholds += refinement["thresholds_if_highest_finite_wins"]
        fractions = sorted(fractions + refinement["fractions_if_smallest_positive_wins"],
                           reverse=True)
    result = {}
    for method, (uses_partition, uses_tau) in METHOD_AXES.items():
        if method == "ols":
            rows = [{"method": method}]
        elif method == "ridge":
            rows = [{"method": method, "ridge_lambda": lam}
                    for lam in protocol["grid"]["ridge_lambdas"]]
        else:
            rows = [{"method": method, "fraction": f, "partition": p, "tau": t}
                    for f, p, t in itertools.product(
                        fractions, partitions if uses_partition else [None],
                        thresholds if uses_tau else [None])]
        ids = [hashlib.sha256(json_bytes(row)).hexdigest() for row in rows]
        result[method] = {"count": len(rows), "candidate_ids": ids, "candidates": rows}
    return result


def calendar_inventory(week_labels: list[str], protocol: dict) -> dict:
    """Create exact half-open chronological slices and shared inner validation dates."""
    c = protocol["calendar"]
    if len(week_labels) != c["weekly_rows"]:
        raise ValueError("Unexpected weekly panel length")
    rows = []
    for origin in range(c["outer_burn_in"], len(week_labels), c["outer_horizon"]):
        test_end = min(origin + c["outer_horizon"], len(week_labels))
        for lookback in c["lookbacks"]:
            start = origin - lookback
            inner = []
            for stop, end in zip(c["inner_train_ends"][str(lookback)],
                                 c["inner_validation_ends"][str(lookback)]):
                inner.append({"train_slice": [start, start + stop],
                              "validation_slice": [start + stop, start + end],
                              "validation_dates": week_labels[start + stop:start + end]})
            rows.append({"origin": origin, "lookback": lookback,
                         "train_slice": [start, origin], "test_slice": [origin, test_end],
                         "train_first": week_labels[start], "train_last": week_labels[origin-1],
                         "test_dates": week_labels[origin:test_end], "inner": inner})
    return {"index_convention": "zero-based half-open positional slices", "rows": rows}
