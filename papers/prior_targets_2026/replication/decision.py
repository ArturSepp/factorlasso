"""Assemble an auditable evidence decision from completed immutable studies."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import (
    HERE, check_snapshot, digest, read_json, save_json, validate_root,
)
from papers.prior_targets_2026.replication.multiasset import check_manifest

RUNS = ["E2_confirmation", "E2_robustness_v1", "E3_calibration_audit_v1",
        "E3_L1_confirmation_v1", "E3_independent_review_v2",
        "E3_groups_pilot_v2", "E3_groups_confirmation_v2", "E3_groups_review_v1",
        "E3_solver_audit_v1", "E3_restricted_v2", "E4_rolling_v1", "E4_independent_review_v1"]


def validate_manifest(root, name):
    """Verify both early nested manifests and later flat hash registries."""
    folder = root / name
    manifest = read_json(folder / "manifest.json")
    if "artifacts" in manifest:
        for relative, value in manifest["artifacts"].items():
            assert digest(folder / relative)==value
    else:
        check_manifest(folder)


def value_at(root, reference):
    """Resolve one unambiguous numerical claim against a hashed source."""
    path = root / reference["file"]
    if reference["kind"]=="json":
        value = read_json(path)
        for key in reference["keys"]:
            value = value[key]
        return float(value)
    frame = pd.read_csv(path)
    for key, value in reference["where"].items():
        frame = frame[frame[key]==value]
    assert len(frame)==1, reference
    return float(frame.iloc[0][reference["column"]])


def build_claims(root):
    """Record exact values with explicit source rows; omit untestable truth claims."""
    claims = []

    def add(name, file, where, column, label):
        """Add one numerical CSV claim with its intended interpretation."""
        ref = dict(kind="csv", file=file, where=where, column=column)
        claims.append(dict(id=name, label=label, reference=ref, value=value_at(root, ref)))

    file = "E3_L1_confirmation_v1/paired_summary.csv"
    for profile in ["baseline", "credit_removed"]:
        for method in ["M1_zero", "M2_auto", "M3_economic", "M4_free_winner", "M5_post_lasso"]:
            where = dict(profile=profile, n=112, method=method)
            for column in ["credit_mse", "credit_mse_delta", "credit_mse_low", "credit_mse_high",
                           "credit_prediction_nmse", "full_covar_error"]:
                add("l1_"+profile+"_"+method+"_"+column, file, where, column,
                    "50-seed original E3 fixed-geometry confirmation; "+column)
    file = "E3_restricted_v2/paired_summary.csv"
    for profile in ["baseline", "credit_removed", "rates_dominant"]:
        for method in ["M1_zero", "small2", "small3"]:
            where = dict(profile=profile, method=method, comparator="M1_zero")
            for column in ["credit_mse", "credit_mse_delta", "credit_mse_low", "credit_mse_high",
                           "scenario_mse_delta", "prediction_nmse_delta"]:
                add("restricted_"+profile+"_"+method+"_"+column, file, where, column,
                    "Fresh-seed follow-up sensitivity, not original confirmation; "+column)
    file = "E3_groups_confirmation_v2/paired_summary.csv"
    table = pd.read_csv(root/file)
    for _, row in table.iterrows():
        for column in ["credit_mse", "credit_mse_delta", "credit_mse_low", "credit_mse_high"]:
            where = dict(profile=row.profile, setting=row.setting, target=row.target)
            add("groups_"+row.profile+"_"+row.setting+"_"+row.target+"_"+column,
                file, where, column, "50-seed group confirmation within specification; "+column)
    file = "E4_independent_review_v1/paired_effects.csv"
    for lane in ["production_snapshot", "public_etf_proxies"]:
        for span in [0, 36]:
            for method in ["M0_ols", "M1_zero", "M2_auto", "M3_fixed", "M3_small2",
                           "M3_small3", "M4_free_winner", "M5_post_lasso"]:
                for column in ["relative_change_pct", "paired_delta", "low", "high"]:
                    where = dict(lane=lane, span=span, cohort="credit", method=method, block=6)
                    add("rolling_"+lane+"_"+str(span)+"_"+method+"_"+column, file, where, column,
                        "Retrospective reconstruction; 29 months; pointwise block6 interval; "+column)
    return claims


def validate_ledger(root, ledger):
    """Reject unsupported references, hidden failures or inflated empirical claims."""
    assert ledger["decision"]=="focused_diagnostic_paper"
    assert ledger["empirical_exposure_truth"] is False
    assert ledger["fresh_public_rebuild"] is False
    assert ledger["universal_prior_dominance"] is False
    assert ledger["theory_developed"] is False
    assert set(ledger["source_manifests"])==set(RUNS)
    for name, value in ledger["source_manifests"].items():
        assert digest(root/name/"manifest.json")==value
    assert len(ledger["disclosed_failures"])==2
    assert ledger["disclosed_failures"][0]["failed_panels"]==2
    assert "preflight" in ledger["disclosed_failures"][1]["stage"]
    failure = ledger["disclosed_failures"][0]
    assert digest(root/failure["file"])==failure["sha256"]
    assert len(read_json(root/failure["file"]))==failure["failed_panels"]
    assert not (root/"E3_restricted_v1/asset_results.csv").exists()
    assert "ref[0]" in (root/"E3_restricted_v1/archive/restricted.py").read_text()
    empirical = read_json(root/"E4_independent_review_v1/summary.json")
    assert ledger["empirical_exposure_truth"]==empirical["independent_exposure_truth"]
    assert ledger["fresh_public_rebuild"]==empirical["fresh_public_rebuild"]
    assert len(ledger["claims"])==362
    assert len({c["id"] for c in ledger["claims"]})==len(ledger["claims"])
    for claim in ledger["claims"]:
        file = claim["reference"]["file"]
        assert file in ledger["source_files"]
        assert digest(root/file)==ledger["source_files"][file]
        assert "pilot" not in file
        np.testing.assert_allclose(claim["value"], value_at(root, claim["reference"]),
                                   atol=1e-12, rtol=1e-12)
    required = ["M0_ols", "M1_zero", "M2_auto", "M3_fixed", "M3_small2",
                "M3_small3", "M4_free_winner", "M5_post_lasso"]
    for method in required:
        assert any(c["reference"].get("where", {}).get("method")==method
                   and c["id"].startswith("rolling_") for c in ledger["claims"])


def verify(root, out):
    """Check a ledger and prove its gates reject four meaningful corruptions."""
    from papers.prior_targets_2026.replication.robustness import reject_assertion
    ledger = read_json(out/"claim_ledger.json")
    validate_ledger(root, ledger)
    bad = deepcopy(ledger)
    bad["claims"][0]["value"] += .1
    reject_assertion(lambda: validate_ledger(root, bad))
    bad = deepcopy(ledger)
    bad["empirical_exposure_truth"] = True
    reject_assertion(lambda: validate_ledger(root, bad))
    bad = deepcopy(ledger)
    bad["disclosed_failures"] = []
    reject_assertion(lambda: validate_ledger(root, bad))
    bad = deepcopy(ledger)
    bad["claims"] = [c for c in bad["claims"]
                     if c["reference"].get("where", {}).get("method")!="M4_free_winner"]
    reject_assertion(lambda: validate_ledger(root, bad))
    return dict(numerical_claims=len(ledger["claims"]), source_manifests=len(RUNS),
                unsupported_value_rejected=True, invented_empirical_truth_rejected=True,
                concealed_failure_rejected=True, missing_comparator_rejected=True,
                decision=ledger["decision"])


def main():
    """Assemble completed evidence, preserving pilots, amendments and limitations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    check_snapshot(root)
    sys.path.insert(0, str(root/"source_snapshot/src"))
    out = root/"E5_decision_v1"
    for name in RUNS:
        validate_manifest(root, name)
    if args.command=="verify":
        check_manifest(out)
        print(json.dumps(verify(root, out), indent=2))
        return
    out.mkdir()
    shutil.copy2(HERE/"decision.py", out/"decision.py")
    claims = build_claims(root)
    failures = read_json(root/"E3_groups_pilot_v1/failures.json")
    assert len(failures)==2
    ledger = dict(
        decision="focused_diagnostic_paper", empirical_exposure_truth=False,
        fresh_public_rebuild=False, universal_prior_dominance=False, theory_developed=False,
        interpretation="Attribution, prediction and covariance accuracy are distinct objectives. "
                       "Target quality and information provenance determine conditional gains "
                       "and harmful failure regions. Empirical reconstruction is exploratory.",
        source_manifests={name: digest(root/name/"manifest.json") for name in RUNS},
        source_files={c["reference"]["file"]: digest(root/c["reference"]["file"]) for c in claims},
        disclosed_failures=[
            dict(stage="E3_groups_pilot_v1", failed_panels=2, planned_panels=10,
                 file="E3_groups_pilot_v1/failures.json",
                 sha256=digest(root/"E3_groups_pilot_v1/failures.json"),
                 amendment="Repeat entire pilot with existing ECOS fallback; "
                           "fresh confirmation seeds; every fallback recorded."),
            dict(stage="E3_restricted_v1_preflight", simulated_panels=0,
                 error="LassoEstimationResult is not subscriptable; corrected to estimated_beta.",
                 amendment="Preserve failed preflight, run unchanged design in v2.")],
        unavailable=["independent dated empirical exposure measurements",
                     "fresh public rebuild and historical-vintage validation"],
        claims=claims)
    save_json(out/"claim_ledger.json", ledger)
    receipt = verify(root, out)
    save_json(out/"verification.json", receipt)
    pd.DataFrame([dict(id=c["id"], value=c["value"], source=c["reference"]["file"],
                       field=c["reference"]["column"], interpretation=c["label"])
                  for c in claims]).to_csv(out/"claims.csv", index=False)
    save_json(out/"manifest.json",
              {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()})
    print(json.dumps(receipt, indent=2))


if __name__=="__main__":
    main()




