"""Freeze a training-calibrated ETF/MATF pilot without running comparative fits.

Only the first eligible 60-month calibration window supplies data-generating
parameters. Existing explicit sleeve membership is authenticated, never inferred.
The synthetic alignment/conflict and reversal designs are controlled stresses,
not assertions that ETF coefficients, economic signs or market premia are truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import factorlasso

if __package__:
    from .french_attribution import external, sha, read, write, digest, require, REPO
else:
    from french_attribution import external, sha, read, write, digest, require, REPO

HERE = Path(__file__).resolve().parent
CALIBRATION_END = "2016-12-31"
CALIBRATION_START = "2012-01-31"


def calibration(x, y, memberships, premia):
    """Build fixed financial-unit truth parameters using training rows only."""
    x, y = [frame.loc[CALIBRATION_START:CALIBRATION_END].copy() for frame in (x, y)]
    dates = pd.date_range(CALIBRATION_START, CALIBRATION_END, freq="ME")
    require(x.index.equals(dates) and y.index.equals(dates), "Incomplete calibration months")
    require(x.columns.is_unique and y.columns.is_unique, "Duplicate named series")
    require(np.isfinite(x).all().all() and np.isfinite(y).all().all(), "Missing calibration data")
    require(memberships.index.equals(y.columns), "Explicit membership order differs")
    require(
        np.isin(memberships, [0, 1]).all() and (memberships.sum(axis=1) == 1).all(),
        "Membership must be explicit one-hot coverage of every asset",
    )
    require(
        list(memberships.columns) == ["Equity", "Fixed Income", "Alternatives"],
        "Unexpected sleeves",
    )
    require(
        premia.index.is_unique and premia.index.is_monotonic_increasing,
        "Premium dates must be unique and ordered",
    )
    require(list(premia.columns) == list(x.columns), "Premium factor basis differs")
    available = premia.index + pd.offsets.MonthEnd(1)
    past = premia.loc[available <= pd.Timestamp(CALIBRATION_END)]
    require(len(past) > 0, "No premium observation available at calibration cutoff")
    pi = past.iloc[-1].to_numpy(dtype=float)
    require(np.isfinite(pi).all(), "Incomplete supplied premium vector")
    xx, yy = x.to_numpy(), y.to_numpy()
    xs, ys = xx.std(0), yy.std(0)
    require(min(xs.min(), ys.min()) > 1e-12, "Degenerate calibration scale")
    z, v = (xx - xx.mean(0)) / xs, (yy - yy.mean(0)) / ys
    corr = z.T @ z / len(z)
    require(np.linalg.eigvalsh(corr)[0] > 1e-8, "Unidentified factor covariance")
    ols = np.linalg.lstsq(z, v, rcond=None)[0].T
    error = v - z @ ols.T
    cov = error.T @ error / len(error)
    noise = 0.9 * cov + 0.1 * np.diag(np.diag(cov))
    require(np.linalg.eigvalsh(noise)[0] > 0, "Nonpositive residual covariance")
    signal = np.sum((ols @ corr) * ols, axis=1)
    require(np.all(signal > 0), "Zero calibrated signal")
    labels = np.argmax(memberships.to_numpy(), axis=1)
    aligned, conflict = np.zeros_like(ols), np.zeros_like(ols)
    supports, details = [], []
    for label, name in enumerate(memberships.columns):
        ids = np.flatnonzero(labels == label)
        require(len(ids) >= 2, "Each explicit sleeve must have at least two assets")
        median = np.median(ols[ids], axis=0)
        support = np.argsort(-abs(median), kind="stable")[:3]
        supports.append(support)
        local_corr = corr[np.ix_(support, support)]
        _, eigenvectors = np.linalg.eigh(local_corr)
        direction = eigenvectors[:, -1]
        if np.dot(direction, median[support]) < 0:
            direction = -direction
        require(np.min(abs(direction)) > 1e-8, "Degenerate aligned loading direction")
        # Leading eigenvector on the active support guarantees partial/marginal alignment.
        a = np.zeros(x.shape[1])
        a[support] = direction
        # Most-correlated active pair; deterministic coordinate tie order, no scored fits.
        pairs = [(j, k) for j in range(3) for k in range(j + 1, 3)]
        j, k = max(pairs, key=lambda pair: abs(local_corr[pair]))
        rho = local_corr[j, k]
        require(abs(rho) >= 0.15, "Conflict construction lacks a sufficiently correlated pair")
        c = np.zeros(x.shape[1])
        c[support[j]], c[support[k]] = 1.0, -2.0 / rho
        remaining = list(set(range(3)) - {j, k})[0]
        c[support[remaining]] = 0.1 * np.sign(direction[remaining])
        # Orient the stress in the same target-coordinate direction as the aligned vector.
        c *= np.sign(a[support[j]])
        for target, vector in ((aligned, a), (conflict, c)):
            scale = np.sqrt(signal[ids] / (vector @ corr @ vector))
            target[ids] = scale[:, None] * vector
        require(np.all(a[support] * (corr @ a)[support] > 0), "Alignment construction failed")
        require(c[support[j]] * (corr @ c)[support[j]] < 0, "Conflict construction failed")
        details.append(
            dict(
                sleeve=name,
                assets=len(ids),
                active_factors=list(x.columns[support]),
                conflict_pair=list(x.columns[support[[j, k]]]),
                target_factor=x.columns[support[j]],
                pair_correlation=float(rho),
            )
        )
    covariance = corr * np.outer(xs, xs)
    arrays = dict(
        factor_covariance=covariance,
        residual_covariance=noise * np.outer(ys, ys),
        factor_sd=xs,
        response_sd=ys,
        proxy_ols=ols,
        truth_labels=labels,
        calibration_x=xx,
        calibration_y=yy,
        annual_premia=pi,
        aligned_beta=aligned * ys[:, None] / xs,
        conflict_beta=conflict * ys[:, None] / xs,
        calibrated_signal_variance=signal * ys**2,
    )
    metadata = dict(
        start=CALIBRATION_START,
        end=CALIBRATION_END,
        months=len(x),
        assets=list(y.columns),
        factors=list(x.columns),
        sleeves=details,
        premium_observation=str(past.index[-1].date()),
        premium_available=str((past.index[-1] + pd.offsets.MonthEnd(1)).date()),
        annual_premia=pi.tolist(),
        factor_min_eigenvalue=float(np.linalg.eigvalsh(corr)[0]),
        population_geometry=geometry(arrays),
        interpretation="Controlled sparse truth calibrated in scale, not true ETF betas",
    )
    return arrays, metadata


def geometry(arrays):
    """Population active-cell alignment before any finite-sample estimator comparison."""
    result = {}
    covariance = arrays["factor_covariance"]
    for kind in ("aligned", "conflict"):
        beta = arrays[kind + "_beta"]
        marginal = beta @ covariance
        active = beta != 0
        result[kind] = dict(
            active_cells=int(active.sum()),
            opposite_active_cells=int(((beta * marginal < 0) & active).sum()),
            minimum_signal_variance=float(np.sum(beta * marginal, axis=1).min()),
        )
        np.testing.assert_allclose(
            np.sum(beta * marginal, axis=1), arrays["calibrated_signal_variance"], atol=1e-12
        )
    return result


def design():
    """Numeric pilot choices fixed independently of comparative outcomes."""
    cells = [
        dict(history=t, alignment=a, strength=s, evolution=e)
        for t, a, s, e in itertools.product(
            (60, 120), ("aligned", "conflict"), ("weak", "strong"), ("stable", "reversal")
        )
    ]
    selected = [
        dict(history=60, alignment="aligned", strength="weak", evolution="stable"),
        dict(history=120, alignment="aligned", strength="strong", evolution="stable"),
        dict(history=60, alignment="conflict", strength="weak", evolution="reversal"),
        dict(history=120, alignment="conflict", strength="strong", evolution="reversal"),
    ]
    for cell in cells:
        cell["key"] = "_".join(
            str(cell[k]) for k in ("history", "alignment", "strength", "evolution")
        )
    selected_keys = [
        c["key"] for c in cells if any(all(c[k] == v for k, v in s.items()) for s in selected)
    ]
    return dict(
        study="etf_matf_known_truth_pilot_v1",
        annualization=12,
        primary_family="hcgl",
        family_reason="Row-separable exposition and pilot cost; not rankings",
        planned_full_core=cells,
        pilot_cells=selected_keys,
        pilot_replications=20,
        pilot_seed=2026102901,
        reserved_full_seed=2026103001,
        test_rows=512,
        random_design="SeedSequence([pilot_seed,replication]); 120+512 rows; tail training history",
        primary_weighting="equal",
        ew_sensitivity_span=120,
        core_arms=["ungated", "pooled_release", "pooled_exclude", "soft_direction", "ridge"],
        conditional_arms=["conditional_release", "conditional_exclude"],
        conditional_cells=[k for k in selected_keys if "conflict" in k],
        gate_diagnostic_arms=["singleton_release", "singleton_exclude", "pooled_zero_only"],
        gate_diagnostic_cells=[k for k in selected_keys if "weak" in k],
        fcgl_cells=[k for k in selected_keys if "strong" in k],
        ew_cells=[k for k in selected_keys if "strong" in k],
        soft_sensitivity_cells=[k for k in selected_keys if "weak" in k],
        soft_strength_primary=1.0,
        soft_strength_sensitivity=[0.1, 10.0],
        zero_strength_primary=0.0,
        soft_zero_sensitivity=1.0,
        sparse_fraction_grid=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0],
        ridge_grid=[0.0, 0.01, 0.1, 1.0, 10.0, 100.0],
        ridge_upper_extensions=[1000.0, 10000.0],
        ridge_boundary_rule="If largest lambda selected, append next extension; stop after 10000",
        sparse_boundary_rule="0 is exact endpoint; 1 is KKT zero endpoint; no extension past 1",
        selection="Three prefix folds; standardized reconstruction MSE; ties larger penalty",
        mechanism_penalty="Select unconstrained family lambda on training; share across gate arms",
        independent_tuning="Ridge uses same folds; separate gate-arm tuning deferred to full study",
        partition_cut=0.6,
        penalty_partition="Ward on weighted standardized training responses; relearn in each fold",
        gate_partition="Same as penalty except singleton arms; truth labels only for diagnostics",
        gate_threshold=2.0,
        l1_weight=0.5,
        geometry_sensitivity=dict(
            cut=[0.4, 0.8], threshold=[1.0, 3.0], scope="deferred; not part of this pilot"
        ),
        conditional_variance="Full WLS of standardized response mean; date HC1 T/(T-K-1); lags=0",
        marginal_variance="Existing package date score with Kish/(Kish-1)",
        threshold_interpretation="Regularization; no selection-adjusted or serial-dependence size",
        standardization="Training weighted means/sds; weights normalized to sum T; SSE/T",
        signal_variance_multiplier=dict(weak=0.25, strong=1.0),
        residuals="IID dates; .9 empirical residual covariance + .1 diagonal; fixed across cells",
        factors="IID Gaussian with fixed calibration covariance and supplied annual_premia/12 mean",
        truth="Three active factors per explicit sleeve; calibrated per-asset signal variance",
        reversal="First half training scale 1; next half linearly 1 to -0.5; test scale -0.5",
        reversal_scope="Coherent sign-reversal stress crossing zero; not typical ETF drift",
        alpha="True alpha zero; primary forecast ignores fitted intercept; full forecast secondary",
        premium_uncertainty="Known supplied vector; no perturbations or estimated-premium arms",
        primary_loss="mean over assets of summed squared annual factor contribution errors",
        secondary_loss="mean squared annual total expected-return error, zero fitted alpha",
        supporting=[
            "full forecast error",
            "annual reconstruction risk",
            "test MSE",
            "loading MSE",
            "component bias",
            "false exclusions",
            "wrong signs",
            "retention",
            "cluster ARI",
            "solver diagnostics",
            "runtime",
            "realized gate threshold crossings",
        ],
        inference="20 paired seeds/cell; MC SE and mean difference +/-1.96 SE; descriptive pilot",
        decision_rule="No positive-results-only gate; retain every condition and failure",
        full_run="Separate future freeze with fresh seeds, counts and precision rules after pilot",
        deferred=[
            "portfolio optimization regret",
            "CMA reoptimisations",
            "universe bootstrap",
            "serial dependence",
            "nonzero alpha",
            "OLS/Lasso/Coop/UniLasso full benchmark block",
        ],
        numerical_acceptance=dict(
            solver="CLARABEL",
            status="optimal or independently resolved_inaccurate after two attempts",
            max_kkt=1e-5,
            max_constraint_violation=1e-7,
            max_objective_discrepancy=1e-7,
            attempts=[1e-7, 1e-8],
            max_iter=[500, 2000],
            objective_compression="Exact weighted thin QR plus orthogonal SSE constant; all arms",
            refinement="FISTA on identical objective; target normalized KKT 1e-7; max 10000 steps",
            inaccurate_max_kkt=1e-7,
            inaccurate_max_relative_dual_gap=1e-8,
            inaccurate_max_solver_primal_dual_residual=1e-7,
        ),
        failure_rule="Stop cell/run on unverified fit; record failure; never average survivors",
    )


def inventory(p):
    """Unique pilot fit inventory; reused baseline fits are not counted twice."""
    rows = []
    for key in p["pilot_cells"]:
        configs = [("hcgl", None)]
        if key in p["fcgl_cells"]:
            configs.append(("fcgl", None))
        if key in p["ew_cells"]:
            configs.append(("hcgl", p["ew_sensitivity_span"]))
        for family, span in configs:
            arms = [
                (a, p["soft_strength_primary"] if a == "soft_direction" else 0.0, 0.0)
                for a in p["core_arms"]
                if a != "ridge" or family == "hcgl"
            ]
            if family == "hcgl" and span is None:
                if key in p["conditional_cells"]:
                    arms += [(a, 0.0, 0.0) for a in p["conditional_arms"]]
                if key in p["gate_diagnostic_cells"]:
                    arms += [(a, 0.0, 0.0) for a in p["gate_diagnostic_arms"]]
                if key in p["soft_sensitivity_cells"]:
                    arms += [("soft_direction", v, 0.0) for v in p["soft_strength_sensitivity"]]
                    arms += [("soft_exclusion", 1.0, p["soft_zero_sensitivity"])]
            for replication, (arm, strength, zero_strength) in itertools.product(
                range(p["pilot_replications"]), arms
            ):
                rows.append(
                    dict(
                        cell=key,
                        replication=replication,
                        family=family,
                        span=span,
                        arm=arm,
                        soft_strength=strength,
                        zero_strength=zero_strength,
                    )
                )
    require(
        len({json.dumps(row, sort_keys=True) for row in rows}) == len(rows),
        "Duplicate planned fits",
    )
    return rows


def generate(arrays, cell, replication, p):
    """Draw one frozen synthetic dataset; no fitting or evaluation of comparators."""
    require(0 <= replication < p["pilot_replications"], "Unexpected pilot seed")
    rng = np.random.default_rng(np.random.SeedSequence([p["pilot_seed"], replication]))
    n, k, q = cell["history"], len(arrays["annual_premia"]), len(arrays["truth_labels"])
    length = 120 + p["test_rows"]
    x = rng.normal(size=(length, k)) @ np.linalg.cholesky(arrays["factor_covariance"]).T
    x += arrays["annual_premia"] / 12
    eps = rng.normal(size=(length, q)) @ np.linalg.cholesky(arrays["residual_covariance"]).T
    beta = arrays[cell["alignment"] + "_beta"] * np.sqrt(
        p["signal_variance_multiplier"][cell["strength"]]
    )
    scale = np.ones(n)
    if cell["evolution"] == "reversal":
        scale[n // 2 :] = np.linspace(1.0, -0.5, n - n // 2)
    train = slice(120 - n, 120)
    target = beta * scale[-1]
    return dict(
        x=x[train],
        y=(x[train] @ beta.T) * scale[:, None] + eps[train],
        x_test=x[120:],
        y_test=x[120:] @ target.T + eps[120:],
        truth=target,
        loading_scales=scale,
        training_beta=beta,
        annual_premia=arrays["annual_premia"].copy(),
    )


def freeze(bundle, parent, validation, output):
    """Authenticate earlier inputs, calibrate, and save a complete pilot-only freeze."""
    output = external(output)
    import_path = Path(factorlasso.__file__).resolve()
    require(
        import_path == (REPO / "src/factorlasso/__init__.py").resolve(),
        "Use the repository source checkout, not an unrelated installed distribution",
    )
    require(not output.exists() or not any(output.iterdir()), "Freeze needs a fresh directory")
    manifest, master = read(bundle / "manifest.json"), read(parent / "master_frozen.json")
    require(digest(master["frozen"]) == master["identity"], "Changed parent CMA freeze")
    source_inputs = master["frozen"]["variants"][0]["inputs"]
    require(
        source_inputs["parent_bundle_manifest_sha256"] == sha(bundle / "manifest.json"),
        "Unbound CMA bundle",
    )
    membership_file = parent / "explicit_class_membership.csv"
    require(
        source_inputs["allocation_policy"]["membership_sha256"] == sha(membership_file),
        "Explicit sleeve membership changed",
    )
    input_hashes = {
        str(bundle / "manifest.json"): sha(bundle / "manifest.json"),
        str(parent / "master_frozen.json"): sha(parent / "master_frozen.json"),
        str(membership_file): sha(membership_file),
    }
    frames = {}
    for name, detail in manifest["files"].items():
        file = bundle / detail["name"]
        require(sha(file) == detail["sha256"], "Changed input bundle file: " + name)
        input_hashes[str(file)] = detail["sha256"]
        if name in ("asset_excess_logreturns", "factor_logreturns", "premia"):
            frames[name] = pd.read_csv(
                file, index_col=0, parse_dates=True, float_precision="round_trip"
            )
    membership = pd.read_csv(membership_file, index_col=0)
    arrays, metadata = calibration(
        frames["factor_logreturns"], frames["asset_excess_logreturns"], membership, frames["premia"]
    )
    v = read(validation / "verification.json")
    require(v["status"] == "passed", "Comparators must be independently validated")
    for name, expected in v["source_sha256"].items():
        require(sha(HERE / name) == expected, "Comparator changed after validation")
    for name, expected in v["files_sha256"].items():
        require(sha(validation / name) == expected, "Changed comparator verification artifacts")
    p = design()
    p.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        calibration=metadata,
        inputs_retrospective=manifest["historical_status"],
        comparative_pilot_run=False,
    )
    jobs = inventory(p)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "calibration.npz", **arrays)
    write(output / "protocol.json", p)
    write(output / "planned_fits.json", jobs)
    source_files = [
        HERE / name
        for name in (
            "matf_comparators.py",
            "verify_matf_comparators.py",
            "matf_pilot_protocol.py",
            "tests/test_matf_comparators.py",
            "french_estimators.py",
            "french_attribution.py",
            "cvxpy_shape_compat.py",
            "estimator_benchmark.py",
        )
    ]
    sources = {file.relative_to(REPO).as_posix(): sha(file) for file in source_files}
    sources["pyproject.toml"] = sha(REPO / "pyproject.toml")
    core = {
        file.relative_to(REPO).as_posix(): sha(file)
        for file in sorted((REPO / "src/factorlasso").rglob("*.py"))
    }
    binding = dict(
        protocol_sha256=sha(output / "protocol.json"),
        calibration_sha256=sha(output / "calibration.npz"),
        planned_fits_sha256=sha(output / "planned_fits.json"),
        source_sha256=sources,
        unchanged_estimator_sha256=core,
        inputs_sha256=input_hashes,
        validation_path=str(validation),
        validation_sha256=sha(validation / "verification.json"),
        python=sys.executable,
        factorlasso_import=str(import_path),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "cvxpy", "clarabel", "factorlasso")
        },
    )
    write(output / "freeze.json", dict(identity=digest(binding), binding=binding))
    for name in sources:
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / name).read_bytes())
    return dict(
        identity=digest(binding),
        status="frozen; comparative pilot not run",
        datasets=len(p["pilot_cells"]) * p["pilot_replications"],
        planned_fits=len(jobs),
        calibration=metadata,
        output=str(output),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.bundle, args.parent, args.validation, args.out), indent=2))
