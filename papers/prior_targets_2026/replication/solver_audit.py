"""Audit every inaccurate group-path candidate and selected fallback without changing results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

from papers.prior_targets_2026.replication.run import (
    HERE, digest, read_json, save_json, validate_root, check_snapshot,
)
from papers.prior_targets_2026.replication.multiasset import calibration, check_manifest, samples_for


def main():
    """Independently resolve flagged candidates and verify the recorded selections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = validate_root(args.output_root)
    check_snapshot(root)
    sys.path.insert(0, str(root/"source_snapshot/src"))
    from papers.prior_targets_2026.replication.groups import model_for
    from papers.prior_targets_2026.replication.estimators import select_index
    source = root/"E3_groups_confirmation_v2"
    check_manifest(source)
    out = root/"E3_solver_audit_v1"
    out.mkdir()
    shutil.copy2(HERE/"solver_audit.py", out/"solver_audit.py")
    config = read_json(source/"archive/groups_protocol_v2.json")
    data = calibration(root)
    universe = pd.read_csv(root/"source_snapshot/data/etf_universe.csv").set_index("ticker")
    data["groups"] = universe.loc[data["tickers"], "sub_asset_class"].to_numpy()
    events = pd.read_csv(source/"solver_events.csv")
    panels = pd.read_csv(source/"panel_results.csv")
    assets = pd.read_csv(source/"asset_results.csv")
    paths = pd.read_csv(source/"validation_paths.csv")
    keys = ["profile", "seed", "setting", "target", "lambda_index"]
    selected = panels.merge(events, on=keys, suffixes=("", "_event"), validate="one_to_one")
    flags = pd.concat([events[events.status=="optimal_inaccurate"],
                       selected[selected.fallback][events.columns]]).drop_duplicates(keys)
    assert len(events[events.status=="optimal_inaccurate"])==2
    assert len(selected[selected.fallback])==3
    rows = []
    for _, event in flags.iterrows():
        samples, _, _, _, _, _ = samples_for(
            data, event.profile, 112, int(event.seed), config)
        (xt, yt), (xv, yv), _ = samples
        xs, ys = xt.std(0), yt.std(0)
        z = pd.DataFrame((xt-xt.mean(0))/xs, columns=data["factors"])
        v = pd.DataFrame((yt-yt.mean(0))/ys, columns=data["tickers"])
        prior = pd.DataFrame(data["prior"]*xs[None]/ys[:, None],
                             index=data["tickers"], columns=data["factors"])
        groups = pd.Series(data["groups"], index=data["tickers"])
        setting = next(s for s in config["settings"] if s["name"]==event.setting)
        fresh = model_for(setting, event.target, prior, groups, solver="ECOS")
        fresh.reg_lambda = event.reg_lambda
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fresh.fit(z, v)
        assert not caught, [str(w.message) for w in caught]
        beta = fresh.coef_.to_numpy()*ys[:, None]/xs[None]
        alpha = yt.mean(0)+fresh.alpha_const_.to_numpy()*ys-beta@xt.mean(0)
        loss = float(np.mean(np.mean((yv-xv@beta.T-alpha)**2, axis=0)/yt.var(0)))
        mask = ((paths.profile==event.profile)&(paths.seed==event.seed)
                &(paths.setting==event.setting)&(paths.target==event.target))
        path = paths[mask].sort_values("lambda_index")
        losses = path.validation_nmse.to_numpy().copy()
        recorded_loss = losses[event.lambda_index]
        losses[event.lambda_index] = loss
        choice = selected[(selected.profile==event.profile)&(selected.seed==event.seed)
                          &(selected.setting==event.setting)&(selected.target==event.target)].iloc[0]
        unchanged = select_index(losses)==choice.lambda_index
        assert unchanged
        difference = None
        is_selected = choice.lambda_index==event.lambda_index
        if is_selected:
            part = assets[(assets.profile==event.profile)&(assets.seed==event.seed)
                          &(assets.setting==event.setting)&(assets.target==event.target)
                          ].sort_values("asset")
            saved = part[[f"beta{f}" for f in range(9)]].to_numpy()
            difference = float(np.max(abs((saved-beta)*xs[None]/ys[:, None])))
            assert difference<5e-4
        rows.append(dict(profile=event.profile, seed=int(event.seed), setting=event.setting,
                         target=event.target, lambda_index=int(event.lambda_index),
                         original_status=event.status, was_selected=bool(is_selected),
                         original_validation_loss=recorded_loss, ecos_validation_loss=loss,
                         loss_difference=loss-recorded_loss, selection_unchanged=bool(unchanged),
                         selected_standardised_beta_error=difference))
    pd.DataFrame(rows).to_csv(out/"audit.csv", index=False)
    receipt = dict(candidates_checked=len(rows), inaccurate_candidates=2,
                   selected_inaccurate=int(selected.status.eq("optimal_inaccurate").sum()),
                   selected_fallbacks_checked=3, all_choices_unchanged=True,
                   independent_solver_warnings=0,
                   selected_max_standardised_error=max(r["selected_standardised_beta_error"]
                                                        or 0 for r in rows),
                   source_results_changed=False)
    save_json(out/"summary.json", receipt)
    save_json(out/"manifest.json",
              {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()})
    print(json.dumps(receipt, indent=2))


if __name__=="__main__":
    main()

