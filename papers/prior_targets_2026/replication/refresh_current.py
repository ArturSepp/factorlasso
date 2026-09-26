"""Refresh the FI prior validation against an explicitly frozen current CMA book.

All research outputs stay in an explicit C-local directory. Previous paper evidence
is preserved. The empirical setup uses current CMA settings, including a fixed
normalized penalty; historical splits are retrospective sensitivity exercises.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
import pandas as pd
from . import fi_validation as f
from . import fi_validation_metrics as metrics
from . import fi_validation_synthetic as synthetic
from . import sign_revision
from . import fi_validation_exhibits as exhibits
from . import fi_validation_stress as stress


def setup(root):
    """Select the requested evidence tree and bind synthetic pricing to its frozen premia."""
    f.ROOT=root
    if (root/'inputs/premia.csv').exists():
        synthetic.PREMIA=f.read(root/'inputs/premia.csv').iloc[0].reindex(synthetic.FACTORS).to_numpy()
        assert np.isfinite(synthetic.PREMIA).all()
    synthetic.MAIN_LAMBDA=f.MAIN_LAMBDA


def initialise(root,previous,cma_root):
    """Freeze same-vintage observations and current economic inputs with exact identities."""
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'inputs').exists(), 'Use a fresh evidence directory'
    for folder in ['inputs','characteristics']:
        shutil.copytree(previous/folder,root/folder)
    for name in ['protocol.json','R0_receipt.json','approved_roadmap.md']:
        shutil.copy2(previous/name,root/name)
    owner=root/'owner_inputs';owner.mkdir()
    filenames=['global_saa_universe_data_cmas_usd_2026q2.xlsx','factor_cma2026.xlsx',
               'factor_cma2026_factor_prices.csv','global_saa_universe_data_metadata.csv']
    for name in filenames:shutil.copy2(cma_root/name,owner/name)
    f.ROOT=root
    assets,bundle=f.final_assets(),f.factor_bundle()
    newx=f.qis.to_returns(bundle.factor_prices[f.FACTORS].resample('ME').last(),is_log_returns=True).loc['2000-01-31':f.CUT]
    oldx=f.read(root/'inputs/x.csv')
    np.testing.assert_allclose(newx.loc[oldx.index],oldx,atol=1e-14)
    roster=pd.read_csv(root/'inputs/roster.csv',index_col=0)
    oldy=f.read(root/'inputs/y.csv')
    for ticker in roster.index[roster.source.eq('CMA')]:
        valid=oldy[ticker].notna()
        np.testing.assert_allclose(oldy.loc[valid,ticker],assets.excess_logreturns[ticker].reindex(oldy.index)[valid],atol=1e-14)
    # The regression observations stay byte-identical; only economic/context inputs refresh.
    meta=pd.read_csv(owner/'global_saa_universe_data_metadata.csv',index_col=0)
    assert meta.loc[roster.index[roster.source.eq('CMA')],'alpha_weight'].eq(0).all()
    for name,data in [('full_y',assets.excess_logreturns),('full_metadata',meta),
                      ('factor_covar',assets.covar_data.x_covar),('full_betas',assets.covar_data.y_betas),
                      ('full_variances',assets.covar_data.y_variances),('full_clusters',assets.covar_data.clusters),
                      ('premia',bundle.select(f.MATF_CUSTOM_IG_HY).cmas.loc[[f.CUT]])]:
        f.csv(data,root/'inputs'/f'{name}.csv')
    f.jsave(root/'inputs/manifest.json',{p.name:f.sha(p) for p in (root/'inputs').iterdir() if p.is_file() and p.name!='manifest.json'})
    protocol=json.loads((root/'protocol.json').read_text())
    spec=f.get_cma_covar_estimation_spec()
    assert spec.loss_normalization=='weight_sum' and spec.auto_sign_use_fit_span and spec.auto_sign_variance=='date' and spec.solver=='MOSEK'
    protocol.update(lambda_main=f.MAIN_LAMBDA,lambda_grid=f.GRID,loss_normalization=spec.loss_normalization,
        revision='approved current calibration',previous_evidence=str(previous),
        sign_revision=dict(ewma='effective native fit span',variance='date-score sandwich',
            missingness='original masks',solver='MOSEK',loss='per-response valid EWMA weight sum'),
        penalty_calibration=dict(monthly=spec.reg_lambda_freq_dict['ME'],quarterly=spec.reg_lambda_freq_dict['QE'],
            historical_use='current-cut calibration held fixed retrospectively; no tuning on these scores',
            span36='same normalized penalty; explicit horizon sensitivity'),
        owner_inputs={str(cma_root/name):f.sha(owner/name) for name in filenames},
        admission='Global baseline PE 50%, ILS 100%; every study FI index has zero admitted alpha',
        zero_penalty_controls={k:f'{v} final signs and clusters, lambda=0' for k,v in f.CONTROL_BASE.items()},
        monte_carlo=dict(arms=synthetic.ARMS,target_r2=synthetic.TARGET_R2,residual_noise='per asset, population R-squared = target_r2'),
        unchanged_observations=['x.csv','y.csv','roster.csv'],
        full_context='182-index current CMA book; reviewed Equity factor selections retained')
    f.jsave(root/'protocol.json',protocol)
    setup(root)
    # Regression guard: legacy runner's 1e-5 would fail this check.
    rr=roster.query("lane=='indices'")
    model=f.make_model(rr,'E2')
    assert model.reg_lambda==spec.reg_lambda_freq_dict['ME']
    assert model.loss_normalization=='weight_sum' and model.solver=='MOSEK'
    assert f.make_model(rr,'E2',lam=0.).reg_lambda==0.
    f.jsave(root/'configuration_validation.json',dict(current_lambda=model.reg_lambda,
        stale_lambda=1e-5,stale_default_would_fail=1e-5!=model.reg_lambda,
        explicit_zero_preserved=True,input_panel_parity=True,solver='MOSEK'))
    print('Current calibration frozen',model.reg_lambda,flush=True)


def compute(stage,root):
    """Run each existing owner computation without changing live settings."""
    setup(root)
    if stage=='endpoint':
        for lane in ['indices','funds']:f.r1(lane)
        f.r2();metrics.full_replay(root);metrics.diagnostics(root)
        stress.compute(root,root/'stress')
    elif stage=='oos':
        for lane in ['indices','funds']:
            f.r3(lane);metrics.scores(root,lane)
    elif stage=='sign':
        out=root/'sign_revision';out.mkdir(exist_ok=True)
        sign_revision.endpoint(out,source=root)
        sign_revision.oos(out,source=root)
        f.jsave(out/'protocol.json',dict(loss_normalization='weight_sum',reg_lambda=f.MAIN_LAMBDA,
            variants=sign_revision.LABELS,solver='MOSEK',source=str(root),
            interpretation='controlled sign ablation at current loss and economic calibration; not an exact archived software replay'))
    elif stage=='mc':
        synthetic.run(root,examples=True,solver='MOSEK')
        synthetic.run(root,solver='MOSEK')
    elif stage=='checks':
        metrics.final_checks(root)
        if (root/'exhibits').exists():exhibits.verify(root/'exhibits')
        else:exhibits.compute(root,root/'exhibits')
    elif stage=='prepare':
        from .current_revision_paper import prepare
        prepare(root,Path(__file__).resolve().parents[1])
    else:raise ValueError(stage)


def main():
    """Require explicit frozen source locations for a reviewable, resumable refresh."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['init','endpoint','oos','sign','mc','checks','prepare'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous',type=Path)
    p.add_argument('--cma-root',type=Path)
    a=p.parse_args();root=a.root.resolve()
    assert root.is_absolute() and 'OneDrive' not in root.parts
    if a.stage=='init':initialise(root,a.previous,a.cma_root)
    else:compute(a.stage,root)

if __name__=='__main__':main()
