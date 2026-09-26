"""Refresh evidence tables and tagged numbers in manuscript.md without rewriting prose.

manuscript.md is the only prose source. ``prepare`` replaces the contents of
``<!-- evidence: key -->`` blocks and ``<!-- value: key -->`` tags from the evidence root,
checks every directional claim the prose makes, and records any claim whose direction changed.
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from . import fi_validation as f
from . import fi_validation_paper as paper


def qualitative_checks(root):
    """Check every retained comparative claim rather than assuming its old direction."""
    checks={}
    for lane in ['indices','funds']:
        scores=pd.read_csv(root/f'summary/{lane}/asset_scores.csv')
        for span in [60,36]:
            avg=scores.query('span==@span').groupby('arm').oos_r2.mean().reindex(list(f.ARMS))
            checks[f'{lane}_{span}_conditional_best']=avg.idxmax()=='E2'
        boot=pd.read_csv(root/f'summary/{lane}/bootstrap.csv').query('span==60 and block==4 and arm=="E2" and comparator=="A"')
        checks[f'{lane}_conditional_gain_positive_interval']=bool((boot.low>0).all())
    scores=pd.read_csv(root/'sign_revision/oos_summary.csv').query('span==60')
    for lane in ['indices','funds']:
        d=scores.query('lane==@lane').pivot(index='arm',columns='variant',values='monthly_rmse_pct')
        checks[f'{lane}_all_sign_policies_improve']=bool((d.loc[list(f.ARMS),'date']<d.loc[list(f.ARMS),'old']).all())
        for arm in ['A','E1','E2']:
            checks[f'{lane}_{arm}_beats_target']=bool(d.loc[arm,'date']<d.loc[arm+'_target_only','date'])
        checks[f'{lane}_conditional_beats_zero']=bool(d.loc['E2','date']<d.loc['E2_lambda_zero','date'])
        # Directions stated in Section 5 and the abstract.
        checks[f'{lane}_conditional_beats_zero_penalty_detected']=bool(d.loc['E2','date']<d.loc['Z_lambda_zero','date'])
        checks[f'{lane}_zero_penalty_beats_zero_prior']=bool(d.loc['Z_lambda_zero','date']<d.loc['Z','date'])
        checks[f'{lane}_prior_signs_beat_detected_at_zero_penalty']=bool(d.loc['E2_lambda_zero','date']<d.loc['Z_lambda_zero','date'])
        # Abstract: most of the gain over Zero prior is the penalty's cost at the zero centre.
        z,z0,e2=(d.loc[a,'date'] for a in ['Z','Z_lambda_zero','E2'])
        checks[f'{lane}_penalty_cost_exceeds_half_gap']=bool(z-z0>.5*(z-e2))
    duration=pd.read_csv(root/'summary/oad_rank_association.csv').set_index('arm')
    checks['duration_does_not_favor_conditional']=duration.spearman_rates_oad.idxmax()!='E2'
    replay=pd.read_csv(root/'R1/full_replay/comparison.csv',index_col=0)
    checks['joint_IL_improves_IL']=bool(replay.loc['LF94TRUH Index','r2_current']>replay.loc['LF94TRUH Index','r2_univariate'])
    checks['joint_IL_lowers_IG_agg_fit']=bool(replay.loc['LEGATRUH Index','r2_current']<replay.loc['LEGATRUH Index','r2_univariate'])
    focal=pd.read_csv(root/'summary/mc_focal_cases.csv')
    wrong=focal.query('case=="credit_wrong" and asset=="IG"').set_index('arm')
    weak=focal.query('case=="il_weak" and asset=="IL"').set_index('arm')
    checks['wrong_mapping_beta_worse']=bool(wrong.loc['E2','mean_beta_mse']>wrong.loc['A','mean_beta_mse'])
    checks['wrong_mapping_cma_better']=bool(wrong.loc['E2','mean_absolute_cma_error_bp']<wrong.loc['A','mean_absolute_cma_error_bp'])
    checks['weak_IL_worse_than_zero']=bool(weak.loc['E2','mean_absolute_cma_error_bp']>weak.loc['Z','mean_absolute_cma_error_bp'])
    f.jsave(root/'qualitative_claim_checks.json',checks)
    return checks


def prepare(root,here):
    """Regenerate all tables and update the approved manuscript with current-method evidence."""
    root,here=Path(root),Path(here)
    blocks,claims=paper.evidence(root,root/'exhibits',root/'stress')
    source=(here/'manuscript.md').read_text(encoding='utf-8')
    for key,value in blocks.items():
        pattern=r'(<!-- evidence: '+key+r' -->\n).*?(\n<!-- /evidence -->)'
        source,count=re.subn(pattern,lambda m:m[1]+value+m[2],source,flags=re.S)
        assert count==1,key
    vals=paper.values(root,root/'stress')
    source=paper.fill_values(source,vals)
    paper.verify_values(source,vals)
    checks=qualitative_checks(root)
    # Any changed ranking must be resolved explicitly in prose before publication.
    expected_false={'indices_conditional_beats_zero','funds_conditional_beats_zero',
                    'funds_conditional_beats_zero_penalty_detected'}
    failures=[key for key,value in checks.items() if bool(value)!=(key not in expected_false)]
    f.jsave(root/'editorial_review_flags.json',dict(changed_comparative_claims=failures))
    paper.verify(source,blocks)
    (here/'manuscript.md').write_text(source,encoding='utf-8')
    metadata=json.loads((here/'paper_sources.json').read_text(encoding='utf-8'))
    if metadata['study_root']!=str(root):metadata['previous_edition_study_root']=metadata['study_root']
    metadata.update(study_root=str(root),
        exhibit_root=str(root/'exhibits'),stress_root=str(root/'stress'),checked_on=str(pd.Timestamp.today().date()),
        calibration_revision='current normalized CMA calibration and approved factor premia',
        monthly_penalty=f.MAIN_LAMBDA,loss_normalization='weight_sum')
    f.jsave(here/'paper_sources.json',metadata)
    f.jsave(root/'numerical_claims.json',claims)
    f.jsave(root/'evidence_manifest.json',{str(p.relative_to(root)).replace('\\','/'):f.sha(p)
        for prefix in ['inputs','characteristics','R1','R2','R3','R4','OSS','summary','sign_revision']
        for p in (root/prefix).rglob('*') if p.is_file()})
    f.jsave(root/'numerical_values.json',vals)
    print('Prepared',len(blocks),'blocks and',len(vals),'values; editorial review flags:',failures,flush=True)
