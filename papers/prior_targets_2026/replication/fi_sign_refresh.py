"""Recompute the prior note under the verified sign revision, keeping old evidence frozen."""
from pathlib import Path
import argparse,json,shutil
from papers.prior_targets_2026.replication import fi_validation as f
from papers.prior_targets_2026.replication import fi_validation_metrics as metrics
from papers.prior_targets_2026.replication import fi_validation_synthetic as synthetic

OLD=Path('C:/Users/artur/AppData/Local/AgentWork/ARTURDESKTOP/FactorLasso/analyses/fi_prior_validation_20260924')
ROOT=Path('C:/Users/artur/AppData/Local/AgentWork/ARTURDESKTOP/FactorLasso/analyses/fi_sign_validation_20260925')

def init():
    """Copy only unchanged inputs and record the methodological revision."""
    ROOT.mkdir(exist_ok=True)
    for folder in ('inputs','characteristics'):
        if not (ROOT/folder).exists():shutil.copytree(OLD/folder,ROOT/folder)
    for name in ('protocol.json','R0_receipt.json','approved_roadmap.md'):
        if not (ROOT/name).exists():shutil.copy2(OLD/name,ROOT/name)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    protocol['sign_revision']=dict(ewma='effective native fit span',variance='date-score sandwich',
        missingness='original response and factor masks restored',solver='MOSEK',loss='unchanged 1/T',
        prior_evidence=str(OLD))
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

def study():
    """Rebuild empirical endpoint, CMA and held-out exhibits using their owners."""
    f.ROOT=ROOT
    for lane in ('indices','funds'):f.r1(lane)
    f.r2()
    metrics.full_replay(ROOT)
    metrics.diagnostics(ROOT)
    for lane in ('indices','funds'):
        f.r3(lane)
        metrics.scores(ROOT,lane)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['init','study','mc'],required=True)
    args=parser.parse_args()
    if args.stage=='init':init()
    elif args.stage=='study':study()
    else:
        synthetic.run(ROOT,examples=True,solver='MOSEK')
        synthetic.run(ROOT,solver='MOSEK')
