"""Refit one-factor sensitivity settings using the extended penalty grid.

Only the completed inner-CV surfaces select penalties. The reference fixes
cut=.5, alpha=.5 and tau=1.5. Every other reported setting changes exactly one
applicable parameter. Can consume finished shards while the primary study runs.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

if __package__:
    from . import tuning_boundary_validation as T
else:
    import tuning_boundary_validation as T


def variant_name(candidate):
    changes = [(name,value) for name,value,base in
               [('cut',candidate.cut,.5),('alpha',candidate.alpha,.5),('tau',candidate.tau,1.5)]
               if value != base]
    if len(changes)>1:
        return None
    return 'reference' if not changes else f'{changes[0][0]}_{changes[0][1]:g}'


def refit(job):
    root,out,design,seed=job
    root,out=Path(root),Path(out)
    warnings.filterwarnings('error',category=RuntimeWarning)
    np.seterr(invalid='raise',divide='raise',over='raise')
    prefix=f'{design.name}_{seed}'
    cv=pd.read_csv(root/'shards'/f'{prefix}_cv.csv')
    if not (cv.seed==seed).all() or not (cv.scenario==design.name).all():
        raise ValueError('Sensitivity input identity mismatch')
    x,y,xt,yt,truth,groups=T.L.simulate(design,seed)
    rows,arrays=[],{}
    for method in T.L.METHODS:
        part=cv[cv.method==method]
        for (cut,alpha,tau),frame in part.groupby(['cut','alpha','tau'],sort=False):
            candidate=T.L.Candidate(cut,alpha,tau)
            variant=variant_name(candidate)
            if variant is None:
                continue
            if set(frame.lambda_fraction)!=set(T.FRACTIONS):
                raise ValueError('Incomplete sensitivity penalty surface')
            choice=frame.iloc[np.argmin(frame.cv_mse)]
            prep=T.B.prepare(x,y,cut)
            b,a,lam,signs=T.path(prep,method,candidate,[choice.lambda_fraction])
            fitted=T.B.Fitted(b[0],a[0],prep,lam[0],choice.lambda_fraction,tau,
                              choice.cv_mse,0.,signs)
            for suffix,value in [('beta',b[0]),('intercept',a[0]),('labels',prep.labels),
                                 ('signs',signs)]:
                arrays[f'{method}__{variant}__{suffix}']=value
            for tolerance in T.B.ZERO_TOLERANCES:
                rows.append(dict(scenario=design.name,seed=seed,method=method,variant=variant,
                                 cut=cut,alpha=alpha,tau=tau,zero_tolerance=tolerance,
                                 lambda_fraction=choice.lambda_fraction,cv_mse=choice.cv_mse,
                                 **T.B.evaluate(fitted,xt,yt,truth,tolerance)))
    pd.DataFrame(rows).to_csv(out/'shards'/f'{prefix}_draws.csv',index=False)
    np.savez_compressed(out/'shards'/f'{prefix}_coefficients.npz',**arrays)
    result=dict(scenario=design.name,seed=seed,rows=len(rows),warnings=0,
                input_cv_sha256=T.digest(root/'shards'/f'{prefix}_cv.csv'))
    (out/'shards'/f'{prefix}_receipt.json').write_text(json.dumps(result,indent=2))
    return result


def run(root,out,workers=2,watch=False):
    root=Path(root).resolve()
    out=T.B.safe_output(out)
    (out/'shards').mkdir()
    declaration=json.loads((root/'protocol.json').read_text())
    if declaration['study']!='learned' or declaration['seeds']!=list(range(1000,1040)):
        raise ValueError('Require full learned endpoint study')
    protocol=dict(parent=str(root),parent_protocol_sha256=T.digest(root/'protocol.json'),
                  selection='Existing training-only CV surfaces; no new test-based choice',
                  reference=dict(cut=.5,alpha=.5,tau=1.5),workers=workers)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    sources=[Path(__file__),Path(T.__file__)]
    for p in sources:
        (out/(p.stem+'_snapshot.py')).write_bytes(p.read_bytes())
    receipt=dict(status='running',completed=0,
                 source_hashes={str(p):T.digest(p) for p in sources})
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    waiting={(d.name,s):(d,s) for d in T.L.DESIGNS for s in range(1000,1040)}
    running={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while waiting or running:
            for key,(design,seed) in list(waiting.items()):
                if len(running)>=workers*2:
                    break
                ready=root/'shards'/f'{design.name}_{seed}_receipt.json'
                if ready.exists():
                    future=pool.submit(refit,(str(root),str(out),design,seed))
                    running[future]=key
                    del waiting[key]
            if not running:
                parent=json.loads((root/'receipt.json').read_text())
                if not watch or parent['status']=='failed':
                    raise RuntimeError('Parent incomplete or failed')
                time.sleep(10)
                continue
            done,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                result=future.result()
                del running[future]
                receipt['completed']+=1
                (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
                print(receipt['completed'],result['scenario'],result['seed'],flush=True)
    draws=pd.concat([pd.read_csv(p) for p in (out/'shards').glob('*_draws.csv')])
    draws.to_csv(out/'estimator_draws.csv',index=False)
    receipt.update(status='complete',outputs={str(p.relative_to(out)):T.digest(p)
                   for p in out.rglob('*') if p.is_file() and p.name!='receipt.json'})
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    run(args.root,args.out,args.workers,args.watch)
