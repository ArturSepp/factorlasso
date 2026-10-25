"""Training-only endpoint extension of the two synthetic revision studies.

The declared finite grid reaches natural domains: cut 0/1, mixing 0/1,
gate threshold 0/infinity, and penalty fraction 0/1. The upper penalty is a
sufficient zero-solution bound. Zero penalty is constrained least squares;
infinite tau rejects every gate (or releases every cell for the release rule).
Previously evaluated CV cells are retained exactly, with hash-checked provenance.
Use --fresh-cv to recompute every cell without any prior result directory.
Only mathematically identical paths within a training fold share computations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

if __package__:
    from . import learned_pooling_validation as L
    from . import learned_pooling_expanded as E
    from .cvxpy_shape_compat import safe_cvxpy_sum_shape, compatibility_record
else:
    import learned_pooling_validation as L
    import learned_pooling_expanded as E
    from cvxpy_shape_compat import safe_cvxpy_sum_shape, compatibility_record

B = L.B
CUTS = (.35, .5, .7, 0., .175, .85, 1.)
ALPHAS = (.25, .5, .75, 1., .1, 0.)
TAUS = (0., 1.5, 3., .75, 6., np.inf)
FRACTIONS = (1., .3, .1, .03, .01, .003, .001, .0003, .0001,
             .00003, .00001, .000003, .000001, 0.)
KEYS = ('cut', 'alpha', 'tau', 'lambda_fraction')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(candidate):
    values = asdict(B.Protocol())
    values.update(alpha=candidate.alpha, cutoff_fraction=candidate.cut,
                  tau_grid=(candidate.tau,))
    return SimpleNamespace(**values)


def candidates(method, study):
    if study == 'original':
        taus = (.75, 1.5, 2.5, 0., 3., 6., np.inf) if method in B.SIGNED_METHODS else (1.5,)
        return tuple(L.Candidate(.5, .5, t) for t in taus)
    if method in ('lasso', 'unilasso'):
        return (L.Candidate(),)
    if method == 'cooperative':
        return tuple(L.Candidate(c) for c in CUTS)
    old = list(L.candidates_for(method)) + list(E.endpoint_candidates(method))
    taus = TAUS if method in B.SIGNED_METHODS else (1.5,)
    new = [L.Candidate(c, a, t) for c in CUTS for a in ALPHAS for t in taus]
    return tuple(dict.fromkeys(old + new))


def zero_penalty(prep, signs):
    """Independent per-response bounded least squares, with pinned zeros removed."""
    beta = np.zeros((prep.y.shape[1], prep.x.shape[1]))
    for k in range(len(beta)):
        keep = signs[k] != 0
        if not keep.any():
            continue
        xx, yy, ss = prep.x[:, keep], prep.y[:, k], signs[k, keep]
        lower = np.where(ss == 1, 0., -np.inf)
        upper = np.where(ss == -1, 0., np.inf)
        if not np.isfinite(lower).any() and not np.isfinite(upper).any():
            value = np.linalg.lstsq(xx, yy, rcond=None)[0]
        else:
            fit = lsq_linear(xx, yy, bounds=(lower, upper), method='bvls',
                             tol=1e-12, max_iter=1000)
            if not fit.success:
                raise RuntimeError(f'Constrained least squares failed: {fit.message}')
            value = fit.x
        beta[k, keep] = value
    return beta


def pure_group_scale(prep, method, labels):
    """Dual-norm zero bound for alpha=0 (row HCGL or column-block FCGL)."""
    corr = 2 * prep.y.T @ prep.x / len(prep.x)
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    axis = 0 if method == 'fcgl_unconstrained' else 1
    return max(float(np.max(np.linalg.norm(corr[g], axis=axis))) /
               np.sqrt(len(g)/len(groups)) for g in groups)


@safe_cvxpy_sum_shape()
def path(prep, method, candidate, fractions, known_labels=None):
    prot = protocol(candidate)
    labels, signs, _ = B.method_inputs(prep, method, candidate.tau, prot, known_labels)
    if np.isinf(candidate.tau) and method in B.SIGNED_METHODS:
        expected = np.isnan(signs).all() if method == 'hcgl_pooled_free' else (signs == 0).all()
        if not expected:
            raise AssertionError('Infinite threshold must fail all gates')
    fractions = np.asarray(fractions)
    if method == 'unilasso':
        return B.fit_path(prep, method, candidate.tau, prot, known_labels, fractions)
    beta = np.zeros((len(fractions), prep.y.shape[1], prep.x.shape[1]))
    intercept = np.zeros((len(fractions), prep.y.shape[1]))
    lambdas = np.zeros(len(fractions))
    positive = fractions > 0
    if positive.any():
        if candidate.alpha > 0 or method in ('lasso', 'cooperative'):
            b, a, lam, _ = B.fit_path(prep, method, candidate.tau, prot,
                                     known_labels, fractions[positive])
            beta[positive], intercept[positive], lambdas[positive] = b, a, lam
        else:
            groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
            variable = cp.Variable(beta.shape[1:])
            axis = 0 if method == 'fcgl_unconstrained' else 1
            penalty = sum(np.sqrt(len(g)/len(groups)) *
                          cp.sum(cp.norm(variable[g], 2, axis=axis)) for g in groups)
            constraints = []
            for sign in (0, 1, -1):
                mask = signs == sign
                if mask.any():
                    constraints.append(variable[mask] == 0 if sign == 0 else
                                       sign * variable[mask] >= 0)
            parameter = cp.Parameter(nonneg=True)
            problem = cp.Problem(cp.Minimize(cp.sum_squares(
                prep.y-prep.x@variable.T)/len(prep.x)+parameter*penalty), constraints)
            scale = max(pure_group_scale(prep, method, labels), 1e-12)
            for i in np.flatnonzero(positive):
                parameter.value = fractions[i]*scale
                if fractions[i] >= 1:
                    # The dual-norm bound proves exact zero is optimal, including
                    # with a sign cone. Avoid conic numerical dust at this kink.
                    lambdas[i] = parameter.value
                    continue
                B._solve(problem, prot.solver)
                beta[i], lambdas[i] = variable.value, parameter.value
    if (~positive).any():
        beta[~positive] = zero_penalty(prep, signs)
    if not np.isfinite(beta).all() or not np.isfinite(intercept).all():
        raise RuntimeError('Nonfinite endpoint path')
    return beta, intercept, lambdas, signs


def canonical_key(prep, method, candidate, known_labels=None):
    labels, signs, _ = B.method_inputs(prep, method, candidate.tau,
                                      protocol(candidate), known_labels)
    # At alpha1 the group penalty vanishes. Keep the gate itself in the key.
    grouping = b'' if candidate.alpha == 1 and method != 'cooperative' else labels.tobytes()
    return method, candidate.alpha, grouping, signs.tobytes()


def surface(x, y, method, seed, study, prior=None, known_labels=None):
    """No test sample/truth inputs; folds and scaling are inherited exactly."""
    choices = candidates(method, study)
    errors = np.zeros((len(choices), len(FRACTIONS)))
    missing = np.ones_like(errors, dtype=bool)
    if prior is not None:
        if prior.duplicated(list(KEYS)).any() or not np.isfinite(prior.cv_mse).all():
            raise ValueError('Duplicate or nonfinite prior cells')
        if not (prior.seed == seed).all() or not (prior.method == method).all():
            raise ValueError('Prior seed/method mismatch')
        lookup = {tuple(r[k] for k in KEYS): r.cv_mse for _, r in prior.iterrows()}
        for i, c in enumerate(choices):
            for j, f in enumerate(FRACTIONS):
                key = (c.cut, c.alpha, c.tau, f)
                if key in lookup:
                    errors[i, j], missing[i, j] = lookup[key], False
    computed_paths = 0
    for train, valid in B.cv_splits(len(x), 3, seed):
        preparations = {cut: B.prepare(x[train], y[train], cut)
                        for cut in {c.cut for c in choices}}
        cache = {}
        for i, c in enumerate(choices):
            js = np.flatnonzero(missing[i])
            if not len(js):
                continue
            prep = preparations[c.cut]
            key = canonical_key(prep, method, c, known_labels)
            needed = [j for j in js if (key, j) not in cache]
            if needed:
                b, a, _, _ = path(prep, method, c, [FRACTIONS[j] for j in needed], known_labels)
                prediction = np.einsum('ij,lkj->lik', prep.transform_x(x[valid]), b)
                prediction += a[:, None, :]
                loss = np.sum((prediction-prep.transform_y(y[valid])[None])**2,
                              axis=(1, 2))/y.shape[1]/len(x)
                if not np.isfinite(loss).all():
                    raise RuntimeError('Nonfinite validation loss')
                cache.update({(key, j): v for j, v in zip(needed, loss)})
                computed_paths += 1
            errors[i, js] += [cache[(key, j)] for j in js]
    records = [dict(seed=seed, method=method, **asdict(c), lambda_fraction=f,
                    cv_mse=errors[i, j], reused=not missing[i, j])
               for i, c in enumerate(choices) for j, f in enumerate(FRACTIONS)]
    return pd.DataFrame(records), computed_paths


def fit_one(job):
    study, design, seed, out, prior_root = job
    warnings.filterwarnings('error', category=RuntimeWarning)
    np.seterr(invalid='raise', divide='raise', over='raise')
    out = Path(out)
    prior_root = None if prior_root is None else Path(prior_root)
    prefix = out/'shards'/f'{design.name}_{seed}'
    start = time.perf_counter()
    simulation = L.simulate if study == 'learned' else B.simulate
    x, y, xt, yt, truth, groups = simulation(design, seed)
    prior_frames = None
    if study == 'learned' and prior_root is not None:
        original = pd.read_csv(prior_root/'main_40/shards'/f'{design.name}_{seed}_cv.csv')
        expanded = pd.read_csv(prior_root/'expanded_alpha1/shards'/f'{design.name}_{seed}_cv.csv')
        prior_frames = pd.concat([original[~original.method.isin(E.EXTENDED)], expanded])
        if not (prior_frames.scenario == design.name).all():
            raise ValueError('Scenario cache mismatch')
        receipt_file = prior_root/'main_40/shards'/f'{design.name}_{seed}_receipt.json'
        receipt = json.loads(receipt_file.read_text())
        hashes = {k: L.array_hash(v) for k, v in zip(
            ('x_train','y_train','x_test','y_test','beta_true','true_groups'),
            (x,y,xt,yt,truth,groups))}
        if hashes != receipt['inputs']:
            raise ValueError('Input cache mismatch')
    methods = L.METHODS if study == 'learned' else B.METHODS
    rows, surfaces, arrays, counts = [], [], {}, {}
    for method in methods:
        known = groups if method == 'hcgl_known_zero' else None
        prior = None if prior_frames is None else prior_frames[prior_frames.method == method]
        cv, count = surface(x, y, method, seed, study, prior, known)
        counts[method] = count
        choice = cv.iloc[int(np.argmin(cv.cv_mse))]
        candidate = L.Candidate(choice.cut, choice.alpha, choice.tau)
        prep = B.prepare(x, y, candidate.cut)
        betas, intercepts, lam, signs = path(prep, method, candidate,
                                           [choice.lambda_fraction], known)
        fitted = B.Fitted(betas[0], intercepts[0], prep, lam[0], choice.lambda_fraction,
                          candidate.tau, choice.cv_mse, time.perf_counter()-start, signs)
        for suffix, values in (('beta',betas[0]), ('intercept',intercepts[0]),
                               ('labels',prep.labels), ('signs',signs)):
            arrays[method+'__joint_cv__'+suffix] = values
        for tolerance in B.ZERO_TOLERANCES:
            rows.append(dict(scenario=design.name, seed=seed, method=method,
                             variant='joint_cv', zero_tolerance=tolerance,
                             **asdict(candidate), lambda_fraction=choice.lambda_fraction,
                             lambda_value=lam[0], cv_mse=choice.cv_mse,
                             lambda_upper_boundary=int(choice.lambda_fraction == 1),
                             lambda_lower_boundary=int(choice.lambda_fraction == 0),
                             **L.group_metrics(prep.labels, groups, truth),
                             **B.evaluate(fitted, xt, yt, truth, tolerance)))
        cv['scenario'] = design.name
        surfaces.append(cv)
        pd.DataFrame(rows).to_csv(str(prefix)+'_draws.csv', index=False)
    pd.concat(surfaces, ignore_index=True).to_csv(str(prefix)+'_cv.csv', index=False)
    np.savez_compressed(str(prefix)+'_coefficients.npz', **arrays)
    result = dict(scenario=design.name, seed=seed, seconds=time.perf_counter()-start,
                  computed_paths=counts, warnings=0)
    Path(str(prefix)+'_receipt.json').write_text(json.dumps(result, indent=2))
    return result


def checked_fit(job):
    try:
        return fit_one(job)
    except Exception:
        error = dict(study=job[0], scenario=job[1].name, seed=job[2], stack=traceback.format_exc())
        failure = Path(job[3])/f'failure_{job[1].name}_{job[2]}.json'
        failure.write_text(json.dumps(error, indent=2))
        raise


def run(out, prior, study, workers=6, reps=None, fresh_cv=False):
    if study == 'learned' and prior is None and not fresh_cv:
        raise ValueError('Supply --prior or explicitly choose --fresh-cv')
    out = B.safe_output(out)
    prior = None if fresh_cv or prior is None else Path(prior).resolve()
    if (out/'protocol.json').exists():
        raise FileExistsError('Use a fresh run directory')
    if study == 'learned' and prior is not None:
        E.validate_prior(prior/'main_40')
        expanded = json.loads((prior/'expanded_alpha1/receipt.json').read_text())
        if expanded['status'] != 'complete' or expanded['completed_replications'] != 200:
            raise ValueError('Require completed alpha1 study')
        for base in (prior/'main_40', prior/'expanded_alpha1'):
            receipt = json.loads((base/'receipt.json').read_text())
            for name, expected in receipt['outputs'].items():
                if name.endswith('_cv.csv') and digest(base/name) != expected:
                    raise ValueError(f'Mutated prior CV: {name}')
    designs = L.DESIGNS if study == 'learned' else B.SCENARIOS
    count = reps if reps is not None else (40 if study == 'learned' else 20)
    first_seed = 1000 if study == 'learned' else 0
    (out/'shards').mkdir()
    declaration = dict(study=study, prior=None if prior is None else str(prior),
                       fresh_cv=prior is None, designs=[asdict(d) for d in designs],
                       seeds=list(range(first_seed, first_seed+count)), workers=workers,
                       candidates={m:[asdict(c) for c in candidates(m,study)] for m in
                                   (L.METHODS if study == 'learned' else B.METHODS)},
                       lambda_fractions=FRACTIONS,
                       endpoint_note='Infinity is complete gate failure',
                       selection=('Minimum training-only inner-CV MSE; candidate-major order, '
                                  'then descending penalty fractions; first minimum on ties'),
                       zero='bounded least squares, no ridge substitution',
                       warning_policy='All RuntimeWarnings and NumPy invalid/divide/overflow raise')
    (out/'protocol.json').write_text(json.dumps(declaration, indent=2))
    sources = [Path(__file__), Path(B.__file__), Path(L.__file__), Path(E.__file__),
               Path(__file__).with_name('cvxpy_shape_compat.py')]
    for p in sources:
        (out/(p.stem+'_snapshot.py')).write_bytes(p.read_bytes())
    receipt = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                   source_hashes={str(p):digest(p) for p in sources}, completed=0,
                   cvxpy_shape_compatibility=compatibility_record())
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2))
    jobs = [(study,d,s,str(out),None if prior is None else str(prior)) for d in designs
            for s in range(first_seed, first_seed+count)]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(checked_fit,j) for j in jobs]
            for f in as_completed(futures):
                r=f.result()
                receipt['completed'] += 1
                print(f"{receipt['completed']}/{len(jobs)} {r['scenario']} {r['seed']} "
                      f"{r['seconds']:.1f}s", flush=True)
                (out/'receipt.json').write_text(json.dumps(receipt, indent=2))
        draws = pd.concat([pd.read_csv(p) for p in (out/'shards').glob('*_draws.csv')])
        draws.to_csv(out/'estimator_draws.csv', index=False)
        keys = ['scenario','method','variant','zero_tolerance']
        reported = list(E.METRICS) + ['null_selection','selected_cells',
                                    'discovered_clusters','adjusted_rand']
        L.summarize(draws[keys + reported], keys).to_csv(
            out/'estimator_summary.csv',index=False)
        receipt.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        receipt.update(status='failed',error=repr(error))
        raise
    finally:
        receipt['outputs'] = {str(p.relative_to(out)):digest(p) for p in out.rglob('*')
                              if p.is_file() and p.name != 'receipt.json'}
        (out/'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--fresh-cv',action='store_true',
                        help='Recompute all training CV cells; read no prior results')
    parser.add_argument('--study',choices=('learned','original'),default='learned')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--reps',type=int)
    args=parser.parse_args()
    if args.fresh_cv and args.prior is not None:
        parser.error('--fresh-cv and --prior are mutually exclusive')
    run(args.out,args.prior,args.study,args.workers,args.reps,args.fresh_cv)
