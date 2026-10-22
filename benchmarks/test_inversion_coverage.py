"""Independent coverage of FCGL conditional tests at known true portfolio loadings.

Only synthetic data. Each outer panel re-estimates the working covariance and
refits full FCGL. Optional --design points to caller-supplied x/y CSVs for a
fixed-design simulation; no private data is distributed by FactorLasso.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from bootstrap_coverage import synthetic_panel
from factorlasso import FactorTestInversion, LassoModel, LassoModelType
import factorlasso


def design_panel(path, seed, noise):
    """Caller-supplied fixed design; simulated responses only, never simulated factors."""
    source = Path(path)
    x = pd.read_csv(source/'x.csv', index_col=0, parse_dates=True)
    original = pd.read_csv(source/'y.csv', index_col=0, parse_dates=True)
    truth = pd.read_csv(source/'betas.csv', index_col=0).to_numpy()
    parameters = json.loads((source/'parameters.json').read_text())
    parameters['model_type'] = LassoModelType.FACTOR_CLUSTER_GROUP_LASSO
    parameters['factors_beta_loading_signs'] = pd.read_csv(source/'signs.csv', index_col=0)
    model = LassoModel(**parameters)
    weights = pd.read_csv(source/'weights.csv', index_col=0).iloc[:, 0]
    scales = pd.read_csv(source/'scales.csv', index_col=0).iloc[:, 0].to_numpy()
    rng = np.random.default_rng(seed)
    # Vary true loadings, including weak alternatives that the observed fit set to zero.
    truth *= rng.uniform(.5, 1.5, size=truth.shape)
    weak = rng.uniform(0., .15, size=(len(truth), 7))
    truth[:, :7] += weak*(rng.uniform(size=weak.shape) < .3)
    truth[:, 9] = 0.
    t, n = original.shape
    if noise == 'high_correlation':
        z = np.sqrt(.99)*rng.normal(size=(t, 1))+.1*rng.normal(size=(t, n))
    else:
        z = .6*rng.normal(size=(t, 1))+.8*rng.normal(size=(t, n))
    if noise == 'tails':
        z *= np.sqrt(3/rng.chisquare(5, size=(t, 1)))
    rho = .85 if noise == 'boundary_persistence' else (.6 if noise == 'persistent' else .35)
    for i in range(1, t):
        z[i] += rho*z[i-1]
    if noise in ('heteroskedastic', 'tails'):
        amplitude = .65+.35*np.abs(x.iloc[:, 0].fillna(0).to_numpy()/x.iloc[:, 0].std())
        z *= amplitude[:, None]
    y = pd.DataFrame(x.fillna(0).to_numpy() @ truth.T+z*scales*rng.uniform(.7, 1.4),
                     index=original.index, columns=original.columns).where(original.notna())
    return model, x, y, truth, weights


def experiment(task):
    case, run, seed_base, draws, solver, design, early_accept = task
    seed = seed_base+run
    base, factor = case.rsplit(':', 1)
    if base.startswith('design_'):
        model, x, y, truth, weights = design_panel(design, seed, base.removeprefix('design_'))
    else:
        model, x, y, truth, weights = synthetic_panel(base, seed)
    model.solver = solver
    contrast = pd.DataFrame(0., index=y.columns, columns=x.columns)
    factor_number = int(factor[1:])
    contrast.iloc[:, factor_number] = weights.to_numpy()
    target = float(weights.to_numpy() @ truth[:, factor_number])
    engine = FactorTestInversion(model, x, y, draws=draws, seed=seed+3000000)
    test = engine.test(contrast, target, accept_above=early_accept)
    test.update(case=case, run=run, seed=seed, truth=target, draws=draws,
                rho=engine.noise_metadata.get('rho'), residual_df=engine.df)
    return test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=199)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed-base', type=int, default=5710000)
    parser.add_argument('--solver', default='CLARABEL')
    parser.add_argument('--design', type=Path)
    parser.add_argument('--early-accept', type=float, default=.01)
    parser.add_argument('--cases', nargs='+', default=[
        'regular_signal:f0', 'weak_selection_boundary:f0', 'short_ragged_collinear:f4'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    package = Path(factorlasso.__file__).parent
    dependencies = {}
    for name in ['numpy', 'pandas', 'scipy', 'cvxpy', 'mosek']:
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    provenance = dict(
        pipeline_sha256={p.relative_to(package).as_posix(): sha256(p.read_bytes()).hexdigest()
                         for p in sorted(package.rglob('*.py'))},
        dependencies=dependencies,
        design_sha256={} if args.design is None else {
            p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(args.design.iterdir())
            if p.is_file()})
    tasks = [(case, run, args.seed_base+i*100000, args.draws, args.solver,
              args.design, args.early_accept)
             for i, case in enumerate(args.cases) for run in range(args.experiments)]
    rows, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(experiment, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 50 == 0 or i == len(tasks):
                df = pd.DataFrame(rows)
                df.to_csv(args.output/'trials.csv', index=False)
                print(f'Panels {i}/{len(tasks)}; {time.monotonic()-started:.0f}s; '
                      f'refits {df.draws_completed.sum()}', flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby('case').pvalue.agg(['count', 'mean', 'min'])
    thresholds = sorted({args.early_accept, 1/(args.draws+1), .05, .025, .01, .005})
    for alpha in thresholds:
        if alpha > args.early_accept+1e-12 or alpha < 1/(args.draws+1)-1e-12:
            continue  # An early-acceptance bound cannot classify a larger threshold.
        summary[f'coverage_at_{alpha}'] = (
            df.assign(ok=df.pvalue > alpha+1e-12).groupby('case').ok.mean())
    summary.to_csv(args.output/'coverage.csv')
    (args.output/'specification.json').write_text(json.dumps(
        {**vars(args), 'output': str(args.output), 'design': str(args.design),
         'elapsed': time.monotonic()-started, **provenance}, indent=2))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
