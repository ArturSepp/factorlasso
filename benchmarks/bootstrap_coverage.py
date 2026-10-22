"""Offline coverage audit of full FCGL bootstrap intervals, never an automatic CI certificate.

Synthetic data only. Run with --output outside the checkout. Each experiment is
an independent panel; binomial intervals are computed per factor, not by treating
correlated coefficients as independent observations. The declared failure rule is
that a Wilson 95% upper bound below .95 rejects nominal 95% coverage for that cell.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from factorlasso import LassoModel, LassoModelType, bootstrap_factor_betas


def synthetic_panel(case, seed):
    """Known positive, zero and weak slopes; dependent heteroskedastic joint errors."""
    rng = np.random.default_rng(seed)
    short = case == 'short_ragged_collinear'
    t, p, n = (57, 13, 7) if short else (144, 3, 4)
    common = rng.normal(size=(t, 1))
    rho = .85 if short else .3
    raw = np.sqrt(rho)*common + np.sqrt(1-rho)*rng.normal(size=(t, p))
    scale = np.array([.035, .015, .009, .010, .013, .015, .014,
                      .014, .050, .020, .025, .020, .070]) if short else np.full(p, .035)
    x = raw*scale
    truth = np.zeros((n, p))
    if short:
        truth[:6, 4] = [.3, .5, .6, .7, .8, .9]
        truth[:6, 0] = [.0, .08, .1, .1, .15, .2]
        truth[0, 5] = .4
        truth[-1, -1] = .8
        truth[-1, 2] = .2
    else:
        truth[:, 0] = [.8, .7, .6, .5]
        truth[:2, 1] = [.2, .3]
        if case == 'weak_selection_boundary':
            truth[:, 0] = [.04, .03, .02, .01]
            truth[:2, 1] = [.02, .03]
    noise = (.6*rng.normal(size=(t, 1)) + .8*rng.normal(size=(t, n)))
    for row in range(1, t):
        noise[row] += .35*noise[row-1]
    noise *= .009*(.65+.35*np.abs(common))
    y = x @ truth.T + noise
    if short:
        for asset in range(n-1):
            y[:t-(23 if asset == 3 else 19), asset] = np.nan
    x = pd.DataFrame(x, columns=[f'f{i}' for i in range(p)])
    y = pd.DataFrame(y, columns=[f'a{i}' for i in range(n)])
    signs = pd.DataFrame(np.nan, index=y.columns, columns=x.columns)
    if short:
        signs.iloc[:, :7] = 1.
        signs.iloc[:, 9] = 0.
        signs.iloc[:, 10] = -1.
    model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-5, span=36, cutoff_fraction=.6, warmup_period=12,
        apply_ols_prior=True, prior_selection_type='highest_r2',
        auto_sign_constraints=True, auto_sign_adaptive_weights=True,
        auto_sign_threshold_t=1., auto_sign_adaptive_floor=.5,
        factors_beta_loading_signs=signs, solver='CLARABEL', loss_normalization='weight_sum')
    w = np.array([.263, .053, .010, .068, .145, .152, .350]) if short else np.full(n, 1/n)
    return model, x, y, truth, pd.Series(w, index=y.columns)


def experiment(task):
    """One independently simulated panel and its complete bootstrap refits."""
    case, run, draws, seed = task
    model, x, y, truth, weights = synthetic_panel(case, seed+run)
    result = bootstrap_factor_betas(model, x, y, draws=draws, seed=seed+100000+run,
                                    bandwidth=3.)
    rows = []
    for method in ['percentile', 'basic']:
        intervals = result.portfolio_intervals(weights, method=method)
        true_portfolio = weights.to_numpy() @ truth
        for f, label in enumerate(x.columns):
            if case == 'short_ragged_collinear' and f == 9:
                continue  # Fixed by specification; not an estimated confidence interval.
            row = intervals.loc[label]
            rows.append(dict(case=case, run=run, method=method, factor=label,
                truth=true_portfolio[f], estimate=row.estimate,
                lower=row.lower, upper=row.upper,
                covered=bool(row.lower-1e-6 <= true_portfolio[f] <= row.upper+1e-6)))
    return rows


def main():
    """Write the simulation specification, individual trials and per-factor coverage audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=int, default=100)
    parser.add_argument('--draws', type=int, default=199)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = ['regular_signal', 'weak_selection_boundary', 'short_ragged_collinear']
    tasks = [(case, run, args.draws, 914810+i*10000)
             for i, case in enumerate(cases) for run in range(args.experiments)]
    started, rows = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, batch in enumerate(pool.map(experiment, tasks, chunksize=1), 1):
            rows.extend(batch)
            if done % 10 == 0:
                print(f'Coverage panels {done}/{len(tasks)}; {time.monotonic()-started:.0f}s',
                      flush=True)
    trials = pd.DataFrame(rows)
    trials.to_csv(args.output/'trials.csv', index=False)
    grouped = trials.groupby(['case', 'method', 'factor'])
    coverage = grouped.agg(coverage=('covered', 'mean'), experiments=('covered', 'size'),
                           truth=('truth', 'first'), mean_estimate=('estimate', 'mean'))
    z = norm.ppf(.975)
    p, n = coverage.coverage, coverage.experiments
    centre = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    coverage['mc_lower'] = centre-radius
    coverage['mc_upper'] = centre+radius
    coverage['reject_95_coverage'] = coverage.mc_upper < .95
    coverage.to_csv(args.output/'coverage.csv')
    (args.output/'validation.json').write_text(json.dumps(dict(
        experiments=args.experiments, bootstrap_draws=args.draws, cases=cases,
        bandwidth=3, probability=.95, confidence_gate='Wilson 95% upper limit below .95',
        rejected_cells=int(coverage.reject_95_coverage.sum()),
        total_cells=len(coverage),
        validated_95_confidence=False,
        conclusion='No universal FCGL coverage claim; inspect per-design, per-factor evidence',
        elapsed_seconds=time.monotonic()-started), indent=2))
    print(coverage.to_string(), flush=True)


if __name__ == '__main__':
    main()
