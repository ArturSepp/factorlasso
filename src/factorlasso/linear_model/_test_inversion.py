"""Conditional Monte Carlo test inversion of the full production estimator.

For Gaussian errors with covariance sigma**2 C and *fixed known* C, conditioning
on the null-model projection and residual length eliminates all other linear
coefficients and sigma. Uniform directions in the null residual space give
exchangeable Monte Carlo tests for any deterministic statistic, including FCGL.
Estimated C requires independent simulation calibration; it has no exact claim.

References
----------
Dufour (2006), Monte Carlo tests with nuisance parameters: A general approach
to finite-sample inference and nonstandard asymptotics, Journal of Econometrics
133, 443-477, doi:10.1016/j.jeconom.2005.06.007. Numerical inversion on a finite
grid is an approximation; disconnected accepted regions are enclosed by a hull.
Patterson and Thompson (1971), Recovery of inter-block information when block
sizes are unequal, Biometrika 58, 545-554, doi:10.1093/biomet/58.3.545.
The working AR1 profile uses restricted likelihood with plug-in marginal scales
and pairwise correlations, not joint REML estimation or an exact covariance law.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, qr, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_distribution

from factorlasso.linear_model._estimator import LassoModel


@dataclass(frozen=True)
class TestInversionCalibration:
    """Independent simulation calibration, conditional on its declared design family.

    Attributes
    ----------
    confidence : float
        Requested marginal coverage.
    rejection_probability : float
        Monte Carlo test rejection threshold selected on calibration data only.
    coverage : pandas.DataFrame
        Per-design validation coverage and simultaneous one-sided binomial bounds.
    passed : bool
        Whether every validation lower bound clears the requested coverage.
    metadata : dict
        Trial counts, seed separation and scope. Never a distribution-free certificate.
    """

    confidence: float
    rejection_probability: float
    coverage: pd.DataFrame
    passed: bool
    metadata: dict


def calibrate_inversion_tests(calibration: pd.DataFrame, validation: pd.DataFrame,
                              confidence: float = .95, minimum_validation: int = 1000
                              ) -> TestInversionCalibration:
    """Select a test threshold on training simulations and audit independent coverage.

    Parameters
    ----------
    calibration, validation : pandas.DataFrame
        Columns ``case``, ``seed``, ``draws`` and ``pvalue`` from tests of true contrasts.
        Seeds must be unique within each case and disjoint across the two datasets.
    confidence : float, default 0.95
        Target pointwise coverage; this is not simultaneous coverage across factors.
    minimum_validation : int, default 1000
        Minimum independent outer panels in every validation design.

    Returns
    -------
    TestInversionCalibration
        Passed only if all one-sided Clopper-Pearson lower bounds, Bonferroni
        adjusted across validation designs, exceed the requested coverage.
    """
    if not 0 < confidence < 1 or minimum_validation < 20:
        raise ValueError('Invalid confidence or minimum_validation')
    for table in (calibration, validation):
        if (not {'case', 'seed', 'draws', 'pvalue'} <= set(table) or table.empty
                or table[['case', 'seed', 'draws', 'pvalue']].isna().any().any()
                or table.duplicated(['case', 'seed']).any()
                or not table.pvalue.between(0, 1).all()):
            raise ValueError('Require unique case/seed and finite pvalues in [0, 1]')
    if set(calibration.seed) & set(validation.seed):
        raise ValueError('Calibration and validation seeds must be disjoint')
    if set(calibration.case) != set(validation.case):
        raise ValueError('Calibration and validation must cover the same designs')
    if validation.groupby('case').size().min() < minimum_validation:
        raise ValueError('Insufficient independent validation panels')
    draws = pd.concat([calibration.draws, validation.draws]).unique()
    if len(draws) != 1 or draws[0] < 19 or draws[0] != int(draws[0]):
        raise ValueError('Calibration and validation require the same Monte Carlo draw count')
    # Predeclared conservative training target, never retuned against validation.
    target = 1-(1-confidence)/2
    maximum = 1-confidence
    for table in (calibration, validation):
        if 'accept_above' in table:
            maximum = min(maximum, table.accept_above.dropna().min())
        elif 'pvalue_exact' in table and not table.pvalue_exact.all():
            raise ValueError('Bounded pvalues require their early acceptance threshold')
    candidates = np.arange(int(np.floor(maximum*(draws[0]+1)+1e-9)), 0, -1)/(draws[0]+1)
    selected = [a for a in candidates if a+1e-12 >= 1/(draws[0]+1) and
                calibration.assign(covered=calibration.pvalue > a+1e-12)
                .groupby('case').covered.mean().min() >= target]
    if not selected:
        raise ValueError('No nonzero test threshold passes calibration; inference unavailable')
    alpha = float(max(selected))
    result = validation.assign(covered=validation.pvalue > alpha+1e-12).groupby('case').agg(
        covered=('covered', 'sum'), experiments=('covered', 'size'))
    result['coverage'] = result.covered/result.experiments
    tail = .05/len(result)
    result['lower_bound'] = [0. if k == 0 else beta_distribution.ppf(tail, k, n-k+1)
                             for k, n in zip(result.covered, result.experiments)]
    result['passed'] = result.lower_bound >= confidence
    return TestInversionCalibration(confidence, alpha, result, bool(result.passed.all()),
        dict(scope='Independent synthetic design family only; estimated covariance',
             calibration_panels=len(calibration), validation_panels=len(validation),
             training_target=target, validation_error_budget=.05,
             minimum_validation=minimum_validation, disjoint_seeds=True, draws=int(draws[0])))


@dataclass(frozen=True)
class FactorTestInversionResult:
    """Production points and numerically inverted Monte Carlo confidence sets.

    Attributes
    ----------
    intervals : pandas.DataFrame
        Contrast-indexed estimate, lower, upper and boundary/search diagnostics.
        Unbounded tails are infinity, never silently clipped to the search range.
    tests : pandas.DataFrame
        Every evaluated candidate and its Monte Carlo p-value.
    metadata : dict
        Conditional model, calibration, numerical resolution and refit counts.
    """

    intervals: pd.DataFrame
    tests: pd.DataFrame
    metadata: dict


def _design(x, y, model):
    valid_x = x.notna().all(axis=1)
    if ((x.notna().any(axis=1) & ~valid_x).any() or y.loc[~valid_x].notna().any().any()):
        raise ValueError('Factor rows must be complete or wholly missing with missing responses')
    valid = y.notna().to_numpy()
    rows, assets = np.where(valid)
    columns, parts = [], []
    signs = model.factors_beta_loading_signs
    if signs is not None:
        signs = signs.reindex(index=y.columns, columns=x.columns)
    for a, asset in enumerate(y.columns):
        mask = assets == a
        if mask.sum() < model.warmup_period:
            raise ValueError('Insufficient response history')
        columns.append((asset, None))
        parts.append(mask.astype(float))
        for f, factor in enumerate(x.columns):
            if signs is not None and signs.loc[asset, factor] == 0:
                continue  # Genuine external zero restrictions only; no fitted selection.
            columns.append((asset, factor))
            parts.append(np.where(mask, x.iloc[rows, f].to_numpy(), 0.))
    return np.column_stack(parts), columns, rows, assets


def _working_covariance(x, y, design, columns, rows, assets, calendar):
    """Common-window correlations and feasible REML AR1; repeated in every audit panel.

    Correlating residuals projected on different histories attenuates shared noise.
    Every pair is instead projected on its common dates and union nuisance design.
    Correlation-noise shrinkage is capped at 50%, but tends to zero for a reliably
    common residual component. It is a working regularisation rule, not exact inference.
    """
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Unidentified nuisance design')
    response = y.to_numpy()[rows, assets]
    residual = response-design @ np.linalg.lstsq(design, response, rcond=None)[0]
    scales = []
    for a in range(y.shape[1]):
        idx = assets == a
        df = idx.sum()-np.linalg.matrix_rank(design[idx])
        if df < 3:
            raise ValueError('At least three unpenalized residual degrees of freedom per asset')
        variance = residual[idx] @ residual[idx]/df
        if not np.isfinite(variance) or variance <= 1e-20:
            raise ValueError('Degenerate residual scale; provide an explicit covariance model')
        scales.append(float(np.sqrt(variance)))
    corr = np.eye(y.shape[1])
    variance, signal, overlaps = 0., 0., []
    for i in range(y.shape[1]):
        for j in range(i):
            mask = y.iloc[:, [i, j]].notna().all(axis=1)
            names = [name for name in x.columns if any(
                factor == name and asset in [y.columns[i], y.columns[j]]
                for asset, factor in columns)]
            d = np.column_stack([np.ones(mask.sum()), x.loc[mask, names].to_numpy()])
            yy = y.loc[mask].iloc[:, [i, j]].to_numpy()
            df = len(d)-np.linalg.matrix_rank(d) if len(d) else 0
            if df < 3:
                raise ValueError('At least three common-window residual degrees of freedom '
                                 'per asset pair; otherwise supply a covariance model')
            e = yy-d @ np.linalg.lstsq(d, yy, rcond=None)[0]
            r = float(np.clip(np.corrcoef(e.T)[0, 1], -1., 1.))
            if not np.isfinite(r):
                raise ValueError('Degenerate common-window residual correlation')
            corr[i, j] = corr[j, i] = r
            variance += 2*(1-r*r)**2/max(df-1, 1)
            signal += 2*r*r
            overlaps.append(int(df))
    shrinkage = float(min(.5, variance/max(signal, 1e-12)))
    corr = (1-shrinkage)*corr+shrinkage*np.eye(len(scales))
    ev, vec = np.linalg.eigh(corr)
    corr = (vec*np.maximum(ev, .001)) @ vec.T
    corr /= np.sqrt(np.outer(np.diag(corr), np.diag(corr)))
    scale = np.asarray(scales)[assets]
    cross = corr[assets[:, None], assets[None, :]]*np.outer(scale, scale)
    gaps = np.abs(calendar[rows, None]-calendar[None, rows])
    df = len(rows)-design.shape[1]

    def objective(rho):
        chol = cholesky(cross*rho**gaps, lower=True)
        a = solve_triangular(chol, design, lower=True)
        z = solve_triangular(chol, response, lower=True)
        q, r = qr(a, mode='economic')
        e = z-q @ (q.T @ z)
        return float(2*np.log(np.diag(chol)).sum()+2*np.log(np.abs(np.diag(r))).sum()
                     +df*np.log((e @ e)/df))

    optimum = minimize_scalar(objective, bounds=(0., .85), method='bounded',
                              options={'xatol': 1e-4, 'maxiter': 40})
    if not optimum.success:
        raise RuntimeError('Working covariance restricted-likelihood profile failed')
    rho = min([0., float(optimum.x), .85], key=objective)
    return cross*rho**gaps, dict(
        noise='common-window correlation with feasible REML AR1', rho=rho,
        asset_scales=scales, correlation=corr.tolist(), correlation_shrinkage=shrinkage,
        correlation_eigenvalue_floor=.001, common_residual_df=overlaps,
        reml_objective=objective(rho), rho_bounds=[0., .85])


class FactorTestInversion:
    """Conditional null simulation using the full production fit as test statistic.

    Parameters
    ----------
    model : LassoModel
        Production configuration. Fresh copies refit every data-dependent stage;
        the caller's estimator is never mutated.
    x, y : pandas.DataFrame
        Aligned native-frequency factors and responses, with original missing dates.
    covariance : numpy.ndarray, optional
        Positive-definite error covariance shape on observed cells in row-major
        (date, asset) order. A common unknown variance multiplier is conditioned out.
        Must be fixed independently of y for the exact Gaussian interpretation.
        None estimates an AR1/cross-asset shape and requires simulation calibration.
    draws : int, default 199
        Independent conditional directions; minimum achievable p-value is 1/(draws+1).
    seed : int, default 0
        Random seed. Common directions are reused across candidate values.
    calendar : array-like, optional
        Integer native-cadence dates, used for the working AR1 shape. Default row dates.

    Notes
    -----
    Conditioning on the null sufficient statistics removes all nuisance regression
    coefficients and the common variance, not covariance-shape uncertainty. No
    large-sample normal approximation or FCGL bias correction is used. The fixed
    design must identify all non-fixed slopes and intercepts. Numerical inversion
    returns a grid hull with explicit resolution, not an exact continuous optimiser.
    """

    def __init__(self, model: LassoModel, x: pd.DataFrame, y: pd.DataFrame, *,
                 covariance: np.ndarray | None = None, draws: int = 199, seed: int = 0,
                 calendar=None):
        if not isinstance(model, LassoModel):
            raise TypeError('model must be a LassoModel')
        if (isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 19
                or not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0):
            raise ValueError('Require draws >= 19 and a nonnegative integer seed')
        for frame in (x, y):
            if (not isinstance(frame, pd.DataFrame) or frame.empty
                    or not frame.index.is_unique or not frame.index.is_monotonic_increasing
                    or not frame.columns.is_unique or np.isinf(frame.to_numpy()).any()):
                raise ValueError('Require finite-or-missing frames with unique ordered labels')
        if not x.index.equals(y.index):
            raise ValueError('x and y indexes must match')
        self.model = deepcopy(model.get_params())
        self.x, self.y = x.copy(), y.copy()
        self.draws, self.seed = int(draws), int(seed)
        d, self.columns, self.rows, self.assets = _design(x, y, model)
        self.observed_index = pd.MultiIndex.from_arrays(
            [y.index[self.rows], y.columns[self.assets]], names=['date', 'asset'])
        calendar = np.arange(len(x)) if calendar is None else np.asarray(calendar)
        if (calendar.shape != (len(x),) or not np.isfinite(calendar).all()
                or (np.diff(calendar) <= 0).any()
                or not np.allclose(calendar, np.round(calendar))):
            raise ValueError('calendar requires strictly increasing integer native dates')
        self.known_shape = covariance is not None
        self.noise_metadata = {'noise': 'caller-supplied fixed covariance shape'}
        if covariance is None:
            covariance, self.noise_metadata = _working_covariance(
                x, y, d, self.columns, self.rows, self.assets, calendar)
        c = np.asarray(covariance, dtype=float)
        if (c.shape != (len(d), len(d)) or not np.isfinite(c).all()
                or not np.allclose(c, c.T, rtol=1e-10, atol=1e-14)):
            raise ValueError('covariance must be finite symmetric on observed cells')
        self.chol = cholesky(c, lower=True)
        self.design = solve_triangular(self.chol, d, lower=True)
        self.response = solve_triangular(self.chol, y.to_numpy()[self.rows, self.assets],
                                         lower=True)
        u, s, vh = np.linalg.svd(self.design, full_matrices=False)
        if len(s) != d.shape[1] or s[-1] <= s[0]*1e-10 or len(d)-len(s) < 3:
            raise ValueError('Unidentified nuisance design or inadequate '
                             'residual degrees of freedom')
        self.u, self.s, self.vh = u, s, vh
        self.fit_mean = u @ (u.T @ self.response)
        self.residual = self.response-self.fit_mean
        self.df = len(d)-len(s)
        self.rss = float(self.residual @ self.residual)
        if self.rss <= 1e-20:
            raise ValueError('Degenerate residual sphere')
        self.gamma = vh.T @ ((u.T @ self.response)/s)
        self.random = np.random.default_rng(seed).normal(size=(draws, len(d)))
        self.random_residual = self.random-(self.random @ u) @ u.T
        self.estimate = LassoModel(**deepcopy(self.model)).fit(x, y).coef_.copy()
        self.refits = 0
        self._cache = {}
        self._fixed_design_cache = {}

    def _geometry(self, contrast):
        if (not isinstance(contrast, pd.DataFrame)
                or not contrast.index.equals(self.y.columns)
                or not contrast.columns.equals(self.x.columns)
                or not np.isfinite(contrast.to_numpy()).all()):
            raise ValueError('contrast must be a finite asset-by-factor frame in exact label order')
        c = np.array([0. if factor is None else contrast.loc[asset, factor]
                      for asset, factor in self.columns])
        if not np.any(c):
            raise ValueError('Contrast is fixed by specification or identically zero')
        z = (self.vh @ c)/self.s
        scale = float(np.linalg.norm(z))
        direction = self.u @ (z/scale)
        nuisance = self.fit_mean-direction*(direction @ self.response)
        directions = self.random_residual + np.outer(self.random @ direction, direction)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        point = float(np.sum(contrast.to_numpy()*self.estimate.to_numpy()))
        return c, scale, direction, nuisance, directions, point

    def test(self, contrast: pd.DataFrame, candidate: float, *, accept_above: float | None = None
             ) -> dict:
        """Evaluate a conservative Monte Carlo rank test of a scalar production contrast.

        Parameters
        ----------
        contrast : pandas.DataFrame
            Signed asset/factor coefficients; no normalisation is imposed.
        candidate : float
            Hypothesised underlying economic loading in contrast units.
        accept_above : float, optional
            Stop once the lower bound on the fixed-draw Monte Carlo p-value exceeds
            this threshold. This certifies acceptance, never early rejection. Returned
            pvalue is then a lower bound; remaining simulations cannot alter that decision.

        Returns
        -------
        dict
            Production estimate, candidate, pvalue, simulation count and known-shape flag.
            Ties count against rejection; failed fits raise rather than being discarded.
        """
        if not np.isfinite(candidate) or (accept_above is not None and not 0 < accept_above < 1):
            raise ValueError('candidate must be finite')
        c, scale, direction, nuisance, directions, point = self._geometry(contrast)
        key = (contrast.to_numpy().tobytes(), float(candidate), accept_above)
        if key in self._cache:
            return self._cache[key].copy()
        observed = abs(point-candidate)
        if observed <= 1e-14:
            result = dict(estimate=point, candidate=float(candidate), pvalue=1., draws=0,
                          pvalue_upper=1., pvalue_exact=True, draws_completed=0,
                          accept_above=accept_above, known_covariance_shape=self.known_shape)
        else:
            radius = np.sqrt(self.rss+((c @ self.gamma-candidate)/scale)**2)
            center = nuisance+direction*(candidate/scale)
            samples = (directions*radius+center) @ self.chol.T
            exceed = 0
            for b in range(self.draws):
                response_values = self.y.to_numpy(copy=True)
                response_values[self.rows, self.assets] = samples[b]
                response = pd.DataFrame(response_values, index=self.y.index, columns=self.y.columns)
                try:
                    fit = LassoModel(**deepcopy(self.model))
                    # Memoise only factor regressions with their exact factor/mask
                    # keys; all response-dependent selections and bounds are refitted.
                    fit._inference_fixed_design_cache = self._fixed_design_cache
                    fit.fit(self.x, response)
                except Exception as error:
                    raise RuntimeError(
                        f'Conditional FCGL refit {b} failed; no draws dropped') from error
                simulated = float(np.sum(contrast.to_numpy()*fit.coef_.to_numpy()))
                if not np.isfinite(simulated):
                    raise RuntimeError('Nonfinite conditional FCGL statistic')
                exceed += abs(simulated-candidate) >= observed-1e-10*max(1., observed)
                self.refits += 1
                if accept_above is not None and (1+exceed)/(self.draws+1) > accept_above+1e-12:
                    break
            completed = b+1
            result = dict(estimate=point, candidate=float(candidate),
                          pvalue=(1+exceed)/(self.draws+1), draws=self.draws,
                          pvalue_upper=(1+exceed+self.draws-completed)/(self.draws+1),
                          pvalue_exact=completed == self.draws, draws_completed=completed,
                          accept_above=accept_above, known_covariance_shape=self.known_shape)
        self._cache[key] = result
        return result.copy()

    def intervals(self, contrasts: Mapping[str, pd.DataFrame], *, confidence: float = .95,
                  calibration: TestInversionCalibration | None = None, grid_size: int = 13,
                  refinements: int = 6, max_expansions: int = 8, progress=None
                  ) -> FactorTestInversionResult:
        """Numerically invert tests, retaining an outer hull of every accepted grid region.

        Parameters
        ----------
        contrasts : mapping of str to pandas.DataFrame
            Named portfolio or asset contrasts in the estimator's exact label order.
        confidence : float, default 0.95
            Requested marginal confidence level.
        calibration : TestInversionCalibration, optional
            Passing independent calibration is mandatory for estimated noise shape.
        grid_size : int, default 13
            Initial grid over the adaptively expanded search range; at least five.
        refinements : int, default 6
            Bisections at both outer acceptance boundaries.
        max_expansions : int, default 8
            Doublings per side before returning an explicitly unbounded interval.
        progress : callable, optional
            Called with contrast name, candidate and pvalue after each candidate test.

        Returns
        -------
        FactorTestInversionResult
            Production points, conservative bracket endpoints, all tested candidates
            and explicit numerical-search and conditional-validity metadata.
        """
        if (not contrasts or not 0 < confidence < 1 or grid_size < 5
                or refinements < 0 or max_expansions < 1):
            raise ValueError('Invalid inversion controls')
        if calibration is None:
            if not self.known_shape:
                raise ValueError('Estimated covariance requires passing independent calibration')
            alpha = 1-confidence
        else:
            if not calibration.passed or not np.isclose(calibration.confidence, confidence):
                raise ValueError('Calibration did not pass or has different confidence')
            if calibration.metadata.get('draws') != self.draws:
                raise ValueError('Draw count differs from the independently validated procedure')
            alpha = calibration.rejection_probability
        if 1/(self.draws+1) > alpha+1e-12:
            raise ValueError('Too few draws to resolve the calibrated rejection probability')
        audit, summaries = [], []
        for name, contrast in contrasts.items():
            c, scale, _, _, _, point = self._geometry(contrast)
            width = max(3*scale*np.sqrt(self.rss/self.df), .1*abs(point), 1e-4)
            evaluated = {}

            def evaluate(value):
                value = float(value)
                if value not in evaluated:
                    test = self.test(contrast, value, accept_above=alpha)
                    test.update(contrast=name, accepted=test['pvalue'] > alpha+1e-12)
                    evaluated[value] = test
                    audit.append(test)
                    if progress is not None:
                        progress(name, value, test['pvalue'])
                return evaluated[value]['accepted']

            evaluate(point)
            edges, open_tails = [], []
            for sign in [-1, 1]:
                edge = point+sign*width
                rejected = 0
                for _ in range(max_expansions):
                    rejected = 0 if evaluate(edge) else rejected+1
                    if rejected >= 2:
                        break
                    edge = point+2*(edge-point)
                edges.append(edge)
                open_tails.append(rejected < 2)
            for value in np.linspace(*edges, grid_size):
                evaluate(value)
            accepted = sorted(v for v, test in evaluated.items() if test['accepted'])
            bounds, resolutions = [], []
            for side in [0, 1]:
                if open_tails[side]:
                    bounds.append(-np.inf if side == 0 else np.inf)
                    resolutions.append(np.inf)
                    continue
                inside = accepted[0 if side == 0 else -1]
                outside_values = [v for v, test in evaluated.items() if not test['accepted']
                                  and (v < inside if side == 0 else v > inside)]
                outside = max(outside_values) if side == 0 else min(outside_values)
                for _ in range(refinements):
                    mid = (inside+outside)/2
                    if evaluate(mid):
                        inside = mid
                    else:
                        outside = mid
                # Use the rejected side of the final bracket, never truncate acceptance.
                bounds.append(outside)
                resolutions.append(abs(inside-outside))
            ordered = [t['accepted'] for _, t in sorted(evaluated.items())]
            regions = sum(a and (i == 0 or not ordered[i-1]) for i, a in enumerate(ordered))
            summaries.append(dict(contrast=name, estimate=point, lower=bounds[0], upper=bounds[1],
                confidence=confidence, rejection_probability=alpha,
                lower_resolution=resolutions[0], upper_resolution=resolutions[1],
                acceptance_regions=regions, candidates=len(evaluated),
                status='Unbounded' if any(open_tails) else 'Calibrated conditional'))
        return FactorTestInversionResult(pd.DataFrame(summaries).set_index('contrast'),
            pd.DataFrame(audit), dict(method='full_FCGL_conditional_test_inversion',
                confidence=confidence, rejection_probability=alpha,
                draws=self.draws, seed=self.seed,
                known_covariance_shape=self.known_shape, residual_degrees_freedom=self.df,
                observed_cells=len(self.rows), nuisance_parameters=len(self.columns)-1,
                refits=self.refits, **self.noise_metadata,
                calibration=None if calibration is None else calibration.metadata,
                search='Finite-grid hull and conservative endpoint brackets; '
                       'not globally certified',
                scope='Pointwise; conditional linear response model and supplied factor history'))
