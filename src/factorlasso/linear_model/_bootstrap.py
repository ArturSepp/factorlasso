"""Full-estimator dependent-wild bootstrap for sparse factor loadings.

References
----------
Shao (2010), The Dependent Wild Bootstrap, JASA 105, 218-235,
doi:10.1198/jasa.2009.tm08744, motivates the common calendar multipliers.
Chatterjee and Lahiri (2011), Bootstrapping Lasso Estimators, JASA 106,
608-625, doi:10.1198/jasa.2011.tm10159, explains why an ordinary bootstrap
need not give valid LASSO confidence intervals. This implementation refits
all data-dependent FCGL stages, but does NOT inherit a post-selection
coverage theorem. Percentiles are stability ranges until separately validated.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from factorlasso.linear_model._estimator import LassoModel
from factorlasso.linear_model._solvers.common import _compute_solver_weights
from factorlasso.utils._hac import bartlett_kernel


@dataclass(frozen=True)
class FactorBootstrapResult:
    """Joint draws from fresh fits of the configured production estimator.

    Attributes
    ----------
    estimate : pandas.DataFrame
        Original fitted coefficients, assets by factors.
    draws : pandas.DataFrame
        Coefficients indexed by (draw, asset), with factor columns. Assets in
        one draw share calendar multipliers and must be aggregated together.
    fit_diagnostics : pandas.DataFrame
        Per-draw cluster count and changes in partition, automatic priors,
        solver-facing signs and adaptive penalty weights relative to the fit.
    response_support : pandas.DataFrame
        Valid counts, effective weighted observations and residual scale.
    metadata : dict
        Resampling settings and explicit conditional-inference limitations.
    """

    estimate: pd.DataFrame
    draws: pd.DataFrame
    fit_diagnostics: pd.DataFrame
    response_support: pd.DataFrame
    metadata: dict

    def intervals(self, confidence: float = .95, method: str = 'percentile') -> pd.DataFrame:
        """Summarize marginal asset coefficients without claiming coverage.

        Parameters
        ----------
        confidence : float, default 0.95
            Central probability mass of the bootstrap distribution.
        method : {'percentile', 'basic'}, default 'percentile'
            Percentile endpoints or reflected estimation-error endpoints.

        Returns
        -------
        pandas.DataFrame
            Asset/factor indexed estimate, lower, upper, bootstrap mean, bias
            and near-zero frequency. Basic bounds can violate sign constraints.
        """
        shape = (self.metadata['draws'], *self.estimate.shape)
        values = self.draws.to_numpy().reshape(shape)
        index = pd.MultiIndex.from_product(
            [self.estimate.index, self.estimate.columns], names=['asset', 'factor'])
        return _summary(values.reshape(shape[0], -1), self.estimate.to_numpy().ravel(),
                        index, confidence, method)

    def portfolio_intervals(self, weights: pd.Series, confidence: float = .95,
                            method: str = 'percentile') -> pd.DataFrame:
        """Aggregate joint asset draws before calculating portfolio intervals.

        Parameters
        ----------
        weights : pandas.Series
            Finite signed exposures in the caller's reporting units, indexed
            by every fitted asset exactly once. They need not sum to one;
            net-equity weights retain borrowing and leverage.
        confidence : float, default 0.95
            Central bootstrap probability mass.
        method : {'percentile', 'basic'}, default 'percentile'
            Same endpoint convention as :meth:`intervals`.

        Returns
        -------
        pandas.DataFrame
            Factor-indexed summaries of jointly aggregated coefficients.
        """
        if (not isinstance(weights, pd.Series) or weights.index.has_duplicates
                or set(weights.index) != set(self.estimate.index)):
            raise ValueError('weights must identify every fitted asset exactly once')
        w = weights.reindex(self.estimate.index).to_numpy(dtype=float)
        if not np.isfinite(w).all():
            raise ValueError('weights must be finite')
        values = self.draws.to_numpy().reshape(self.metadata['draws'], *self.estimate.shape)
        joint = np.einsum('baf,a->bf', values, w)
        return _summary(joint, w @ self.estimate.to_numpy(), self.estimate.columns,
                        confidence, method)


def _summary(draws, estimate, index, confidence, method):
    if isinstance(confidence, (bool, np.bool_)) or not 0 < confidence < 1:
        raise ValueError('confidence must be between zero and one')
    if method not in ('percentile', 'basic'):
        raise ValueError("method must be 'percentile' or 'basic'")
    tail = (1-confidence)/2
    lo, hi = np.quantile(draws, [tail, 1-tail], axis=0)
    if method == 'basic':
        lo, hi = 2*estimate-hi, 2*estimate-lo
    mean = draws.mean(axis=0)
    return pd.DataFrame(dict(estimate=estimate, lower=lo, upper=hi, bootstrap_mean=mean,
                             bootstrap_bias=mean-estimate,
                             near_zero_frequency=(np.abs(draws) <= 1e-7).mean(axis=0),
                             probability=confidence, method=method,
                             status='coverage_not_validated'), index=index)


def _partition(model, n):
    groups = model.clusters_
    return np.eye(n, dtype=bool) if groups is None else (
        groups.to_numpy()[:, None] == groups.to_numpy()[None, :])


def _changed(first, second):
    if first is None or second is None:
        return int(first is not second)
    a, b = np.asarray(first), np.asarray(second)
    if a.shape != b.shape:
        return int(max(a.size, b.size))
    return int(np.count_nonzero(~np.isclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True)))


def _selected_prior(model):
    prior = model.ols_beta_prior_
    return None if prior is None else np.abs(prior.to_numpy()) > 1e-8


def bootstrap_factor_betas(
    model: LassoModel, x: pd.DataFrame, y: pd.DataFrame, *, draws: int = 2000,
    seed: int = 0, bandwidth: float = 3., calendar=None,
    multipliers: pd.DataFrame | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FactorBootstrapResult:
    """Refit all production estimation stages on dependent-wild response draws.

    Parameters
    ----------
    model : LassoModel
        Configuration, fitted or unfitted. Fresh deep-copied models are fitted
        on the original sample and each draw; the supplied object is untouched.
        Set native-frequency spans and penalties before calling. Explicit expert
        inputs remain fixed; automatic clustering, priors, bounds, signs and
        adaptive weights are recomputed. This is a current fit, not a replay of
        a rolling smoother or of external model/penalty selection.
    x, y : pandas.DataFrame
        Aligned factor and response panels with unique, increasing index and
        unique columns. Missing responses retain their exact dates. No price,
        return-type or frequency conversion is performed. A partially missing
        factor row is rejected; an all-missing opening row requires missing y.
    draws : int, default 2000
        Number of complete joint refits, at least 20 (use 2000 or more for reports).
    seed : int, default 0
        NumPy random-generator seed, recorded for reproducibility.
    bandwidth : float, default 3
        Bartlett dependence bandwidth, zero correlation at that calendar distance.
    calendar : array-like, optional
        Increasing numeric observation times in chosen units; defaults to row
        numbers. Supply actual month/quarter ordinals when calendar gaps exist.
    multipliers : pandas.DataFrame, optional
        Explicit finite draws-by-dates multipliers, with columns exactly x.index.
        Overrides random generation. Supports externally coordinated resampling;
        the caller owns cadence aggregation and dependence for supplied draws.
    progress : callable, optional
        Called after each successful draw with (completed, total).

    Returns
    -------
    FactorBootstrapResult
        Original production estimates, joint coefficient draws and diagnostics.

    Notes
    -----
    Conditional on observed factors, fitted linear mean, histories, penalty and
    expert constraints, generate y* = fitted_mean + centred_residual * multiplier.
    Residual centring uses the production squared EWMA weights. Multipliers have
    unit variance and a Bartlett time covariance and are shared across assets.
    No residual degrees-of-freedom correction is imposed. Full refitting includes
    selection instability but does not eliminate shrinkage bias or guarantee
    95% coverage, especially near selection boundaries or with short histories.
    Solver failures abort the run, never silently remove difficult draws.
    """
    if not isinstance(model, LassoModel):
        raise TypeError('model must be a LassoModel')
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 20:
        raise ValueError('draws must be an integer of at least 20')
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    for name, frame in [('x', x), ('y', y)]:
        if (not isinstance(frame, pd.DataFrame) or frame.empty or frame.index.has_duplicates
                or not frame.index.is_monotonic_increasing or frame.columns.has_duplicates):
            raise ValueError(f'{name} requires unique ordered dates and unique columns')
        if np.isinf(frame.to_numpy(dtype=float)).any():
            raise ValueError(f'{name} cannot contain infinity')
    if not x.index.equals(y.index):
        raise ValueError('x and y indexes must match exactly')
    valid_x = x.notna().all(axis=1)
    if ((x.notna().any(axis=1) & ~valid_x).any()
            or y.loc[~valid_x].notna().any().any()):
        raise ValueError('factor rows must be complete or wholly missing with missing responses')
    valid = y.notna().to_numpy()
    counts = valid.sum(axis=0)
    if (counts < max(model.warmup_period, 3)).any():
        raise ValueError('every response needs at least warmup_period and three observations')
    t, n = y.shape
    calendar = np.arange(t) if calendar is None else np.asarray(calendar)
    if len(calendar) != t:
        raise ValueError('calendar length must match the panel')
    kernel = bartlett_kernel(calendar, bandwidth)
    if multipliers is not None:
        if (not isinstance(multipliers, pd.DataFrame) or multipliers.shape != (draws, t)
                or not multipliers.columns.equals(x.index)
                or not np.isfinite(multipliers.to_numpy(dtype=float)).all()):
            raise ValueError('multipliers must be finite draws-by-dates with exact date columns')
        innovations = multipliers.to_numpy(dtype=float, copy=True)
    else:
        eig, vec = np.linalg.eigh(kernel)
        root = vec * np.sqrt(np.maximum(eig, 0.))
        innovations = np.random.default_rng(seed).standard_normal((draws, t)) @ root.T
    parameters = deepcopy(model.get_params())
    baseline = LassoModel(**deepcopy(parameters)).fit(x, y)
    fitted = baseline.predict(x.fillna(0.)).to_numpy()
    residual = np.where(valid, y.to_numpy()-fitted, 0.)
    weights = _compute_solver_weights(t, n, model.span, valid)**2
    mass = weights.sum(axis=0)
    if (mass <= 0).any():
        raise ValueError('every response needs positive observation weight')
    residual -= (weights*residual).sum(axis=0)/mass
    residual = np.where(valid, residual, 0.)
    support = pd.DataFrame(dict(observations=counts,
        effective_observations=mass**2/(weights**2).sum(axis=0),
        residual_std=np.sqrt((weights*residual**2).sum(axis=0)/mass)), index=y.columns)
    values, audits = [], []
    original_partition = _partition(baseline, n)
    for b, innovation in enumerate(innovations):
        response = pd.DataFrame(np.where(valid, fitted + residual*innovation[:, None], np.nan),
                                index=y.index, columns=y.columns)
        try:
            fit = LassoModel(**deepcopy(parameters)).fit(x, response)
        except Exception as error:
            raise RuntimeError(f'bootstrap refit {b} failed; no draws were discarded') from error
        value = fit.coef_.reindex(index=y.columns, columns=x.columns).to_numpy()
        if not np.isfinite(value).all():
            raise RuntimeError(f'bootstrap refit {b} produced nonfinite coefficients')
        values.append(value)
        audits.append(dict(draw=b, clusters=0 if fit.clusters_ is None else fit.clusters_.nunique(),
            partition_changed=bool(np.any(_partition(fit, n) != original_partition)),
            prior_cells_changed=_changed(baseline.effective_beta_prior_, fit.effective_beta_prior_),
            prior_selection_cells_changed=_changed(_selected_prior(baseline), _selected_prior(fit)),
            sign_cells_changed=_changed(baseline.derived_signs_, fit.derived_signs_),
            adaptive_cells_changed=_changed(baseline.sign_penalty_weights_,
                                            fit.sign_penalty_weights_),
            block_weights_changed=_changed(baseline.sign_block_weights_, fit.sign_block_weights_)))
        if progress is not None:
            progress(b+1, draws)
    index = pd.MultiIndex.from_product([range(draws), y.columns], names=['draw', 'asset'])
    return FactorBootstrapResult(baseline.coef_.copy(),
        pd.DataFrame(np.concatenate(values), index=index, columns=x.columns),
        pd.DataFrame(audits).set_index('draw'), support,
        dict(method='full_estimator_dependent_wild', draws=int(draws), seed=int(seed),
             bandwidth=float(bandwidth), supplied_multipliers=multipliers is not None,
             calendar=list(map(float, calendar)), coverage='not_validated',
             scope=('Fixed factors, histories, penalty and explicit expert inputs; '
                    'full current refits'),
             residual_adjustment='EWMA centred; no degrees-of-freedom rescaling',
             failed_draws=0))
