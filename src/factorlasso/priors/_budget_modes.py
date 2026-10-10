"""Resolve named-factor conditioning and loading-budget provenance.

For two selected factors the largest variance inflation factor is the largest
diagonal of their inverse weighted correlation matrix. The weights and complete
rows are those of the joint OLS prior, including the response's missing values.
A rejected pair keeps its first factor. The second becomes a competitor.

Cluster budgets are leave-one-response-out means of absolute first-pass
loadings. They are loading limits, not confidence bounds. A standard-error
slack has no unique allocation among competitors. Its diagnostic therefore
reports each implied one-SE budget separately, without an additive allocation.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
import warnings

import numpy as np
import pandas as pd

from factorlasso.priors._bounds import _competitor_geometry, _observation_weights


class BudgetRule(str, Enum):
    N_STD = 'n_std'
    CLUSTER_MEAN = 'cluster_mean'


class NamedStatus(str, Enum):
    AUTOMATIC = 'automatic'
    NAMED = 'named'
    ACCEPTED = 'named_set_accepted'
    DEMOTED = 'named_set_demoted'
    DISABLED = 'conditioning_disabled'


class BudgetSource(str, Enum):
    DECLARED = 'declared'
    DECLARED_ZERO = 'declared_zero'
    CLUSTER_MEAN = 'cluster_mean'
    UNDECLARED_ZERO = 'undeclared_zero'
    N_STD = 'n_std'


class ClusterStatus(str, Enum):
    OK = 'cluster_mean'
    SMALL = 'cluster_too_small'


def weighted_max_vif(*, x: pd.DataFrame, y: pd.Series, span: float | None,
                     min_periods: int) -> float:
    """Largest VIF on the prior's original calendar grid, infinity if unidentified."""
    values = x.to_numpy(dtype=float)
    weights = _observation_weights(n=len(x), span=span)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(y.to_numpy(dtype=float))
    valid &= weights > 0
    if valid.sum() < max(min_periods, len(x.columns) + 1, 3):
        return np.inf
    values, weights = values[valid], weights[valid]
    weights = weights / weights.sum()
    values = values - values[-1]
    values = values - weights @ values
    covariance = values.T @ (weights[:, None] * values)
    sd = np.sqrt(np.diag(covariance))
    if np.any(sd <= 0):
        return np.inf
    correlation = covariance / np.outer(sd, sd)
    if np.linalg.eigvalsh(correlation).min() <= np.finfo(float).eps * len(sd):
        return np.inf
    return float(np.diag(np.linalg.inv(correlation)).max())


def resolve_named_sets(*, selections: dict[Any, tuple], x: pd.DataFrame,
                       y: pd.DataFrame, span: float | None, min_periods: int,
                       threshold: float | None) -> tuple[dict[Any, tuple], pd.DataFrame]:
    """Retain order and record both requested and effective sets without mutating input."""
    effective, records, demoted = {}, [], []
    for response, requested in selections.items():
        absent = [factor for factor in requested if factor not in x.columns]
        if absent:
            raise ValueError(f'factor_for_prior[{response!r}] names unknown factors {absent!r}')
        chosen, vif = requested, 1.0
        status = NamedStatus.NAMED if requested else NamedStatus.AUTOMATIC
        if len(requested) == 2:
            vif = weighted_max_vif(x=x.loc[:, list(requested)], y=y[response],
                                   span=span, min_periods=min_periods)
            status = NamedStatus.DISABLED if threshold is None else NamedStatus.ACCEPTED
            if threshold is not None and vif > threshold:
                chosen, status = requested[:1], NamedStatus.DEMOTED
                demoted.append(response)
        effective[response] = chosen
        records.append(dict(response=response, requested_set=requested, named_set=chosen,
                            named_set_status=status.value, largest_vif=vif))
    if demoted:
        warnings.warn(f'named_set_demoted at VIF threshold {threshold!r}: {demoted!r}',
                      UserWarning, stacklevel=3)
    return effective, pd.DataFrame(records).set_index('response')


def leave_one_out_budgets(*, coefficients: pd.DataFrame,
                         clusters: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Mean absolute loading of other members, with NaN for clusters below three."""
    labels = clusters.reindex(coefficients.index)
    budgets = pd.DataFrame(np.nan, index=coefficients.index, columns=coefficients.columns)
    status = pd.Series(ClusterStatus.SMALL.value, index=coefficients.index)
    for label in labels.dropna().unique():
        members = labels.index[labels.eq(label)]
        if len(members) < 3:
            continue
        values = coefficients.loc[members].abs()
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f'nonfinite first-pass loadings in cluster {label!r}')
        budgets.loc[members] = (values.sum(axis=0) - values) / (len(members) - 1)
        status.loc[members] = ClusterStatus.OK.value
    return budgets, status


def resolve_budgets(*, responses: pd.Index, factors: pd.Index,
                    declared: pd.DataFrame | None,
                    automatic: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finite declarations override the data rule cell by cell, including explicit zero."""
    budgets = pd.DataFrame(np.nan, index=responses, columns=factors)
    sources = pd.DataFrame(BudgetSource.N_STD.value, index=responses, columns=factors)
    if automatic is not None:
        budgets.loc[:, :] = automatic.reindex(index=responses, columns=factors)
        sources = sources.mask(budgets.notna(), BudgetSource.CLUSTER_MEAN.value)
    if declared is not None:
        unknown = declared.columns.difference(factors).tolist()
        if unknown:
            raise ValueError(f'expert_prior_exposure_budget names unknown factors {unknown!r}')
        for response in responses.intersection(declared.index):
            row = declared.loc[response].reindex(factors).astype(float)
            if not row.notna().any():
                continue
            missing = budgets.loc[response].isna() & row.isna()
            budgets.loc[response, missing] = 0.0
            sources.loc[response, missing] = BudgetSource.UNDECLARED_ZERO.value
            finite = row.notna()
            budgets.loc[response, finite] = row[finite]
            sources.loc[response, finite] = BudgetSource.DECLARED.value
            sources.loc[response, row.eq(0.0)] = BudgetSource.DECLARED_ZERO.value
    return budgets, sources


def contribution_diagnostics(*, x: pd.DataFrame, y: pd.DataFrame,
                            diagnostics: pd.DataFrame, named: pd.DataFrame,
                            budgets: pd.DataFrame, sources: pd.DataFrame,
                            signs: pd.DataFrame, span: float | None,
                            min_periods: int, cluster_status: pd.Series | None,
                            geometry_cache: dict | None = None) -> pd.DataFrame:
    """Additive budget terms and separate one-SE equivalents, one row per selected cell."""
    rows, missing, cache = [], set(), {} if geometry_cache is None else geometry_cache
    for row in diagnostics.to_dict('records'):
        response, factor, selection = row['asset'], row['factor'], row['selection']
        if response not in cache:
            cache[response] = _competitor_geometry(
                x=x, y=y[response], factors=selection, span=span, min_periods=min_periods)
        slopes = cache[response][0].loc[factor]
        for competitor, slope in slopes.items():
            sign = signs.at[response, competitor]
            weight = (0.0 if sign == 0.0 else abs(slope) if not np.isfinite(sign)
                      else max(row['direction'] * slope * sign, 0.0))
            budget, source = budgets.at[response, competitor], sources.at[response, competitor]
            term = (np.nan if source == BudgetSource.N_STD.value else
                    0.0 if budget == 0.0 or weight == 0.0 else weight * budget)
            if source == BudgetSource.UNDECLARED_ZERO.value and weight > 0:
                missing.add((response, competitor))
            rows.append(dict(
                response=response, named_factor=factor, named_set=tuple(selection),
                named_set_status=named.at[response, 'named_set_status'],
                largest_vif=named.at[response, 'largest_vif'], prior=row['reference_beta'],
                se_hac=row['se_hac'], competitor=competitor, gamma=slope, sign=sign,
                budget=budget, budget_source=source, contribution=term, slack=row['slack'],
                implied_one_se_budget=row['se_hac'] / weight if weight > 0 else np.nan,
                floor=row['floor'], direction=row['direction'], binding=False,
                bound_status=row['status'],
                cluster_status=None if cluster_status is None else cluster_status.at[response]))
    if missing:
        warnings.warn(f'undeclared_zero for permitted competitors {sorted(missing, key=str)!r}',
                      UserWarning, stacklevel=3)
    return pd.DataFrame(rows)
