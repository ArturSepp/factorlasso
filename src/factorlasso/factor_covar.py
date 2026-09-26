"""
Factor covariance decomposition: Σ_y = β Σ_x β' + D.

Given the factor model ``Y_t = α + β X_t + ε_t``, this module provides
data containers and assembly logic for the covariance decomposition.
Sparse factor loadings β estimated by :class:`~factorlasso.LassoModel`,
factor covariance Σ_x, and idiosyncratic residual variances D are
assembled into the full response-variable covariance matrix.

Convention
----------
- β is ``(N × M)`` with ``index = response_names``, ``columns = factor_names``
- α is ``(N × 1)`` intercept (EWMA-weighted mean residual)
- Σ_x is ``(M × M)`` factor covariance
- Σ_y is ``(N × N)`` response covariance
- D is ``(N × N)`` diagonal residual variances by default, or an empirical
  common-period EWMA correlation scaled by current residual standard deviations;
  both modes assume zero factor-residual covariance
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from factorlasso.ewm_utils import compute_ewm
from factorlasso.residual_covar import ResidualCorrelationData


class ResidualType(str, Enum):
    """Residual covariance structure, with zero factor-residual cross covariance."""

    ORTHOGONAL = 'orthogonal'
    EMPIRICAL = 'empirical'


def _validate_residual_options(residual_type, residual_var_weight, residual_corr_weight):
    """Validate public assembly options even when no snapshots or risk are requested."""
    kind = ResidualType(residual_type)
    if not np.isfinite(residual_corr_weight) or not 0 <= residual_corr_weight <= 1:
        raise ValueError("residual_corr_weight must be finite and in [0, 1]")
    if kind == ResidualType.ORTHOGONAL and residual_corr_weight != 1.:
        raise ValueError("residual_corr_weight applies only to empirical, not orthogonal residuals")
    if kind == ResidualType.EMPIRICAL:
        if not np.isfinite(residual_var_weight) or residual_var_weight < 0:
            raise ValueError("Empirical residual_var_weight must be finite and nonnegative")
    return kind


class VarianceColumns(str, Enum):
    """Column labels for the variance diagnostics DataFrame."""
    EWMA_VARIANCE = 'ewma_var'
    RESIDUAL_VARS = 'residual_var'
    INSAMPLE_ALPHA = 'insample_alpha'
    R2 = 'r2'
    ALPHA = 'stat_alpha'
    TOTAL_VOL = 'total_vol'
    SYST_VOL = 'sys_vol'
    RESID_VOL = 'resid_vol'
    CLUSTER = 'cluster'


@dataclass(frozen=True)
class CurrentFactorCovarData:
    """
    Factor model covariance snapshot: Σ_y = β Σ_x β' + D.

    Stores all components of the factor decomposition at a single estimation
    date and provides methods to assemble the full covariance matrix.

    Parameters
    ----------
    x_covar : pd.DataFrame, shape (M, M)
        Factor covariance Σ_x.
    y_betas : pd.DataFrame, shape (N, M)
        Factor loadings β.
    y_variances : pd.DataFrame, shape (N, K)
        Per-variable diagnostics (ewma_var, residual_var, r2, cluster, …).
        Cluster assignment is persisted here (column ``'cluster'``) so that
        it round-trips through save/load and filter_on_tickers along with
        the other per-variable diagnostics.
    estimation_date : pd.Timestamp, optional
    residuals : pd.DataFrame, optional
        In-sample residuals ε_t = y_t − x_t β', in producer-declared units.
        OptimalPortfolios stores these multiplied by native periods per year.
    clusters : pd.Series, optional
        Cluster assignment per asset (index = asset names, values = cluster
        labels, typically freq-prefixed strings like ``"ME:3"``, ``"QE:1"``).
        Also mirrored into ``y_variances['cluster']`` on construction
        (see ``__post_init__``) for persistence through save/load.
    linkages : pd.DataFrame, optional
        SciPy linkage matrix for the HCGL dendrogram, stacked across
        frequencies. Columns: ``left``, ``right``, ``distance``,
        ``n_samples``. Index is freq-prefixed (e.g. ``"ME:step_0"``,
        ``"QE:step_7"``) so the per-freq block can be recovered by prefix
        match. See :func:`factorlasso.cluster_utils.get_linkage_array` for
        reconstructing a scipy-compatible ndarray.
    cutoffs : pd.Series, optional
        Dendrogram cutoff distance per frequency (index = freq code).

    residual_metadata : pd.DataFrame, optional
        Native frequency, beta span, annualisation factor and stored/raw residual
        multiplier per asset; see estimate_residual_correlation.
    residual_correlation : ResidualCorrelationData, optional
        Dimensionless common-period correlation with observation and availability
        dates. Empirical assembly scales this by current MATF residual variances.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from factorlasso.factor_covar import CurrentFactorCovarData, VarianceColumns
    >>> M, N = 3, 5
    >>> x_covar = pd.DataFrame(np.eye(M), columns=[f'f{i}' for i in range(M)],
    ...                         index=[f'f{i}' for i in range(M)])
    >>> betas = pd.DataFrame(np.random.randn(N, M),
    ...                       index=[f'y{i}' for i in range(N)],
    ...                       columns=[f'f{i}' for i in range(M)])
    >>> diag = pd.DataFrame({VarianceColumns.RESIDUAL_VARS: np.ones(N) * 0.01},
    ...                      index=[f'y{i}' for i in range(N)])
    >>> data = CurrentFactorCovarData(x_covar=x_covar, y_betas=betas, y_variances=diag)
    >>> cov = data.get_y_covar()
    >>> cov.shape
    (5, 5)
    """

    # --- Core components ---
    x_covar: pd.DataFrame
    y_betas: pd.DataFrame
    y_variances: pd.DataFrame

    # --- Metadata ---
    estimation_date: Optional[pd.Timestamp] = None

    # --- Optional time series ---
    residuals: Optional[pd.DataFrame] = None

    # --- Clustering outputs (HCGL) ---
    clusters: Optional[pd.Series] = None
    linkages: Optional[pd.DataFrame] = None
    cutoffs: Optional[pd.Series] = None

    # --- Sign-constraint outputs (v0.3.10+) ---
    # ``(N × M)`` matrix of solver-facing sign constraints actually consumed
    # by the LASSO problem (auto-derived layer optionally overlaid with the
    # practitioner-set ``factors_beta_loading_signs`` layer). Values are
    # ``-1``, ``0``, ``+1``, or ``NaN``; NaN entries were unconstrained,
    # ``0`` entries forced ``β_kj = 0``, ``±1`` entries imposed a one-sided
    # constraint. Stored only when sign constraints were actually used.
    # Downstream pipelines write this as a ``derived_signs`` Excel sheet
    # for diagnostic and audit purposes.
    derived_signs: Optional[pd.DataFrame] = None

    # Native annual-alpha residual metadata and a prepared common-period risk estimate.
    residual_metadata: Optional[pd.DataFrame] = None
    residual_correlation: Optional[ResidualCorrelationData] = None

    def __post_init__(self):
        """
        Mirror ``clusters`` (if a per-asset Series) into
        ``y_variances['cluster']`` so that cluster assignment is persisted
        through save/load and survives filter_on_tickers along with the
        other per-variable diagnostics. No-op if clusters is None, not a
        Series, or already present in y_variances.
        """
        if self.clusters is None:
            return
        if not isinstance(self.clusters, pd.Series):
            return
        if VarianceColumns.CLUSTER.value in self.y_variances.columns:
            return

        # frozen dataclass — write through object.__setattr__
        y_var = self.y_variances.copy()
        y_var[VarianceColumns.CLUSTER.value] = self.clusters.reindex(y_var.index)
        object.__setattr__(self, 'y_variances', y_var)

    # ── Covariance assembly ──────────────────────────────────────────

    def get_y_covar(
        self,
        residual_var_weight: float = 1.0,
        assets: Optional[Union[List[str], pd.Index]] = None,
        *,
        residual_type: Union[ResidualType, str] = ResidualType.ORTHOGONAL,
        residual_corr_weight: float = 1.0,
    ) -> pd.DataFrame:
        """Assemble B F B' + w D, assuming zero factor-residual cross covariance.

        Orthogonal (default) preserves the stored MATF residual diagonal. Empirical
        retains that same diagonal and adds rho times common-period residual
        correlations scaled by current MATF residual standard deviations:
        D = S [(1-rho) I + rho R] S. ``residual_corr_weight`` is rho in [0, 1];
        ``residual_var_weight`` is w and scales the entire residual block.

        Prepare correlation before retrieval; configure its grid and span during
        estimation. Correlation retrieval has no covariance scale conversion.
        """
        residual = self.get_residual_covar(
            residual_var_weight, assets, residual_type=residual_type,
            residual_corr_weight=residual_corr_weight,
        )
        betas = self.y_betas if assets is None else self.y_betas.loc[assets, :]
        beta = betas.to_numpy()
        y_covar = beta @ self.x_covar.to_numpy() @ beta.T
        if not np.isclose(residual_var_weight, 0.0):
            y_covar += residual.to_numpy()
        return pd.DataFrame(y_covar, index=betas.index, columns=betas.index)

    def get_residual_covar(
        self,
        residual_var_weight: float = 1.0,
        assets: Optional[Union[List[str], pd.Index]] = None,
        *,
        residual_type: Union[ResidualType, str] = ResidualType.ORTHOGONAL,
        residual_corr_weight: float = 1.0,
    ) -> pd.DataFrame:
        """Return w D using current marginal variances and available residual dependence.

        Units are those of y_variances (annual variance in OptimalPortfolios).
        Zero variance weight needs no residual state. Zero correlation retention
        yields the orthogonal matrix exactly and also needs no prepared state.
        """
        kind = _validate_residual_options(residual_type, residual_var_weight,
                                         residual_corr_weight)
        names = self.y_betas.index if assets is None else self.y_betas.loc[assets].index
        resid = self.y_variances[VarianceColumns.RESIDUAL_VARS.value]
        resid = resid if assets is None else resid.loc[assets]
        if not names.equals(resid.index):
            raise ValueError("y_betas and y_variances residual index disagree")
        values = resid.to_numpy(dtype=float)
        covariance = np.diag(values)
        if np.isclose(residual_var_weight, 0.):
            return pd.DataFrame(np.zeros_like(covariance), index=names, columns=names)
        if kind == ResidualType.EMPIRICAL:
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("MATF residual variances must be finite and nonnegative")
            if residual_corr_weight > 0:
                if self.residual_correlation is None:
                    raise ValueError(
                        "Empirical residual covariance requires prepared residual correlation"
                    )
                corr = self.residual_correlation.get_corr(self.estimation_date, names).to_numpy()
                vol = np.sqrt(values)
                covariance = residual_corr_weight * corr * vol[:, None] * vol[None, :]
                # Preserve the MATF diagonal bit-for-bit, independently of sqrt rounding.
                np.fill_diagonal(covariance, values)
        return pd.DataFrame(residual_var_weight * covariance, index=names, columns=names)

    @property
    def y_covar(self) -> pd.DataFrame:
        """Shorthand for ``get_y_covar()``."""
        return self.get_y_covar()

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_model_vols(
        self, assets: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Total, systematic, and residual volatilities per variable.

        Returns
        -------
        pd.DataFrame
            Columns: ``total_vol``, ``sys_vol``, ``resid_vol``.
        """
        if assets is None:
            assets = self.y_betas.index.tolist()
        betas_np = self.y_betas.loc[assets, :].values
        sys_var = np.diag(betas_np @ self.x_covar.values @ betas_np.T)
        res_var = self.y_variances.loc[assets, VarianceColumns.RESIDUAL_VARS.value].values
        return pd.DataFrame({
            VarianceColumns.TOTAL_VOL.value: np.sqrt(sys_var + res_var),
            VarianceColumns.SYST_VOL.value: np.sqrt(sys_var),
            VarianceColumns.RESID_VOL.value: np.sqrt(res_var),
        }, index=assets)

    def estimate_alpha(
            self,
            alpha_span: Union[int, Dict[str, int]] = 120,
            asset_frequencies: Union[str, pd.Series, None] = None,
            default_freq: str = 'ME',
    ) -> pd.Series:
        """
        Estimate alpha from EWMA of residuals, respecting per-asset frequency.

        Parameters
        ----------
        alpha_span : int or dict
            If int: single EWMA span applied to all columns.
            If dict: keys are pandas freq codes ('ME', 'QE'), values are the
            EWMA span in observations at that frequency, e.g.
            ``{'ME': 120, 'QE': 40}`` (~10y calendar half-life for both).
        asset_frequencies : str or pd.Series, optional
            - str: a single freq code applied to all assets (e.g. 'ME').
            - pd.Series: index = asset names, values = freq codes. Assets
              absent from the index fall back to ``default_freq``.
            - None: all assets use ``default_freq``.
        default_freq : str, default 'ME'
            Frequency assumed for assets not covered by ``asset_frequencies``.
        """
        if self.residuals is None:
            raise ValueError("Residuals required for alpha estimation")

        # Legacy scalar path
        if isinstance(alpha_span, (int, float)):
            alphas = compute_ewm(self.residuals, span=int(alpha_span))
            return alphas.iloc[-1, :].rename(VarianceColumns.ALPHA.value)

        # Metadata produced at fitting time supplies each asset's native cadence.
        if asset_frequencies is None and self.residual_metadata is not None:
            asset_frequencies = self.residual_metadata['frequency']
        # Normalise asset_frequencies to a per-column lookup
        if asset_frequencies is None:
            freq_lookup: Dict[str, str] = {}
        elif isinstance(asset_frequencies, str):
            freq_lookup = {c: asset_frequencies for c in self.residuals.columns}
        elif isinstance(asset_frequencies, pd.Series):
            freq_lookup = asset_frequencies.to_dict()
        else:
            raise TypeError(
                f"asset_frequencies must be str, pd.Series, or None; "
                f"got {type(asset_frequencies).__name__}"
            )

        # Group columns by their native frequency
        by_freq: Dict[str, List[str]] = {}
        for col in self.residuals.columns:
            freq = freq_lookup.get(col, default_freq)
            by_freq.setdefault(freq, []).append(col)

        last_values: Dict[str, float] = {}
        for freq, cols in by_freq.items():
            if freq not in alpha_span:
                raise KeyError(
                    f"alpha_span missing entry for frequency '{freq}' "
                    f"(assets e.g. {cols[:3]})"
                )
            sub = self.residuals.loc[:, cols]
            # Relies on upstream factor_covar_estimator preserving NaN on
            # non-event rows. pandas ewm carries forward through NaN, so
            # span is in observations at the column's native frequency.
            ewm = compute_ewm(sub, span=int(alpha_span[freq]))
            last = ewm.iloc[-1, :]
            for c in cols:
                last_values[c] = float(last.get(c, np.nan))

        return pd.Series(
            {c: last_values[c] for c in self.residuals.columns},
            name=VarianceColumns.ALPHA.value,
        )

    def get_snapshot(
            self,
            assets: Optional[List[str]] = None,
            alpha_span: Union[int, Dict[str, int]] = 120,
            asset_frequencies: Union[str, pd.Series, None] = None,
            default_freq: str = 'ME',
    ) -> pd.DataFrame:
        """
        Summary table: betas, R², volatilities, alpha per variable.

        Columns are the factor loadings, ``r2``, ``stat_alpha``, ``insample_alpha``,
        ``total_vol``, ``sys_vol`` and ``resid_vol``. ``stat_alpha`` is the EWMA of the
        stored residuals; without stored residuals it falls back to the in-sample alpha,
        so the column set does not depend on what was stored.

        Raises
        ------
        ValueError
            If ``y_variances`` lacks ``r2`` or ``insample_alpha``.
        """
        required = (VarianceColumns.R2.value, VarianceColumns.INSAMPLE_ALPHA.value)
        missing = [name for name in required if name not in self.y_variances.columns]
        if missing:
            raise ValueError(
                f"get_snapshot needs y_variances columns {missing}, "
                f"got {list(self.y_variances.columns)!r}"
            )
        assets = assets or self.y_betas.index.tolist()
        df = self.y_betas.loc[assets, :].copy()
        vols = self.get_model_vols(assets=assets)

        if self.residuals is not None:
            alphas = self.estimate_alpha(
                alpha_span=alpha_span,
                asset_frequencies=asset_frequencies,
                default_freq=default_freq,
            ).loc[assets]
        else:
            # Fallback under the same column name, so that the table has one
            # 'insample_alpha' column, not two.
            alphas = self.y_variances.loc[assets, VarianceColumns.INSAMPLE_ALPHA.value].rename(
                VarianceColumns.ALPHA.value
            )

        diag = pd.concat([
            self.y_variances.loc[assets, VarianceColumns.R2.value],
            alphas,
            self.y_variances.loc[assets, VarianceColumns.INSAMPLE_ALPHA.value],
        ], axis=1)

        return pd.concat([df, diag, vols], axis=1)

    # ── Subsetting ───────────────────────────────────────────────────

    def filter_on_tickers(
            self, assets: Union[List[str], pd.Index, Dict[str, str]],
    ) -> CurrentFactorCovarData:
        """
        Subset to selected response variables (optionally renaming).

        Notes
        -----
        ``linkages`` and ``cutoffs`` are freq-level objects (one dendrogram
        per frequency, one cutoff per frequency) that describe the global
        clustering geometry, not per-asset metadata. They pass through
        unchanged under an asset-level subset — the clustering hierarchy
        is not "re-cut" for a filtered universe. If you need a per-subset
        clustering, run the estimator again on the subset.

        ``clusters`` is asset-indexed and is subset/renamed accordingly.
        """
        if isinstance(assets, dict):
            keys = list(assets.keys())
            y_betas = self.y_betas.loc[keys, :].rename(index=assets)
            y_var = self.y_variances.loc[keys].rename(index=assets)
            resid = (self.residuals.loc[:, keys].rename(columns=assets)
                     if self.residuals is not None else None)
            clusters = (self.clusters.loc[keys].rename(assets)
                        if self.clusters is not None else None)
            derived_signs = (
                self.derived_signs.loc[keys, :].rename(index=assets)
                if self.derived_signs is not None else None
            )
        else:
            keys = list(assets) if not isinstance(assets, list) else assets
            y_betas = self.y_betas.loc[keys, :]
            y_var = self.y_variances.loc[keys]
            resid = self.residuals[keys] if self.residuals is not None else None
            clusters = (self.clusters.loc[keys]
                        if self.clusters is not None else None)
            derived_signs = (
                self.derived_signs.loc[keys, :]
                if self.derived_signs is not None else None
            )

        # linkages and cutoffs are freq-level, not asset-level — pass through.
        return CurrentFactorCovarData(
            x_covar=self.x_covar,
            y_betas=y_betas,
            y_variances=y_var,
            residuals=resid,
            estimation_date=self.estimation_date,
            clusters=clusters,
            linkages=self.linkages,
            cutoffs=self.cutoffs,
            derived_signs=derived_signs,
            residual_metadata=(self.residual_metadata.loc[keys].rename(
                index=assets if isinstance(assets, dict) else {})
                if self.residual_metadata is not None else None),
            residual_correlation=(self.residual_correlation.filter_on_tickers(assets)
                                  if self.residual_correlation is not None else None),
        )

    # ── Serialisation ────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save core data to an Excel file (one sheet per component)."""
        with pd.ExcelWriter(path) as writer:
            self.x_covar.to_excel(writer, sheet_name='x_covar')
            self.y_betas.to_excel(writer, sheet_name='y_betas')
            self.y_variances.to_excel(writer, sheet_name='y_variances')
            if self.estimation_date is not None:
                pd.Series({'estimation_date': self.estimation_date.isoformat()}).to_excel(
                    writer, sheet_name='snapshot_metadata')
            if self.residual_metadata is not None:
                self.residual_metadata.to_excel(writer, sheet_name='residual_metadata')
            if self.residual_correlation is not None:
                for name, frame in self.residual_correlation.to_sheets().items():
                    frame.to_excel(writer, sheet_name=name)
            if self.residuals is not None:
                self.residuals.to_excel(writer, sheet_name='residuals')
            if self.linkages is not None:
                self.linkages.to_excel(writer, sheet_name='linkages')
            if self.cutoffs is not None:
                self.cutoffs.to_excel(writer, sheet_name='cutoffs')
            if self.derived_signs is not None:
                self.derived_signs.to_excel(writer, sheet_name='derived_signs')

    @classmethod
    def load(cls, path: str) -> CurrentFactorCovarData:
        """Load from an Excel file created by :meth:`save`."""
        sheets = pd.read_excel(path, sheet_name=None, index_col=0)
        y_var = sheets['y_variances']

        # Reconstruct clusters from y_variances if present
        clusters: Optional[pd.Series] = None
        if VarianceColumns.CLUSTER.value in y_var.columns:
            clusters = y_var[VarianceColumns.CLUSTER.value].copy()

        # Linkages — stacked DataFrame with freq-prefixed index
        linkages: Optional[pd.DataFrame] = sheets.get('linkages')

        # Cutoffs — Series with freq index. pd.read_excel returns a DataFrame
        # with one column; take the first (and only) column as a Series.
        cutoffs: Optional[pd.Series] = None
        cutoffs_df = sheets.get('cutoffs')
        if cutoffs_df is not None:
            cutoffs = cutoffs_df.iloc[:, 0]
            cutoffs.name = 'cluster_cutoff'

        # Derived signs sheet — optional, only present when sign constraints
        # were used at the originating fit.
        derived_signs: Optional[pd.DataFrame] = sheets.get('derived_signs')

        return cls(
            x_covar=sheets['x_covar'],
            y_betas=sheets['y_betas'],
            y_variances=y_var,
            residuals=sheets.get('residuals'),
            estimation_date=(pd.Timestamp(sheets['snapshot_metadata'].iloc[0, 0])
                             if 'snapshot_metadata' in sheets else None),
            residual_metadata=sheets.get('residual_metadata'),
            residual_correlation=(ResidualCorrelationData.from_sheets(sheets)
                                  if 'residual_corr_info' in sheets else None),
            clusters=clusters,
            linkages=linkages,
            cutoffs=cutoffs,
            derived_signs=derived_signs,
        )


# ═══════════════════════════════════════════════════════════════════════
# Rolling container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RollingFactorCovarData:
    """
    Time series of :class:`CurrentFactorCovarData` snapshots.

    Stores ``Dict[Timestamp, CurrentFactorCovarData]`` and provides
    panel accessors for betas, R², variances, etc.
    """

    data: Dict[pd.Timestamp, CurrentFactorCovarData] = field(default_factory=dict)

    # ── Container protocol ───────────────────────────────────────────

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.data.keys()))

    @property
    def n_observations(self) -> int:
        return len(self.data)

    def __getitem__(self, date: pd.Timestamp) -> CurrentFactorCovarData:
        return self.data[date]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sorted(self.data.keys()))

    def add(self, date: pd.Timestamp, estimation: CurrentFactorCovarData):
        self.data[date] = estimation

    def get_latest(self) -> CurrentFactorCovarData:
        return self.data[max(self.data.keys())]

    # ── Matrix time series ───────────────────────────────────────────

    def get_x_covars(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Factor covariance matrices over time."""
        return {d: e.x_covar for d, e in sorted(self.data.items())}

    def _snapshots_asof(self, dates=None):
        """Yield only snapshots available by each query date, without backdating fits."""
        requested = self.dates if dates is None else pd.DatetimeIndex(dates).unique().sort_values()
        available = self.dates
        for date in requested:
            position = available.searchsorted(date, side='right') - 1
            if position < 0:
                raise ValueError("No fitted covariance was available at the requested date")
            estimation = self.data[available[position]]
            if estimation.estimation_date is not None and estimation.estimation_date > date:
                raise ValueError("Fitted covariance was not available at the requested date")
            estimation_date = (
                min(date, estimation.estimation_date)
                if estimation.estimation_date is not None
                else date
            )
            yield date, replace(estimation, estimation_date=estimation_date)

    def get_y_covars(
        self,
        residual_var_weight: float = 1.0,
        assets: Optional[Union[List[str], pd.Index]] = None,
        *,
        residual_type: Union[ResidualType, str] = ResidualType.ORTHOGONAL,
        dates: Optional[pd.DatetimeIndex] = None,
        residual_corr_weight: float = 1.0,
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Return fitted or as-of total covariances with current MATF marginal risk.

        Options match CurrentFactorCovarData.get_y_covar. Between common-period
        updates R may be held, while B, F and residual variances come from the
        latest available snapshot. Retrieval never refits any component.
        """
        _validate_residual_options(residual_type, residual_var_weight, residual_corr_weight)
        return {date: estimation.get_y_covar(
            residual_var_weight, assets, residual_type=residual_type,
            residual_corr_weight=residual_corr_weight,
        ) for date, estimation in self._snapshots_asof(dates)}

    def get_residual_covars(
        self,
        residual_var_weight: float = 1.0,
        assets: Optional[Union[List[str], pd.Index]] = None,
        *,
        residual_type: Union[ResidualType, str] = ResidualType.ORTHOGONAL,
        residual_corr_weight: float = 1.0,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Assemble w D at every fit/query date, refreshing MATF marginal variances."""
        _validate_residual_options(residual_type, residual_var_weight, residual_corr_weight)
        return {date: estimation.get_residual_covar(
            residual_var_weight, assets, residual_type=residual_type,
            residual_corr_weight=residual_corr_weight,
        ) for date, estimation in self._snapshots_asof(dates)}

    def get_residual_correlations(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Return distinct dimensionless R vintages keyed by fit/availability date."""
        history = {}
        for _, snapshot in self._snapshots_asof():
            prepared = snapshot.residual_correlation
            if prepared is not None:
                history[prepared.estimation_date] = prepared.get_corr(snapshot.estimation_date)
        return dict(sorted(history.items()))

    def get_y_betas(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Factor loadings over time.  Each DataFrame is (N × M)."""
        return {d: e.y_betas for d, e in sorted(self.data.items())}

    # ── Panel DataFrame accessors ────────────────────────────────────

    def get_residual_vars(self) -> pd.DataFrame:
        """Residual variances: index = dates, columns = variables."""
        return pd.DataFrame({
            d: e.y_variances[VarianceColumns.RESIDUAL_VARS.value]
            for d, e in sorted(self.data.items())
        }).T

    def get_ewma_vars(self) -> pd.DataFrame:
        return pd.DataFrame({
            d: e.y_variances[VarianceColumns.EWMA_VARIANCE.value]
            for d, e in sorted(self.data.items())
        }).T

    def get_r2(self) -> pd.DataFrame:
        """R² panel: index = dates, columns = variables."""
        return pd.DataFrame({
            d: e.y_variances[VarianceColumns.R2.value]
            for d, e in sorted(self.data.items())
        }).T

    def get_systematic_vars(self) -> pd.DataFrame:
        """Systematic variances diag(β Σ_x β'): index = dates, columns = variables."""
        records = {}
        for d, e in sorted(self.data.items()):
            betas_np = e.y_betas.values  # (N × M)
            sys_var = np.diag(betas_np @ e.x_covar.values @ betas_np.T)
            records[d] = pd.Series(sys_var, index=e.y_betas.index)
        return pd.DataFrame(records).T

    def get_total_vols(self) -> pd.DataFrame:
        return np.sqrt(self.get_systematic_vars() + self.get_residual_vars())

    def get_residual_vols(self) -> pd.DataFrame:
        return np.sqrt(self.get_residual_vars())

    def get_alphas(self,
                   alpha_span: Union[int, Dict[str, int]] = 120,
                   asset_frequencies: Union[str, pd.Series, None] = None,
                   default_freq: str = 'ME',
                   ) -> pd.DataFrame:
        """Rolling alpha panel: index = dates, columns = variables.

        Thin wrapper over :meth:`CurrentFactorCovarData.estimate_alpha`, one
        call per estimation date. Dates whose entry carries no residuals fall
        back to the in-sample alpha recorded on ``y_variances``.

        Parameters
        ----------
        alpha_span : int or dict
            EWMA span for the residual smoother. An int applies one span to
            every column; a dict keyed by pandas frequency code
            (``{'ME': 60, 'QE': 20}``) applies a span **in observations at the
            column's own cadence**, which is how one calendar horizon is
            expressed across mixed-frequency responses.
        asset_frequencies : str or pd.Series, optional
            Per-response frequency codes, forwarded to ``estimate_alpha``.
            Required for a dict ``alpha_span`` to mean anything: without it
            every response falls back to ``default_freq``, so a
            ``{'ME': 60, 'QE': 20}`` span would silently apply 60 to the
            quarterly responses.
        default_freq : str, default 'ME'
            Cadence assumed for a response absent from ``asset_frequencies``.

        Returns
        -------
        pd.DataFrame
            Alphas by date and response; empty when the container is empty.
        """
        records = {}
        for d, e in sorted(self.data.items()):
            if e.residuals is not None:
                records[d] = e.estimate_alpha(alpha_span=alpha_span,
                                              asset_frequencies=asset_frequencies,
                                              default_freq=default_freq)
            else:
                records[d] = e.y_variances[VarianceColumns.INSAMPLE_ALPHA.value]
        return pd.DataFrame(records).T if records else pd.DataFrame()

    def get_factor_var(self, factor: str) -> pd.Series:
        return pd.Series(
            {d: e.x_covar.loc[factor, factor] for d, e in sorted(self.data.items())},
            name=factor,
        )

    def get_beta(self, factor: str) -> pd.DataFrame:
        """Single factor loadings over time: index = dates, columns = variables."""
        return pd.DataFrame(
            {d: e.y_betas[factor] for d, e in sorted(self.data.items())}
        ).T

    def filter_on_tickers(
        self, tickers: Union[List[str], pd.Index],
    ) -> RollingFactorCovarData:
        return RollingFactorCovarData(
            data={d: e.filter_on_tickers(tickers) for d, e in self.data.items()}
        )

    def get_snapshot(self, alpha_span: int = 120) -> Dict[pd.Timestamp, pd.DataFrame]:
        return {d: e.get_snapshot(alpha_span=alpha_span) for d, e in self.data.items()}
