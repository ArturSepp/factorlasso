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
- D is ``(N × N)`` diagonal residual variances
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from factorlasso.ewm_utils import compute_ewm


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
        In-sample residuals ε_t = y_t − x_t β'.
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
    ) -> pd.DataFrame:
        """
        Assemble response covariance matrix.

        .. math::

            \\Sigma_y(w) = \\beta\\,\\Sigma_x\\,\\beta^\\top + w\\,D

        Parameters
        ----------
        residual_var_weight : float, default 1.0
            Scaling on the diagonal residual variances.
        assets : list of str, optional
            Subset of response variables.

        Returns
        -------
        pd.DataFrame, shape (N, N) or (len(assets), len(assets))

        Raises
        ------
        ValueError
            If ``y_betas`` and ``y_variances`` indices disagree for the
            requested asset set.  Silent row-ordering mismatch between β
            and D is a subtle bug class that surfaces as wrong but
            non-throwing covariance matrices in production; the explicit
            check here converts it into a loud error.
        """
        betas = self.y_betas if assets is None else self.y_betas.loc[assets, :]
        resid = self.y_variances[VarianceColumns.RESIDUAL_VARS.value]
        resid = resid if assets is None else resid.loc[assets]

        # Row-ordering guard: β and D must agree on the asset order,
        # otherwise β Σ_x β' + diag(resid) silently mixes rows. This
        # guards against partial filter_on_tickers or any upstream
        # reindex that desynchronises the two containers.
        if not betas.index.equals(resid.index):
            raise ValueError(
                "y_betas and y_variances residual index disagree; "
                f"betas: {list(betas.index)[:5]}...; "
                f"resid: {list(resid.index)[:5]}..."
            )

        names = betas.index
        betas_np = betas.values  # (N × M)
        y_covar = betas_np @ self.x_covar.to_numpy() @ betas_np.T

        if not np.isclose(residual_var_weight, 0.0):
            y_covar += residual_var_weight * np.diag(resid.to_numpy())

        return pd.DataFrame(y_covar, index=names, columns=names)

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
        """
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
            alphas = self.y_variances.loc[assets, VarianceColumns.INSAMPLE_ALPHA.value]

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
        else:
            keys = list(assets) if not isinstance(assets, list) else assets
            y_betas = self.y_betas.loc[keys, :]
            y_var = self.y_variances.loc[keys]
            resid = self.residuals[keys] if self.residuals is not None else None
            clusters = (self.clusters.loc[keys]
                        if self.clusters is not None else None)

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
        )

    # ── Serialisation ────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save core data to an Excel file (one sheet per component)."""
        with pd.ExcelWriter(path) as writer:
            self.x_covar.to_excel(writer, sheet_name='x_covar')
            self.y_betas.to_excel(writer, sheet_name='y_betas')
            self.y_variances.to_excel(writer, sheet_name='y_variances')
            if self.residuals is not None:
                self.residuals.to_excel(writer, sheet_name='residuals')
            if self.linkages is not None:
                self.linkages.to_excel(writer, sheet_name='linkages')
            if self.cutoffs is not None:
                self.cutoffs.to_excel(writer, sheet_name='cutoffs')

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

        return cls(
            x_covar=sheets['x_covar'],
            y_betas=sheets['y_betas'],
            y_variances=y_var,
            residuals=sheets.get('residuals'),
            clusters=clusters,
            linkages=linkages,
            cutoffs=cutoffs,
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

    def get_y_covars(
        self,
        residual_var_weight: float = 1.0,
        assets: Optional[Union[List[str], pd.Index]] = None,
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Response covariance matrices over time."""
        return {
            d: e.get_y_covar(residual_var_weight=residual_var_weight, assets=assets)
            for d, e in sorted(self.data.items())
        }

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

    def get_alphas(self, alpha_span: int = 120) -> pd.DataFrame:
        records = {}
        for d, e in sorted(self.data.items()):
            if e.residuals is not None:
                records[d] = e.estimate_alpha(alpha_span=alpha_span)
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
