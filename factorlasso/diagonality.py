"""
Regularisation selection by residual diagonality.

Provides :class:`LassoModelDiagonalityCV`, a sibling of :class:`~factorlasso.LassoModelCV` that
selects ``reg_lambda`` by a different criterion. Where ``LassoModelCV`` maximises out-of-sample
R², this selects the SPARSEST penalty whose held-out residuals remain consistent with the strict
factor structure the model asserts, ``Sigma = beta Sigma_F beta' + D`` with ``D`` diagonal.

The two answer different questions and can disagree. R² asks how well the model predicts
returns; diagonality asks whether the systematic and idiosyncratic blocks have been separated
correctly. A model can predict well while leaving a whole factor in the residual, and downstream
uses that invert ``D`` — GLS cross-sectional regression, risk budgeting, any decomposition of
squared Sharpe into systematic and idiosyncratic parts — are wrong when it does.

Selection rule
--------------
Lower is better, but the criterion is NOT minimised. The statistic falls with model density and
flattens, so its minimum is not identified. Instead each penalty is compared against a fixed
chi-square threshold and the sparsest passing penalty is taken:

    best_lambda_ = max { lambda : mean fold sphericity <= threshold }.

When no penalty passes, ``passed_`` is False, ``best_lambda_`` falls back to the minimiser, and
``missing_factors_`` reports the residual components above the Marchenko-Pastur edge. That is the
informative outcome: no penalty repairs a factor the model does not carry, and the remedy is to
extend the factor set.

Taking the most parsimonious model a specification test does not reject is standard, and the rule
here is not new. Gagliardini, Ossola and Scaillet (2019) use the same shape on the same kind of
statistic, selecting the factor count as ``min { k : xi(k) < 0 }`` where ``xi`` is the residual
eigenvalue less a vanishing penalty. This class differs by indexing a regularisation path rather
than a factor count, and by calibrating against a fixed chi-square threshold rather than a
vanishing penalty. Neither difference is a result. See the references in
:mod:`factorlasso.residual_diagnostics`, and prefer their criterion when the conclusion rests on
the count of missing factors rather than on the choice of penalty.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from factorlasso import LassoModel, LassoModelType, LassoModelDiagonalityCV
>>> rng = np.random.default_rng(0)
>>> T, M, N = 240, 3, 8
>>> idx = pd.date_range('2006-01-31', periods=T, freq='ME')
>>> X = pd.DataFrame(rng.standard_normal((T, M)), index=idx,
...                  columns=[f'f{i}' for i in range(M)])
>>> beta = rng.standard_normal((N, M))
>>> Y = pd.DataFrame(X.to_numpy() @ beta.T + rng.standard_normal((T, N)),
...                  index=idx, columns=[f'y{i}' for i in range(N)])
>>> sel = LassoModelDiagonalityCV(n_splits=3).fit(x=X, y=Y)
>>> isinstance(sel.best_lambda_, float)
True
>>> sorted(sel.diagnostics_.columns)[:3]
['fold_sphericity_mean', 'mean_abs_offdiag', 'mp_edge']
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cvxpy as cvx
import numpy as np
import pandas as pd

from factorlasso.cv import DEFAULT_LAMBDA_GRID, expanding_window_splits
from factorlasso.lasso_estimator import LassoModel, LassoModelType
from factorlasso.residual_diagnostics import (
    ResidualDiagnostics,
    diagnose_residuals,
    effective_sparsity,
    missing_factor_components,
)

__all__ = ["LassoModelDiagonalityCV"]

_FOLD_ERRORS: Tuple[type, ...] = (
    cvx.error.SolverError,
    cvx.error.DCPError,
    ValueError,
    np.linalg.LinAlgError,
)

_GROUP_FAMILY = (
    LassoModelType.GROUP_LASSO,
    LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
    LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
)


@dataclass
class LassoModelDiagonalityCV:
    """
    Select ``reg_lambda`` by the diagonality of held-out residuals.

    Parameters
    ----------
    lambdas : sequence of float, optional
        Regularisation grid. Default: the 20-point grid of :mod:`factorlasso.cv`.
    n_splits : int, default 5
        Expanding-window folds, as in :class:`~factorlasso.LassoModelCV`.
    base_model : LassoModel, optional
        Template. All hyperparameters except ``reg_lambda`` are inherited.
    refit : bool, default True
        Refit at ``best_lambda_`` on the full sample and store as ``best_model_``.
    use_lambda_path : bool, default True
        For the group-LASSO family, derive clustering, signs and adaptive weights once per fold
        and sweep the grid with one canonical form. Defaults True here because this selector
        always sweeps a full grid, where the path solve is the faster route.
    significance : float, default 0.05
        Size of the diagonality test.
    zero_rtol : float, default 1e-4
        Relative tolerance below which a loading counts as zero when computing the degrees of
        freedom, applied to the largest absolute loading. Interior-point solvers do not return
        exact zeros, and an absolute cut is not portable across units; see
        :func:`~factorlasso.effective_sparsity`.
    min_periods : int, default 30
        Minimum overlap for a pair of residual series to enter the statistic.

    Attributes
    ----------
    best_lambda_ : float
        Sparsest penalty whose held-out residuals pass, or the minimiser if none pass.
    best_score_ : float
        Mean fold sphericity at ``best_lambda_``. Lower is closer to diagonal.
    passed_ : bool
        Whether any penalty on the grid passed.
    threshold_ : float
        The fixed chi-square critical value the statistic is compared against.
    diagnostics_ : pd.DataFrame
        Index = lambdas. Columns include ``fold_sphericity_mean``, ``threshold``,
        ``top_eigenvalue``, ``mp_edge``, ``n_above_edge``, ``mean_abs_offdiag``,
        ``raw_offdiag_ss``, ``n_nonzero``, ``n_empty_factors``, ``passes``.
    fold_scores_ : pd.DataFrame
        Index = lambdas, columns = folds, values = per-fold sphericity. NaN where a fold failed.
    missing_factors_ : pd.DataFrame or None
        Residual components above the edge at ``best_lambda_``, populated when ``passed_`` is
        False. Empty frame when the model passes.
    best_model_ : LassoModel or None

    Notes
    -----
    Residuals are formed on the HELD-OUT window using loadings fitted on the training window, so
    the statistic carries no in-sample optimism. The degrees of freedom subtract the loadings the
    fit actually kept, counted at ``zero_rtol``, so a denser model is not rewarded for having
    fewer effective observations left. ``diagnostics_['n_empty_factors']`` flags penalties that
    leave a regressor with no carrier, which makes ``beta' D^-1 beta`` singular.

    See Also
    --------
    factorlasso.LassoModelCV : selection by out-of-sample R².
    factorlasso.diagnose_residuals : the underlying test.
    """

    lambdas: Optional[Sequence[float]] = None
    n_splits: int = 5
    base_model: Optional[LassoModel] = None
    refit: bool = True
    use_lambda_path: bool = True
    significance: float = 0.05
    zero_rtol: float = 1e-4
    min_periods: int = 30

    # ── Fitted state (trailing underscore) ───────────────────────────
    best_lambda_: Optional[float] = None
    best_score_: Optional[float] = None
    passed_: Optional[bool] = None
    threshold_: Optional[float] = None
    diagnostics_: Optional[pd.DataFrame] = field(default=None, repr=False)
    fold_scores_: Optional[pd.DataFrame] = field(default=None, repr=False)
    missing_factors_: Optional[pd.DataFrame] = field(default=None, repr=False)
    best_model_: Optional[LassoModel] = field(default=None, repr=False)

    # ── Core API ─────────────────────────────────────────────────────

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        verbose: bool = False,
    ) -> "LassoModelDiagonalityCV":
        """
        Sweep the grid, score held-out residual diagonality, and select.

        Parameters
        ----------
        x : pd.DataFrame, shape (T, M)
            Regressor (factor) returns.
        y : pd.DataFrame, shape (T, N)
            Response (asset) returns. May contain NaNs.
        verbose : bool, default False
            Forwarded to the estimator; warns on each failed fold.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``x`` and ``y`` do not share an index, or the grid is empty.
        RuntimeError
            If every fold of every penalty failed.
        """
        if isinstance(x, pd.Series):
            x = x.to_frame()
        if isinstance(y, pd.Series):
            y = y.to_frame()
        if not x.index.equals(y.index):
            raise ValueError(
                f"x and y must share the same index: "
                f"x has {len(x)} rows, y has {len(y)} rows"
            )
        lambdas = (
            list(self.lambdas) if self.lambdas is not None else list(DEFAULT_LAMBDA_GRID)
        )
        if not lambdas:
            raise ValueError("lambdas must be non-empty")

        splits = list(expanding_window_splits(len(x), self.n_splits))
        scores = np.full((len(lambdas), len(splits)), np.nan)
        records: dict = {}

        template = self._make_model(lambdas[0])
        use_path = self.use_lambda_path and template.model_type in _GROUP_FAMILY

        for j, (train_idx, test_idx) in enumerate(splits):
            fold_models = self._fit_fold(
                x=x, y=y, train_idx=train_idx, lambdas=lambdas,
                use_path=use_path, verbose=verbose, fold=j,
            )
            if fold_models is None:
                continue                                  # whole column stays NaN
            for i, model in enumerate(fold_models):
                if model is None:
                    continue
                try:
                    diag = self._score_fold(model, x.iloc[test_idx], y.iloc[test_idx])
                except _FOLD_ERRORS as err:
                    if verbose:
                        warnings.warn(
                            f"diagonality fold (lambda={lambdas[i]:.2e}, split={j}) "
                            f"scoring failed: {type(err).__name__}: {err}",
                            RuntimeWarning, stacklevel=2,
                        )
                    continue
                scores[i, j] = diag.sphericity
                records.setdefault(lambdas[i], []).append((diag, model))

        fold_scores = pd.DataFrame(
            scores,
            index=pd.Index(lambdas, name="reg_lambda"),
            columns=pd.RangeIndex(len(splits), name="fold"),
        )
        mean_scores = fold_scores.mean(axis=1, skipna=True)
        if mean_scores.isna().all():
            raise RuntimeError(
                "All folds failed; cannot select a penalty. "
                "Check input data and solver settings."
            )

        self.fold_scores_ = fold_scores
        self.diagnostics_ = self._build_diagnostics(lambdas, mean_scores, records)
        self.threshold_ = float(self.diagnostics_["threshold"].dropna().iloc[0])

        passing = self.diagnostics_.index[self.diagnostics_["passes"].fillna(False)]
        self.passed_ = bool(len(passing) > 0)
        self.best_lambda_ = (
            float(max(passing)) if self.passed_ else float(mean_scores.idxmin())
        )
        self.best_score_ = float(mean_scores.loc[self.best_lambda_])

        if self.refit:
            self.best_model_ = self._make_model(self.best_lambda_).fit(
                x=x, y=y, verbose=verbose,
            )
        self.missing_factors_ = self._report_missing(x, y, verbose=verbose)
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Delegate to the refitted ``best_model_``."""
        if self.best_model_ is None:
            raise RuntimeError(
                "predict() requires refit=True or a manually fitted best_model_"
            )
        return self.best_model_.predict(x)

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Sphericity of the residuals of ``best_model_`` on the supplied sample.

        Lower is closer to diagonal, so this is NOT comparable with
        :meth:`LassoModelCV.score`, where higher is better.
        """
        if self.best_model_ is None:
            raise RuntimeError(
                "score() requires refit=True or a manually fitted best_model_"
            )
        return float(self._score_fold(self.best_model_, x, y).sphericity)

    # ── Helpers ──────────────────────────────────────────────────────

    def _make_model(self, reg_lambda: float) -> LassoModel:
        """Build a fresh LassoModel inheriting hyperparameters from base_model."""
        if self.base_model is None:
            return LassoModel(reg_lambda=reg_lambda)
        params = self.base_model.get_params()
        params["reg_lambda"] = reg_lambda
        return LassoModel(**params)

    def _fit_fold(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        train_idx: np.ndarray,
        lambdas: Sequence[float],
        use_path: bool,
        verbose: bool,
        fold: int,
    ) -> Optional[Sequence[Optional[LassoModel]]]:
        """Fit every penalty on one training window, by path solve or per-lambda loop."""
        if use_path:
            try:
                return self._make_model(lambdas[0]).fit_reg_lambda_path(
                    x=x.iloc[train_idx], y=y.iloc[train_idx],
                    reg_lambdas=list(lambdas), verbose=verbose,
                )
            except _FOLD_ERRORS as err:
                if verbose:
                    warnings.warn(
                        f"diagonality fold (split={fold}) failed during path solve: "
                        f"{type(err).__name__}: {err}",
                        RuntimeWarning, stacklevel=2,
                    )
                    print(
                        f"[LassoModelDiagonalityCV] fold failed (split={fold}): "
                        f"{type(err).__name__}: {err}",
                        file=sys.stderr,
                    )
                return None
        fitted: list = []
        for lam in lambdas:
            model = self._make_model(lam)
            try:
                model.fit(x=x.iloc[train_idx], y=y.iloc[train_idx], verbose=verbose)
                fitted.append(model)
            except _FOLD_ERRORS as err:
                if verbose:
                    warnings.warn(
                        f"diagonality fold (lambda={lam:.2e}, split={fold}) failed: "
                        f"{type(err).__name__}: {err}",
                        RuntimeWarning, stacklevel=2,
                    )
                fitted.append(None)
        return fitted

    def _score_fold(
        self, model: LassoModel, x_test: pd.DataFrame, y_test: pd.DataFrame,
    ) -> ResidualDiagnostics:
        """Residual diagnostics on a held-out window at the fold's loadings."""
        resid = y_test - model.predict(x_test)
        sparsity = effective_sparsity(model.estimated_betas, rtol=self.zero_rtol)
        return diagnose_residuals(
            resid,
            n_fitted_per_asset=sparsity.per_asset,
            significance=self.significance,
            min_periods=self.min_periods,
        )

    def _build_diagnostics(
        self, lambdas: Sequence[float], mean_scores: pd.Series, records: dict,
    ) -> pd.DataFrame:
        """Fold-averaged diagnostics, one row per penalty."""
        rows = []
        for lam in lambdas:
            entries = records.get(lam, [])
            if not entries:
                rows.append({"reg_lambda": lam, "fold_sphericity_mean": np.nan,
                             "passes": False})
                continue
            diags = [d for d, _ in entries]
            models = [m for _, m in entries]
            mean_sphericity = float(mean_scores.loc[lam])
            threshold = float(np.mean([d.threshold for d in diags]))
            rows.append({
                "reg_lambda": lam,
                "fold_sphericity_mean": mean_sphericity,
                "threshold": threshold,
                "top_eigenvalue": float(np.mean([d.top_eigenvalue for d in diags])),
                "mp_edge": float(np.mean([d.mp_edge for d in diags])),
                "n_above_edge": float(np.mean([d.n_above_edge for d in diags])),
                "mean_abs_offdiag": float(np.mean([d.mean_abs_offdiag for d in diags])),
                "raw_offdiag_ss": float(np.mean([d.raw_offdiag_ss for d in diags])),
                "n_nonzero": float(np.mean([
                    effective_sparsity(m.estimated_betas, rtol=self.zero_rtol).n_nonzero
                    for m in models
                ])),
                "n_empty_factors": float(np.mean([
                    len(effective_sparsity(m.estimated_betas,
                                           rtol=self.zero_rtol).empty_factors)
                    for m in models
                ])),
                "passes": bool(
                    mean_sphericity <= threshold
                    and np.mean([d.top_eigenvalue for d in diags])
                    <= np.mean([d.mp_edge for d in diags])
                ),
            })
        return pd.DataFrame(rows).set_index("reg_lambda")

    def _report_missing(
        self, x: pd.DataFrame, y: pd.DataFrame, verbose: bool,
    ) -> pd.DataFrame:
        """Residual components above the edge at the selected penalty."""
        model = self.best_model_
        if model is None:
            model = self._make_model(self.best_lambda_).fit(x=x, y=y, verbose=verbose)
        resid = y - model.predict(x)
        sparsity = effective_sparsity(model.estimated_betas, rtol=self.zero_rtol)
        if self.passed_:
            return pd.DataFrame(columns=["component", "eigenvalue", "series", "loading"])
        return missing_factor_components(
            resid,
            n_fitted_per_asset=sparsity.per_asset,
            min_periods=self.min_periods,
        )
