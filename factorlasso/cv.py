"""
Cross-validated regularisation for :class:`~factorlasso.LassoModel`.

Provides :class:`LassoModelCV`, which selects ``reg_lambda`` from a grid
using **expanding-window time-series splits**.  Random K-fold CV is
inappropriate for the temporal data this package targets — it leaks
information from the future into training folds and produces optimistic
scores.  Expanding-window CV mirrors how the model is actually used in
production (refit on a growing history, score the next out-of-sample
window).

Examples
--------
>>> import numpy as np, pandas as pd
>>> from factorlasso import LassoModel, LassoModelCV
>>> rng = np.random.default_rng(0)
>>> T, M, N = 200, 3, 4
>>> idx = pd.date_range('2020-01-31', periods=T, freq='MS')
>>> X = pd.DataFrame(rng.standard_normal((T, M)), index=idx,
...                   columns=[f'f{i}' for i in range(M)])
>>> beta = rng.standard_normal((N, M))
>>> Y = pd.DataFrame(X.values @ beta.T + 0.1 * rng.standard_normal((T, N)),
...                   index=idx, columns=[f'y{i}' for i in range(N)])
>>> cv = LassoModelCV(n_splits=4).fit(x=X, y=Y)
>>> isinstance(cv.best_lambda_, float)
True
>>> cv.cv_scores_.shape  # (n_lambdas, n_splits)
(20, 4)
>>> y_hat = cv.predict(X)  # delegates to refitted best_model_
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence, Tuple

import cvxpy as cvx
import numpy as np
import pandas as pd

from factorlasso.lasso_estimator import LassoModel

# Errors we treat as "this fold failed, record NaN and continue".
# Anything else (KeyboardInterrupt, MemoryError, attribute errors from
# bad user kwargs) should propagate to the caller.
_FOLD_ERRORS: Tuple[type, ...] = (
    cvx.error.SolverError,
    cvx.error.DCPError,
    ValueError,
    np.linalg.LinAlgError,
)


# ═══════════════════════════════════════════════════════════════════════
# Time-series splits
# ═══════════════════════════════════════════════════════════════════════

def expanding_window_splits(
    n_samples: int, n_splits: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate expanding-window train / test index splits.

    Mirrors :class:`sklearn.model_selection.TimeSeriesSplit` semantics:
    each successive fold uses an expanding training window and a
    fixed-size test window immediately following it.

    Parameters
    ----------
    n_samples : int
        Total number of observations.
    n_splits : int
        Number of folds.  Must be ``>= 1``.

    Yields
    ------
    (train_idx, test_idx) : tuple of np.ndarray
        Integer position indices into the time axis.

    Raises
    ------
    ValueError
        If ``n_splits < 1`` or ``n_samples`` is too small to produce
        at least one observation per fold.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    test_size = n_samples // (n_splits + 1)
    if test_size < 1:
        raise ValueError(
            f"n_samples={n_samples} is too small for n_splits={n_splits}; "
            f"need at least {n_splits + 1} observations"
        )
    for i in range(n_splits):
        train_end = test_size * (i + 1)
        test_end = train_end + test_size
        yield np.arange(train_end), np.arange(train_end, test_end)


# ═══════════════════════════════════════════════════════════════════════
# Cross-validated estimator
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_LAMBDA_GRID: Tuple[float, ...] = tuple(
    float(x) for x in np.logspace(-6, -1, 20)
)


@dataclass
class LassoModelCV:
    """
    Time-series cross-validated ``reg_lambda`` selection for LassoModel.

    Fits :class:`LassoModel` on every (lambda, fold) combination using
    expanding-window splits, picks the lambda that maximises mean
    out-of-sample R², and (optionally) refits a final model on the full
    dataset.

    Parameters
    ----------
    lambdas : sequence of float, optional
        Regularisation grid.  Default: 20 points log-spaced on ``[1e-6, 1e-1]``.
    n_splits : int, default 5
        Number of expanding-window folds.
    base_model : LassoModel, optional
        Template model.  All hyperparameters except ``reg_lambda`` are
        inherited; if ``None``, a default :class:`LassoModel` is used.
    refit : bool, default True
        After CV, refit a fresh :class:`LassoModel` with ``best_lambda_``
        on the full dataset and store it as ``best_model_``.

    Attributes
    ----------
    best_lambda_ : float
        Lambda with the highest mean fold R².
    best_score_ : float
        Mean R² achieved at ``best_lambda_``.
    cv_scores_ : pd.DataFrame
        Index = lambdas, columns = fold indices ``0..n_splits-1``,
        values = R² on the held-out window.  NaN where a fold's solver
        failed or scored on degenerate data.
    best_model_ : LassoModel or None
        Refitted model (only when ``refit=True``).

    Notes
    -----
    Scoring uses :meth:`LassoModel.score`, which is mean R² across
    response variables.  Higher is better.

    Fold failures in the solver (``cvx.error.SolverError``,
    ``LinAlgError``, ``ValueError``, ``cvx.error.DCPError``) are caught
    and recorded as NaN in ``cv_scores_``.  Any other exception —
    ``KeyboardInterrupt``, ``MemoryError``, ``AttributeError`` from
    an unexpected kwarg — propagates so that real bugs are not silently
    swallowed by the CV loop.
    """

    lambdas: Optional[Sequence[float]] = None
    n_splits: int = 5
    base_model: Optional[LassoModel] = None
    refit: bool = True

    # ── Fitted state (trailing underscore) ───────────────────────────
    best_lambda_: Optional[float] = None
    best_score_: Optional[float] = None
    cv_scores_: Optional[pd.DataFrame] = field(default=None, repr=False)
    best_model_: Optional[LassoModel] = field(default=None, repr=False)

    # ── Core API ─────────────────────────────────────────────────────

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        verbose: bool = False,
    ) -> "LassoModelCV":
        """
        Run cross-validation and (optionally) refit the best model.

        Parameters
        ----------
        x : pd.DataFrame, shape (T, M)
            Regressor (factor) returns.
        y : pd.DataFrame, shape (T, N)
            Response (asset) returns.  May contain NaNs.
        verbose : bool, default False
            If True, forwarded to :meth:`LassoModel.fit` and a warning
            is emitted for every fold that fails.

        Returns
        -------
        self
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

        lambdas = list(self.lambdas) if self.lambdas is not None else list(DEFAULT_LAMBDA_GRID)
        if not lambdas:
            raise ValueError("lambdas must be non-empty")

        splits = list(expanding_window_splits(len(x), self.n_splits))

        scores = np.full((len(lambdas), len(splits)), np.nan)
        for i, lam in enumerate(lambdas):
            for j, (tr, te) in enumerate(splits):
                model = self._make_model(lam)
                try:
                    model.fit(
                        x=x.iloc[tr], y=y.iloc[tr], verbose=verbose,
                    )
                    scores[i, j] = model.score(x.iloc[te], y.iloc[te])
                except _FOLD_ERRORS as err:
                    if verbose:
                        warnings.warn(
                            f"CV fold (lambda={lam:.2e}, split={j}) failed: "
                            f"{type(err).__name__}: {err}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        print(
                            f"[LassoModelCV] fold failed "
                            f"(lambda={lam:.2e}, split={j}): "
                            f"{type(err).__name__}: {err}",
                            file=sys.stderr,
                        )
                    # Fall through with NaN score

        cv_scores = pd.DataFrame(
            scores,
            index=pd.Index(lambdas, name="reg_lambda"),
            columns=pd.RangeIndex(len(splits), name="fold"),
        )
        mean_scores = cv_scores.mean(axis=1, skipna=True)
        if mean_scores.isna().all():
            raise RuntimeError(
                "All CV folds failed; cannot select a best lambda. "
                "Check input data and solver settings."
            )

        best_lambda = float(mean_scores.idxmax())
        self.cv_scores_ = cv_scores
        self.best_lambda_ = best_lambda
        self.best_score_ = float(mean_scores.loc[best_lambda])

        if self.refit:
            self.best_model_ = self._make_model(best_lambda).fit(
                x=x, y=y, verbose=verbose,
            )

        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Delegate to the refitted ``best_model_``."""
        if self.best_model_ is None:
            raise RuntimeError(
                "predict() requires refit=True or a manually fitted best_model_"
            )
        return self.best_model_.predict(x)

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        """Delegate to the refitted ``best_model_``."""
        if self.best_model_ is None:
            raise RuntimeError(
                "score() requires refit=True or a manually fitted best_model_"
            )
        return self.best_model_.score(x, y)

    # ── Helpers ──────────────────────────────────────────────────────

    def _make_model(self, reg_lambda: float) -> LassoModel:
        """Build a fresh LassoModel inheriting hyperparameters from base_model."""
        if self.base_model is None:
            return LassoModel(reg_lambda=reg_lambda)
        params = self.base_model.get_params()
        params["reg_lambda"] = reg_lambda
        return LassoModel(**params)
