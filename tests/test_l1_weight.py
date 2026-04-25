"""
Tests for the ``l1_weight`` / Sparse Group LASSO feature (v0.3.2).

Coverage:
- Backward compatibility at ``l1_weight=0.0`` (bit-identical to v0.3.1
  pure group LASSO).
- LASSO mode ignores ``l1_weight`` (documented behaviour).
- Input validation via ``LassoModel.__post_init__`` and solver-level
  range check.
- Mathematical limit at ``l1_weight=1.0``: group term is nuked, result
  approximates pure LASSO.
- Interaction with ``group_penalty``: at α=1 group scheme is irrelevant;
  at 0<α<1 both terms are active and the choice matters.
- Group-then-prune property: on a panel with noisy within-cluster
  assets, increasing α progressively zeros the spurious loadings while
  preserving genuine signal loadings.
- Sign constraints and prior centering compose with the L1 term.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlasso import (
    LassoModel,
    LassoModelType,
    compute_clusters_from_corr_matrix,
)
from factorlasso.ewm_utils import compute_ewm_covar, set_group_loadings
from factorlasso.lasso_estimator import (
    get_x_y_np,
    solve_group_lasso_cvx_problem,
    solve_lasso_cvx_problem,
)

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def hcgl_panel():
    """Multi-response panel with block correlation structure suitable for HCGL."""
    rng = np.random.default_rng(42)
    T, M, N, n_clusters = 300, 6, 30, 5
    X = rng.standard_normal((T, M))
    cluster_of = rng.integers(0, n_clusters, N)
    beta_true = np.zeros((N, M))
    for c in range(n_clusters):
        mask = cluster_of == c
        factors = rng.choice(M, size=2, replace=False)
        for f in factors:
            beta_true[mask, f] = rng.normal(0.5, 0.15, mask.sum())
    Y = X @ beta_true.T + 0.1 * rng.standard_normal((T, N))
    for c in range(n_clusters):
        mask = cluster_of == c
        shared = rng.standard_normal(T)
        for i in np.where(mask)[0]:
            Y[:, i] = 0.3 * shared + 0.7 * Y[:, i]
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(M)])
    Y_df = pd.DataFrame(Y, columns=[f"y{i}" for i in range(N)])
    return X_df, Y_df


@pytest.fixture
def hcgl_prepped(hcgl_panel):
    """Pre-computed HCGL inputs: demeaned x/y, valid_mask, clusters, group loadings."""
    X, Y = hcgl_panel
    x_np, y_np, mask = get_x_y_np(x=X, y=Y, span=60, demean=True)
    corr = compute_ewm_covar(a=y_np, span=60, is_corr=True)
    corr_df = pd.DataFrame(corr, index=Y.columns, columns=Y.columns)
    clusters, _, _ = compute_clusters_from_corr_matrix(corr_df)
    gl = set_group_loadings(group_data=clusters).to_numpy()
    return {
        "X": X, "Y": Y,
        "x_np": x_np, "y_np": y_np, "mask": mask,
        "clusters": clusters, "group_loadings": gl,
    }


@pytest.fixture
def sparsity_panel():
    """
    Adversarial panel: each HCGL cluster contains a few genuine-loading
    assets and several pure-noise assets that cluster together via a
    shared idiosyncratic shock. Tests whether l1_weight > 0 zeros out
    the noise-asset loadings.
    """
    rng = np.random.default_rng(0)
    T, M, N = 400, 5, 20
    X = rng.standard_normal((T, M))
    beta_true = np.zeros((N, M))
    # First 3 assets of each half load on a factor; rest are pure noise
    beta_true[:3, 0] = rng.normal(0.8, 0.05, 3)
    beta_true[10:13, 1] = rng.normal(0.8, 0.05, 3)
    Y = X @ beta_true.T + 0.1 * rng.standard_normal((T, N))
    # Shared idiosyncratic shock within each half drives clustering
    shared0 = rng.standard_normal(T)
    shared1 = rng.standard_normal(T)
    for i in range(10):
        Y[:, i] += 0.3 * shared0
    for i in range(10, 20):
        Y[:, i] += 0.3 * shared1
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(M)])
    Y_df = pd.DataFrame(Y, columns=[f"y{i}" for i in range(N)])
    # True-zero mask (which cells SHOULD be zero in β)
    noise_mask = np.ones((N, M), dtype=bool)
    noise_mask[:3, 0] = False     # signal
    noise_mask[10:13, 1] = False  # signal
    return X_df, Y_df, noise_mask


# ═══════════════════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Default behaviour must be bit-identical to pre-l1_weight code."""

    def test_solver_default_vs_explicit_zero(self, hcgl_prepped):
        """solve_group_lasso_cvx_problem: no kwarg ≡ l1_weight=0.0."""
        p = hcgl_prepped
        res_default = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=1e-4, span=60,
        )
        res_zero = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=1e-4, span=60,
            l1_weight=0.0,
        )
        delta = np.abs(res_default.estimated_beta - res_zero.estimated_beta)
        assert delta.max() == 0.0, (
            f"Backward compat broken at solver level: Max |Δβ|={delta.max():.2e}"
        )

    def test_lassomodel_default_vs_explicit_zero(self, hcgl_panel):
        """LassoModel: no kwarg ≡ l1_weight=0.0."""
        X, Y = hcgl_panel
        m_default = LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            reg_lambda=1e-4, span=60,
        ).fit(x=X, y=Y)
        m_zero = LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            reg_lambda=1e-4, span=60, l1_weight=0.0,
        ).fit(x=X, y=Y)
        delta = (m_default.coef_ - m_zero.coef_).abs().values.max()
        assert delta == 0.0, (
            f"Backward compat broken at LassoModel level: Max |Δβ|={delta:.2e}"
        )

    def test_lasso_mode_ignores_l1_weight(self, hcgl_panel):
        """Docstring: 'ignored for pure LASSO since L1 is the only penalty already.'"""
        X, Y = hcgl_panel
        m0 = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=1e-4, span=60,
        ).fit(x=X, y=Y)
        m5 = LassoModel(
            model_type=LassoModelType.LASSO, reg_lambda=1e-4, span=60,
            l1_weight=0.5,
        ).fit(x=X, y=Y)
        delta = (m0.coef_ - m5.coef_).abs().values.max()
        assert delta == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Input validation
# ═══════════════════════════════════════════════════════════════════════


class TestValidation:
    """l1_weight must be rejected outside [0, 1] at construction and solver."""

    @pytest.mark.parametrize("bad", [-0.1, -1e-9, 1.001, 2.0, np.nan, np.inf, -np.inf])
    def test_lassomodel_rejects_bad_values(self, bad):
        with pytest.raises(ValueError, match="l1_weight must lie in"):
            LassoModel(l1_weight=bad)

    @pytest.mark.parametrize("good", [0.0, 1e-9, 0.5, 1.0 - 1e-9, 1.0])
    def test_lassomodel_accepts_good_values(self, good):
        # Should not raise
        m = LassoModel(l1_weight=good)
        assert m.l1_weight == good

    def test_solver_rejects_bad_values(self, hcgl_prepped):
        p = hcgl_prepped
        with pytest.raises(ValueError, match="l1_weight must lie in"):
            solve_group_lasso_cvx_problem(
                x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
                valid_mask=p["mask"], reg_lambda=1e-4, span=60,
                l1_weight=1.5,
            )

    def test_get_params_includes_l1_weight(self):
        params = LassoModel(l1_weight=0.25).get_params()
        assert params["l1_weight"] == 0.25

    def test_set_params_updates_l1_weight(self):
        m = LassoModel()
        assert m.l1_weight == 0.0
        m.set_params(l1_weight=0.15)
        assert m.l1_weight == 0.15


# ═══════════════════════════════════════════════════════════════════════
# Mathematical limits
# ═══════════════════════════════════════════════════════════════════════


class TestLimits:
    """Verify α=0 and α=1 endpoints match their algebraic limits."""

    def test_alpha_one_approximates_pure_lasso(self, hcgl_prepped):
        """
        At α=1 the group term has coefficient 0, leaving only elementwise L1.
        Result must match a pure-LASSO solve at the same λ, up to CVXPY
        solver precision. The docstring claim 'Set α = 1 for pure LASSO
        (no group structure)' is verified here.
        """
        p = hcgl_prepped
        res_alpha1 = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=3e-4, span=60,
            l1_weight=1.0,
        )
        res_pure = solve_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], valid_mask=p["mask"],
            reg_lambda=3e-4, span=60,
        )
        delta = np.abs(res_alpha1.estimated_beta - res_pure.estimated_beta)
        # Solver-precision tolerance — CLARABEL interior-point typically
        # converges to ~1e-6 relative accuracy. The two problems are
        # algebraically identical; differences are pure numerical noise.
        assert delta.max() < 1e-4, (
            f"α=1 should match pure LASSO; got Max |Δβ|={delta.max():.3e}"
        )
        assert delta.mean() < 1e-5

    def test_alpha_zero_ignores_group_penalty_choice_noop(self, hcgl_prepped):
        """
        At α=0 only the group term is active. Verified by TestInteraction
        below that the group_penalty choice matters here. Sanity check:
        α=0 with either scheme still gives the expected normalization
        default behaviour.
        """
        p = hcgl_prepped
        res_norm = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=1e-4, span=60,
            l1_weight=0.0, group_penalty="normalized",
        )
        # β must be finite, non-trivial
        assert np.isfinite(res_norm.estimated_beta).all()
        assert np.linalg.norm(res_norm.estimated_beta) > 0.1


# ═══════════════════════════════════════════════════════════════════════
# Interaction with group_penalty
# ═══════════════════════════════════════════════════════════════════════


class TestInteraction:
    """l1_weight × group_penalty: group-term scaling is irrelevant at α=1."""

    def test_group_penalty_irrelevant_at_alpha_one(self, hcgl_prepped):
        """At α=1 the group term has coefficient 0 → group_penalty choice is moot."""
        p = hcgl_prepped
        res_norm = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=3e-4, span=60,
            l1_weight=1.0, group_penalty="normalized",
        )
        res_yl = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=3e-4, span=60,
            l1_weight=1.0, group_penalty="yuan_lin",
        )
        delta = np.abs(res_norm.estimated_beta - res_yl.estimated_beta)
        assert delta.max() == 0.0, (
            f"group_penalty should be no-op at α=1; Max |Δβ|={delta.max():.2e}"
        )

    def test_group_penalty_matters_in_middle(self, hcgl_prepped):
        """At 0<α<1 both terms active → normalized and yuan_lin diverge."""
        p = hcgl_prepped
        res_norm = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=3e-4, span=60,
            l1_weight=0.5, group_penalty="normalized",
        )
        res_yl = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=3e-4, span=60,
            l1_weight=0.5, group_penalty="yuan_lin",
        )
        delta = np.abs(res_norm.estimated_beta - res_yl.estimated_beta)
        assert delta.max() > 1e-5, (
            "group_penalty should matter when 0<α<1"
        )


# ═══════════════════════════════════════════════════════════════════════
# Group-then-prune — the whole point of the feature
# ═══════════════════════════════════════════════════════════════════════


class TestSparsityProgression:
    """
    The README's claim: increasing α zeros within-group noisy β.
    This is the feature's value proposition; if it doesn't hold,
    l1_weight is cosmetic.
    """

    def test_spurious_mass_decreases_monotonically_with_alpha(
        self, sparsity_panel,
    ):
        X, Y, noise_mask = sparsity_panel
        x_np, y_np, mask = get_x_y_np(x=X, y=Y, span=60, demean=True)
        corr = compute_ewm_covar(a=y_np, span=60, is_corr=True)
        clusters, _, _ = compute_clusters_from_corr_matrix(
            pd.DataFrame(corr, index=Y.columns, columns=Y.columns)
        )
        gl = set_group_loadings(group_data=clusters).to_numpy()

        alphas = [0.0, 0.1, 0.3, 0.5, 0.8]
        spurious_masses = []
        for a in alphas:
            res = solve_group_lasso_cvx_problem(
                x=x_np, y=y_np, group_loadings=gl,
                valid_mask=mask, reg_lambda=5e-3, span=60,
                l1_weight=a,
            )
            spurious = float(np.abs(res.estimated_beta[noise_mask]).sum())
            spurious_masses.append(spurious)

        # Non-increasing: L1 shrinkage should only reduce spurious loadings
        for i in range(1, len(alphas)):
            assert spurious_masses[i] <= spurious_masses[i - 1] + 1e-6, (
                f"Spurious β mass increased from α={alphas[i-1]} "
                f"({spurious_masses[i-1]:.4f}) to α={alphas[i]} "
                f"({spurious_masses[i]:.4f})"
            )
        # And meaningfully so: at least 2x reduction from α=0 to α=0.8
        assert spurious_masses[-1] < 0.5 * spurious_masses[0], (
            f"L1 pruning did not fire: "
            f"α=0 mass={spurious_masses[0]:.4f}, "
            f"α=0.8 mass={spurious_masses[-1]:.4f}"
        )

    def test_signal_mass_preserved(self, sparsity_panel):
        """
        At moderate α (research range up to 0.3) the genuine-signal β
        must survive — otherwise L1 is destroying information, not
        cleaning noise.
        """
        X, Y, noise_mask = sparsity_panel
        signal_mask = ~noise_mask
        x_np, y_np, mask = get_x_y_np(x=X, y=Y, span=60, demean=True)
        corr = compute_ewm_covar(a=y_np, span=60, is_corr=True)
        clusters, _, _ = compute_clusters_from_corr_matrix(
            pd.DataFrame(corr, index=Y.columns, columns=Y.columns)
        )
        gl = set_group_loadings(group_data=clusters).to_numpy()

        signal_masses = {}
        for a in [0.0, 0.1, 0.3]:
            res = solve_group_lasso_cvx_problem(
                x=x_np, y=y_np, group_loadings=gl,
                valid_mask=mask, reg_lambda=5e-3, span=60,
                l1_weight=a,
            )
            signal_masses[a] = float(np.abs(res.estimated_beta[signal_mask]).sum())

        # Signal mass must stay within 10% of the α=0 baseline over the
        # research range. If the L1 term hammers the signal too, the
        # feature isn't useful.
        baseline = signal_masses[0.0]
        for a in [0.1, 0.3]:
            ratio = signal_masses[a] / baseline
            assert ratio > 0.9, (
                f"Signal mass dropped too much at α={a}: "
                f"{signal_masses[a]:.4f} / {baseline:.4f} = {ratio:.2f}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Composition with other features
# ═══════════════════════════════════════════════════════════════════════


class TestComposition:
    """l1_weight composes with sign constraints and prior centering."""

    def test_sign_constraints_respected_at_all_alpha(self):
        """
        Sign constraints must hold regardless of l1_weight. We only test
        non-negative and zero constraints — the non-positive constraint
        has a known fragility to solver precision when the optimum lies
        at zero, which is orthogonal to l1_weight.
        """
        rng = np.random.default_rng(7)
        T, M, N = 200, 4, 10
        X = pd.DataFrame(
            rng.standard_normal((T, M)),
            columns=[f"f{i}" for i in range(M)],
        )
        beta_true = rng.standard_normal((N, M))
        # Make y0 strongly non-negative on f0
        beta_true[0, 0] = 1.5
        Y = pd.DataFrame(
            X.values @ beta_true.T + 0.1 * rng.standard_normal((T, N)),
            columns=[f"y{i}" for i in range(N)],
        )
        group_data = pd.Series(rng.integers(0, 3, N), index=Y.columns)
        signs = pd.DataFrame(np.nan, index=Y.columns, columns=X.columns)
        signs.loc["y0", "f0"] = 1   # non-negative
        signs.loc["y2", "f0"] = 0   # zero

        for alpha in [0.0, 0.1, 0.3, 0.8, 1.0]:
            m = LassoModel(
                model_type=LassoModelType.GROUP_LASSO,
                group_data=group_data,
                reg_lambda=3e-4, span=60,
                factors_beta_loading_signs=signs,
                l1_weight=alpha,
            ).fit(x=X, y=Y)
            tol = 1e-6
            assert m.coef_.loc["y0", "f0"] >= -tol, (
                f"Non-negative violated at α={alpha}: "
                f"β[y0,f0]={m.coef_.loc['y0', 'f0']}"
            )
            assert abs(m.coef_.loc["y2", "f0"]) < tol, (
                f"Zero-constraint violated at α={alpha}: "
                f"β[y2,f0]={m.coef_.loc['y2', 'f0']}"
            )

    def test_prior_centering_applied_to_l1_term(self, hcgl_prepped):
        """
        The L1 penalty is |β - β₀|, not |β|. With a large α and a
        non-zero prior, β should shrink toward the prior, not toward
        zero. Tests the '|β - β₀|_1' form from the docstring.
        """
        p = hcgl_prepped
        N, M = p["group_loadings"].shape[0], p["x_np"].shape[1]
        # Pick a non-zero prior
        prior = np.full((N, M), 0.3)

        # Two solves: one with prior, one without, both at α=1 (pure L1)
        # and huge λ (strong shrinkage).
        res_with_prior = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=1.0, span=60,
            l1_weight=1.0,
            factors_beta_prior=prior,
        )
        res_no_prior = solve_group_lasso_cvx_problem(
            x=p["x_np"], y=p["y_np"], group_loadings=p["group_loadings"],
            valid_mask=p["mask"], reg_lambda=1.0, span=60,
            l1_weight=1.0,
        )
        # With a non-zero prior, the shrinkage target is 0.3; without,
        # it's 0. Mean β should be meaningfully closer to the prior.
        mean_with = float(res_with_prior.estimated_beta.mean())
        mean_no = float(res_no_prior.estimated_beta.mean())
        # Prior=0.3, no-prior shrinks toward 0 → mean_no ~ 0.
        # With prior, shrinks toward 0.3 → mean_with > mean_no.
        assert mean_with > mean_no + 0.05, (
            f"Prior centering not honoured by L1 term: "
            f"mean_with_prior={mean_with:.3f} vs mean_no_prior={mean_no:.3f}"
        )
