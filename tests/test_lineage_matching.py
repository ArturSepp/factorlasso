"""Independent oracle tests for deterministic lineage bipartite matching."""

from itertools import combinations, permutations

import numpy as np
import pytest

from factorlasso.cluster_lineage import solve_max_weight_matching


def _matching_weight(matching: dict[int, int], weights: dict[tuple[int, int], float]) -> float:
    """Return the unperturbed total weight of one matching."""
    return sum(weights[(left, right)] for left, right in matching.items())


def _brute_force_max_weight(
        n_left: int,
        n_right: int,
        weights: dict[tuple[int, int], float],
) -> float:
    """Return the exact partial-matching optimum by exhaustive enumeration."""
    best = 0.0
    for cardinality in range(1, min(n_left, n_right) + 1):
        for left_nodes in combinations(range(n_left), cardinality):
            for right_nodes in combinations(range(n_right), cardinality):
                for permuted_right in permutations(right_nodes):
                    pairs = zip(left_nodes, permuted_right)
                    values = [weights.get(pair) for pair in pairs]
                    if all(value is not None for value in values):
                        best = max(best, sum(values))
    return best


def test_solver_matches_brute_force_on_seeded_sparse_panels() -> None:
    """The solver attains the exact optimum on more than 100 random panels up to 6x6."""
    rng = np.random.default_rng(20260811)
    for _ in range(120):
        n_left = int(rng.integers(1, 7))
        n_right = int(rng.integers(1, 7))
        weights = {
            (left, right): float(rng.integers(1, 10_000))
            for left in range(n_left)
            for right in range(n_right)
            if rng.random() < 0.55
        }
        edges = [(left, right, weight) for (left, right), weight in weights.items()]

        matching = solve_max_weight_matching(n_left, n_right, edges)

        assert len(set(matching.values())) == len(matching)
        assert _matching_weight(matching, weights) == pytest.approx(
            _brute_force_max_weight(n_left, n_right, weights), abs=0.0
        )


def test_solver_reproduces_exact_ties_deterministically() -> None:
    """The documented perturbation and traversal order make exact ties reproducible."""
    edges = [(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)]

    first = solve_max_weight_matching(2, 2, edges)

    assert first == {0: 1, 1: 0}
    assert all(solve_max_weight_matching(2, 2, edges) == first for _ in range(10))


def test_solver_matches_networkx_reference_objective() -> None:
    """Development installs reproduce the former NetworkX oracle's total weight."""
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(614)
    for _ in range(40):
        n_left, n_right = 8, 9
        weights = {
            (left, right): float(rng.integers(1, 100_000))
            for left in range(n_left)
            for right in range(n_right)
            if rng.random() < 0.35
        }
        edges = [(left, right, weight) for (left, right), weight in weights.items()]
        graph = nx.Graph()
        graph.add_weighted_edges_from(
            (("L", left), ("R", right), weight) for left, right, weight in edges
        )
        reference = nx.algorithms.matching.max_weight_matching(graph)
        reference_weight = sum(graph[left][right]["weight"] for left, right in reference)

        matching = solve_max_weight_matching(n_left, n_right, edges)

        assert _matching_weight(matching, weights) == pytest.approx(reference_weight, abs=0.0)


@pytest.mark.parametrize(
    ("n_left", "n_right", "edges", "message"),
    [
        (-1, 1, [], "non-negative"),
        (1, 1, [(1, 0, 1.0)], "left index"),
        (1, 1, [(0, 1, 1.0)], "right index"),
        (1, 1, [(0, 0, 0.0)], "positive"),
        (1, 1, [(0, 0, np.inf)], "finite"),
        (1, 1, [(0, 0, 1.0), (0, 0, 2.0)], "duplicate"),
    ],
)
def test_solver_rejects_invalid_graphs(
        n_left: int,
        n_right: int,
        edges: list[tuple[int, int, float]],
        message: str,
) -> None:
    """Invalid dimensions, endpoints, weights and duplicate pairs fail explicitly."""
    with pytest.raises(ValueError, match=message):
        solve_max_weight_matching(n_left, n_right, edges)
