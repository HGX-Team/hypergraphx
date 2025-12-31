import math
import pytest
import numpy as np
from scipy.stats import spearmanr

from hypergraphx.generation.scale_free import scale_free_hypergraph


def _iter_all_edges(hg):
    """Best-effort iterator over all hyperedges as tuples."""
    if hasattr(hg, "edges"):
        try:
            for e in hg.edges:
                yield tuple(e)
            return
        except TypeError:
            pass

    if hasattr(hg, "get_edges"):
        # try no-arg form
        try:
            for e in hg.get_edges():
                yield tuple(e)
            return
        except TypeError:
            pass

    raise AttributeError("Cannot iterate edges: expected hg.edges or hg.get_edges().")


def _edges_of_size(hg, size: int):
    """Return all hyperedges of given cardinality as sorted tuples."""
    if hasattr(hg, "get_edges"):
        # common patterns: get_edges(size=...), get_edges(size)
        try:
            edges = hg.get_edges(size=size)
        except TypeError:
            edges = hg.get_edges(size)
        return [tuple(e) for e in edges]

    # fallback: filter
    return [e for e in _iter_all_edges(hg) if len(e) == size]


def _sizes_present(hg):
    """Return set of hyperedge cardinalities present."""
    if hasattr(hg, "get_sizes"):
        return set(hg.get_sizes())
    return {len(e) for e in _iter_all_edges(hg)}


def _edges_by_size_as_sets(hg, sizes):
    return {s: set(_edges_of_size(hg, s)) for s in sizes}


def _spearman_from_gaussian_rho(rho: float) -> float:
    """
    For (X,Y) jointly normal with Pearson corr rho, the population Spearman corr is:
        rho_S = (6/pi) * asin(rho/2).
    """
    return (6.0 / math.pi) * math.asin(rho / 2.0)


def _activities_from_seed(
    num_nodes: int, sizes: list[int], alpha_by_size, rho: float, seed: int
):
    """
    Recompute the hidden-variable layer activities/fitness sequences w_{i,s} used by the generator.
    This isolates the inter-layer Spearman control from hyperedge sampling noise.
    """
    sizes = sorted(sizes)
    m = len(sizes)

    if isinstance(alpha_by_size, dict):
        alpha_for = {int(k): float(v) for k, v in alpha_by_size.items()}
    else:
        alpha_for = {s: float(alpha_by_size) for s in sizes}

    if m > 1:
        C = np.full((m, m), rho, dtype=float)
        np.fill_diagonal(C, 1.0)
        L = np.linalg.cholesky(C)
    else:
        L = np.array([[1.0]], dtype=float)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((num_nodes, m))
    Y = Z @ L.T

    w = {}
    for j, s in enumerate(sizes):
        alpha = alpha_for[s]
        scores = Y[:, j]

        # rank 1 = largest score
        order = np.argsort(scores)[::-1]
        ranks = np.empty(num_nodes, dtype=float)
        ranks[order] = np.arange(1, num_nodes + 1, dtype=float)

        w[s] = ranks ** (-alpha)
    return w


@pytest.fixture
def basic_params():
    return dict(
        num_nodes=200,
        edges_by_size={2: 300, 3: 150, 4: 80},
        alpha_by_size=1.0,
        rho=0.6,
        seed=42,
    )


def test_hyperedge_cardinalities_and_counts(basic_params):
    """
    Each s-uniform layer has the requested number of hyperedges; each hyperedge
    has the correct cardinality and valid node labels.
    """
    p = dict(basic_params)
    p["seed"] = 42
    hg = scale_free_hypergraph(**p)

    requested_sizes = set(p["edges_by_size"].keys())
    assert _sizes_present(hg) == requested_sizes

    n = p["num_nodes"]
    for s, m in p["edges_by_size"].items():
        edges = _edges_of_size(hg, s)
        assert len(edges) == m
        for e in edges:
            assert len(e) == s
            assert tuple(e) == tuple(sorted(e))
            assert all(0 <= int(v) < n for v in e)


def test_simple_hypergraph_no_parallel_hyperedges(basic_params):
    """
    With enforce_unique_edges=True, there are no parallel hyperedges within each s-layer.
    """
    p = dict(basic_params)
    p["seed"] = 42
    p["enforce_unique_edges"] = True
    hg = scale_free_hypergraph(**p)

    for s, m in p["edges_by_size"].items():
        edges = _edges_of_size(hg, s)
        assert len(edges) == m
        assert len(set(edges)) == m


def test_reproducible_given_seed(basic_params):
    p = dict(basic_params)
    p["seed"] = 123
    hg1 = scale_free_hypergraph(**p)
    hg2 = scale_free_hypergraph(**p)

    sizes = list(p["edges_by_size"].keys())
    assert _edges_by_size_as_sets(hg1, sizes) == _edges_by_size_as_sets(hg2, sizes)


def test_different_seed_changes_realization(basic_params):
    p1 = dict(basic_params)
    p2 = dict(basic_params)
    p1["seed"] = 1
    p2["seed"] = 2

    hg1 = scale_free_hypergraph(**p1)
    hg2 = scale_free_hypergraph(**p2)

    sizes = list(basic_params["edges_by_size"].keys())
    e1 = _edges_by_size_as_sets(hg1, sizes)
    e2 = _edges_by_size_as_sets(hg2, sizes)

    assert any(e1[s] != e2[s] for s in sizes)


def test_invalid_edges_by_size_empty():
    with pytest.raises(ValueError, match="edges_by_size must be a non-empty dict"):
        scale_free_hypergraph(
            num_nodes=10, edges_by_size={}, alpha_by_size=1.0, rho=0.0, seed=0
        )


def test_invalid_hyperedge_size_exceeds_num_nodes():
    with pytest.raises(ValueError, match="cannot exceed num_nodes"):
        scale_free_hypergraph(
            num_nodes=5, edges_by_size={6: 1}, alpha_by_size=1.0, rho=0.0, seed=0
        )


def test_invalid_negative_num_hyperedges():
    with pytest.raises(ValueError, match="nonnegative int"):
        scale_free_hypergraph(
            num_nodes=10, edges_by_size={2: -1}, alpha_by_size=1.0, rho=0.0, seed=0
        )


def test_impossible_simple_hypergraph_raises():
    # For n=6, number of distinct 2-hyperedges is C(6,2)=15
    with pytest.raises(ValueError, match="Requested .* unique hyperedges"):
        scale_free_hypergraph(
            num_nodes=6,
            edges_by_size={2: 16},
            alpha_by_size=1.0,
            rho=0.0,
            seed=0,
            enforce_unique_edges=True,
        )


def test_alpha_by_size_missing_key_raises():
    with pytest.raises(ValueError, match="Missing alpha"):
        scale_free_hypergraph(
            num_nodes=50,
            edges_by_size={2: 10, 3: 10},
            alpha_by_size={2: 1.0},  # missing size-3 exponent
            rho=0.2,
            seed=0,
        )


def test_alpha_nonpositive_raises():
    with pytest.raises(ValueError, match="alpha must be > 0"):
        scale_free_hypergraph(
            num_nodes=50,
            edges_by_size={2: 10, 3: 10},
            alpha_by_size=0.0,
            rho=0.2,
            seed=0,
        )


def test_rho_out_of_range_for_equicorrelation():
    # With 3 sizes, m=3 => rho must be >= -1/(m-1) = -0.5
    with pytest.raises(ValueError, match=r"rho must be in \["):
        scale_free_hypergraph(
            num_nodes=50,
            edges_by_size={2: 10, 3: 10, 4: 10},
            alpha_by_size=1.0,
            rho=-0.6,
            seed=0,
        )
    with pytest.raises(ValueError, match=r"rho must be in \["):
        scale_free_hypergraph(
            num_nodes=50,
            edges_by_size={2: 10, 3: 10, 4: 10},
            alpha_by_size=1.0,
            rho=1.1,
            seed=0,
        )


def test_inter_layer_activity_spearman_matches_gaussian_copula_theory():
    """
    Hidden-variable test: inter-layer Spearman rank correlation of activities.

    Activities are generated as a monotone rank–power transform of latent Gaussian scores,
    so Spearman(activity_a, activity_b) should match the Gaussian-copula prediction:
        rho_S = (6/pi) * asin(rho/2),
    up to finite-sample fluctuations.
    """
    num_nodes = 5000
    sizes = [2, 3]
    alpha = 1.2
    seed = 123

    for rho in (0.7, -0.4):
        w = _activities_from_seed(
            num_nodes=num_nodes,
            sizes=sizes,
            alpha_by_size=alpha,
            rho=rho,
            seed=seed,
        )
        r_emp, _ = spearmanr(w[2], w[3])
        r_theory = _spearman_from_gaussian_rho(rho)

        assert math.copysign(1.0, r_emp) == math.copysign(1.0, r_theory)
        assert abs(r_emp - r_theory) < 0.03
