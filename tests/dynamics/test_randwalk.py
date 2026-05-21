import numpy as np
import pytest

from hypergraphx import Hypergraph
from hypergraphx.dynamics.randwalk import (
    transition_matrix,
    random_walk,
    RW_stationary_state,
    random_walk_density,
)


def _make_connected_hypergraph():
    return Hypergraph(edge_list=[(0, 1), (1, 2)])


def test_transition_matrix_rows_sum_to_one():
    """Test transition matrix row normalization."""
    hg = _make_connected_hypergraph()
    T = transition_matrix(hg).toarray()

    assert np.allclose(T.sum(axis=1), 1.0)


def test_transition_matrix_handles_non_contiguous_node_ids():
    hg = Hypergraph(edge_list=[(492, 938), (938, 1200), (1200, 5000)])

    T = transition_matrix(hg).toarray()
    _, mapping = hg.binary_incidence_matrix(return_mapping=True)
    node_to_idx = {node: idx for idx, node in mapping.items()}

    assert T.shape == (4, 4)
    assert np.allclose(T.sum(axis=1), 1.0)
    assert T[node_to_idx[492], node_to_idx[938]] == 1.0
    assert T[node_to_idx[938], node_to_idx[492]] == 0.5
    assert T[node_to_idx[938], node_to_idx[1200]] == 0.5
    assert T[node_to_idx[1200], node_to_idx[938]] == 0.5
    assert T[node_to_idx[1200], node_to_idx[5000]] == 0.5
    assert T[node_to_idx[5000], node_to_idx[1200]] == 1.0


def test_random_walk_length():
    """Test random walk length equals time + 1."""
    np.random.seed(0)
    hg = _make_connected_hypergraph()
    path = random_walk(hg, s=0, time=3)

    assert len(path) == 4


def test_random_walk_returns_non_contiguous_node_ids():
    hg = Hypergraph(edge_list=[(492, 938), (938, 1200), (1200, 5000)])

    path = random_walk(hg, s=492, time=5, seed=0)

    assert path[0] == 492
    assert set(path).issubset({492, 938, 1200, 5000})


def test_stationary_state_properties():
    """Test stationary state is a valid distribution."""
    hg = _make_connected_hypergraph()
    pi = RW_stationary_state(hg)
    assert np.all(pi >= 0)
    assert np.isclose(pi.sum(), 1.0)


def test_random_walk_density_normalization():
    """Test density evolution keeps normalization."""
    hg = _make_connected_hypergraph()
    s = np.array([1.0, 0.0, 0.0])
    densities = random_walk_density(hg, s, time=2)

    assert np.allclose([d.sum() for d in densities], 1.0)


def test_random_walk_density_invalid():
    """Test invalid density vector raises."""
    hg = _make_connected_hypergraph()
    with pytest.raises(ValueError, match="probability"):
        random_walk_density(hg, np.array([0.2, 0.2, 0.2]), time=1)


def test_transition_matrix_large_sparse_chain():
    """Smoke test: large N should remain sparse and fast enough."""
    N = 10_000
    edges = [(i, i + 1) for i in range(N - 1)]
    hg = Hypergraph(edge_list=edges)
    T = transition_matrix(hg).tocsr()
    assert T.shape == (N, N)
    assert T.nnz == 2 * (N - 1)
    rowsum = np.asarray(T.sum(axis=1)).ravel()
    assert np.allclose(rowsum, 1.0)
