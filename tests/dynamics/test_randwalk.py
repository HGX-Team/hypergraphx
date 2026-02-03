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


def test_random_walk_length():
    """Test random walk length equals time + 1."""
    np.random.seed(0)
    hg = _make_connected_hypergraph()
    path = random_walk(hg, s=0, time=3)

    assert len(path) == 4


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
