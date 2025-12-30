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
    """Test stationary state raises on singular system for small graphs."""
    hg = _make_connected_hypergraph()
    with pytest.raises(np.linalg.LinAlgError):
        RW_stationary_state(hg)


def test_random_walk_density_normalization():
    """Test density evolution keeps normalization."""
    hg = _make_connected_hypergraph()
    s = np.array([1.0, 0.0, 0.0])
    densities = random_walk_density(hg, s, time=2)

    assert np.allclose([d.sum() for d in densities], 1.0)


def test_random_walk_density_invalid():
    """Test invalid density vector raises assertion."""
    hg = _make_connected_hypergraph()
    with pytest.raises(AssertionError, match="probability"):
        random_walk_density(hg, np.array([0.2, 0.2, 0.2]), time=1)
