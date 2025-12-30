import numpy as np
import pytest

from hypergraphx import Hypergraph
from hypergraphx.generation.configuration_model import configuration_model


def _make_hypergraph():
    edges = [(0, 1), (1, 2), (0, 1, 2), (2, 3, 4)]
    return Hypergraph(edge_list=edges)


def test_configuration_model_preserves_edge_count():
    """Test configuration model returns a hypergraph with same number of edges."""
    np.random.seed(0)
    hg = _make_hypergraph()

    sampled = configuration_model(hg, n_steps=5, label="edge")

    assert sampled.num_edges() == hg.num_edges()


def test_configuration_model_size_filter():
    """Test configuration model shuffles only specified size."""
    np.random.seed(1)
    hg = _make_hypergraph()

    sampled = configuration_model(hg, n_steps=5, size=2)

    assert set(sampled.get_edges(size=3)) == set(hg.get_edges(size=3))


def test_configuration_model_invalid_args():
    """Test configuration model rejects order+size combo."""
    hg = _make_hypergraph()
    with pytest.raises(ValueError, match="Only one"):
        configuration_model(hg, order=1, size=2)
