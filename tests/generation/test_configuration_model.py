import numpy as np
import pytest

from hypergraphx import Hypergraph
from hypergraphx.exceptions import InvalidParameterError
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


def test_configuration_model_rejects_stub_label():
    """Test configuration model rejects the removed stub label."""
    hg = _make_hypergraph()
    with pytest.raises(InvalidParameterError, match="label must be one of"):
        configuration_model(hg, label="stub")


def test_configuration_model_duplicate_output_merge_default():
    """Test repeated sampled hyperedges are merged by default."""
    hg = Hypergraph(edge_list=[(0, 1), (2, 3), (0, 2), (1, 3)], weighted=False)

    sampled = configuration_model(hg, n_steps=50, label="edge", seed=6)

    assert not sampled.is_weighted()
    assert sampled.num_edges() == 2
    assert set(sampled.get_edges()) == {(0, 2), (1, 3)}


def test_configuration_model_duplicate_output_count():
    """Test repeated sampled hyperedges can be encoded as weights."""
    hg = Hypergraph(edge_list=[(0, 1), (2, 3), (0, 2), (1, 3)], weighted=False)

    sampled = configuration_model(
        hg, n_steps=50, label="edge", seed=6, duplicate_output="count"
    )

    assert sampled.is_weighted()
    assert sampled.num_edges() == 2
    assert sampled.get_weight((0, 2)) == 2
    assert sampled.get_weight((1, 3)) == 2


def test_configuration_model_duplicate_output_error():
    """Test repeated sampled hyperedges can be rejected."""
    hg = Hypergraph(edge_list=[(0, 1), (2, 3), (0, 2), (1, 3)], weighted=False)

    with pytest.raises(InvalidParameterError, match="Repeated sampled hyperedges"):
        configuration_model(
            hg, n_steps=50, label="edge", seed=6, duplicate_output="error"
        )


def test_configuration_model_duplicate_output_count_rejects_weighted_input():
    """Test multiplicity-as-weight output is rejected for weighted inputs."""
    hg = Hypergraph(edge_list=[(0, 1), (1, 2)], weighted=True, weights=[2.0, 3.0])

    with pytest.raises(InvalidParameterError, match="only supported for unweighted"):
        configuration_model(hg, n_steps=5, duplicate_output="count", seed=0)
