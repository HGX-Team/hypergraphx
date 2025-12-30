import numpy as np
import pytest

from hypergraphx import Hypergraph
from hypergraphx.measures.degree import (
    degree,
    degree_sequence,
    degree_distribution,
    degree_correlation,
)


def _make_hypergraph():
    edges = [(0, 1), (1, 2), (2, 3), (0, 1, 2), (0, 2, 3)]
    return Hypergraph(edge_list=edges)


def test_degree_basic_and_by_size():
    """Test degree for overall and size-filtered edges."""
    hg = _make_hypergraph()

    assert degree(hg, 0) == 3
    assert degree(hg, 0, size=2) == 1
    assert degree(hg, 0, size=3) == 2


def test_degree_invalid_args():
    """Test degree raises for invalid order/size combination."""
    hg = _make_hypergraph()
    with pytest.raises(ValueError, match="both specified"):
        degree(hg, 0, order=1, size=3)


def test_degree_sequence_and_distribution():
    """Test degree sequence and distribution outputs."""
    hg = _make_hypergraph()

    seq = degree_sequence(hg)
    assert seq[0] == 3
    assert seq[1] == 3

    dist = degree_distribution(hg)
    assert dist == {3: 2, 4: 1, 2: 1}


def test_degree_correlation_matrix_shape_and_values():
    """Test degree correlation matrix has expected shape and no NaNs."""
    hg = _make_hypergraph()

    corr = degree_correlation(hg)
    assert corr.shape == (2, 2)
    assert np.all(np.isfinite(corr))
    assert np.all(corr <= 1.0 + 1e-8)
    assert np.all(corr >= -1.0 - 1e-8)
