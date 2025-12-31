import numpy as np
import pytest

from hypergraphx import Hypergraph
from hypergraphx.measures.eigen_centralities import (
    CEC_centrality,
    ZEC_centrality,
    HEC_centrality,
)


def _make_uniform_connected():
    edges = [(0, 1, 2), (1, 2, 3)]
    return Hypergraph(edge_list=edges)


def test_cec_centrality_basic():
    """Test CEC centrality returns positive entries for uniform connected hypergraph."""
    np.random.seed(0)
    hg = _make_uniform_connected()

    cec = CEC_centrality(hg, tol=1e-6, max_iter=100)

    assert set(cec.keys()) == set(hg.get_nodes())
    assert all(value > 0 for value in cec.values())


def test_zec_centrality_normalization():
    """Test ZEC centrality returns normalized non-negative values."""
    np.random.seed(1)
    hg = _make_uniform_connected()

    zec = ZEC_centrality(hg, max_iter=50, tol=1e-6)

    assert set(zec.keys()) == set(hg.get_nodes())
    assert all(value >= 0 for value in zec.values())
    assert np.isclose(sum(zec.values()), 1.0, atol=1e-6)


def test_hec_centrality_normalization():
    """Test HEC centrality returns normalized non-negative values."""
    np.random.seed(2)
    hg = _make_uniform_connected()

    hec = HEC_centrality(hg, max_iter=50, tol=1e-6)

    assert set(hec.keys()) == set(hg.get_nodes())
    assert all(value >= 0 for value in hec.values())
    assert np.isclose(sum(hec.values()), 1.0, atol=1e-6)


def test_eigen_centralities_require_uniform():
    """Test eigen centralities reject non-uniform hypergraphs."""
    hg = Hypergraph(edge_list=[(0, 1), (1, 2, 3)])

    with pytest.raises(Exception, match="not uniform"):
        CEC_centrality(hg)
    with pytest.raises(Exception, match="not uniform"):
        ZEC_centrality(hg)
    with pytest.raises(Exception, match="not uniform"):
        HEC_centrality(hg)
