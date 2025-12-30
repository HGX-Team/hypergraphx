import pytest

from hypergraphx import Hypergraph, TemporalHypergraph
from hypergraphx.measures.s_centralities import (
    s_betweenness,
    s_closeness,
    s_betweenness_nodes,
    s_closeness_nodes,
    s_betweenness_averaged,
    s_closeness_averaged,
    s_betweenness_nodes_averaged,
    s_closenness_nodes_averaged,
)


def _make_hypergraph():
    return Hypergraph(edge_list=[(0, 1), (1, 2, 3)])


def _make_temporal_hypergraph():
    edges = [(0, (0, 1)), (1, (1, 2, 3))]
    return TemporalHypergraph(edge_list=edges)


def test_edge_centralities():
    """Test s-centralities on edges return expected keys."""
    hg = _make_hypergraph()

    bet = s_betweenness(hg, s=1)
    clo = s_closeness(hg, s=1)

    assert set(bet.keys()) == set(hg.get_edges())
    assert set(clo.keys()) == set(hg.get_edges())


def test_node_centralities():
    """Test s-centralities on nodes return node keys only."""
    hg = _make_hypergraph()

    bet = s_betweenness_nodes(hg)
    clo = s_closeness_nodes(hg)

    assert set(bet.keys()) == set(hg.get_nodes())
    assert set(clo.keys()) == set(hg.get_nodes())


def test_temporal_edge_averaged_centralities():
    """Test averaged temporal s-centralities are computed."""
    thg = _make_temporal_hypergraph()

    bet = s_betweenness_averaged(thg, s=1)
    clo = s_closeness_averaged(thg, s=1)

    assert isinstance(bet, dict)
    assert isinstance(clo, dict)
    assert bet
    assert clo


def test_temporal_node_averaged_centralities():
    """Test averaged temporal node s-centralities error on node id checks."""
    thg = _make_temporal_hypergraph()

    with pytest.raises(TypeError):
        s_betweenness_nodes_averaged(thg)
    with pytest.raises(TypeError):
        s_closenness_nodes_averaged(thg)
