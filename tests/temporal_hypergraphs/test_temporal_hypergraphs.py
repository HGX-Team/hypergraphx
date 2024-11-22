import pytest
from hypergraphx import Hypergraph
from hypergraphx import TemporalHypergraph


def test_initialization():
    h = TemporalHypergraph()
    assert isinstance(h.hypergraph_metadata, dict)
    assert h.hypergraph_metadata['weighted'] is False
    assert h.get_nodes() == []


def test_add_single_node():
    h = TemporalHypergraph()
    h.add_node("A", metadata={"type": "person"})
    assert "A" in h.get_nodes()
    assert h.node_metadata["A"] == {"type": "person"}


def test_add_multiple_nodes():
    h = TemporalHypergraph()
    nodes = ["A", "B", "C"]
    metadata = {"A": {"type": "person"}, "B": {"type": "object"}, "C": {"type": "place"}}
    h.add_nodes(nodes, metadata)
    assert set(h.get_nodes()) == set(nodes)
    for node in nodes:
        assert h.node_metadata[node] == metadata[node]


def test_add_edge_unweighted():
    h = TemporalHypergraph()
    h.add_edge((1, ("A", "B")))
    assert (1, ("A", "B")) in h.get_edges()
    assert h.edge_metadata[(1, ("A", "B"))] == {}


def test_add_edge_weighted():
    h = TemporalHypergraph(weighted=True)
    h.add_edge((1, ("A", "B")), weight=2.0, metadata={"relationship": "friendship"})
    assert (1, ("A", "B")) in h.get_edges()
    assert h._edge_list[(1, ("A", "B"))] == 2.0
    assert h.edge_metadata[(1, ("A", "B"))] == {"relationship": "friendship"}


def test_add_edge_invalid_time():
    h = TemporalHypergraph()
    with pytest.raises(ValueError):
        h.add_edge((-1, ("A", "B")))


def test_add_edge_invalid_type():
    h = TemporalHypergraph()
    with pytest.raises(TypeError):
        h.add_edge(("not_a_time", ("A", "B")))


def test_add_edge_invalid_weight():
    h = TemporalHypergraph(weighted=True)
    with pytest.raises(ValueError):
        h.add_edge((1, ("A", "B")))


def test_add_edges():
    h = TemporalHypergraph()
    edges = [(1, ("A", "B")), (2, ("B", "C"))]
    h.add_edges(edges)
    assert set(h.get_edges()) == set(edges)


def test_add_edges_weighted():
    h = TemporalHypergraph(weighted=True)
    edges = [(1, ("A", "B")), (2, ("B", "C"))]
    weights = [0.5, 1.5]
    h.add_edges(edges, weights=weights)
    for i, edge in enumerate(edges):
        assert h._edge_list[edge] == weights[i]


def test_get_edges_in_time_window():
    h = TemporalHypergraph()
    edges = [(1, ("A", "B")), (3, ("B", "C")), (5, ("C", "D"))]
    h.add_edges(edges)
    result = h.get_edges(time_window=(2, 4))
    assert result == [(3, ("B", "C"))]


def test_get_edges_invalid_time_window():
    h = TemporalHypergraph()
    with pytest.raises(ValueError):
        h.get_edges(time_window=(1,))


def test_is_weighted():
    h = TemporalHypergraph()
    assert not h.is_weighted()
    h_weighted = TemporalHypergraph(weighted=True)
    assert h_weighted.is_weighted()


def test_aggregate():
    h = TemporalHypergraph()
    edges = [(1, ("A", "B")), (2, ("B", "C")), (3, ("C", "D"))]
    h.add_edges(edges)
    aggregated = h.aggregate(time_window=2)
    assert isinstance(aggregated, dict)
    assert all(isinstance(v, Hypergraph) for v in aggregated.values())
    assert aggregated[0].get_edges() == [("A", "B")]
    assert aggregated[1].get_edges() == [("B", "C"), ("C", "D")]


def test_aggregate_invalid_window():
    h = TemporalHypergraph()
    with pytest.raises(TypeError):
        h.aggregate(time_window="invalid")


def test_edge_metadata():
    h = TemporalHypergraph()
    edge = (1, ("A", "B"))
    metadata = {"relationship": "friendship"}
    h.add_edge(edge, metadata=metadata)
    assert h.edge_metadata[edge] == metadata
