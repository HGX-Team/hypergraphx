from os import times

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
    h.add_edge(("A", "B"), time=1)
    assert (1, ("A", "B")) in h.get_edges()
    assert h.edge_metadata[(1, ("A", "B"))] == {}


def test_add_edge_weighted():
    h = TemporalHypergraph(weighted=True)
    h.add_edge(("A", "B"), time=1, weight=2.0, metadata={"relationship": "friendship"})
    assert (1, ("A", "B")) in h.get_edges()
    assert h._edge_list[(1, ("A", "B"))] == 2.0
    assert h.edge_metadata[(1, ("A", "B"))] == {"relationship": "friendship"}


def test_add_edge_invalid_time():
    h = TemporalHypergraph()
    with pytest.raises(ValueError):
        h.add_edge(("A", "B"), time=-1)


def test_add_edge_invalid_type():
    h = TemporalHypergraph()
    with pytest.raises(TypeError):
        h.add_edge(("A", "B"), time="not_a_time")

def test_add_edge_invalid_type2():
    h = TemporalHypergraph()
    with pytest.raises(ValueError):
        h.add_edge(("A", "B"), time=-1)


def test_add_edges():
    h = TemporalHypergraph()
    edges = [("A", "B"), ("B", "C")]
    times = [1, 2]
    h.add_edges(edges, times)
    assert set(h.get_edges()) == {(1, ("A", "B")), (2, ("B", "C"))}


def test_add_edges_weighted():
    h = TemporalHypergraph(weighted=True)
    edges = [("A", "B"), ("B", "C")]
    weights = [0.5, 1.5]
    times = [1, 2]
    h.add_edges(edges, times, weights=weights)
    for i, edge in enumerate(edges):
        assert h.get_weight(edges[i], times[i]) == weights[i]


def test_get_edges_in_time_window():
    h = TemporalHypergraph()
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    times = [1, 2, 3]
    h.add_edges(edges, times)
    result = h.get_edges(time_window=(2, 4))
    assert set(result) == {(2, ("B", "C")), (3, ("C", "D"))}


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
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    times = [0, 1, 2]
    h.add_edges(edges, times)
    aggregated = h.aggregate(time_window=2)
    assert isinstance(aggregated, dict)
    assert all(isinstance(v, Hypergraph) for v in aggregated.values())
    assert aggregated[0].get_edges() == [("A", "B"), ("B", "C")]
    assert aggregated[1].get_edges() == [("C", "D")]


def test_aggregate_invalid_window():
    h = TemporalHypergraph()
    with pytest.raises(TypeError):
        h.aggregate(time_window="invalid")


def test_edge_metadata():
    h = TemporalHypergraph()
    edge = ("A", "B")
    time = 1
    metadata = {"relationship": "friendship"}
    h.add_edge(edge, time, metadata=metadata)
    assert h.get_edge_metadata(edge, time) == metadata
