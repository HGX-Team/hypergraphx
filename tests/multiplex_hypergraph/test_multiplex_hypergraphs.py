import pytest
from hypergraphx import Hypergraph
from hypergraphx import MultiplexHypergraph  # Replace 'your_module' with the module name containing your class


def test_initialization():
    h = MultiplexHypergraph()
    assert isinstance(h.hypergraph_metadata, dict)
    assert h.hypergraph_metadata['weighted'] is False
    assert h.get_nodes() == []


def test_add_single_node():
    h = MultiplexHypergraph()
    h.add_node("A", metadata={"color": "red"})
    assert "A" in h.get_nodes()
    assert h.node_metadata["A"] == {"color": "red"}


def test_add_multiple_nodes():
    h = MultiplexHypergraph()
    nodes = ["A", "B", "C"]
    metadata = {"A": {"color": "red"}, "B": {"color": "blue"}, "C": {"color": "green"}}
    h.add_nodes(nodes, metadata)
    assert set(h.get_nodes()) == set(nodes)
    for node in nodes:
        assert h.node_metadata[node] == metadata[node]


def test_add_edges_unweighted():
    h = MultiplexHypergraph()
    edges = [("A", "B"), ("B", "C")]
    layers = ["layer1", "layer2"]
    h.add_edges(edges, edge_layer=layers)
    assert ("A", "B") in [edge[0] for edge in h._edge_list.keys()]
    assert ("B", "C") in [edge[0] for edge in h._edge_list.keys()]
    assert "layer1" in h.existing_layers
    assert "layer2" in h.existing_layers


def test_add_edges_weighted():
    h = MultiplexHypergraph(weighted=True)
    edges = [("A", "B"), ("B", "C")]
    layers = ["layer1", "layer2"]
    weights = [0.5, 1.0]
    h.add_edges(edges, edge_layer=layers, weights=weights)
    assert h._weighted is True
    for i, edge in enumerate(edges):
        assert h._edge_list[(tuple(sorted(edge)), layers[i])] == weights[i]


def test_add_edges_metadata():
    h = MultiplexHypergraph()
    edges = [("A", "B")]
    layers = ["layer1"]
    metadata = [{"type": "friendship"}]
    h.add_edges(edges, edge_layer=layers, metadata=metadata)
    assert h.edge_metadata[(tuple(sorted(("A", "B"))), "layer1")] == {"type": "friendship"}


def test_hypergraph_metadata():
    h = MultiplexHypergraph()
    h.set_dataset_metadata({"name": "test_dataset"})
    assert h.get_dataset_metadata() == {"name": "test_dataset"}


def test_layer_metadata():
    h = MultiplexHypergraph()
    h.set_layer_metadata("layer1", {"description": "Layer 1"})
    assert h.get_layer_metadata("layer1") == {"description": "Layer 1"}


def test_aggregated_hypergraph():
    h = MultiplexHypergraph(weighted=True)
    h.add_node("A", metadata={"color": "red"})
    h.add_edges([("A", "B")], edge_layer=["layer1"], weights=[0.5])
    aggregated = h.aggregated_hypergraph()
    assert isinstance(aggregated, Hypergraph)
    assert "A" in aggregated.get_nodes()
    assert aggregated.edge_weight(("A", "B")) == 0.5
