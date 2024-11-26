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
    assert ("A", "B") in [edge[0] for edge in h.get_edges()]
    assert ("B", "C") in [edge[0] for edge in h.get_edges()]
    assert "layer1" in h.existing_layers
    assert "layer2" in h.existing_layers


def test_add_edges_weighted():
    h = MultiplexHypergraph(weighted=True)
    edges = [("A", "B"), ("B", "C")]
    layers = ["layer1", "layer2"]
    weights = [0.5, 1.0]
    h.add_edges(edges, edge_layer=layers, weights=weights)
    assert h.is_weighted()
    for i, edge in enumerate(edges):
        assert h.get_weight(edges[i], layers[i]) == weights[i]


def test_add_edges_metadata():
    h = MultiplexHypergraph()
    edges = [("A", "B")]
    layers = ["layer1"]
    metadata = [{"type": "friendship"}]
    h.add_edges(edges, edge_layer=layers, metadata=metadata)
    assert h.get_edge_metadata(edges[0], layers[0]) == {"type": "friendship"}


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
    h.add_node("B", metadata={"color": "red"})
    h.add_node("C", metadata={"color": "red"})
    h.add_edge(("A", "B"), layer="layer1", weight=0.5)
    h.add_edge(("B", "C"), layer="layer2", weight=0.5)
    aggregated = h.aggregated_hypergraph()
    assert isinstance(aggregated, Hypergraph)
    assert "A" in aggregated.get_nodes()
    assert "B" in aggregated.get_nodes()
    assert "C" in aggregated.get_nodes()
    assert ("A", "B") in aggregated.get_edges()
    assert ("B", "C") in aggregated.get_edges()
    assert aggregated.get_weight(("A", "B")) == 0.5
    assert aggregated.get_weight(("B", "C")) == 0.5

def test_add_edge_without_weight_in_weighted_hypergraph():
    """
    Test adding an edge without specifying a weight in a weighted hypergraph.
    """
    h = MultiplexHypergraph(weighted=True)
    h.add_edge(("A", "B"), layer="layer1")
    assert h.get_weight(("A", "B"), "layer1") == 1.0, "Default weight should be 1.0 in a weighted hypergraph."


def test_update_edge_weight():
    """
    Test updating the weight of an existing edge in a weighted hypergraph.
    """
    h = MultiplexHypergraph(weighted=True)
    h.add_edge(("A", "B"), layer="layer1", weight=0.5)
    h.set_weight(("A", "B"), "layer1", 1.5)
    assert h.get_weight(("A", "B"), "layer1") == 1.5, "Edge weight should be updated to the latest value."


def test_aggregate_weights_of_duplicate_edges():
    """
    Test adding duplicate edges in a weighted hypergraph and summing their weights.
    """
    h = MultiplexHypergraph(weighted=True)
    h.add_edge(("A", "B"), layer="layer1", weight=1.0)
    h.add_edge(("A", "B"), layer="layer1", weight=2.0)
    assert h.get_weight(("A", "B"), "layer1") == 3.0, "Weights should be aggregated for duplicate edges."


def test_add_edge_metadata_in_unweighted_hypergraph():
    """
    Test adding metadata to an edge in an unweighted hypergraph.
    """
    h = MultiplexHypergraph(weighted=False)
    h.add_edge(("A", "B"), layer="layer1", metadata={"type": "friendship"})
    assert h.get_edge_metadata(("A", "B"), "layer1") == {"type": "friendship"}, "Edge metadata should match the provided metadata."


def test_add_edge_metadata_in_weighted_hypergraph():
    """
    Test adding metadata to an edge in a weighted hypergraph.
    """
    h = MultiplexHypergraph(weighted=True)
    h.add_edge(("A", "B"), layer="layer1", weight=0.5, metadata={"type": "collaboration"})
    assert h.get_edge_metadata(("A", "B"), "layer1") == {"type": "collaboration"}, "Edge metadata should match the provided metadata."


def test_add_edges_with_mixed_metadata_and_weights():
    """
    Test adding multiple edges with mixed metadata and weights in a weighted hypergraph.
    """
    h = MultiplexHypergraph(weighted=True)
    edges = [("A", "B"), ("B", "C")]
    layers = ["layer1", "layer2"]
    weights = [0.5, 1.0]
    metadata = [{"type": "friendship"}, {"type": "work"}]
    h.add_edges(edges, edge_layer=layers, weights=weights, metadata=metadata)
    for i, edge in enumerate(edges):
        assert h.get_weight(edge, layers[i]) == weights[i], f"Weight of edge {edge} should be {weights[i]}."
        assert h.get_edge_metadata(edge, layers[i]) == metadata[i], f"Metadata of edge {edge} should be {metadata[i]}."


def test_get_nonexistent_edge_weight():
    """
    Test retrieving the weight of a nonexistent edge in a weighted hypergraph.
    """
    h = MultiplexHypergraph(weighted=True)
    with pytest.raises(ValueError):
        h.get_weight(("A", "B"), "layer1")


def test_get_nonexistent_edge_metadata():
    """
    Test retrieving metadata of a nonexistent edge.
    """
    h = MultiplexHypergraph()
    with pytest.raises(ValueError, match="Edge .* not in hypergraph."):
        h.get_edge_metadata(("A", "B"), "layer1")


def test_aggregated_hypergraph_with_metadata_and_weights():
    """
    Test creating an aggregated hypergraph and ensuring weights and metadata are preserved.
    """
    h = MultiplexHypergraph(weighted=True)
    h.add_node("A", metadata={"color": "red"})
    h.add_node("B", metadata={"color": "blue"})
    h.add_node("C", metadata={"color": "green"})
    h.add_edge(("A", "B"), layer="layer1", weight=0.5, metadata={"type": "friendship"})
    h.add_edge(("B", "C"), layer="layer2", weight=1.0, metadata={"type": "work"})

    aggregated = h.aggregated_hypergraph()

    assert isinstance(aggregated, Hypergraph), "Aggregated object should be of type Hypergraph."
    assert ("A", "B") in aggregated.get_edges(), "Edge ('A', 'B') should be in the aggregated hypergraph."
    assert ("B", "C") in aggregated.get_edges(), "Edge ('B', 'C') should be in the aggregated hypergraph."
    assert aggregated.get_weight(("A", "B")) == 0.5, "Weight of ('A', 'B') should be preserved in the aggregated hypergraph."
    assert aggregated.get_weight(("B", "C")) == 1.0, "Weight of ('B', 'C') should be preserved in the aggregated hypergraph."
    assert aggregated.get_edge_metadata(("A", "B")) == {"type": "friendship"}, "Metadata of ('A', 'B') should be preserved."
    assert aggregated.get_edge_metadata(("B", "C")) == {"type": "work"}, "Metadata of ('B', 'C') should be preserved."
