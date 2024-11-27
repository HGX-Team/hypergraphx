import pytest
from hypergraphx import (
    Hypergraph,
    DirectedHypergraph,
    TemporalHypergraph,
    MultiplexHypergraph,
)
from hypergraphx.filters import filter_hypergraph


@pytest.fixture
def sample_hypergraph():
    """Fixture for a basic Hypergraph."""
    hg = Hypergraph()
    hg.add_node("A", metadata={"type": "person", "age": 25})
    hg.add_node("B", metadata={"type": "location", "country": "USA"})
    hg.add_node("C", metadata={"type": "person", "age": 30})
    hg.add_edge(("A", "B", "C"), metadata={"type": "interaction", "weight": 1.0})
    hg.add_edge(("A", "C"), metadata={"type": "friendship", "weight": 0.5})
    return hg


@pytest.fixture
def sample_temporal_hypergraph():
    """Fixture for a TemporalHypergraph."""
    thg = TemporalHypergraph()
    thg.add_node("A", metadata={"type": "person"})
    thg.add_node("B", metadata={"type": "location"})
    thg.add_edge(
        ("A", "B"), time=10, metadata={"type": "interaction", "event": "visit"}
    )
    thg.add_edge(("A", "C"), time=20, metadata={"type": "interaction", "event": "call"})
    return thg


@pytest.fixture
def sample_multiplex_hypergraph():
    """Fixture for a MultiplexHypergraph."""
    mhg = MultiplexHypergraph()
    mhg.add_node("A", metadata={"type": "person"})
    mhg.add_node("B", metadata={"type": "location"})
    mhg.add_edge(("A", "B"), layer="social", metadata={"type": "friendship"})
    mhg.add_edge(("A", "C"), layer="work", metadata={"type": "colleague"})
    return mhg


def test_filter_nodes_keep(sample_hypergraph):
    """Test keeping nodes that match the criteria."""
    hg = sample_hypergraph
    node_criteria = {"type": ["person"]}
    filter_hypergraph(hg, node_criteria=node_criteria, mode="keep")
    assert set(hg.get_nodes()) == {"A", "C"}  # Only persons are retained
    assert hg.get_edges() == [("A", "C")]  # Edges updated with remaining nodes


def test_filter_nodes_remove(sample_hypergraph):
    """Test removing nodes that match the criteria."""
    hg = sample_hypergraph
    node_criteria = {"type": ["location"]}
    filter_hypergraph(hg, node_criteria=node_criteria, mode="remove")
    assert set(hg.get_nodes()) == {"A", "C"}  # Location removed
    assert hg.get_edges() == [("A", "C")]  # Edge updated without "B"


def test_filter_edges_keep(sample_hypergraph):
    """Test keeping edges that match the criteria."""
    hg = sample_hypergraph
    edge_criteria = {"type": ["friendship"]}
    filter_hypergraph(hg, edge_criteria=edge_criteria, mode="keep")
    assert hg.get_edges() == [("A", "C")]  # Only "friendship" edge kept


def test_filter_edges_remove(sample_hypergraph):
    """Test removing edges that match the criteria."""
    hg = sample_hypergraph
    edge_criteria = {"type": ["interaction"]}
    filter_hypergraph(hg, edge_criteria=edge_criteria, mode="remove")
    assert hg.get_edges() == [("A", "C")]  # "interaction" edge removed


def test_filter_nodes_and_edges(sample_hypergraph):
    """Test filtering both nodes and edges."""
    hg = sample_hypergraph
    node_criteria = {"type": ["location"]}
    edge_criteria = {"type": ["interaction"]}
    filter_hypergraph(
        hg, node_criteria=node_criteria, edge_criteria=edge_criteria, mode="remove"
    )
    assert set(hg.get_nodes()) == {"A", "C"}  # "B" removed
    assert hg.get_edges() == [("A", "C")]  # "interaction" edge removed


def test_filter_nodes_remove_keep_edges_false(sample_hypergraph):
    """
    Test removing nodes that match criteria and removing all edges they participate in.
    """
    hg = sample_hypergraph
    node_criteria = {"type": ["person"]}
    filter_hypergraph(hg, node_criteria=node_criteria, mode="remove", keep_edges=False)
    assert set(hg.get_nodes()) == {"B"}  # Only "B" (location) remains
    assert hg.get_edges() == []  # All edges involving "A" or "C" are removed


def test_filter_nodes_remove_keep_edges_true(sample_hypergraph):
    """
    Test removing nodes that match criteria but retaining edges by excluding those nodes.
    """
    hg = sample_hypergraph
    node_criteria = {"type": ["location"]}
    filter_hypergraph(hg, node_criteria=node_criteria, mode="remove", keep_edges=True)
    assert set(hg.get_nodes()) == {"A", "C"}  # "B" (location) is removed
    assert hg.get_edges() == [("A", "C")]  # Edge updated to exclude "B"


def test_filter_edges_remove_complex_metadata(sample_hypergraph):
    """
    Test removing edges that match complex metadata criteria.
    """
    hg = sample_hypergraph
    edge_criteria = {"type": ["interaction"], "weight": [1.0]}
    filter_hypergraph(hg, edge_criteria=edge_criteria, mode="remove")
    assert set(hg.get_nodes()) == {"A", "B", "C"}  # Nodes remain unchanged
    assert hg.get_edges() == [("A", "C")]  # Only the "friendship" edge remains


def test_combined_node_and_edge_filter_keep(sample_hypergraph):
    """
    Test combining node and edge filters with keep mode.
    """
    hg = sample_hypergraph
    node_criteria = {"type": ["person"]}
    edge_criteria = {"type": ["friendship"]}
    filter_hypergraph(
        hg, node_criteria=node_criteria, edge_criteria=edge_criteria, mode="keep"
    )
    assert set(hg.get_nodes()) == {"A", "C"}  # Only persons are retained
    assert hg.get_edges() == [("A", "C")]  # Only the friendship edge remains


def test_combined_node_and_edge_filter_remove(sample_hypergraph):
    """
    Test combining node and edge filters with remove mode.
    """
    hg = sample_hypergraph
    node_criteria = {"type": ["location"]}
    edge_criteria = {"type": ["interaction"]}
    filter_hypergraph(
        hg,
        node_criteria=node_criteria,
        edge_criteria=edge_criteria,
        mode="remove",
        keep_edges=False,
    )
    assert set(hg.get_nodes()) == {"A", "C"}  # "B" (location) is removed
    assert hg.get_edges() == [("A", "C")]  # "interaction" edge removed


def test_temporal_hypergraph_filter_remove(sample_temporal_hypergraph):
    """
    Test removing edges from a TemporalHypergraph based on metadata criteria.
    """
    thg = sample_temporal_hypergraph
    edge_criteria = {"event": ["call"]}
    filter_hypergraph(thg, edge_criteria=edge_criteria, mode="remove")
    assert set(thg.get_nodes()) == {"A", "B", "C"}  # Nodes remain unchanged
    assert thg.get_edges() == [(10, ("A", "B"))]  # Only the "visit" edge remains


def test_multiplex_hypergraph_filter_layers(sample_multiplex_hypergraph):
    """
    Test filtering edges from a MultiplexHypergraph based on metadata and layer-specific criteria.
    """
    mhg = sample_multiplex_hypergraph
    edge_criteria = {"type": ["colleague"]}
    filter_hypergraph(mhg, edge_criteria=edge_criteria, mode="remove")
    assert set(mhg.get_nodes()) == {"A", "B", "C"}  # Nodes remain unchanged
    assert mhg.get_edges() == [
        (("A", "B"), "social")
    ]  # Only "friendship" in "social" layer remains


def test_multiplex_hypergraph_filter_nodes_keep_edges():
    """
    Test removing nodes from a MultiplexHypergraph while keeping edges updated.
    """
    mhg = MultiplexHypergraph()
    mhg.add_node("A", metadata={"type": "person"})
    mhg.add_node("B", metadata={"type": "location"})
    mhg.add_node("C", metadata={"type": "person"})
    mhg.add_node("D", metadata={"type": "person"})
    mhg.add_edge(("A", "B", "D"), layer="social", metadata={"type": "friendship"})
    mhg.add_edge(("A", "C"), layer="work", metadata={"type": "colleague"})
    node_criteria = {"type": ["location"]}
    filter_hypergraph(mhg, node_criteria=node_criteria, mode="remove", keep_edges=True)
    assert set(mhg.get_nodes()) == {"A", "C", "D"}
    assert mhg.get_edges() == [(("A", "C"), "work"), (("A", "D"), "social")]
