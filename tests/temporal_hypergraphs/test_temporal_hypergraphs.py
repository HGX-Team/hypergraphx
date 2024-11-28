from hypergraphx import Hypergraph


def test_initialization():
    h = TemporalHypergraph()
    assert isinstance(h._hypergraph_metadata, dict)
    assert h._hypergraph_metadata["weighted"] is False
    assert h.get_nodes() == []


def test_add_single_node():
    h = TemporalHypergraph()
    h.add_node("A", metadata={"type": "person"})
    assert "A" in h.get_nodes()
    assert h._node_metadata["A"] == {"type": "person"}


def test_add_multiple_nodes():
    h = TemporalHypergraph()
    nodes = ["A", "B", "C"]
    metadata = {
        "A": {"type": "person"},
        "B": {"type": "object"},
        "C": {"type": "place"},
    }
    h.add_nodes(nodes, metadata)
    assert set(h.get_nodes()) == set(nodes)
    for node in nodes:
        assert h._node_metadata[node] == metadata[node]


def test_add_edge_unweighted():
    h = TemporalHypergraph()
    h.add_edge(("A", "B"), time=1)
    assert (1, ("A", "B")) in h.get_edges()
    assert h.get_edge_metadata(("A", "B"), 1) == {}


def test_add_edge_weighted():
    h = TemporalHypergraph(weighted=True)
    h.add_edge(("A", "B"), time=1, weight=2.0, metadata={"relationship": "friendship"})
    assert (1, ("A", "B")) in h.get_edges()
    assert h.get_weight(("A", "B"), 1) == 2.0
    assert h.get_edge_metadata(("A", "B"), 1) == {"relationship": "friendship"}


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


def test_aggregate_multiple_windows():
    """Test aggregation of edges across multiple windows."""
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge((1, 2), time=1, weight=1.0, metadata={"type": "A"})
    thg.add_edge((3, 4), time=4, weight=2.0, metadata={"type": "B"})
    thg.add_edge((1, 3), time=8, weight=3.0, metadata={"type": "C"})
    thg.add_edge((2, 4), time=11, weight=4.0, metadata={"type": "D"})

    aggregated = thg.aggregate(time_window=5)
    assert len(aggregated) == 3

    # First window (0-5)
    window_0 = aggregated[0]
    assert set(window_0.get_edges()) == {(1, 2), (3, 4)}
    assert window_0.get_edge_metadata((1, 2)) == {"type": "A"}
    assert window_0.get_edge_metadata((3, 4)) == {"type": "B"}

    # Second window (5-10)
    window_1 = aggregated[1]
    assert set(window_1.get_edges()) == {(1, 3)}
    assert window_1.get_edge_metadata((1, 3)) == {"type": "C"}

    # Third window (10-15)
    window_2 = aggregated[2]
    assert set(window_2.get_edges()) == {(2, 4)}
    assert window_2.get_edge_metadata((2, 4)) == {"type": "D"}


def test_aggregate_empty():
    """Test aggregation when no edges exist."""
    thg = TemporalHypergraph()
    aggregated = thg.aggregate(time_window=5)
    assert len(aggregated) == 0


def test_aggregate_single_window():
    """Test aggregation of all edges into a single window."""
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge((1, 2), time=1, weight=1.0, metadata={"type": "A"})
    thg.add_edge((3, 4), time=3, weight=2.0, metadata={"type": "B"})

    aggregated = thg.aggregate(time_window=10)
    assert len(aggregated) == 1

    # Only one window (0-10)
    window_0 = aggregated[0]
    assert set(window_0.get_edges()) == {(1, 2), (3, 4)}
    assert window_0.get_edge_metadata((1, 2)) == {"type": "A"}
    assert window_0.get_edge_metadata((3, 4)) == {"type": "B"}


def test_aggregate_with_isolated_nodes():
    """Test aggregation ensures isolated nodes are preserved."""
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge((1, 2), time=1, weight=1.0)
    thg.add_node(3)  # Isolated node
    aggregated = thg.aggregate(time_window=5)
    assert len(aggregated) == 1

    # Ensure isolated nodes are preserved in the window
    window_0 = aggregated[0]
    assert 3 in window_0.get_nodes()
    assert (1, 2) in window_0.get_edges()


def test_aggregate_no_edges_in_window():
    """Test aggregation where no edges fall into the time window."""
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge((1, 2), time=10, weight=1.0)
    thg.add_edge((3, 4), time=20, weight=2.0)

    aggregated = thg.aggregate(time_window=5)
    assert len(aggregated) == 5  # Empty windows between edges

    # Check the windows
    window_0 = aggregated[0]
    assert len(window_0.get_edges()) == 0  # No edges in this window

    window_1 = aggregated[1]
    assert len(window_1.get_edges()) == 0  # No edges in this window

    window_2 = aggregated[2]
    assert (1, 2) in window_2.get_edges()

    window_3 = aggregated[3]
    assert len(window_3.get_edges()) == 0  # No edges in this window

    window_4 = aggregated[4]
    assert (3, 4) in window_4.get_edges()


import pytest
from hypergraphx import TemporalHypergraph  # Replace with the correct module name


def test_add_edge_without_weight_in_weighted_temporal_hypergraph():
    """
    Test adding an edge without specifying a weight in a weighted temporal hypergraph.
    """
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge(("A", "B"), time=10)
    assert (
        thg.get_weight(("A", "B"), 10) == 1.0
    ), "Default weight should be 1.0 for an edge added without a weight."


def test_update_edge_weight_in_temporal_hypergraph():
    """
    Test updating the weight of an existing edge in a weighted temporal hypergraph.
    """
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge(("A", "B"), time=10, weight=0.5)
    thg.add_edge(("A", "B"), time=10, weight=1.5)  # Update weight
    assert (
        thg.get_weight(("A", "B"), 10) == 2.0
    ), "Edge weight should be updated to the latest value."


def test_aggregate_weights_for_duplicate_edges_in_temporal_hypergraph():
    """
    Test aggregating weights for duplicate edges in a weighted temporal hypergraph.
    """
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge(("A", "B"), time=10, weight=2.0)
    thg.add_edge(("A", "B"), time=10, weight=3.0)  # Aggregate weight
    assert (
        thg.get_weight(("A", "B"), 10) == 5.0
    ), "Weights should be aggregated for duplicate edges."


def test_add_edge_metadata_in_temporal_hypergraph():
    """
    Test adding metadata to an edge in a temporal hypergraph.
    """
    thg = TemporalHypergraph()
    thg.add_edge(("A", "B"), time=10, metadata={"type": "interaction"})
    assert thg.get_edge_metadata(("A", "B"), 10)


def test_set_weight_overwrite_existing_weight():
    """
    Test overwriting the weight of an existing edge in a weighted temporal hypergraph.
    """
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge(("A", "B"), time=10, weight=2.0)  # Initial weight
    thg.set_weight(("A", "B"), 10, weight=5.0)  # Overwrite weight
    assert (
        thg.get_weight(("A", "B"), 10) == 5.0
    ), "Weight should be overwritten to the new value."


def test_set_weight_on_nonexistent_edge():
    """
    Test setting a weight on a nonexistent edge in a weighted temporal hypergraph.
    """
    thg = TemporalHypergraph(weighted=True)
    with pytest.raises(ValueError, match="Edge .* not in hypergraph."):
        thg.set_weight(("A", "B"), 10, weight=3.0)


def test_set_weight_in_unweighted_temporal_hypergraph():
    """
    Test attempting to set a weight in an unweighted temporal hypergraph.
    """
    thg = TemporalHypergraph(weighted=False)
    thg.add_edge(("A", "B"), time=10)  # Edge with no weight
    with pytest.raises(ValueError):
        thg.set_weight(("A", "B"), 10, weight=3.0)


def test_set_weight_with_metadata():
    """
    Test overwriting the weight of an edge and verifying metadata remains unchanged.
    """
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge(("A", "B"), time=10, weight=2.0, metadata={"type": "interaction"})
    thg.set_weight(("A", "B"), 10, weight=4.0)  # Overwrite weight
    assert (
        thg.get_weight(("A", "B"), 10) == 4.0
    ), "Weight should be updated to the new value."
    assert thg.get_edge_metadata(("A", "B"), 10) == {
        "type": "interaction"
    }, "Metadata should remain unchanged."


def test_set_weight_multiple_edges():
    """
    Test overwriting weights of multiple edges in a temporal hypergraph.
    """
    thg = TemporalHypergraph(weighted=True)
    edges = [("A", "B"), ("B", "C")]
    times = [10, 15]
    weights = [1.0, 2.0]
    new_weights = [3.0, 4.0]

    # Add edges with initial weights
    for edge, time, weight in zip(edges, times, weights):
        thg.add_edge(edge, time, weight=weight)

    # Overwrite weights
    for edge, time, new_weight in zip(edges, times, new_weights):
        thg.set_weight(edge, time, weight=new_weight)

    # Check updated weights
    for edge, time, new_weight in zip(edges, times, new_weights):
        assert (
            thg.get_weight(edge, time) == new_weight
        ), f"Weight of edge {edge} at time {time} should be updated to {new_weight}."


def test_get_incident_edges_no_edges():
    """
    Test retrieving incident edges for a node with no incident edges.
    """
    thg = TemporalHypergraph()
    thg.add_node("A")
    incident_edges = thg.get_incident_edges("A")
    assert incident_edges == [], "Node A should have no incident edges."


def test_get_incident_edges_single_edge():
    """
    Test retrieving incident edges for a node with a single incident edge.
    """
    thg = TemporalHypergraph()
    thg.add_edge(("A", "B"), time=5)
    incident_edges = thg.get_incident_edges("A")
    assert len(incident_edges) == 1, "Node A should have 1 incident edge."
    assert (
        5,
        ("A", "B"),
    ) in incident_edges, "Edge (5, ('A', 'B')) should be incident to node A."


def test_get_incident_edges_multiple_edges():
    """
    Test retrieving incident edges for a node with multiple incident edges.
    """
    thg = TemporalHypergraph()
    thg.add_edge(("A", "B"), time=5)
    thg.add_edge(("A", "C"), time=10)
    thg.add_edge(("D", "A"), time=15)
    thg.add_edge(("C", "D"), time=20)  # Duplicate edge
    incident_edges = thg.get_incident_edges("A")
    assert len(incident_edges) == 3, "Node A should have 3 incident edges."
    assert (
        5,
        ("A", "B"),
    ) in incident_edges, "Edge (5, ('A', 'B')) should be incident to node A."
    assert (
        10,
        ("A", "C"),
    ) in incident_edges, "Edge (10, ('A', 'C')) should be incident to node A."
    assert (
        15,
        ("A", "D"),
    ) in incident_edges, "Edge (15, ('D', 'A')) should be incident to node A."
    assert (
        20,
        ("C", "D"),
    ) not in incident_edges, "Edge (20, ('C', 'D')) should not be incident to node A."


def test_get_incident_edges_nonexistent_node():
    """
    Test retrieving incident edges for a node not in the hypergraph.
    """
    thg = TemporalHypergraph()
    thg.add_edge(("A", "B"), time=5)
    with pytest.raises(ValueError, match="Node .* not in hypergraph."):
        thg.get_incident_edges("Z")


def test_get_incident_edges_isolated_node():
    """
    Test retrieving incident edges for an isolated node (node exists but has no edges).
    """
    thg = TemporalHypergraph()
    thg.add_node("A")
    thg.add_edge(("B", "C"), time=5)
    incident_edges = thg.get_incident_edges("A")
    assert (
        incident_edges == []
    ), "Node A should have no incident edges as it is isolated."


def test_get_incident_edges_different_times():
    """
    Test retrieving incident edges for a node with edges at different times.
    """
    thg = TemporalHypergraph()
    thg.add_edge(("A", "B"), time=5)
    thg.add_edge(("A", "C"), time=10)
    thg.add_edge(("A", "B"), time=15)  # Same edge, different time
    incident_edges = thg.get_incident_edges("A")
    assert len(incident_edges) == 3, "Node A should have 3 incident edges."
    assert (
        5,
        ("A", "B"),
    ) in incident_edges, "Edge (5, ('A', 'B')) should be incident to node A."
    assert (
        10,
        ("A", "C"),
    ) in incident_edges, "Edge (10, ('A', 'C')) should be incident to node A."
    assert (
        15,
        ("A", "B"),
    ) in incident_edges, "Edge (15, ('A', 'B')) should be incident to node A."
