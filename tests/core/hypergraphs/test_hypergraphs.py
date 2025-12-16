import pytest

from hypergraphx import Hypergraph


def test_hypergraph_initialization_with_metadata():
    """Test initializing a hypergraph with metadata."""
    node_metadata = {"A": {"type": "node"}, "B": {"type": "node"}}
    hypergraph_metadata = {"description": "Test Hypergraph"}
    hg = Hypergraph(
        node_metadata=node_metadata, hypergraph_metadata=hypergraph_metadata
    )

    assert hg._node_metadata == node_metadata
    assert hg._hypergraph_metadata["description"] == "Test Hypergraph"
    assert hg._hypergraph_metadata["weighted"] == False


def test_hypergraph_initialization_with_edges():
    """Test initializing a hypergraph with edges."""
    edges = [(1, 2, 3), (3, 4)]
    hg = Hypergraph(edge_list=edges)

    assert len(hg._edge_list) == 2  # 2 edges
    assert len(hg._adj) == 4  # 4 nodes


def test_hypergraph_initialization_weighted():
    """Test initializing a weighted hypergraph."""
    edges = [(1, 2), (2, 3)]
    weights = [1, 2]
    hg = Hypergraph(edge_list=edges, weighted=True, weights=weights)

    assert hg.is_weighted() is True
    assert hg.get_weight((1, 2)) == 1
    assert hg.get_weight((2, 3)) == 2


def test_is_uniform_empty_hypergraph():
    """Test `is_uniform` method on an empty hypergraph."""
    hg = Hypergraph()
    assert hg.is_uniform() is True  # An empty hypergraph is trivially uniform


def test_is_uniform_single_edge():
    """Test `is_uniform` method with a single hyperedge."""
    edges = [(1, 2, 3)]
    hg = Hypergraph(edge_list=edges)
    assert hg.is_uniform() is True  # Only one hyperedge, so it's uniform


def test_is_uniform_uniform_hypergraph():
    """Test `is_uniform` method on a uniform hypergraph."""
    edges = [(1, 2), (3, 4), (5, 6)]
    hg = Hypergraph(edge_list=edges)
    assert hg.is_uniform() is True  # All edges have size 2


def test_is_uniform_non_uniform_hypergraph():
    """Test `is_uniform` method on a non-uniform hypergraph."""
    edges = [(1, 2), (3, 4, 5), (6,)]
    hg = Hypergraph(edge_list=edges)
    assert hg.is_uniform() is False  # Edge sizes are 2, 3, and 1


def test_is_uniform_mixed_size_edges():
    """Test `is_uniform` method with mixed edge sizes."""
    edges = [(1,), (2, 3), (4, 5, 6)]
    hg = Hypergraph(edge_list=edges)
    assert hg.is_uniform() is False  # Edge sizes are 1, 2, and 3


def test_get_neighbors_no_order_no_size():
    """Test `get_neighbors` with no order or size specified."""
    edges = [(1, 2), (2, 3), (3, 4, 5)]
    hg = Hypergraph(edge_list=edges)
    neighbors = hg.get_neighbors(2)
    assert neighbors == {1, 3}


def test_get_neighbors_with_size():
    """Test `get_neighbors` with size specified."""
    edges = [(1, 2), (2, 3), (3, 4, 5)]
    hg = Hypergraph(edge_list=edges)
    neighbors = hg.get_neighbors(2, size=2)
    assert neighbors == {1, 3}  # Only consider edges of size 2


def test_get_neighbors_with_order1():
    """Test `get_neighbors` with order specified."""
    edges = [(1, 2), (2, 3), (3, 4, 5)]
    hg = Hypergraph(edge_list=edges)
    neighbors = hg.get_neighbors(3, order=1)
    assert neighbors == {2}  # Order 1 implies edges with size 2
    neighbors = hg.get_neighbors(3, order=2)
    assert neighbors == {4, 5}  # Order 2 implies edges with size 3


def test_get_neighbors_order_and_size_error():
    """Test `get_neighbors` raises ValueError when both order and size are specified."""
    edges = [(1, 2), (2, 3), (3, 4, 5)]
    hg = Hypergraph(edge_list=edges)
    with pytest.raises(ValueError, match="Order and size cannot be both specified."):
        hg.get_neighbors(2, order=1, size=2)


def test_get_neighbors_node_not_in_hypergraph():
    """Test `get_neighbors` for a node not in the hypergraph."""
    edges = [(1, 2), (2, 3), (3, 4, 5)]
    hg = Hypergraph(edge_list=edges)
    with pytest.raises(ValueError, match="Node 10 not in hypergraph."):
        neighbors = hg.get_neighbors(10)


def test_get_neighbors_self_loop():
    """Test `get_neighbors` when the node has a self-loop."""
    edges = [(1, 2), (2, 2), (2, 3)]
    hg = Hypergraph(edge_list=edges)
    neighbors = hg.get_neighbors(2)
    assert neighbors == {1, 3}


def test_get_incident_edges_no_order_no_size():
    """Test `get_incident_edges` with no order or size specified."""
    edges = [(1, 2), (2, 3), (3, 4, 5), (2, 4, 7, 8)]
    hg = Hypergraph(edge_list=edges)
    incident_edges = hg.get_incident_edges(2)
    assert set(incident_edges) == {(1, 2), (2, 3), (2, 4, 7, 8)}


def test_get_incident_edges_with_size():
    """Test `get_incident_edges` with size specified."""
    edges = [(1, 2), (2, 3), (3, 4, 5), (2, 4, 7, 8)]
    hg = Hypergraph(edge_list=edges)
    incident_edges = hg.get_incident_edges(2, size=2)
    assert set(incident_edges) == {(1, 2), (2, 3)}  # Only edges of size 2


def test_get_incident_edges_with_order():
    """Test `get_incident_edges` with order specified."""
    edges = [(1, 2), (2, 3), (3, 4, 5)]
    hg = Hypergraph(edge_list=edges)
    incident_edges = hg.get_incident_edges(2, order=1)
    assert set(incident_edges) == {(1, 2), (2, 3)}  # Order 1 implies edges of size 2


def test_get_incident_edges_order_and_size_error():
    """Test `get_incident_edges` raises ValueError when both order and size are specified."""
    edges = [(1, 2), (2, 3), (3, 4, 5)]
    hg = Hypergraph(edge_list=edges)
    with pytest.raises(ValueError, match="Order and size cannot be both specified."):
        hg.get_incident_edges(2, order=1, size=2)


def test_get_incident_edges_node_not_in_hypergraph():
    """Test `get_incident_edges` for a node not in the hypergraph."""
    edges = [(1, 2), (2, 3), (3, 4, 5)]
    hg = Hypergraph(edge_list=edges)
    with pytest.raises(ValueError, match="Node 10 not in hypergraph."):
        incident_edges = hg.get_incident_edges(10)


def test_get_incident_edges_self_loop():
    """Test `get_incident_edges` when the node is part of a self-loop."""
    edges = [(1, 2), (2, 2), (2, 3)]
    hg = Hypergraph(edge_list=edges)
    incident_edges = hg.get_incident_edges(2)
    assert set(incident_edges) == {(1, 2), (2, 2), (2, 3)}  # Includes self-loop


def test_add_edge_unweighted():
    """Test adding an unweighted edge to the hypergraph."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    assert hg.check_edge((1, 2, 3))
    assert hg.get_weight((1, 2, 3)) == 1


def test_add_edge_weighted():
    """Test adding a weighted edge to the hypergraph."""
    hg = Hypergraph(weighted=True)
    hg.add_edge((1, 2, 3), weight=2.5)
    assert hg.check_edge((1, 2, 3))
    assert hg.get_weight((1, 2, 3)) == 2.5


def test_add_edge_weighted_without_weight():
    """Test adding a weighted edge without providing a weight."""
    hg = Hypergraph(weighted=True)
    hg.add_edge((1, 2, 3))
    assert hg.get_weight((1, 2, 3)) == 1  # Default weight is 1


def test_add_edge_unweighted_with_weight():
    """Test adding an unweighted edge with a weight."""
    hg = Hypergraph()
    with pytest.raises(
        ValueError, match="If the hypergraph is not weighted, weight can be 1 or None."
    ):
        hg.add_edge((1, 2, 3), weight=2.5)


def test_add_edge_duplicate_edge_unweighted():
    """Test adding the same unweighted edge multiple times."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    hg.add_edge((1, 2, 3))
    assert hg.check_edge((1, 2, 3))
    assert hg.get_weight((1, 2, 3)) == 1


def test_add_edge_duplicate_edge_weighted():
    """Test adding the same weighted edge multiple times."""
    hg = Hypergraph(weighted=True)
    hg.add_edge((1, 2, 3), weight=1.5)
    hg.add_edge((1, 2, 3), weight=2.5)
    assert hg.get_weight((1, 2, 3)) == 4.0  # Weight is updated to the sum


def test_add_edge_with_metadata():
    """Test adding an edge with metadata."""
    hg = Hypergraph()
    metadata = {"type": "test_edge"}
    hg.add_edge((1, 2, 3), metadata=metadata)
    assert hg.get_edge_metadata((1, 2, 3)) == metadata


def test_add_edge_unsorted():
    """Test that the edge is stored in sorted order."""
    hg = Hypergraph()
    hg.add_edge((3, 1, 2))
    assert (1, 2, 3) in hg._edge_list  # Edge should be stored as sorted tuple


def test_add_edge_with_new_nodes():
    """Test adding an edge introduces new nodes."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    assert 1 in hg._node_metadata
    assert 2 in hg._node_metadata
    assert 3 in hg._node_metadata


def test_add_edge_existing_nodes():
    """Test adding an edge when nodes already exist."""
    hg = Hypergraph()
    hg.add_node(1)
    hg.add_node(2)
    hg.add_edge((1, 2, 3))
    assert 1 in hg._node_metadata
    assert 2 in hg._node_metadata
    assert 3 in hg._node_metadata  # New node added


def test_add_edges_unweighted():
    """Test adding multiple unweighted edges."""
    hg = Hypergraph()
    edges = [(1, 2), (2, 3, 4), (4, 5)]
    hg.add_edges(edges)
    for edge in edges:
        assert hg.check_edge(edge)
    assert all(hg.get_weight(edge) == 1 for edge in edges)


def test_add_edges_weighted():
    """Test adding multiple weighted edges."""
    hg = Hypergraph(weighted=True)
    edges = [(1, 2), (2, 3), (4, 5)]
    weights = [0.5, 1.2, 2.3]
    hg.add_edges(edges, weights=weights)
    for edge, weight in zip(edges, weights):
        assert hg.check_edge(edge)
        assert hg.get_weight(edge) == weight


def test_add_edges_metadata():
    """Test adding multiple edges with metadata."""
    hg = Hypergraph()
    edges = [(1, 2), (3, 4)]
    metadata = [{"type": "edge1"}, {"type": "edge2"}]
    hg.add_edges(edges, metadata=metadata)
    for edge, meta in zip(edges, metadata):
        assert hg.get_edge_metadata(edge) == meta


def test_add_edges_repeated_edges_with_weights():
    """Test adding repeated edges with weights in a weighted hypergraph."""
    hg = Hypergraph(weighted=True)
    edges = [(1, 2), (1, 2)]
    weights = [1.0, 2.0]
    with pytest.raises(
        ValueError,
        match="If weights are provided, the edge list must not contain repeated edges.",
    ):
        hg.add_edges(edges, weights=weights)


def test_add_edges_mismatch_weights_and_edges():
    """Test adding edges when the number of weights and edges mismatch."""
    hg = Hypergraph(weighted=True)
    edges = [(1, 2), (3, 4)]
    weights = [1.0]
    with pytest.raises(
        ValueError, match="The number of edges and weights must be the same."
    ):
        hg.add_edges(edges, weights=weights)


def test_add_edges_empty_list():
    """Test adding an empty list of edges."""
    hg = Hypergraph()
    hg.add_edges([])
    assert len(hg._edge_list) == 0


def test_add_edges_with_existing_edges():
    """Test adding edges when some edges already exist."""
    hg = Hypergraph()
    edges = [(1, 2), (3, 4)]
    hg.add_edges(edges)
    assert len(hg._edge_list) == 2  # Two edges added
    hg.add_edges([(3, 4), (5, 6)])  # Add a duplicate and a new edge
    assert len(hg._edge_list) == 3  # Only one new edge added


def test_remove_edge_existing_edge():
    """Test removing an existing edge from the hypergraph."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    assert (1, 2, 3) in hg._edge_list
    hg.remove_edge((1, 2, 3))
    assert (1, 2, 3) not in hg._edge_list
    with pytest.raises(ValueError):
        hg.get_weight((1, 2, 3))


def test_remove_edge_nonexistent_edge():
    """Test attempting to remove a non-existent edge."""
    hg = Hypergraph()
    hg.add_edge((1, 2))
    assert (3, 4) not in hg._edge_list
    with pytest.raises(KeyError):
        hg.remove_edge((3, 4))


def test_remove_edge_updates_adj():
    """Test that adjacency structure is updated when an edge is removed."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    edge_id = hg._edge_list[(1, 2, 3)]
    assert edge_id in hg._adj[1]
    assert edge_id in hg._adj[2]
    assert edge_id in hg._adj[3]
    hg.remove_edge((1, 2, 3))
    assert edge_id not in hg._adj[1]
    assert edge_id not in hg._adj[2]
    assert edge_id not in hg._adj[3]


def test_remove_edge_weighted_hypergraph():
    """Test removing an edge from a weighted hypergraph."""
    hg = Hypergraph(weighted=True)
    hg.add_edge((1, 2, 3), weight=2.5)
    assert (1, 2, 3) in hg._edge_list
    assert hg._weights[hg._edge_list[(1, 2, 3)]] == 2.5
    hg.remove_edge((1, 2, 3))
    assert (1, 2, 3) not in hg._edge_list
    assert hg._weights == {}


def test_remove_edge_unsorted_input():
    """Test removing an edge with unsorted input."""
    hg = Hypergraph()
    hg.add_edge((3, 1, 2))  # Edge stored as (1, 2, 3)
    assert (1, 2, 3) in hg._edge_list
    hg.remove_edge((2, 1, 3))  # Input unsorted
    assert (1, 2, 3) not in hg._edge_list


def test_remove_node_with_edges():
    """Test removing a node along with its incident edges."""
    hg = Hypergraph()
    hg.add_edge((1, 2))
    hg.add_edge((1, 3))
    hg.add_edge((2, 3))
    assert 1 in hg._adj
    assert (1, 2) in hg._edge_list
    assert (1, 3) in hg._edge_list
    hg.remove_node(1)
    assert 1 not in hg._adj
    assert (1, 2) not in hg._edge_list
    assert (1, 3) not in hg._edge_list
    assert (2, 3) in hg._edge_list  # Other edges unaffected


def test_remove_node_keep_edges():
    """Test removing a node while keeping the incident edges."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    hg.add_edge((1, 4))
    assert 1 in hg.get_nodes()
    hg.remove_node(1, keep_edges=True)
    assert 1 not in hg.get_nodes()
    assert hg.check_edge((2, 3)) is True
    assert hg.check_edge((4,)) is True  # Incident edge reduced to (4,)


def test_remove_node_not_in_hypergraph():
    """Test removing a node that does not exist in the hypergraph."""
    hg = Hypergraph()
    hg.add_edge((1, 2))
    with pytest.raises(KeyError, match="Node 10 not in hypergraph."):
        hg.remove_node(10)


def test_remove_node_weighted_hypergraph_keep_edges():
    """Test removing a node in a weighted hypergraph while keeping edges."""
    hg = Hypergraph(weighted=True)
    hg.add_edge((1, 2, 3), weight=1.5)
    hg.add_edge((1, 4), weight=2.0)
    hg.remove_node(1, keep_edges=True)
    assert (2, 3) in hg._edge_list
    assert hg._weights[hg._edge_list[(2, 3)]] == 1.5
    assert (4,) in hg._edge_list
    assert hg._weights[hg._edge_list[(4,)]] == 2.0


def test_remove_node_unconnected():
    """Test removing a node with no edges."""
    hg = Hypergraph()
    hg.add_node(1)
    assert 1 in hg._adj
    hg.remove_node(1)
    assert 1 not in hg._adj


def test_remove_node_multiple_edges():
    """Test removing a node with multiple incident edges."""
    hg = Hypergraph()
    hg.add_edge((1, 2))
    hg.add_edge((1, 3))
    hg.add_edge((1, 4))
    assert 1 in hg._adj
    hg.remove_node(1)
    assert 1 not in hg._adj
    assert not any((1 in edge for edge in hg._edge_list))


def test_get_edges_all_edges():
    """Test retrieving all edges without any filters."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    hg.add_edge((2, 3))
    edges = hg.get_edges()
    assert set(edges) == {(1, 2, 3), (2, 3)}


def test_get_edges_by_order():
    """Test retrieving edges of a specific order."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))  # Order 2
    hg.add_edge((2, 3))  # Order 1
    edges = hg.get_edges(order=1)
    assert set(edges) == {(2, 3)}


def test_get_edges_by_size():
    """Test retrieving edges of a specific size."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))  # Size 3
    hg.add_edge((2, 3))  # Size 2
    edges = hg.get_edges(size=3)
    assert set(edges) == {(1, 2, 3)}


def test_get_edges_up_to_order():
    """Test retrieving edges up to a specific order."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))  # Order 2
    hg.add_edge((2, 3))  # Order 1
    hg.add_edge((3,))  # Order 0
    edges = hg.get_edges(order=1, up_to=True)
    assert set(edges) == {(2, 3), (3,)}


def test_get_edges_up_to_size():
    """Test retrieving edges up to a specific size."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))  # Size 3
    hg.add_edge((2, 3))  # Size 2
    hg.add_edge((3,))  # Size 1
    edges = hg.get_edges(size=2, up_to=True)
    assert set(edges) == {(2, 3), (3,)}


def test_get_edges_metadata():
    """Test retrieving edges along with metadata."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3), metadata={"type": "triangle"})
    edges = hg.get_edges(metadata=True)
    assert edges == {(1, 2, 3): {"type": "triangle"}}


def test_get_edges_subhypergraph():
    """Test retrieving edges as a subhypergraph."""
    hg = Hypergraph(weighted=True)
    hg.add_edge((1, 2, 3), weight=1.5)
    hg.add_edge((2, 3), weight=2.0)
    sub_hypergraph = hg.get_edges(order=1, subhypergraph=True)
    assert sub_hypergraph.get_edges() == [(2, 3)]


def test_get_edges_subhypergraph_keep_isolated_nodes():
    """Test retrieving a subhypergraph with isolated nodes preserved."""
    hg = Hypergraph(weighted=True)
    hg.add_edge((1, 2, 3), weight=1.5)
    hg.add_edge((2, 3), weight=2.0)
    hg.add_node(4)  # Isolated node
    sub_hypergraph = hg.get_edges(order=1, subhypergraph=True, keep_isolated_nodes=True)
    assert 4 in sub_hypergraph.get_nodes()


def test_get_edges_order_and_size_error():
    """Test error when both order and size are specified."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    with pytest.raises(ValueError, match="Order and size cannot be both specified."):
        hg.get_edges(order=1, size=3)


def test_get_edges_keep_nodes_without_subhypergraph_error():
    """Test error when keep_isolated_nodes is True but subhypergraph is False."""
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    with pytest.raises(
        ValueError, match="Cannot keep nodes if not returning subhypergraphs."
    ):
        hg.get_edges(order=1, keep_isolated_nodes=True)


import pytest
from hypergraphx import Hypergraph  # Replace with the actual module name


def test_get_incident_edges_no_filters():
    """
    Test retrieving all incident edges for a node without order or size filters.
    """
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    hg.add_edge((2, 3, 4))
    hg.add_edge((1, 4))

    incident_edges = hg.get_incident_edges(1)
    assert len(incident_edges) == 2, "Node 1 should have 2 incident edges."
    assert (1, 2, 3) in incident_edges, "Edge (1, 2, 3) should be incident to node 1."
    assert (1, 4) in incident_edges, "Edge (1, 4) should be incident to node 1."


def test_get_incident_edges_by_order():
    """
    Test retrieving incident edges for a node with a specific order.
    """
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))  # Order 2
    hg.add_edge((1, 3))  # Order 1
    hg.add_edge((2, 3, 4))  # Order 2

    incident_edges = hg.get_incident_edges(1, order=1)
    assert len(incident_edges) == 1, "Node 1 should have 1 incident edge of order 1."
    assert (
        1,
        3,
    ) in incident_edges, "Edge (1, 3) should be incident to node 1 and of order 1."


def test_get_incident_edges_by_size():
    """
    Test retrieving incident edges for a node with a specific size.
    """
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))  # Size 3
    hg.add_edge((1, 3))  # Size 2
    hg.add_edge((2, 3, 4))  # Size 3

    incident_edges = hg.get_incident_edges(1, size=3)
    assert len(incident_edges) == 1, "Node 1 should have 1 incident edge of size 3."
    assert (
        1,
        2,
        3,
    ) in incident_edges, "Edge (1, 2, 3) should be incident to node 1 and of size 3."


def test_get_incident_edges_nonexistent_node():
    """
    Test retrieving incident edges for a node not in the hypergraph.
    """
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    with pytest.raises(ValueError, match="Node .* not in hypergraph."):
        hg.get_incident_edges(4)


def test_get_incident_edges_order_and_size_specified():
    """
    Test error when both order and size are specified.
    """
    hg = Hypergraph()
    hg.add_edge((1, 2, 3))
    with pytest.raises(ValueError, match="Order and size cannot be both specified."):
        hg.get_incident_edges(1, order=2, size=3)


def test_get_incident_edges_empty_adj_list():
    """
    Test retrieving incident edges when the node has no adjacent edges.
    """
    hg = Hypergraph()
    hg.add_node(1)
    incident_edges = hg.get_incident_edges(1)
    assert incident_edges == [], "Node 1 should have no incident edges."
