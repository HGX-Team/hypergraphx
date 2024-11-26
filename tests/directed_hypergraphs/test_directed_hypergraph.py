import pytest

from hypergraphx import DirectedHypergraph


@pytest.fixture
def directed_hypergraph():
    """
    Fixture to provide a fresh instance of DirectedHypergraph for each test.
    """
    return DirectedHypergraph()

def test_add_single_node(directed_hypergraph):
    """
    Test adding a single node to the hypergraph.
    """
    node = 'A'
    directed_hypergraph.add_node(node)
    assert node in directed_hypergraph.get_nodes(), f"Node {node} should be in the hypergraph."

def test_add_multiple_nodes(directed_hypergraph):
    """
    Test adding multiple nodes to the hypergraph.
    """
    nodes = ['A', 'B', 'C']
    directed_hypergraph.add_nodes(nodes)
    assert all(node in directed_hypergraph.get_nodes() for node in nodes), "All nodes should be in the hypergraph."

def test_add_edge(directed_hypergraph):
    """
    Test adding an edge to the hypergraph.
    """
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    directed_hypergraph.add_edge(edge)
    assert edge in directed_hypergraph._edge_list, "The edge should be added to the hypergraph."

def test_add_weighted_edge():
    """
    Test adding a weighted edge to the hypergraph.
    """
    hg = DirectedHypergraph(weighted=True)
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    weight = 2.5
    hg.add_edge(edge, weight=weight)
    assert hg.get_weight(edge) == weight, "The edge weight should be set correctly."

def test_add_edges(directed_hypergraph):
    """
    Test adding multiple edges to the hypergraph.
    """
    edge_list = [(('A', 'B'), ('C',)), (('C',), ('D', 'E'))]
    directed_hypergraph.add_edges(edge_list)
    assert all(edge in directed_hypergraph._edge_list for edge in edge_list), "All edges should be added to the hypergraph."

def test_remove_edge(directed_hypergraph):
    """
    Test removing an edge from the hypergraph.
    """
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    directed_hypergraph.add_edge(edge)
    directed_hypergraph.remove_edge(edge)
    assert edge not in directed_hypergraph._edge_list, "The edge should be removed from the hypergraph."

def test_remove_node_with_edges(directed_hypergraph):
    """
    Test removing a node and ensure associated edges are also removed.
    """
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    directed_hypergraph.add_edge(edge)
    directed_hypergraph.remove_node('A')
    assert edge not in directed_hypergraph._edge_list, "Edges associated with the node should be removed."
    assert 'A' not in directed_hypergraph.get_nodes(), "The node should be removed from the hypergraph."

def test_weighted_hypergraph():
    """
    Test if the hypergraph is properly recognized as weighted.
    """
    hg = DirectedHypergraph(weighted=True)
    assert hg.is_weighted(), "The hypergraph should be weighted."

def test_unweighted_hypergraph():
    """
    Test if the hypergraph is properly recognized as unweighted.
    """
    hg = DirectedHypergraph(weighted=False)
    assert not hg.is_weighted(), "The hypergraph should be unweighted."

def test_add_duplicate_edge_unweighted(directed_hypergraph):
    """
    Test adding a duplicate edge in an unweighted hypergraph.
    """
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    directed_hypergraph.add_edge(edge)
    directed_hypergraph.add_edge(edge)  # Adding the same edge again
    edges = list(directed_hypergraph._edge_list.keys())
    assert edges.count(edge) == 1, "Duplicate edges should not be added in an unweighted hypergraph."

def test_add_duplicate_edge_weighted():
    """
    Test adding a duplicate edge in a weighted hypergraph.
    """
    hg = DirectedHypergraph(weighted=True)
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    hg.add_edge(edge, weight=2.0)
    hg.add_edge(edge, weight=3.0)  # Adding the same edge with a new weight
    assert hg.get_weight(edge) == 5.0, "The weight of the duplicate edge should be updated to the latest value."

def test_add_edge_without_weight_in_weighted_hypergraph():
    """
    Test adding an edge without specifying a weight in a weighted hypergraph.
    """
    hg = DirectedHypergraph(weighted=True)
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    hg.add_edge(edge)  # No weight specified
    assert hg.get_weight(edge) == 1.0, "Default weight should be set to 1.0 in a weighted hypergraph."

def test_remove_edge_weighted_hypergraph():
    """
    Test removing an edge from a weighted hypergraph and its weight.
    """
    hg = DirectedHypergraph(weighted=True)
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    weight = 2.0
    hg.add_edge(edge, weight=weight)
    hg.remove_edge(edge)
    assert edge not in hg._edge_list, "The edge should be removed from the hypergraph."
    with pytest.raises(ValueError):
        hg.get_weight(edge)  # Accessing weight of a removed edge should raise an error

def test_edge_metadata_with_weights():
    """
    Test adding metadata to a weighted edge and verifying it.
    """
    hg = DirectedHypergraph(weighted=True)
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    metadata = {"type": "interaction", "priority": 5}
    hg.add_edge(edge, weight=2.5, metadata=metadata)
    assert hg.get_edge_metadata(edge) == metadata, "The edge metadata should match the provided metadata."

def test_add_edge_with_different_weights():
    """
    Test adding the same edge with different weights in a weighted hypergraph.
    """
    hg = DirectedHypergraph(weighted=True)
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    hg.add_edge(edge, weight=1.5)
    hg.add_edge(edge, weight=3.0)  # Update weight
    assert hg.get_weight(edge) == 4.5, "The edge weight should be updated to the latest value."

def test_remove_node_weighted_edges(directed_hypergraph):
    """
    Test removing a node from a weighted hypergraph and ensuring its edges and weights are removed.
    """
    directed_hypergraph = DirectedHypergraph(weighted=True)
    source = ('A', 'B')
    target = ('C',)
    edge = (source, target)
    directed_hypergraph.add_edge(edge, weight=2.5)
    directed_hypergraph.remove_node('A')
    assert edge not in directed_hypergraph._edge_list, "Edges associated with the removed node should be deleted."
    with pytest.raises(ValueError):
        directed_hypergraph.get_weight(edge)  # Accessing weight of a removed edge should raise an error

def test_add_edges_with_different_weights():
    """
    Test adding multiple edges with different weights to a weighted hypergraph.
    """
    hg = DirectedHypergraph(weighted=True)
    edge_list = [(('A',), ('B',)), (('B', 'C'), ('D',))]
    weights = [1.0, 2.5]
    hg.add_edges(edge_list, weights=weights)
    for edge, weight in zip(edge_list, weights):
        assert hg.get_weight(edge) == weight, f"The weight of {edge} should be {weight}."

def test_duplicate_edges_different_weights():
    """
    Test adding duplicate edges with different weights in a weighted hypergraph.
    """
    hg = DirectedHypergraph(weighted=True)
    edge = (('A', 'B'), ('C',))
    hg.add_edge(edge, weight=2.0)
    hg.add_edge(edge, weight=5.0)  # Add duplicate edge with a different weight
    assert hg.get_weight(edge) == 7.0, "The weight should be updated to the latest value for duplicate edges."

def test_default_weight_is_one_in_weighted_hypergraph():
    """
    Test that the weight defaults to 1 when adding an edge with no weight in a weighted hypergraph.
    """
    hg = DirectedHypergraph(weighted=True)
    source = ('A',)
    target = ('B',)
    edge = (source, target)
    hg.add_edge(edge)  # No weight specified
    assert hg.get_weight(edge) == 1.0, "The default weight should be 1.0 for an edge added without a weight."

