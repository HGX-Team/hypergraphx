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



