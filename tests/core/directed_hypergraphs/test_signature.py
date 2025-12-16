import numpy as np

from hypergraphx import DirectedHypergraph
from hypergraphx.measures.directed import hyperedge_signature_vector


def test_hyperedge_signature_vector_basic():
    """Test basic functionality of hyperedge signature vector calculation."""
    edges = [
        ((1, 2), (3, 4)),  # Hyperedge with source size 2, target size 2
        ((5, ), (6, 7, 8)),  # Hyperedge with source size 1, target size 3
    ]

    hypergraph = DirectedHypergraph(edges)

    # Test without max_hyperedge_size
    result = hyperedge_signature_vector(hypergraph)
    print(result)
    expected = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0])  # 3x3 matrix flattened
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

def test_hyperedge_signature_vector_with_max_size():
    """Test hyperedge signature vector with max_hyperedge_size specified."""
    edges = [
        ((1, 2), (3, 4)),  # Hyperedge with size 4 (2 + 2)
        ((5,), (6, 7, 8)),  # Hyperedge with size 4 (1 + 3)
        ((9, 10, 11), (12,)),  # Hyperedge with size 4 (3 + 1)
    ]
    hypergraph = DirectedHypergraph(edges)

    # Specify max_hyperedge_size = 2 (no edges should be counted)
    result = hyperedge_signature_vector(hypergraph, max_hyperedge_size=2)
    print(result)
    expected = np.array([0])  # all edges exceed max size 2
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

def test_hyperedge_signature_vector_empty_hypergraph():
    """Test the function with an empty hypergraph."""
    hypergraph = DirectedHypergraph(edge_list=[])
    result = hyperedge_signature_vector(hypergraph)
    expected = np.array([])  # Expecting an empty array for an empty hypergraph
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

def test_hyperedge_signature_vector_large_edges_ignored():
    """Test that edges larger than max_hyperedge_size are ignored."""
    edges = [
        ((1, 2), (3, 4)),  # Hyperedge with size 4 (2 + 2)
        ((5,), (6, 7, 8)),  # Hyperedge with size 4 (1 + 3)
        ((9, 10, 11, 12), (13, 14, 15, 16)),  # Hyperedge with size 8 (4 + 4)
    ]
    hypergraph = DirectedHypergraph(edges)

    # Specify max_hyperedge_size = 4 (only edges with size <= 4 should be counted)
    result = hyperedge_signature_vector(hypergraph, max_hyperedge_size=4)
    expected = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0])  # 3x3 matrix flattened
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"