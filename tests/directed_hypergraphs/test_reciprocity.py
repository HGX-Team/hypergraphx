import pytest

from hypergraphx import DirectedHypergraph
from hypergraphx.measures.directed.reciprocity import exact_reciprocity, strong_reciprocity, weak_reciprocity

# STRONG RECIPROCITY

def test_basic_exact_reciprocity():
    # Test with simple reciprocated edges
    edges = [
        ((1,), (2,)),
        ((2,), (1,)),
        ((3,), (4,)),
        ((4,), (3,))
    ]
    h = DirectedHypergraph(edge_list=edges)
    result = exact_reciprocity(h, 3)

    # Expect 100% reciprocity for edges of size 2
    assert result[2] == 1.0, "Expected 100% reciprocity for edges of size 2"
    assert result[3] == 0, f"Expected no reciprocity for edges of size 3"


def test_no_exact_reciprocity():
    # Test with no reciprocated edges
    edges = [
        ((1,), (2,)),
        ((3,), (4,)),
        ((5,), (6,))
    ]
    h = DirectedHypergraph(edge_list=edges)
    result = exact_reciprocity(h, 3)

    # Expect 0% reciprocity across all edge sizes
    for i in range(2, 3):
        assert result[i] == 0, f"Expected no reciprocity for edges of size {i}"


def test_mixed_edge_sizes_exact_reciprocity():
    # Test with mixed edge sizes
    edges = [
        ((1,), (2,)),  # size 2, no reciprocation
        ((2,), (1,)),  # size 2, reciprocated with above
        ((3, 4), (5,)),  # size 3, no reciprocation
        ((5,), (3, 4)),  # size 3, reciprocated with above
        ((6,), (7, 8)),  # size 3, no reciprocation
        ((9, 10), (11, 12, 13)),  # size 5, no reciprocation
    ]
    h = DirectedHypergraph(edge_list=edges)
    result = exact_reciprocity(h, 6)

    assert result[2] == 1.0, "Expected 100% reciprocity for edges of size 2"
    assert result[3] == 2/3, "Expected 50% reciprocity for edges of size 3"
    assert result[5] == 0.0, "Expected 0% reciprocity for edges of size 5"
    for i in [4, 6]:
        assert result[i] == 0, f"Expected no reciprocity for edges of size {i}"


# WEAK RECIPROCITY

def test_basic_weak_reciprocity():
    # Test with simple weak reciprocity where there are reverse node pairs
    edges = [
        ((1,), (2,)),      # Size 2, reciprocated with (2,), (1,)
        ((2,), (1,)),      # Size 2, reciprocated with above
        ((3,), (4,)),      # Size 2, no reciprocation
        ((5,), (6, 7)),    # Size 3, no reciprocation
    ]
    hypergraph = DirectedHypergraph(edge_list=edges)
    result = weak_reciprocity(hypergraph, max_hyperedge_size=3)

    # Expected: 50% reciprocity for size 2, 0% for size 3
    assert result[2] == 2/3, "Expected 50% weak reciprocity for edges of size 2"
    assert result[3] == 0.0, "Expected no weak reciprocity for edges of size 3"

def test_no_weak_reciprocity():
    # Test with no reciprocated edges, no reverse pairs
    edges = [
        ((1,), (2,)),   # Size 2, no reciprocation
        ((3,), (4,)),   # Size 2, no reciprocation
        ((5,), (6,)),   # Size 2, no reciprocation
    ]
    hypergraph = DirectedHypergraph(edge_list=edges)
    result = weak_reciprocity(hypergraph, max_hyperedge_size=2)

    # Expected: 0 reciprocity for all sizes
    for i in range(2, 3):
        assert result[i] == 0, f"Expected no reciprocity for edges of size {i}"

def test_mixed_edge_sizes_weak_reciprocity():
    # Test with a mix of reciprocated and non-reciprocated edges across sizes
    edges = [
        ((1,), (2,)),           # Size 2, reciprocated with (2,), (1,)
        ((2,), (1,)),           # Size 2, reciprocated with above
        ((3, 4), (5,)),         # Size 3, no reciprocation
        ((6,), (7, 8)),         # Size 3, no reciprocation
        ((9, 10), (11, 12)),    # Size 4, reciprocated with (11, 12), (9, 10)
        ((11,), (10,))          # Size 2, reciprocated with above
    ]
    hypergraph = DirectedHypergraph(edge_list=edges)
    result = weak_reciprocity(hypergraph, max_hyperedge_size=4)

    assert result[2] == 1.0, "Expected 66.6% weak reciprocity for edges of size 2"
    assert result[3] == 0.0, "Expected 0% weak reciprocity for edges of size 3"
    assert result[4] == 1.0, "Expected 100% weak reciprocity for edges of size 4"

def test_empty_hypergraph():
    # Test with no edges
    edges = []
    hypergraph = DirectedHypergraph(edge_list=edges)
    result = weak_reciprocity(hypergraph, max_hyperedge_size=5)

    # Expected: 0 reciprocity for all edge sizes up to max_hyperedge_size
    for i in range(2, 6):
        assert result[i] == 0, f"Expected no reciprocity for edges of size {i}"

def test_large_edge_size_out_of_bounds():
    # Test with edges larger than max_hyperedge_size (should be ignored)
    edges = [
        ((1, 2, 3), (4, 5, 6)),  # Size 6, should be ignored
        ((10,), (11,)),           # Size 2, no reciprocity
    ]
    hypergraph = DirectedHypergraph(edge_list=edges)
    result = weak_reciprocity(hypergraph, max_hyperedge_size=5)

    # Only edge size 2 should be considered, with no reciprocity
    assert result[2] == 0, "Expected no weak reciprocity for edge size 2"
    for i in range(3, 6):
        assert result[i] == 0, f"Expected no reciprocity for edges of size {i}"
