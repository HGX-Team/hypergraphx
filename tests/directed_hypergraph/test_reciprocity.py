import pytest

from hypergraphx import DirectedHypergraph
from hypergraphx.measures.directed.reciprocity import strong_reciprocity


def test_basic_reciprocity():
    # Test with simple reciprocated edges
    edges = [
        ((1,), (2,)),
        ((2,), (1,)),
        ((3,), (4,)),
        ((4,), (3,))
    ]
    h = DirectedHypergraph(edge_list=edges)
    result = strong_reciprocity(h, 3)

    # Expect 100% reciprocity for edges of size 2
    assert result[2] == 1.0, "Expected 100% reciprocity for edges of size 2"
    assert result[3] == 0, f"Expected no reciprocity for edges of size 3"


def test_no_reciprocity():
    # Test with no reciprocated edges
    edges = [
        ((1,), (2,)),
        ((3,), (4,)),
        ((5,), (6,))
    ]
    h = DirectedHypergraph(edge_list=edges)
    result = strong_reciprocity(h, 3)

    # Expect 0% reciprocity across all edge sizes
    for i in range(2, 3):
        assert result[i] == 0, f"Expected no reciprocity for edges of size {i}"


def test_mixed_edge_sizes():
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
    result = strong_reciprocity(h, 6)

    assert result[2] == 1.0, "Expected 100% reciprocity for edges of size 2"
    assert result[3] == 2/3, "Expected 50% reciprocity for edges of size 3"
    assert result[5] == 0.0, "Expected 0% reciprocity for edges of size 5"
    for i in [4, 6]:
        assert result[i] == 0, f"Expected no reciprocity for edges of size {i}"
