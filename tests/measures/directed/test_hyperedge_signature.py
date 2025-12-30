import numpy as np
import pytest

from hypergraphx import DirectedHypergraph, Hypergraph
from hypergraphx.measures.directed.hyperedge_signature import hyperedge_signature_vector


def test_hyperedge_signature_vector_counts():
    """Test hyperedge signature vector counts source/target sizes."""
    edges = [((0, 1), (2,)), ((0,), (1, 2, 3))]
    hg = DirectedHypergraph(edge_list=edges)

    signature = hyperedge_signature_vector(hg)

    assert isinstance(signature, np.ndarray)
    assert signature.sum() == 2


def test_hyperedge_signature_empty_hypergraph():
    """Test empty directed hypergraph returns empty signature."""
    hg = DirectedHypergraph(edge_list=[])
    signature = hyperedge_signature_vector(hg)
    assert signature.size == 0


def test_hyperedge_signature_requires_directed():
    """Test signature rejects non-directed hypergraph."""
    with pytest.raises(ValueError, match="DirectedHypergraph"):
        hyperedge_signature_vector(Hypergraph(edge_list=[(0, 1)]))
