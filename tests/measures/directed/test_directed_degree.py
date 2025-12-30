from hypergraphx import DirectedHypergraph
from hypergraphx.measures.directed.degree import (
    in_degree,
    out_degree,
    in_degree_sequence,
    out_degree_sequence,
)


def _make_directed():
    edges = [((0,), (1, 2)), ((2,), (0,)), ((1,), (2,))]
    return DirectedHypergraph(edge_list=edges)


def test_in_out_degree():
    """Test in/out degree counts on directed hypergraph."""
    hg = _make_directed()

    assert in_degree(hg, 0) == 1
    assert out_degree(hg, 0) == 1


def test_in_out_degree_sequence():
    """Test in/out degree sequences include all nodes."""
    hg = _make_directed()

    in_seq = in_degree_sequence(hg)
    out_seq = out_degree_sequence(hg)

    assert set(in_seq.keys()) == set(hg.get_nodes())
    assert set(out_seq.keys()) == set(hg.get_nodes())
