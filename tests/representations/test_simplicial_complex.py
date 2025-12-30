from hypergraphx import Hypergraph
from hypergraphx.representations.simplicial_complex import simplicial_complex


def test_simplicial_complex_contains_subsets():
    """Test simplicial complex includes all subsets of edges."""
    hg = Hypergraph(edge_list=[(0, 1, 2)])
    sc = simplicial_complex(hg)

    assert (0, 1, 2) in sc.get_edges()
    assert (0, 1) in sc.get_edges()
    assert (0,) in sc.get_edges()
    assert () in sc.get_edges()
