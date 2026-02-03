import networkx as nx

from hypergraphx import Hypergraph, DirectedHypergraph
from hypergraphx.representations.projections import (
    bipartite_projection,
    clique_projection,
    line_graph,
    directed_line_graph,
)


def test_bipartite_projection_structure():
    """Test bipartite projection returns expected node types."""
    hg = Hypergraph(edge_list=[(0, 1), (1, 2, 3)])
    g, id_to_obj = bipartite_projection(hg)

    assert isinstance(g, nx.Graph)
    assert any(node.startswith("N") for node in g.nodes)
    assert any(node.startswith("E") for node in g.nodes)
    assert len(id_to_obj) == g.number_of_nodes()

    g2, id_to_obj2, obj_to_id2 = bipartite_projection(
        hg,
        node_order=[0, 1, 2, 3],
        edge_order=[(0, 1), (1, 2, 3)],
        return_obj_to_id=True,
    )
    assert isinstance(obj_to_id2, dict)
    assert id_to_obj2[obj_to_id2[0]] == 0


def test_clique_projection_edges():
    """Test clique projection adds pairwise edges for hyperedges."""
    hg = Hypergraph(edge_list=[(0, 1, 2)])
    g = clique_projection(hg)

    assert g.has_edge(0, 1)
    assert g.has_edge(0, 2)
    assert g.has_edge(1, 2)


def test_line_graph_mapping():
    """Test line graph returns mapping of edges."""
    hg = Hypergraph(edge_list=[(0, 1), (1, 2)])
    g, id_to_edge = line_graph(hg, distance="intersection", s=1, weighted=True)

    assert g.number_of_nodes() == len(hg.get_edges())
    assert set(id_to_edge.values()) == set(hg.get_edges())

    # Deterministic mapping with explicit order.
    g2, id_to_edge2 = line_graph(
        hg, distance="intersection", s=1, weighted=True, edge_order=[(1, 2), (0, 1)]
    )
    assert id_to_edge2[0] == (1, 2)
    assert id_to_edge2[1] == (0, 1)


def test_directed_line_graph():
    """Test directed line graph creates nodes for directed hyperedges."""
    hg = DirectedHypergraph(edge_list=[((0,), (1,)), ((1,), (2,))])
    g, id_to_edge = directed_line_graph(hg, distance="intersection", s=1)

    assert g.number_of_nodes() == len(hg.get_edges())
    assert set(id_to_edge.values()) == set(hg.get_edges())
