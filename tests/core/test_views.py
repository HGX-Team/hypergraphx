import pytest

from hypergraphx import Hypergraph, MultiplexHypergraph, TemporalHypergraph


def test_node_view_dynamic_and_membership():
    hg = Hypergraph(edge_list=[(1, 2), (2, 3, 4)], weighted=False)

    nodes = hg.nodes
    assert set(nodes) == {1, 2, 3, 4}
    assert 2 in nodes
    assert 99 not in nodes
    assert len(nodes) == 4

    hg.add_node(5)
    assert 5 in nodes
    assert len(nodes) == 5


def test_edge_view_basic_and_size_filter():
    hg = Hypergraph(edge_list=[(2, 1), (2, 3, 4)], weighted=False)

    edges = hg.edges
    assert len(edges) == 2
    assert (1, 2) in edges
    assert (2, 1) in edges  # normalized membership

    size2 = hg.edges.size(2)
    assert set(size2) == {(1, 2)}
    assert (1, 2) in size2
    assert (2, 3, 4) not in size2


def test_temporal_edge_view_time_window_and_size_filter():
    th = TemporalHypergraph(
        edge_list=[(1, 2), (1, 2, 3)],
        time_list=[0, 5],
        weighted=False,
    )

    tw = th.edges.time_window((0, 3))
    assert (0, (2, 1)) in tw
    assert (5, (1, 2, 3)) not in tw
    assert list(tw) == [(0, (1, 2))]

    size2 = th.edges.size(2)
    assert (0, (2, 1)) in size2
    assert (5, (1, 2, 3)) not in size2


def test_multiplex_edge_view_layer_and_membership_compat():
    mh = MultiplexHypergraph(
        edge_list=[(1, 2), (2, 3)],
        edge_layer=["L1", "L2"],
        weighted=False,
    )

    l1 = mh.edges.layer("L1")
    assert ("L1", (2, 1)) in l1
    assert ((2, 1), "L1") in l1  # backward compatible packed form
    assert ("L2", (2, 3)) not in l1


def test_multiplex_get_edges_filters():
    mh = MultiplexHypergraph(
        edge_list=[(1, 2), (2, 3, 4), (5, 6)],
        edge_layer=["L1", "L1", "L2"],
        weighted=False,
    )

    assert set(mh.get_edges(layer="L1")) == {("L1", (1, 2)), ("L1", (2, 3, 4))}
    assert set(mh.get_edges(size=2)) == {("L1", (1, 2)), ("L2", (5, 6))}
    assert set(mh.get_edges(layer="L1", size=2)) == {("L1", (1, 2))}
    assert list(mh.get_edges(layer="L1", metadata=True).keys()) == list(
        mh.get_edges(layer="L1")
    )


def test_iter_helpers_match_views():
    hg = Hypergraph(edge_list=[(1, 2), (2, 3, 4)], weighted=False)
    assert list(hg.iter_nodes()) == list(hg.nodes)
    assert list(hg.iter_edges()) == list(hg.edges)


def test_layer_filter_on_non_multiplex_errors():
    hg = Hypergraph(edge_list=[(1, 2)], weighted=False)
    with pytest.raises(TypeError):
        _ = hg.edges.layer("L1")


def test_time_window_filter_on_non_temporal_errors():
    hg = Hypergraph(edge_list=[(1, 2)], weighted=False)
    with pytest.raises(TypeError):
        _ = hg.edges.time_window((0, 1))
