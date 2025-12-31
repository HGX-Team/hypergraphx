import pytest

from hypergraphx import DirectedHypergraph, Hypergraph, TemporalHypergraph
from hypergraphx.exceptions import MissingNodeError
from hypergraphx.utils.components import (
    connected_components,
    node_connected_component,
    num_connected_components,
    largest_component,
    largest_component_size,
    isolated_nodes,
    is_isolated,
    is_connected,
)


def _make_hypergraph():
    hg = Hypergraph(edge_list=[(0, 1), (2, 3)])
    hg.add_node(4)
    return hg


def test_connected_components_and_counts():
    """Test connected components and related helpers."""
    hg = _make_hypergraph()

    components = connected_components(hg)
    assert len(components) == 3
    assert num_connected_components(hg) == 3

    largest = largest_component(hg)
    assert len(largest) == 2
    assert largest_component_size(hg) == 2


def test_node_connected_component():
    """Test node component extraction."""
    hg = _make_hypergraph()
    comp = node_connected_component(hg, 0)
    assert set(comp) == {0, 1}


def test_isolated_nodes():
    """Test isolated node detection helpers."""
    hg = _make_hypergraph()

    assert isolated_nodes(hg) == [4]
    assert is_isolated(hg, 4) is True
    assert is_connected(hg) is False


def test_cc_invalid_args():
    """Test order/size validation."""
    hg = _make_hypergraph()
    with pytest.raises(ValueError, match="both specified"):
        connected_components(hg, order=1, size=2)


def test_connected_components_order_size_filters():
    hg = Hypergraph(edge_list=[(0, 1), (2, 3), (1, 2, 3)])
    hg.add_node(4)

    comps_order_1 = connected_components(hg, order=1)
    assert len(comps_order_1) == 3
    assert {frozenset(c) for c in comps_order_1} == {
        frozenset({0, 1}),
        frozenset({2, 3}),
        frozenset({4}),
    }

    comps_order_2 = connected_components(hg, order=2)
    assert len(comps_order_2) == 3
    assert {frozenset(c) for c in comps_order_2} == {
        frozenset({1, 2, 3}),
        frozenset({0}),
        frozenset({4}),
    }

    comp_size_2 = node_connected_component(hg, 2, size=2)
    assert set(comp_size_2) == {2, 3}
    comp_size_3 = node_connected_component(hg, 2, size=3)
    assert set(comp_size_3) == {1, 2, 3}

    assert num_connected_components(hg, order=1) == 3
    assert num_connected_components(hg, order=2) == 3
    assert set(largest_component(hg, order=1)) in ({0, 1}, {2, 3})
    assert set(largest_component(hg, order=2)) == {1, 2, 3}
    assert largest_component_size(hg, order=1) == 2
    assert largest_component_size(hg, order=2) == 3


def test_connected_components_size_equivalence():
    hg = Hypergraph(edge_list=[(0, 1), (1, 2, 3)])
    hg.add_node(4)

    comps_order_1 = connected_components(hg, order=1)
    comps_size_2 = connected_components(hg, size=2)
    assert {frozenset(c) for c in comps_order_1} == {frozenset(c) for c in comps_size_2}

    comps_order_2 = connected_components(hg, order=2)
    comps_size_3 = connected_components(hg, size=3)
    assert {frozenset(c) for c in comps_order_2} == {frozenset(c) for c in comps_size_3}


def test_connected_components_isolated_nodes_only():
    hg = Hypergraph()
    hg.add_nodes([0, 1, 2])
    comps = connected_components(hg)
    assert {frozenset(c) for c in comps} == {
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
    }


def test_node_connected_component_filtered_isolated():
    hg = Hypergraph(edge_list=[(0, 1), (1, 2, 3)])
    hg.add_node(4)

    comp = node_connected_component(hg, 4, order=1)
    assert set(comp) == {4}
    comp = node_connected_component(hg, 4, order=2)
    assert set(comp) == {4}


def test_connected_components_missing_order_returns_isolates():
    hg = Hypergraph(edge_list=[(0, 1), (1, 2)])
    comps = connected_components(hg, order=2)
    assert {frozenset(c) for c in comps} == {
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
    }
    assert num_connected_components(hg, order=2) == 3
    assert is_connected(hg, order=2) is False


def test_is_connected_filtering():
    hg = Hypergraph(edge_list=[(0, 1), (2, 3), (0, 1, 2, 3)])
    assert is_connected(hg, order=1) is False
    assert is_connected(hg, order=3) is True
    assert is_connected(hg, size=4) is True


def test_node_connected_component_missing_node_raises():
    hg = Hypergraph(edge_list=[(0, 1)])
    with pytest.raises(MissingNodeError):
        node_connected_component(hg, 99)


def test_connected_components_directed_hypergraph():
    hg = DirectedHypergraph(edge_list=[((0,), (1,)), ((2,), (3,))])
    hg.add_node(4)
    comps = connected_components(hg, order=1)
    assert {frozenset(c) for c in comps} == {
        frozenset({0, 1}),
        frozenset({2, 3}),
        frozenset({4}),
    }
    assert num_connected_components(hg, order=1) == 3


def test_connected_components_temporal_hypergraph():
    hg = TemporalHypergraph(edge_list=[(0, (0, 1)), (1, (1, 2, 3))])
    hg.add_node(4)
    comps_order_1 = connected_components(hg, order=1)
    assert {frozenset(c) for c in comps_order_1} == {
        frozenset({0, 1}),
        frozenset({2}),
        frozenset({3}),
        frozenset({4}),
    }
    comps_order_2 = connected_components(hg, order=2)
    assert {frozenset(c) for c in comps_order_2} == {
        frozenset({1, 2, 3}),
        frozenset({0}),
        frozenset({4}),
    }


def test_isolated_nodes_with_filter():
    hg = Hypergraph(edge_list=[(0, 1), (1, 2, 3)])
    assert set(isolated_nodes(hg, order=2)) == {0}
    assert set(isolated_nodes(hg, order=1)) == {2, 3}
    assert set(isolated_nodes(hg, order=3)) == {0, 1, 2, 3}


def test_connected_components_chain_pairs():
    hg = Hypergraph(edge_list=[(0, 1), (1, 2), (2, 3), (3, 4)])
    comps = connected_components(hg, order=1)
    assert len(comps) == 1
    assert set(comps[0]) == {0, 1, 2, 3, 4}
    assert is_connected(hg, order=1) is True
    assert largest_component_size(hg, order=1) == 5


def test_connected_components_two_cliques():
    clique_a = [(0, 1), (0, 2), (1, 2)]
    clique_b = [(3, 4), (3, 5), (4, 5)]
    hg = Hypergraph(edge_list=clique_a + clique_b)
    comps = connected_components(hg, order=1)
    assert {frozenset(c) for c in comps} == {
        frozenset({0, 1, 2}),
        frozenset({3, 4, 5}),
    }
    assert largest_component_size(hg, order=1) == 3


def test_connected_components_mixed_sizes_disconnected():
    hg = Hypergraph(edge_list=[(0, 1, 2), (3, 4), (4, 5, 6)])
    hg.add_node(7)
    comps_order_1 = connected_components(hg, order=1)
    assert {frozenset(c) for c in comps_order_1} == {
        frozenset({3, 4}),
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({5}),
        frozenset({6}),
        frozenset({7}),
    }
    comps_order_2 = connected_components(hg, order=2)
    assert {frozenset(c) for c in comps_order_2} == {
        frozenset({0, 1, 2}),
        frozenset({4, 5, 6}),
        frozenset({3}),
        frozenset({7}),
    }


def test_connected_components_uniform_hypergraph():
    hg = Hypergraph(edge_list=[(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)])
    comps = connected_components(hg, order=2)
    assert len(comps) == 1
    assert set(comps[0]) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
    assert is_connected(hg, order=2) is True


def test_connected_components_large_isolates():
    hg = Hypergraph()
    hg.add_nodes(list(range(10)))
    comps = connected_components(hg)
    assert len(comps) == 10
    assert all(len(c) == 1 for c in comps)
    assert is_connected(hg) is False


def test_connected_components_large_hyperedges_chain():
    hg = Hypergraph(
        edge_list=[
            (0, 1, 2, 3, 4),
            (4, 5, 6, 7, 8),
            (8, 9, 10, 11, 12),
        ]
    )
    comps = connected_components(hg, order=4)
    assert len(comps) == 1
    assert set(comps[0]) == set(range(13))
    assert is_connected(hg, order=4) is True


def test_connected_components_large_hyperedges_disconnected():
    hg = Hypergraph(
        edge_list=[
            (0, 1, 2, 3, 4),
            (5, 6, 7, 8, 9),
        ]
    )
    comps = connected_components(hg, size=5)
    assert {frozenset(c) for c in comps} == {
        frozenset({0, 1, 2, 3, 4}),
        frozenset({5, 6, 7, 8, 9}),
    }
    assert largest_component_size(hg, size=5) == 5


def test_connected_components_large_hyperedges_with_bridge_sizes():
    hg = Hypergraph(
        edge_list=[
            (0, 1, 2, 3, 4, 5),
            (5, 6, 7, 8, 9, 10),
            (10, 11, 12),
            (12, 13),
        ]
    )
    comps_order_5 = connected_components(hg, order=5)
    assert {frozenset(c) for c in comps_order_5} == {
        frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
        frozenset({11}),
        frozenset({12}),
        frozenset({13}),
    }
    comps_order_2 = connected_components(hg, order=2)
    assert {frozenset(c) for c in comps_order_2} == {
        frozenset({10, 11, 12}),
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
        frozenset({4}),
        frozenset({5}),
        frozenset({6}),
        frozenset({7}),
        frozenset({8}),
        frozenset({9}),
        frozenset({13}),
    }
    comps_order_1 = connected_components(hg, order=1)
    assert {frozenset(c) for c in comps_order_1} == {
        frozenset({12, 13}),
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
        frozenset({4}),
        frozenset({5}),
        frozenset({6}),
        frozenset({7}),
        frozenset({8}),
        frozenset({9}),
        frozenset({10}),
        frozenset({11}),
    }


def test_connected_components_mixed_sizes_with_large_overlap():
    hg = Hypergraph(
        edge_list=[
            (0, 1, 2, 3, 4, 5, 6),
            (6, 7, 8),
            (8, 9),
            (9, 10, 11, 12),
            (12, 13, 14, 15, 16),
        ]
    )
    comps_all = connected_components(hg)
    assert len(comps_all) == 1
    assert set(comps_all[0]) == set(range(17))

    comps_order_3 = connected_components(hg, order=3)
    assert {frozenset(c) for c in comps_order_3} == {
        frozenset({9, 10, 11, 12}),
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
        frozenset({4}),
        frozenset({5}),
        frozenset({6}),
        frozenset({7}),
        frozenset({8}),
        frozenset({13}),
        frozenset({14}),
        frozenset({15}),
        frozenset({16}),
    }

    comps_order_6 = connected_components(hg, order=6)
    assert {frozenset(c) for c in comps_order_6} == {
        frozenset({0, 1, 2, 3, 4, 5, 6}),
        frozenset({7}),
        frozenset({8}),
        frozenset({9}),
        frozenset({10}),
        frozenset({11}),
        frozenset({12}),
        frozenset({13}),
        frozenset({14}),
        frozenset({15}),
        frozenset({16}),
    }
