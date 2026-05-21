import pytest

from hypergraphx import Hypergraph
from hypergraphx.filters import filter_by_weight


def test_filter_by_weight_not_inplace_keeps_top_percent():
    hg = Hypergraph(weighted=True)
    hg.add_edge((0, 1), weight=1)
    hg.add_edge((1, 2), weight=2)
    hg.add_edge((2, 3), weight=3)
    hg.add_edge((3, 4), weight=4)

    out = filter_by_weight(hg, top_percent=50, inplace=False)

    assert out is not hg
    assert set(out.get_edges()) == {(2, 3), (3, 4)}
    assert set(hg.get_edges()) == {(0, 1), (1, 2), (2, 3), (3, 4)}


def test_filter_by_weight_keeps_ties_at_threshold():
    hg = Hypergraph(weighted=True)
    hg.add_edge((0, 1), weight=1)
    hg.add_edge((1, 2), weight=2)
    hg.add_edge((2, 3), weight=2)
    hg.add_edge((3, 4), weight=4)

    filter_by_weight(hg, top_percent=50)

    assert set(hg.get_edges()) == {(1, 2), (2, 3), (3, 4)}


def test_filter_by_weight_drops_isolated_nodes_after_filter():
    hg = Hypergraph(weighted=True)
    hg.add_node(99, metadata={"isolated": True})
    hg.add_edge((0, 1), weight=1)
    hg.add_edge((1, 2), weight=10)

    out = filter_by_weight(
        hg,
        top_percent=50,
        inplace=False,
        drop_isolated_nodes_after_filter=True,
    )

    assert set(out.get_edges()) == {(1, 2)}
    assert set(out.get_nodes()) == {1, 2}
    assert set(hg.get_nodes()) == {0, 1, 2, 99}


def test_filter_by_weight_method_shortcut():
    hg = Hypergraph(weighted=True)
    hg.add_edge((0, 1), weight=1)
    hg.add_edge((1, 2), weight=10)

    out = hg.filter_by_weight(top_percent=50, inplace=False)

    assert set(out.get_edges()) == {(1, 2)}
    assert set(hg.get_edges()) == {(0, 1), (1, 2)}


def test_filter_by_weight_validates_weighted_hypergraph_and_percent():
    hg = Hypergraph(weighted=False)
    hg.add_edge((0, 1))

    with pytest.raises(ValueError, match="weighted"):
        filter_by_weight(hg, top_percent=50)

    weighted = Hypergraph(weighted=True)
    weighted.add_edge((0, 1), weight=1)

    with pytest.raises(ValueError, match="top_percent"):
        filter_by_weight(weighted, top_percent=0)

    with pytest.raises(ValueError, match="top_percent"):
        filter_by_weight(weighted, top_percent=101)
