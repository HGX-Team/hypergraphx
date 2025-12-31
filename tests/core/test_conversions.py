import pytest

from hypergraphx import (
    DirectedHypergraph,
    Hypergraph,
    MultiplexHypergraph,
    TemporalHypergraph,
)


def test_directed_to_hypergraph_merges_weights_and_metadata():
    dhg = DirectedHypergraph(weighted=True)
    dhg.add_edge(((0, 1), (2,)), weight=2.0, metadata={"a": 1})
    dhg.add_edge(((2,), (0, 1)), weight=3.0, metadata={"a": 2, "b": "x"})
    dhg.add_node(5, metadata={"role": "isolated"})

    hg = dhg.to_hypergraph()

    assert isinstance(hg, Hypergraph)
    assert hg.get_weight((0, 1, 2)) == 5.0
    assert hg.get_edge_metadata((0, 1, 2)) == {"a": [1, 2], "b": "x"}
    assert hg.get_node_metadata(5) == {"role": "isolated"}
    assert hg.get_hypergraph_metadata().get("converted_from") == "DirectedHypergraph"


def test_directed_to_hypergraph_unweighted_counts():
    dhg = DirectedHypergraph(weighted=False)
    dhg.add_edge(((0,), (1,)))
    dhg.add_edge(((1,), (0,)))
    hg = dhg.to_hypergraph()
    assert hg.is_weighted()
    assert hg.get_weight((0, 1)) == 2


def test_temporal_to_hypergraph_merges_by_edge():
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge((0, 1), time=1, weight=1.0, metadata={"t": 1})
    thg.add_edge((0, 1), time=2, weight=2.0, metadata={"t": 2})
    thg.add_edge((1, 2), time=3, weight=3.0)

    hg = thg.to_hypergraph()

    assert hg.get_weight((0, 1)) == 3.0
    assert hg.get_edge_metadata((0, 1)) == {"t": [1, 2]}
    assert hg.get_weight((1, 2)) == 3.0
    assert hg.get_hypergraph_metadata().get("converted_from") == "TemporalHypergraph"


def test_multiplex_to_hypergraph_merges_by_edge():
    mhg = MultiplexHypergraph(weighted=True)
    mhg.add_edge((0, 1), layer="L1", weight=1.0, metadata={"layer": "L1"})
    mhg.add_edge((0, 1), layer="L2", weight=2.0, metadata={"layer": "L2"})
    mhg.add_edge((1, 2), layer="L2", weight=3.0)

    hg = mhg.to_hypergraph()

    assert hg.get_weight((0, 1)) == 3.0
    assert hg.get_edge_metadata((0, 1)) == {"layer": ["L1", "L2"]}
    assert hg.get_weight((1, 2)) == 3.0
    assert hg.get_hypergraph_metadata().get("converted_from") == "MultiplexHypergraph"


def test_conversion_respects_keep_metadata_flags():
    mhg = MultiplexHypergraph(weighted=True)
    mhg.add_node("A", metadata={"role": "node"})
    mhg.add_edge(("A", "B"), layer="L1", weight=1.0, metadata={"tag": "x"})

    hg = mhg.to_hypergraph(
        keep_node_metadata=False,
        keep_edge_metadata=False,
        keep_hypergraph_metadata=False,
    )

    assert hg.get_node_metadata("A") == {}
    assert hg.get_edge_metadata(("A", "B")) == {}
    assert "converted_from" not in hg.get_hypergraph_metadata()


def test_merge_metadata_list_accumulation():
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge((0, 1), time=1, weight=1.0, metadata={"tag": ["a", "b"]})
    thg.add_edge((0, 1), time=2, weight=1.0, metadata={"tag": "c"})
    thg.add_edge((0, 1), time=3, weight=1.0, metadata={"tag": ["b", "d"]})

    hg = thg.to_hypergraph()

    assert hg.get_edge_metadata((0, 1)) == {"tag": ["a", "b", "c", "d"]}


def test_merge_metadata_preserves_equal_values():
    dhg = DirectedHypergraph(weighted=True)
    dhg.add_edge(((0,), (1,)), weight=1.0, metadata={"kind": "same"})
    dhg.add_edge(((1,), (0,)), weight=1.0, metadata={"kind": "same"})

    hg = dhg.to_hypergraph()

    assert hg.get_edge_metadata((0, 1)) == {"kind": "same"}


def test_multiplex_to_hypergraph_unweighted_counts():
    mhg = MultiplexHypergraph(weighted=False)
    mhg.add_edge((0, 1), layer="L1")
    mhg.add_edge((0, 1), layer="L2")
    mhg.add_edge((1, 2), layer="L2")

    hg = mhg.to_hypergraph()

    assert hg.is_weighted()
    assert hg.get_weight((0, 1)) == 2
    assert hg.get_weight((1, 2)) == 1


def test_temporal_to_hypergraph_unweighted_counts():
    thg = TemporalHypergraph(weighted=False)
    thg.add_edge((0, 1), time=1)
    thg.add_edge((0, 1), time=2)
    thg.add_edge((1, 2), time=3)

    hg = thg.to_hypergraph()

    assert hg.is_weighted()
    assert hg.get_weight((0, 1)) == 2
    assert hg.get_weight((1, 2)) == 1


def test_merge_metadata_mixed_types_to_list():
    mhg = MultiplexHypergraph(weighted=True)
    mhg.add_edge((0, 1), layer="L1", weight=1.0, metadata={"tag": "a"})
    mhg.add_edge((0, 1), layer="L2", weight=1.0, metadata={"tag": ["b", "c"]})

    hg = mhg.to_hypergraph()

    assert hg.get_edge_metadata((0, 1)) == {"tag": ["a", "b", "c"]}


def test_merge_metadata_with_none_is_noop():
    dhg = DirectedHypergraph(weighted=True)
    dhg.add_edge(((0,), (1,)), weight=1.0, metadata={"kind": "x"})
    dhg.add_edge(((1,), (0,)), weight=1.0, metadata=None)

    hg = dhg.to_hypergraph()

    assert hg.get_edge_metadata((0, 1)) == {"kind": "x"}


def test_conversion_preserves_isolated_node_metadata():
    thg = TemporalHypergraph(weighted=True)
    thg.add_node("isolated", metadata={"role": "ghost"})
    thg.add_edge((0, 1), time=1, weight=1.0)

    hg = thg.to_hypergraph()

    assert hg.get_node_metadata("isolated") == {"role": "ghost"}


def test_conversion_does_not_mutate_source_metadata():
    mhg = MultiplexHypergraph(weighted=True)
    mhg.add_edge((0, 1), layer="L1", weight=1.0, metadata={"tag": ["a"]})
    mhg.add_edge((0, 1), layer="L2", weight=1.0, metadata={"tag": "b"})
    original = mhg.get_edge_metadata((0, 1), "L1")

    _ = mhg.to_hypergraph()

    assert original == {"tag": ["a"]}


def test_directed_to_hypergraph_distinct_edges():
    dhg = DirectedHypergraph(weighted=True)
    dhg.add_edge(((0,), (1,)), weight=1.0)
    dhg.add_edge(((2,), (3,)), weight=2.0)

    hg = dhg.to_hypergraph()

    assert hg.get_weight((0, 1)) == 1.0
    assert hg.get_weight((2, 3)) == 2.0


def test_conversion_keep_edge_metadata_false():
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge((0, 1), time=1, weight=1.0, metadata={"t": 1})
    hg = thg.to_hypergraph(keep_edge_metadata=False)
    assert hg.get_edge_metadata((0, 1)) == {}
