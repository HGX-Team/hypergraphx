from hypergraphx import (
    DirectedHypergraph,
    Hypergraph,
    MultiplexHypergraph,
    TemporalHypergraph,
)


def test_hypergraph_normalize_ids_preserves_data():
    hg = Hypergraph(weighted=True, hypergraph_metadata={"name": "plain"})
    hg.add_node(20, metadata={"label": "twenty"})
    hg.add_node(10, metadata={"label": "ten"})
    hg.add_node(30, metadata={"label": "thirty"})
    hg.add_edge((20, 10), weight=2.5, metadata={"kind": "pair"})
    hg.set_incidence_metadata((10, 20), 10, {"role": "left"})

    mapping = hg.normalize_ids()

    assert mapping == {10: 0, 20: 1, 30: 2}
    assert set(hg.get_nodes()) == {0, 1, 2}
    assert hg.get_node_metadata(0) == {
        "label": "ten",
        "id_before_normalization": 10,
    }
    assert hg.get_node_metadata(1) == {
        "label": "twenty",
        "id_before_normalization": 20,
    }
    assert hg.get_node_metadata(2) == {
        "label": "thirty",
        "id_before_normalization": 30,
    }
    assert hg.get_edges() == [(0, 1)]
    assert hg.get_weight((0, 1)) == 2.5
    assert hg.get_edge_metadata((0, 1)) == {"kind": "pair"}
    assert hg.get_incidence_metadata((0, 1), 0) == {"role": "left"}
    assert hg.get_hypergraph_metadata()["name"] == "plain"


def test_directed_hypergraph_normalize_ids_preserves_direction():
    dhg = DirectedHypergraph(weighted=True)
    dhg.add_edge(((20, 10), (30,)), weight=4, metadata={"kind": "directed"})

    mapping = dhg.normalize_ids()

    assert mapping == {10: 0, 20: 1, 30: 2}
    assert dhg.get_edges() == [((0, 1), (2,))]
    assert dhg.get_weight(((0, 1), (2,))) == 4
    assert dhg.get_edge_metadata(((0, 1), (2,))) == {"kind": "directed"}
    assert dhg.get_node_metadata(0)["id_before_normalization"] == 10
    assert dhg.get_node_metadata(1)["id_before_normalization"] == 20
    assert dhg.get_node_metadata(2)["id_before_normalization"] == 30


def test_temporal_hypergraph_normalize_ids_preserves_time():
    thg = TemporalHypergraph(weighted=True)
    thg.add_edge((20, 10), time=5, weight=3, metadata={"kind": "temporal"})
    thg.add_node(30, metadata={"isolated": True})

    mapping = thg.normalize_ids()

    assert mapping == {10: 0, 20: 1, 30: 2}
    assert thg.get_edges() == [(5, (0, 1))]
    assert thg.get_weight((0, 1), time=5) == 3
    assert thg.get_edge_metadata((0, 1), time=5) == {"kind": "temporal"}
    assert thg.get_node_metadata(2) == {
        "isolated": True,
        "id_before_normalization": 30,
    }


def test_multiplex_hypergraph_normalize_ids_preserves_layer():
    mhg = MultiplexHypergraph(weighted=True)
    mhg.add_edge((20, 10), layer="L1", weight=7, metadata={"kind": "multiplex"})
    mhg.add_edge((10, 30), layer="L2", weight=1)

    mapping = mhg.normalize_ids()

    assert mapping == {10: 0, 20: 1, 30: 2}
    assert set(mhg.get_edges()) == {("L1", (0, 1)), ("L2", (0, 2))}
    assert mhg.get_weight((0, 1), layer="L1") == 7
    assert mhg.get_edge_metadata((0, 1), layer="L1") == {"kind": "multiplex"}
    assert mhg.get_existing_layers() == {"L1", "L2"}
    assert mhg.get_node_metadata(0)["id_before_normalization"] == 10
