from hypergraphx import Hypergraph


def test_expose_and_populate_data_structures_roundtrip():
    """Test serialization roundtrip with weights and metadata."""
    hg = Hypergraph(
        edge_list=[(0, 1), (1, 2, 3)],
        weighted=True,
        weights=[1.5, 2.5],
        node_metadata={0: {"role": "a"}, 2: {"role": "b"}},
        edge_metadata=[{"kind": "pair"}, {"kind": "tri"}],
        hypergraph_metadata={"name": "serial"},
    )

    data = hg.expose_data_structures()
    restored = Hypergraph()
    restored.populate_from_dict(data)

    assert set(restored.get_nodes()) == set(hg.get_nodes())
    assert set(restored.get_edges()) == set(hg.get_edges())
    assert restored.is_weighted() is True
    assert restored.get_weight((0, 1)) == 1.5
    assert restored.get_edge_metadata((1, 2, 3)) == {"kind": "tri"}
    assert restored.get_hypergraph_metadata()["name"] == "serial"


def test_expose_attributes_for_hashing_consistency():
    """Test hashing attributes include nodes and edges metadata."""
    hg = Hypergraph(
        edge_list=[(0, 1)],
        weighted=True,
        weights=[2.0],
        node_metadata={0: {"role": "a"}},
        edge_metadata=[{"kind": "pair"}],
        hypergraph_metadata={"name": "hash"},
    )

    attrs = hg.expose_attributes_for_hashing()

    assert attrs["type"] == "Hypergraph"
    assert attrs["weighted"] is True
    assert attrs["hypergraph_metadata"]["name"] == "hash"
    assert attrs["edges"][0]["weight"] == 2.0
    assert attrs["nodes"][0]["metadata"] == {"role": "a"}
