import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from jsonschema import Draft7Validator

from hypergraphx import DirectedHypergraph, Hypergraph
from hypergraphx.readwrite import (
    HIFJson,
    from_hif_dict,
    read_hif,
    to_hif_dict,
    write_hif,
)

HIF_SCHEMA_PATH = Path(__file__).with_name("hif_schema_v0.1.0.json")
HIF_VALIDATOR = Draft7Validator(json.loads(HIF_SCHEMA_PATH.read_text(encoding="utf-8")))


def assert_valid_hif(data: HIFJson) -> None:
    HIF_VALIDATOR.validate(data)


def assert_round_trip(
    data: HIFJson, *, check_identity: bool = True
) -> Hypergraph | DirectedHypergraph:
    assert_valid_hif(data)
    H = from_hif_dict(data)
    converted = to_hif_dict(H)
    assert_valid_hif(converted)
    if check_identity:
        assert converted == data
    with TemporaryDirectory() as directory:
        path = Path(directory) / "network.json"
        write_hif(H, path)
        assert to_hif_dict(read_hif(path)) == converted
    return H


def test_undirected_hif_roundtrip():
    data: HIFJson = {
        "network-type": "undirected",
        "metadata": {"name": "in-memory"},
        "nodes": [
            {"node": "isolated", "weight": 3, "attrs": {"color": "red"}},
            {"node": "alice"},
            {"node": "bob"},
        ],
        "edges": [
            {"edge": 0, "weight": 2, "attrs": {"kind": "social"}},
            {"edge": "empty", "weight": 4},
        ],
        "incidences": [
            {
                "edge": 0,
                "node": "alice",
                "weight": 0.5,
                "attrs": {"since": 2020},
            },
            {"edge": 0, "node": "bob"},
        ],
    }
    from_dict = assert_round_trip(data)

    assert isinstance(from_dict, Hypergraph)
    assert from_dict.get_edges() == [("alice", "bob")]
    assert from_dict.get_hypergraph_metadata() == {"name": "in-memory"}
    assert from_dict.get_weight(("alice", "bob")) == 2
    assert from_dict.get_edge_metadata(("alice", "bob")) == {"kind": "social"}
    assert from_dict.get_node_metadata("isolated") == {
        "color": "red",
        "weight": 3,
    }
    assert from_dict.get_incidence_metadata(("alice", "bob"), "alice") == {
        "since": 2020,
        "weight": 0.5,
    }
    assert (("alice", "bob"), "bob") not in from_dict.get_all_incidences_metadata()
    assert from_dict.expose_data_structures()["empty_edges"] == {"empty": {"weight": 4}}


def test_only_incidences_are_required_and_node_ids_are_preserved():
    data: HIFJson = {
        "incidences": [
            {"edge": "10", "node": "20"},
            {"edge": "10", "node": "30"},
        ],
    }
    H = assert_round_trip(data, check_identity=False)  # network-type will be added

    assert H.get_edges() == [("20", "30")]
    assert not H.is_weighted()


@pytest.mark.parametrize(
    ("data", "expected_type"),
    [
        ({"incidences": []}, Hypergraph),
        ({"network-type": "directed", "incidences": []}, DirectedHypergraph),
    ],
)
def test_empty_incidence_list_creates_empty_hypergraph(data, expected_type):
    H = assert_round_trip(data, check_identity=False)
    assert isinstance(H, expected_type)
    assert H.num_nodes() == 0
    assert H.num_edges() == 0


def test_weighted_metadata_does_not_control_hypergraph_type():
    data: HIFJson = {
        "metadata": {"weighted": True},
        "incidences": [{"edge": 0, "node": 0}],
    }
    H = assert_round_trip(data, check_identity=False)

    assert not H.is_weighted()
    assert H.get_hypergraph_metadata()["weighted"] is True


def test_to_hif_dict_uses_standard_fields():
    H = Hypergraph(edge_list=[("alice", "bob")], weighted=False)
    H.set_hypergraph_metadata({"name": "example"})
    H.set_node_metadata("alice", {"color": "blue", "weight": 2})
    H.set_edge_metadata(("alice", "bob"), {"kind": "social"})
    H.set_incidence_metadata(("alice", "bob"), "alice", {"weight": 0.5})
    expected: HIFJson = {
        "network-type": "undirected",
        "metadata": {"name": "example"},
        "nodes": [
            {"node": "alice", "weight": 2, "attrs": {"color": "blue"}},
            {"node": "bob"},
        ],
        "edges": [{"edge": 0, "attrs": {"kind": "social"}}],
        "incidences": [
            {"edge": 0, "node": "alice", "weight": 0.5},
            {"edge": 0, "node": "bob"},
        ],
    }

    assert to_hif_dict(H) == expected
    assert_round_trip(expected)


def test_directed_hif_roundtrip():
    data: HIFJson = {
        "network-type": "directed",
        "metadata": {},
        "nodes": [{"node": "a"}, {"node": "b"}],
        "edges": [{"edge": 0, "weight": 2}],
        "incidences": [
            {"edge": 0, "node": "a", "direction": "tail"},
            {"edge": 0, "node": "b", "direction": "head"},
        ],
    }
    H = assert_round_trip(data)

    assert isinstance(H, DirectedHypergraph)
    assert H.get_edges() == [(("a",), ("b",))]
    assert H.get_weight((("a",), ("b",))) == 2


def test_directed_incidence_requires_a_direction():
    data: HIFJson = {
        "network-type": "directed",
        "incidences": [{"edge": 0, "node": 0}],
    }
    with pytest.raises(ValueError, match="require direction"):
        from_hif_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        None,
        {"metadata": [], "incidences": []},
        {"nodes": [{"node": 0, "attrs": []}], "incidences": []},
    ],
)
def test_from_hif_dict_rejects_invalid_dictionary_fields(data: Any):
    with pytest.raises(TypeError, match="dictionary"):
        from_hif_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        {"nodes": [{"node": 0, "weight": "heavy"}], "incidences": []},
        {"edges": [{"edge": 0, "weight": "heavy"}], "incidences": []},
        {"incidences": [{"edge": 0, "node": 0, "weight": "heavy"}]},
        {"nodes": [{"node": 0, "weight": True}], "incidences": []},
    ],
)
def test_from_hif_dict_rejects_non_numeric_weights(data: Any):
    with pytest.raises(TypeError, match="integers or floats"):
        from_hif_dict(data)


def test_abstract_simplicial_complex_is_not_silently_converted():
    data: HIFJson = {"network-type": "asc", "incidences": []}

    with pytest.raises(NotImplementedError, match="simplicial complexes"):
        from_hif_dict(data)


def test_parallel_hif_edges_fail_instead_of_being_merged():
    data: HIFJson = {
        "incidences": [
            {"edge": "first", "node": 0},
            {"edge": "second", "node": 0},
        ]
    }

    with pytest.raises(ValueError, match="Duplicate edge"):
        from_hif_dict(data)


def test_empty_edge_ids_do_not_collide_with_generated_ids():
    H = Hypergraph(edge_list=[(0, 1)], weighted=False)
    H.add_empty_edge(0, {})
    data = to_hif_dict(H)
    restored = assert_round_trip(data)
    assert isinstance(restored, Hypergraph)
    assert restored.get_edges() == [(0, 1)]
    assert {record["edge"] for record in data.get("edges", [])} == {0, 1}


def test_hif_conversions_deep_copy_all_metadata():
    H = Hypergraph(
        edge_list=[(0, 1)],
        weighted=False,
        hypergraph_metadata={"nested": {"name": "original"}},
    )
    H.set_node_metadata(0, {"nested": {"color": "blue"}})
    H.set_edge_metadata((0, 1), {"nested": {"kind": "ordinary"}})
    H.set_incidence_metadata((0, 1), 0, {"nested": {"role": "member"}})
    H.add_empty_edge("empty", {"nested": {"kind": "empty"}})
    data = to_hif_dict(H)
    expected = copy.deepcopy(data)
    restored = from_hif_dict(data)
    data["metadata"]["nested"]["name"] = "changed"
    for records in (data["nodes"], data["edges"], data["incidences"]):
        for record in records:
            if "attrs" in record:
                record["attrs"]["nested"].clear()

    assert to_hif_dict(H) == expected
    assert to_hif_dict(restored) == expected
