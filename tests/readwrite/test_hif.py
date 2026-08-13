import json
from typing import Any

import pytest

from hypergraphx import DirectedHypergraph, Hypergraph
from hypergraphx.readwrite import (
    from_hif_dict,
    read_hif,
    to_hif_dict,
    write_hif,
)


def test_from_hif_dict_matches_read_hif(tmp_path):
    data = {
        "network-type": "undirected",
        "metadata": {"name": "in-memory"},
        "nodes": [{"node": "isolated", "weight": 3, "attrs": {"color": "red"}}],
        "edges": [
            {"edge": "friendship", "weight": 2, "attrs": {"kind": "social"}},
            {"edge": "empty", "weight": 4},
        ],
        "incidences": [
            {
                "edge": "friendship",
                "node": "alice",
                "weight": 0.5,
                "attrs": {"since": 2020},
            },
            {"edge": "friendship", "node": "bob"},
        ],
    }
    path = tmp_path / "network.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    from_dict = from_hif_dict(data)
    from_file = read_hif(path)

    assert to_hif_dict(from_file) == to_hif_dict(from_dict)
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
    assert {"edge": "empty", "weight": 4} in to_hif_dict(from_dict)["edges"]


def test_only_incidences_are_required_and_ids_are_preserved():
    H = from_hif_dict(
        {
            "incidences": [
                {"edge": "10", "node": "20"},
                {"edge": "10", "node": "30"},
            ],
        }
    )

    assert H.get_edges() == [("20", "30")]
    assert not H.is_weighted()


def test_weighted_metadata_does_not_control_hypergraph_type():
    H = from_hif_dict(
        {
            "metadata": {"weighted": True},
            "incidences": [{"edge": 0, "node": 0}],
        }
    )

    assert not H.is_weighted()
    assert H.get_hypergraph_metadata()["weighted"] is True


def test_to_hif_dict_matches_write_hif_and_uses_standard_fields(tmp_path):
    H = Hypergraph(edge_list=[("alice", "bob")], weighted=False)
    H.set_hypergraph_metadata({"name": "example"})
    H.set_node_metadata("alice", {"color": "blue", "weight": 2})
    H.set_edge_metadata(("alice", "bob"), {"kind": "social"})
    H.set_incidence_metadata(("alice", "bob"), "alice", {"weight": 0.5})
    path = tmp_path / "network.json"
    expected = {
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

    write_hif(H, path)

    assert to_hif_dict(H) == expected
    assert json.loads(path.read_text(encoding="utf-8")) == expected


def test_directed_hif_roundtrip():
    data = {
        "network-type": "directed",
        "edges": [{"edge": "reaction", "weight": 2}],
        "incidences": [
            {"edge": "reaction", "node": "a", "direction": "tail"},
            {"edge": "reaction", "node": "b", "direction": "head"},
        ],
    }

    H = from_hif_dict(data)
    converted = to_hif_dict(H)

    assert isinstance(H, DirectedHypergraph)
    assert H.get_edges() == [(("a",), ("b",))]
    assert H.get_weight((("a",), ("b",))) == 2
    assert converted["incidences"] == [
        {"edge": 0, "node": "a", "direction": "tail"},
        {"edge": 0, "node": "b", "direction": "head"},
    ]
    assert from_hif_dict(converted).get_edges() == H.get_edges()


def test_directed_incidence_requires_a_direction():
    with pytest.raises(ValueError, match="require direction"):
        from_hif_dict(
            {
                "network-type": "directed",
                "incidences": [{"edge": 0, "node": 0}],
            }
        )


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


def test_abstract_simplicial_complex_is_not_silently_converted():
    with pytest.raises(NotImplementedError, match="simplicial complexes"):
        from_hif_dict({"network-type": "asc", "incidences": []})


def test_parallel_hif_edges_fail_instead_of_being_merged():
    with pytest.raises(ValueError, match="Duplicate edge"):
        from_hif_dict(
            {
                "incidences": [
                    {"edge": "first", "node": 0},
                    {"edge": "second", "node": 0},
                ]
            }
        )


def test_empty_edge_ids_do_not_collide_with_generated_ids():
    H = Hypergraph(edge_list=[(0, 1)], weighted=False)
    H.add_empty_edge(0, {})

    data = to_hif_dict(H)

    assert {record["edge"] for record in data["edges"]} == {0, 1}


def test_hif_dict_roundtrip_does_not_alias_metadata():
    H = Hypergraph(
        edge_list=[(0, 1)],
        weighted=False,
        hypergraph_metadata={"nested": {"name": "original"}},
    )
    H.set_node_metadata(0, {"nested": {"color": "blue"}})
    data = to_hif_dict(H)

    restored = from_hif_dict(data)
    data["metadata"]["nested"]["name"] = "changed"
    data["nodes"][0]["attrs"]["nested"]["color"] = "red"

    assert restored.get_hypergraph_metadata()["nested"]["name"] == "original"
    assert restored.get_node_metadata(0)["nested"]["color"] == "blue"
    assert H.get_hypergraph_metadata()["nested"]["name"] == "original"
    assert H.get_node_metadata(0)["nested"]["color"] == "blue"


def test_write_hif_rejects_nan(tmp_path):
    H = Hypergraph(weighted=False)
    H.set_hypergraph_metadata({"nested": {"missing": float("nan")}})
    path = tmp_path / "network.json"
    path.write_text("existing data", encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float values"):
        write_hif(H, path)

    assert path.read_text(encoding="utf-8") == "existing data"
