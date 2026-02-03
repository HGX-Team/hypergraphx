from hypergraphx.utils.labeling import LabelEncoder

from hypergraphx.utils.labeling import (
    relabel_edge,
    relabel_edges,
    inverse_relabel_edge,
    inverse_relabel_edges,
    map_node,
    map_nodes,
    inverse_map_nodes,
    get_inverse_mapping,
)


def _make_encoder():
    enc = LabelEncoder()
    enc.fit(["a", "b", "c"])
    return enc


def test_relabel_and_inverse_edges():
    """Test relabeling and inverse relabeling of edges."""
    enc = _make_encoder()
    edge = ("a", "c")

    relabeled = relabel_edge(enc, edge)
    assert relabeled == (0, 2)

    restored = inverse_relabel_edge(enc, relabeled)
    assert restored == edge


def test_relabel_edges_list():
    """Test relabeling list of edges."""
    enc = _make_encoder()
    edges = [("a", "b"), ("b", "c")]

    relabeled = relabel_edges(enc, edges)
    assert relabeled == [(0, 1), (1, 2)]

    restored = inverse_relabel_edges(enc, relabeled)
    assert restored == edges


def test_map_nodes_and_inverse():
    """Test mapping nodes and inverse mapping."""
    enc = _make_encoder()

    assert map_node(enc, "b") == 1
    assert list(map_nodes(enc, ["a", "c"])) == [0, 2]
    assert list(inverse_map_nodes(enc, [0, 2])) == ["a", "c"]


def test_get_inverse_mapping():
    """Test inverse mapping dictionary."""
    enc = _make_encoder()
    inv = get_inverse_mapping(enc)

    assert inv[0] == "a"
    assert inv[2] == "c"
