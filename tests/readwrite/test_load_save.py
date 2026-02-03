import gzip

import pytest
from urllib.error import URLError

from hypergraphx import (
    Hypergraph,
    DirectedHypergraph,
    MultiplexHypergraph,
    TemporalHypergraph,
)
from hypergraphx.readwrite.load import (
    load,
    load_hypergraph,
    load_hypergraph_from_server,
)
from hypergraphx.readwrite.save import save_hypergraph


def _make_weighted_hypergraph():
    return Hypergraph(
        edge_list=[(0, 1), (1, 2, 3)],
        weighted=True,
        weights=[2.0, 3.5],
        node_metadata={0: {"role": "a"}},
        edge_metadata=[{"kind": "pair"}, {"kind": "tri"}],
        hypergraph_metadata={"name": "test"},
    )


def _roundtrip_json(tmp_path, hypergraph, name="hg.json"):
    path = tmp_path / name
    save_hypergraph(hypergraph, str(path), fmt="json")
    return load_hypergraph(str(path))


def test_save_load_json_hypergraph(tmp_path):
    """Test JSON roundtrip for a weighted Hypergraph."""
    hg = _make_weighted_hypergraph()
    loaded = _roundtrip_json(tmp_path, hg)

    assert isinstance(loaded, Hypergraph)
    assert loaded.is_weighted() is True
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert loaded.get_weight((0, 1)) == 2.0
    assert loaded.get_weight((1, 2, 3)) == 3.5
    assert loaded.get_edge_metadata((1, 2, 3)) == {"kind": "tri", "weight": 3.5}


def test_save_load_json_directed(tmp_path):
    """Test JSON roundtrip for a DirectedHypergraph."""
    edges = [((0,), (1, 2)), ((2,), (0,))]
    hg = DirectedHypergraph(edge_list=edges, weighted=True, weights=[1.0, 2.0])
    loaded = _roundtrip_json(tmp_path, hg, name="dg.json")

    assert isinstance(loaded, DirectedHypergraph)
    assert loaded.is_weighted() is True
    assert set(loaded.get_edges()) == set(hg.get_edges())


def test_save_load_json_multiplex(tmp_path):
    """Test JSON roundtrip for a MultiplexHypergraph with layers."""
    edges = [(0, 1), (1, 2, 3)]
    layers = ["L1", "L2"]
    hg = MultiplexHypergraph(
        edge_list=edges,
        edge_layer=layers,
        weighted=True,
        weights=[1.5, 2.5],
    )
    loaded = _roundtrip_json(tmp_path, hg, name="mx.json")

    assert isinstance(loaded, MultiplexHypergraph)
    assert loaded.is_weighted() is True
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert loaded.get_weight((0, 1), "L1") == 1.5


def test_save_load_json_temporal(tmp_path):
    """Test JSON roundtrip for a TemporalHypergraph with times."""
    edges = [(0, (0, 1)), (1, (1, 2, 3))]
    hg = TemporalHypergraph(edge_list=edges, weighted=True, weights=[1.0, 2.0])
    loaded = _roundtrip_json(tmp_path, hg, name="tg.json")

    assert isinstance(loaded, TemporalHypergraph)
    assert loaded.is_weighted() is True
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert loaded.get_weight((1, 2, 3), 1) == 2.0


def test_save_load_binary_hypergraph(tmp_path):
    """Test binary (hgx) roundtrip using pickle serialization."""
    hg = _make_weighted_hypergraph()
    path = tmp_path / "hg.hgx"
    save_hypergraph(hg, str(path), fmt="pickle")
    loaded = load_hypergraph(str(path))

    assert isinstance(loaded, Hypergraph)
    assert loaded.is_weighted() is True
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert loaded.get_weight((0, 1)) == 2.0


def test_load_hgr_file(tmp_path):
    """Test loading a simple weighted .hgr file."""
    hgr = tmp_path / "toy.hgr"
    hgr.write_text("2 3 1\n2 1 2\n3 2 3\n")

    loaded = load_hypergraph(str(hgr))

    assert isinstance(loaded, Hypergraph)
    assert loaded.is_weighted() is True
    assert set(loaded.get_edges()) == {(1, 2), (2, 3)}
    assert loaded.get_weight((1, 2)) == 2
    assert loaded.get_weight((2, 3)) == 3


def test_load_hypergraph_invalid_extension(tmp_path):
    """Test invalid file extension handling."""
    bogus = tmp_path / "bogus.txt"
    bogus.write_text("nope")
    with pytest.raises(ValueError, match="Invalid file type"):
        load_hypergraph(str(bogus))


def test_load_hypergraph_fmt_override(tmp_path):
    hg = _make_weighted_hypergraph()
    path = tmp_path / "data.unknown"
    save_hypergraph(hg, str(path), fmt="json")

    loaded = load_hypergraph(str(path), fmt="json")
    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())


def test_load_hypergraph_from_server_json(monkeypatch, tmp_path):
    """Test JSON loading from server using a mocked downloader."""
    hg = _make_weighted_hypergraph()
    json_path = tmp_path / "hg.json"
    save_hypergraph(hg, str(json_path), fmt="json")
    payload = json_path.read_bytes()
    gz_payload = gzip.compress(payload)

    def fake_download(url, timeout=30):
        return gz_payload

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    loaded = load_hypergraph_from_server("toy", fmt="json", allow_network=True)
    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())


def test_load_hypergraph_from_server_binary(monkeypatch, tmp_path):
    """Test binary loading from server using a mocked downloader."""
    hg = _make_weighted_hypergraph()
    hgx_path = tmp_path / "hg.hgx"
    save_hypergraph(hg, str(hgx_path), fmt="pickle")
    gz_payload = gzip.compress(hgx_path.read_bytes())

    def fake_download(url, timeout=30):
        return gz_payload

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    loaded = load_hypergraph_from_server("toy", fmt="binary", allow_network=True)
    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())


def test_load_hypergraph_from_server_requires_opt_in(monkeypatch):
    called = False

    def fake_download(url, timeout=30):
        nonlocal called
        called = True
        raise AssertionError("Should not download without explicit opt-in.")

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    with pytest.raises(PermissionError, match="Network loading is disabled by default"):
        load_hypergraph_from_server("toy", fmt="json")
    assert called is False


def test_load_hypergraph_from_server_offline_error_is_actionable(monkeypatch):
    def fake_download(url, timeout=30):
        raise URLError("offline")

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    with pytest.raises(ConnectionError, match="Are you offline\\?"):
        load_hypergraph_from_server("toy", fmt="json", allow_network=True)


def test_load_accepts_hypergraph_instances():
    hg = _make_weighted_hypergraph()
    loaded = load(hg)
    assert loaded is hg


def test_load_accepts_dicts():
    hg = _make_weighted_hypergraph()
    data = hg.expose_data_structures()
    loaded = load(data)
    assert loaded == data


def test_load_accepts_iterables_of_objects():
    hg = _make_weighted_hypergraph()
    dh = DirectedHypergraph(edge_list=[((0,), (1,))])
    loaded = load([hg, dh])
    assert loaded == [hg, dh]
