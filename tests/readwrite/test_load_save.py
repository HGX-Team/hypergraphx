import gzip
import json

import pytest
from urllib.error import URLError

from hypergraphx import (
    Hypergraph,
    DirectedHypergraph,
    MultiplexHypergraph,
    TemporalHypergraph,
)
from hypergraphx.readwrite.load import (
    download_remote_dataset,
    download_remote_datasets,
    get_remote_dataset_info,
    iter_remote_hypergraphs,
    list_remote_datasets,
    load,
    load_hypergraph,
    load_hypergraph_from_server,
    search_remote_datasets,
)
from hypergraphx.readwrite.save import save_hypergraph

DEFAULT_CATALOG_URL = (
    "https://hgx-team.github.io/hypergraphx-data/static/js/related-data.js"
)


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
    return load_hypergraph(path)


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


def test_save_load_json_multiplex_numeric_layers(tmp_path):
    """Test JSON roundtrip for a MultiplexHypergraph with numeric layers."""
    hg = MultiplexHypergraph(
        edge_list=[(0, 1), (1, 2, 3)],
        edge_layer=[0, 1],
        weighted=True,
        weights=[1.5, 2.5],
        edge_metadata=[{"kind": "pair"}, {"kind": "triple"}],
    )
    loaded = _roundtrip_json(tmp_path, hg, name="mx_numeric_layers.json")

    assert isinstance(loaded, MultiplexHypergraph)
    assert loaded.is_weighted() is True
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert loaded.get_weight((0, 1), 0) == 1.5
    assert loaded.get_edge_metadata((0, (0, 1)))["kind"] == "pair"


def test_save_load_json_temporal(tmp_path):
    """Test JSON roundtrip for a TemporalHypergraph with times."""
    edges = [(0, (0, 1)), (1, (1, 2, 3))]
    hg = TemporalHypergraph(edge_list=edges, weighted=True, weights=[1.0, 2.0])
    loaded = _roundtrip_json(tmp_path, hg, name="tg.json")

    assert isinstance(loaded, TemporalHypergraph)
    assert loaded.is_weighted() is True
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert loaded.get_weight((1, 2, 3), 1) == 2.0


def test_load_json_temporal_directed_edge_lists(tmp_path):
    """JSON directed edge parts are lists after serialization."""
    path = tmp_path / "temporal_directed.json"
    path.write_text(
        json.dumps(
            [
                {
                    "hypergraph_type": "TemporalHypergraph",
                    "hypergraph_metadata": {
                        "weighted": False,
                        "type": "TemporalHypergraph",
                    },
                },
                {"type": "node", "idx": 0, "metadata": {}},
                {"type": "node", "idx": 1, "metadata": {}},
                {"type": "node", "idx": 2, "metadata": {}},
                {
                    "type": "edge",
                    "interaction": [[0], [1, 2]],
                    "metadata": {"time": 123},
                },
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_hypergraph(str(path), fmt="json")

    assert isinstance(loaded, TemporalHypergraph)
    assert loaded.get_edges() == [(123, ((0,), (1, 2)))]


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


def test_load_gzipped_json_hypergraph(tmp_path):
    hg = _make_weighted_hypergraph()
    json_path = tmp_path / "hg.json"
    gz_path = tmp_path / "hg.json.gz"
    save_hypergraph(hg, str(json_path), fmt="json")
    gz_path.write_bytes(gzip.compress(json_path.read_bytes()))

    loaded = load_hypergraph(str(gz_path))

    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert loaded.get_weight((0, 1)) == 2.0


def test_load_gzipped_binary_hypergraph(tmp_path):
    hg = _make_weighted_hypergraph()
    hgx_path = tmp_path / "hg.hgx"
    gz_path = tmp_path / "hg.hgx.gz"
    save_hypergraph(hg, str(hgx_path), fmt="pickle")
    gz_path.write_bytes(gzip.compress(hgx_path.read_bytes()))

    loaded = load_hypergraph(str(gz_path))

    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert loaded.get_weight((0, 1)) == 2.0


def test_load_gzipped_hypergraph_fmt_override(tmp_path):
    hg = _make_weighted_hypergraph()
    json_path = tmp_path / "data.unknown"
    gz_path = tmp_path / "data.unknown.gz"
    save_hypergraph(hg, str(json_path), fmt="json")
    gz_path.write_bytes(gzip.compress(json_path.read_bytes()))

    loaded = load_hypergraph(str(gz_path), fmt="json")

    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())


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


def test_load_gzipped_hgr_file(tmp_path):
    hgr = tmp_path / "toy.hgr.gz"
    hgr.write_bytes(gzip.compress(b"2 3 1\n2 1 2\n3 2 3\n"))

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

    requested = []

    def fake_download(url, timeout=30, verify_ssl=True):
        requested.append((url, verify_ssl))
        return gz_payload

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    loaded = load_hypergraph_from_server(
        "toy", fmt="json", cache_dir=tmp_path / "cache", use_catalog=False
    )
    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert requested == [
        (
            "https://cricca.disi.unitn.it/datasets/hypergraphx-data/toy/toy.json.gz",
            False,
        )
    ]
    assert (tmp_path / "cache" / "toy" / "toy.json").exists()


def test_load_hypergraph_from_server_binary(monkeypatch, tmp_path):
    """Test binary loading from server using a mocked downloader."""
    hg = _make_weighted_hypergraph()
    hgx_path = tmp_path / "hg.hgx"
    save_hypergraph(hg, str(hgx_path), fmt="pickle")
    gz_payload = gzip.compress(hgx_path.read_bytes())

    requested = []

    def fake_download(url, timeout=30, verify_ssl=True):
        requested.append((url, verify_ssl))
        return gz_payload

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    loaded = load_hypergraph_from_server(
        "toy", fmt="binary", cache_dir=tmp_path / "cache", use_catalog=False
    )
    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert requested == [
        (
            "https://cricca.disi.unitn.it/datasets/hypergraphx-data/toy/toy.hgx.gz",
            False,
        )
    ]
    assert (tmp_path / "cache" / "toy" / "toy.hgx").exists()


def test_load_hypergraph_from_server_hgx_alias(monkeypatch, tmp_path):
    hg = _make_weighted_hypergraph()
    hgx_path = tmp_path / "hg.hgx"
    save_hypergraph(hg, str(hgx_path), fmt="pickle")
    gz_payload = gzip.compress(hgx_path.read_bytes())
    requested = []

    def fake_download(url, timeout=30, verify_ssl=True):
        requested.append((url, timeout, verify_ssl))
        return gz_payload

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    loaded = load_hypergraph_from_server(
        "toy", cache_dir=tmp_path / "cache", use_catalog=False
    )

    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert requested == [
        (
            "https://cricca.disi.unitn.it/datasets/hypergraphx-data/toy/toy.hgx.gz",
            30,
            False,
        )
    ]
    assert (tmp_path / "cache" / "toy" / "toy.hgx").exists()


def test_load_hypergraph_from_server_uses_cache(monkeypatch, tmp_path):
    hg = _make_weighted_hypergraph()
    cache_path = tmp_path / "cache" / "toy" / "toy.hgx"
    cache_path.parent.mkdir(parents=True)
    save_hypergraph(hg, str(cache_path), fmt="pickle")
    called = False

    def fake_download(url, timeout=30, verify_ssl=True):
        nonlocal called
        called = True
        raise AssertionError("Should use cached file.")

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    loaded = load_hypergraph_from_server(
        "toy",
        fmt="hgx",
        cache_dir=tmp_path / "cache",
        use_catalog=False,
    )

    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert called is False


def test_download_remote_dataset_uses_catalog_url(monkeypatch, tmp_path):
    payload = gzip.compress(b"downloaded payload")
    catalog_payload = json.dumps(
        {
            "schema_version": 1,
            "datasets": [
                {
                    "name": "toy",
                    "versions": [
                        {
                            "version": "1.0.0",
                            "binary_download": "https://example.org/files/custom-name.hgx.gz",
                            "json_download": "https://example.org/files/custom-name.json.gz",
                        }
                    ],
                }
            ],
        }
    ).encode()
    requested = []

    def fake_download(url, timeout=30, verify_ssl=True):
        requested.append(url)
        if url == DEFAULT_CATALOG_URL:
            return catalog_payload
        if url == "https://example.org/files/custom-name.hgx.gz":
            return payload
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    path = download_remote_dataset("toy", cache_dir=tmp_path / "cache")

    assert path == tmp_path / "cache" / "toy" / "custom-name.hgx"
    assert path.read_bytes() == b"downloaded payload"
    assert requested == [
        DEFAULT_CATALOG_URL,
        "https://example.org/files/custom-name.hgx.gz",
    ]


def test_download_remote_dataset_json_format(monkeypatch, tmp_path):
    payload = gzip.compress(b"json payload")
    catalog_payload = json.dumps(
        {
            "schema_version": 1,
            "datasets": [
                {
                    "name": "toy",
                    "versions": [
                        {
                            "version": "1.0.0",
                            "binary_download": "https://example.org/files/custom-name.hgx.gz",
                            "json_download": "https://example.org/files/custom-name.json.gz",
                        }
                    ],
                }
            ],
        }
    ).encode()
    requested = []

    def fake_download(url, timeout=30, verify_ssl=True):
        requested.append(url)
        if url == DEFAULT_CATALOG_URL:
            return catalog_payload
        if url == "https://example.org/files/custom-name.json.gz":
            return payload
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    path = download_remote_dataset("toy", fmt="json", cache_dir=tmp_path / "cache")

    assert path == tmp_path / "cache" / "toy" / "custom-name.json"
    assert path.read_bytes() == b"json payload"
    assert requested == [
        DEFAULT_CATALOG_URL,
        "https://example.org/files/custom-name.json.gz",
    ]


def test_download_remote_dataset_uses_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache" / "toy" / "toy.hgx"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"cached")

    def fake_download(url, timeout=30, verify_ssl=True):
        raise AssertionError("Should use cached file.")

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    path = download_remote_dataset(
        "toy",
        cache_dir=tmp_path / "cache",
        use_catalog=False,
    )

    assert path == cache_path
    assert path.read_bytes() == b"cached"


def test_download_remote_datasets_by_name_reuses_catalog(monkeypatch, tmp_path):
    catalog_payload = json.dumps(
        {
            "schema_version": 1,
            "datasets": [
                {
                    "name": "zoo",
                    "tags": ["Undirected", "Biology"],
                    "versions": [
                        {
                            "version": "1.0.0",
                            "binary_download": "https://example.org/zoo/zoo.hgx.gz",
                        }
                    ],
                },
                {
                    "name": "Marvel",
                    "filename": "Marvel",
                    "directory": "Marvel",
                    "tags": ["Undirected"],
                    "versions": [
                        {
                            "version": "1.0.0",
                            "binary_download": "https://example.org/Marvel/Marvel.hgx.gz",
                        }
                    ],
                },
            ],
        }
    ).encode()
    requested = []
    progress = []

    def fake_download(url, timeout=30, verify_ssl=True):
        requested.append(url)
        if url == DEFAULT_CATALOG_URL:
            return catalog_payload
        if url == "https://example.org/zoo/zoo.hgx.gz":
            return gzip.compress(b"zoo payload")
        if url == "https://example.org/Marvel/Marvel.hgx.gz":
            return gzip.compress(b"Marvel payload")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    results = download_remote_datasets(
        ["zoo", "Marvel"],
        cache_dir=tmp_path / "cache",
        progress_callback=progress.append,
    )

    assert list(results) == ["zoo", "Marvel"]
    assert results["zoo"]["path"] == tmp_path / "cache" / "zoo" / "zoo.hgx"
    assert results["Marvel"]["path"] == tmp_path / "cache" / "Marvel" / "Marvel.hgx"
    assert results["zoo"]["path"].read_bytes() == b"zoo payload"
    assert results["Marvel"]["path"].read_bytes() == b"Marvel payload"
    assert [result["status"] for result in progress] == ["downloaded", "downloaded"]
    assert requested == [
        DEFAULT_CATALOG_URL,
        "https://example.org/zoo/zoo.hgx.gz",
        "https://example.org/Marvel/Marvel.hgx.gz",
    ]


def test_download_remote_datasets_filters_and_continues_on_error(monkeypatch, tmp_path):
    catalog_payload = json.dumps(
        {
            "datasets": [
                {
                    "name": "ok",
                    "tags": ["Undirected"],
                    "versions": [
                        {
                            "binary_download": "https://example.org/ok/ok.hgx.gz",
                        }
                    ],
                },
                {
                    "name": "broken",
                    "tags": ["Undirected"],
                    "versions": [
                        {
                            "binary_download": "https://example.org/broken/broken.hgx.gz",
                        }
                    ],
                },
                {
                    "name": "other",
                    "tags": ["Directed"],
                    "versions": [
                        {
                            "binary_download": "https://example.org/other/other.hgx.gz",
                        }
                    ],
                },
            ],
        }
    ).encode()

    def fake_download(url, timeout=30, verify_ssl=True):
        if url == DEFAULT_CATALOG_URL:
            return catalog_payload
        if url == "https://example.org/ok/ok.hgx.gz":
            return gzip.compress(b"ok payload")
        raise FileNotFoundError(url)

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    results = download_remote_datasets(
        attributes="Undirected",
        cache_dir=tmp_path / "cache",
        continue_on_error=True,
    )

    assert list(results) == ["ok", "broken"]
    assert results["ok"]["status"] == "downloaded"
    assert results["ok"]["path"].read_bytes() == b"ok payload"
    assert results["broken"]["status"] == "error"
    assert isinstance(results["broken"]["error"], FileNotFoundError)
    assert results["broken"]["path"] is None


def test_list_remote_datasets(monkeypatch):
    payload = b"""window.RELATED_DATASETS = [
        {"name":"zoo","tags":["Undirected","Biology"],"vertices":100,"edges":41},
        {"name":"email-Enron","tags":["Directed","Temporal"],"vertices":84172,"edges":235395}
    ];"""
    requested = []

    def fake_download(url, timeout=30, verify_ssl=True):
        requested.append((url, verify_ssl))
        return payload

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    datasets = list_remote_datasets(verify_ssl=False)

    assert datasets == [
        {
            "name": "zoo",
            "tags": ["Undirected", "Biology"],
            "vertices": 100,
            "edges": 41,
            "categories": ["Undirected", "Biology"],
            "filename": "zoo",
            "directory": "zoo",
        },
        {
            "name": "email-Enron",
            "tags": ["Directed", "Temporal"],
            "vertices": 84172,
            "edges": 235395,
            "categories": ["Directed", "Temporal"],
            "filename": "email-Enron",
            "directory": "email-Enron",
        },
    ]
    assert requested == [
        (
            DEFAULT_CATALOG_URL,
            False,
        )
    ]


def test_list_remote_datasets_catalog_json(monkeypatch):
    payload = json.dumps(
        {
            "schema_version": 1,
            "datasets": [
                {
                    "name": "zoo",
                    "directory": "zoo",
                    "tags": ["Undirected", "Biology"],
                    "description": "Animal attribute hypergraph",
                    "source": "https://example.org/zoo",
                    "license": "CC0-1.0",
                    "vertices": 100,
                    "edges": 41,
                    "versions": [
                        {
                            "version": "1.0.0",
                            "json_download": "https://example.org/zoo.json.gz",
                            "binary_download": "https://example.org/zoo.hgx.gz",
                        }
                    ],
                }
            ],
        }
    ).encode()
    requested = []

    def fake_download(url, timeout=30, verify_ssl=True):
        requested.append((url, verify_ssl))
        return payload

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    datasets = list_remote_datasets(verify_ssl=False)

    assert datasets[0]["name"] == "zoo"
    assert datasets[0]["directory"] == "zoo"
    assert datasets[0]["description"] == "Animal attribute hypergraph"
    assert datasets[0]["source"] == "https://example.org/zoo"
    assert datasets[0]["license"] == "CC0-1.0"
    assert (
        datasets[0]["versions"][0]["binary_download"]
        == "https://example.org/zoo.hgx.gz"
    )
    assert requested == [
        (
            DEFAULT_CATALOG_URL,
            False,
        )
    ]


def test_get_remote_dataset_info(monkeypatch):
    catalog = [
        {
            "name": "zoo",
            "directory": "zoo",
            "tags": ["Undirected", "Biology"],
            "source": "https://example.org/zoo",
        }
    ]
    monkeypatch.setattr(
        "hypergraphx.readwrite.load.list_remote_datasets",
        lambda **kwargs: catalog,
    )

    info = get_remote_dataset_info("zoo")

    assert info["source"] == "https://example.org/zoo"


def test_load_hypergraph_from_server_uses_catalog_download_url(monkeypatch, tmp_path):
    hg = _make_weighted_hypergraph()
    hgx_path = tmp_path / "custom-name.hgx"
    save_hypergraph(hg, str(hgx_path), fmt="pickle")
    gz_payload = gzip.compress(hgx_path.read_bytes())
    catalog_payload = json.dumps(
        {
            "schema_version": 1,
            "datasets": [
                {
                    "name": "toy",
                    "versions": [
                        {
                            "version": "1.0.0",
                            "binary_download": "https://example.org/files/custom-name.hgx.gz",
                        }
                    ],
                }
            ],
        }
    ).encode()
    requested = []

    def fake_download(url, timeout=30, verify_ssl=True):
        requested.append(url)
        if url == DEFAULT_CATALOG_URL:
            return catalog_payload
        if url == "https://example.org/files/custom-name.hgx.gz":
            return gz_payload
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    loaded = load_hypergraph_from_server("toy", cache_dir=tmp_path / "cache")

    assert isinstance(loaded, Hypergraph)
    assert set(loaded.get_edges()) == set(hg.get_edges())
    assert requested == [
        DEFAULT_CATALOG_URL,
        "https://example.org/files/custom-name.hgx.gz",
    ]
    assert (tmp_path / "cache" / "toy" / "custom-name.hgx").exists()


def test_iter_remote_hypergraphs_filters_and_loads_lazily(monkeypatch):
    catalog = [
        {
            "name": "contacts-hospital",
            "tags": ["Undirected", "Temporal", "Social"],
            "categories": ["Undirected", "Temporal", "Social"],
            "vertices": 75,
            "edges": 27835,
        },
        {
            "name": "zoo",
            "tags": ["Undirected", "Biology"],
            "categories": ["Undirected", "Biology"],
            "description": "Animal attribute hypergraph",
            "source": "https://example.org/biology/zoo",
            "license": "CC0-1.0",
            "vertices": 100,
            "edges": 41,
        },
    ]
    loaded_names = []
    load_kwargs = []

    def fake_list_remote_datasets(**kwargs):
        return catalog

    def fake_load_hypergraph_from_server(name, **kwargs):
        loaded_names.append(name)
        load_kwargs.append(kwargs)
        return Hypergraph(edge_list=[(0, 1)], weighted=False)

    monkeypatch.setattr(
        "hypergraphx.readwrite.load.list_remote_datasets",
        fake_list_remote_datasets,
    )
    monkeypatch.setattr(
        "hypergraphx.readwrite.load.load_hypergraph_from_server",
        fake_load_hypergraph_from_server,
    )

    iterator = iter_remote_hypergraphs(["Undirected", "Temporal"])

    assert loaded_names == []
    first = next(iterator)
    assert isinstance(first, Hypergraph)
    assert loaded_names == ["contacts-hospital"]
    assert load_kwargs[0]["use_catalog"] is False
    assert load_kwargs[0]["dataset_info"] is catalog[0]
    with pytest.raises(StopIteration):
        next(iterator)


def test_iter_remote_hypergraphs_accepts_explicit_names(monkeypatch):
    catalog = [
        {
            "name": "contacts-hospital",
            "tags": ["Undirected", "Temporal", "Social"],
        },
        {
            "name": "zoo",
            "tags": ["Undirected", "Biology"],
        },
    ]
    loaded_names = []
    load_kwargs = []

    monkeypatch.setattr(
        "hypergraphx.readwrite.load.list_remote_datasets",
        lambda **kwargs: catalog,
    )

    def fake_load_hypergraph_from_server(name, **kwargs):
        loaded_names.append(name)
        load_kwargs.append(kwargs)
        return Hypergraph(edge_list=[(name,)], weighted=False)

    monkeypatch.setattr(
        "hypergraphx.readwrite.load.load_hypergraph_from_server",
        fake_load_hypergraph_from_server,
    )

    results = list(iter_remote_hypergraphs(names=["zoo", "contacts-hospital"]))

    assert len(results) == 2
    assert loaded_names == ["zoo", "contacts-hospital"]
    assert load_kwargs[0]["dataset_info"] is catalog[1]
    assert load_kwargs[1]["dataset_info"] is catalog[0]


def test_iter_remote_hypergraphs_match_any_and_include_metadata(monkeypatch):
    catalog = [
        {
            "name": "contacts-hospital",
            "tags": ["Undirected", "Temporal", "Social"],
            "categories": ["Undirected", "Temporal", "Social"],
            "vertices": 75,
            "edges": 27835,
        },
        {
            "name": "zoo",
            "tags": ["Undirected", "Biology"],
            "categories": ["Undirected", "Biology"],
            "description": "Animal attribute hypergraph",
            "source": "https://example.org/biology/zoo",
            "license": "CC0-1.0",
            "vertices": 100,
            "edges": 41,
        },
    ]

    monkeypatch.setattr(
        "hypergraphx.readwrite.load.list_remote_datasets",
        lambda **kwargs: catalog,
    )
    monkeypatch.setattr(
        "hypergraphx.readwrite.load.load_hypergraph_from_server",
        lambda name, **kwargs: Hypergraph(edge_list=[(name,)], weighted=False),
    )

    results = list(
        iter_remote_hypergraphs(
            ["Temporal", "Biology"],
            match_all=False,
            include_metadata=True,
        )
    )

    assert [metadata["name"] for _, metadata in results] == [
        "contacts-hospital",
        "zoo",
    ]
    assert all(isinstance(hypergraph, Hypergraph) for hypergraph, _ in results)


def test_iter_remote_hypergraphs_requires_attributes():
    with pytest.raises(ValueError, match="dataset name or attribute"):
        list(iter_remote_hypergraphs([]))


def test_search_remote_datasets(monkeypatch):
    catalog = [
        {
            "name": "contacts-hospital",
            "tags": ["Undirected", "Temporal", "Social"],
            "categories": ["Undirected", "Temporal", "Social"],
            "vertices": 75,
            "edges": 27835,
        },
        {
            "name": "zoo",
            "tags": ["Undirected", "Biology"],
            "categories": ["Undirected", "Biology"],
            "description": "Animal attribute hypergraph",
            "source": "https://example.org/biology/zoo",
            "license": "CC0-1.0",
            "vertices": 100,
            "edges": 41,
        },
        {
            "name": "email-Enron",
            "tags": ["Directed", "Temporal", "Social", "Technology"],
            "categories": ["Directed", "Temporal", "Social", "Technology"],
            "description": "Email communication dataset",
            "source": "https://example.org/email",
            "license": "GPL-3.0",
            "vertices": 84172,
            "edges": 235395,
        },
    ]
    monkeypatch.setattr(
        "hypergraphx.readwrite.load.list_remote_datasets",
        lambda **kwargs: catalog,
    )

    assert [d["name"] for d in search_remote_datasets("contact")] == [
        "contacts-hospital"
    ]
    assert [d["name"] for d in search_remote_datasets("biology")] == ["zoo"]
    assert [
        d["name"] for d in search_remote_datasets(tags=["Undirected", "Temporal"])
    ] == ["contacts-hospital"]
    assert [
        d["name"]
        for d in search_remote_datasets(
            tags=["Biology", "Technology"],
            match_all_tags=False,
        )
    ] == ["zoo", "email-Enron"]
    assert [
        d["name"]
        for d in search_remote_datasets(
            min_nodes=80,
            max_nodes=1000,
            max_edges=1000,
        )
    ] == ["zoo"]
    assert [d["name"] for d in search_remote_datasets("attribute")] == ["zoo"]
    assert [d["name"] for d in search_remote_datasets(source="biology")] == ["zoo"]
    assert [d["name"] for d in search_remote_datasets(license="cc0")] == ["zoo"]


def test_load_hypergraph_from_server_offline_error_is_actionable(monkeypatch):
    def fake_download(url, timeout=30, verify_ssl=True):
        raise URLError("offline")

    monkeypatch.setattr("hypergraphx.readwrite.load._download", fake_download)

    with pytest.raises(ConnectionError, match="Are you offline\\?"):
        load_hypergraph_from_server("toy", fmt="json", store=False)


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
