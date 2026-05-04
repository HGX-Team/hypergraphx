from __future__ import annotations

import gzip
import json
import os
import re
import ssl
import tempfile

from typing import Any, Iterable, List, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from hypergraphx.core.undirected import Hypergraph
from hypergraphx.core.directed import DirectedHypergraph
from hypergraphx.core.multiplex import MultiplexHypergraph
from hypergraphx.core.temporal import TemporalHypergraph
from hypergraphx.exceptions import InvalidFileTypeError, InvalidFormatError
from hypergraphx.readwrite.io_json import (
    _parse_json_bytes_to_hypergraph,
    load_json_file,
)
from hypergraphx.readwrite.io_pickle import load_pickle

_BASE = "https://cricca.disi.unitn.it/datasets/hypergraphx-data"
_CATALOG_URL = (
    "https://raw.githubusercontent.com/HGX-Team/hypergraphx-data/"
    "main/dist/static/js/related-data.js"
)

__all__ = [
    "iter_remote_hypergraphs",
    "list_remote_datasets",
    "load",
    "load_hypergraph",
    "load_hypergraph_from_server",
]


def _decompress_gzip_if_needed(raw: bytes) -> bytes:
    try:
        return gzip.decompress(raw)
    except OSError:
        return raw


def _ensure_hypergraph_obj(obj: Any):
    allowed = (
        Hypergraph,
        DirectedHypergraph,
        MultiplexHypergraph,
        TemporalHypergraph,
        dict,
    )
    if not isinstance(obj, allowed):
        raise TypeError(f"Object has type {type(obj)!r}, expected one of {allowed}.")


def _download(url: str, *, timeout: int = 30, verify_ssl: bool = True) -> bytes:
    try:
        if verify_ssl:
            context = ssl.create_default_context()
            try:
                import certifi  # type: ignore

                context = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                pass
        else:
            context = ssl._create_unverified_context()  # noqa: SLF001
        req = Request(url, headers={"User-Agent": "hypergraphx-loader/1.0"})
        with urlopen(req, timeout=timeout, context=context) as resp:
            return resp.read()
    except HTTPError as exc:
        raise FileNotFoundError(f"Not found at {url} (HTTP {exc.code}).") from exc
    except URLError as exc:
        raise ConnectionError(
            f"Network error reaching {url}: {exc.reason}. "
            "Are you offline? For offline use, download the dataset and use load_hypergraph(...) on a local file."
        ) from exc


def _network_opt_in_allowed(allow_network: bool) -> bool:
    if allow_network:
        return True
    env = os.environ.get("HYPERGRAPHX_ALLOW_NETWORK", "").strip().lower()
    return env in {"1", "true", "yes", "y", "on"}


def _server_urls(dataset_name: str, fmt: str | None):
    urls = {
        "json": (
            f"{_BASE}/{dataset_name}/{dataset_name}.json.gz",
            f"{_BASE}/{dataset_name}/{dataset_name}.json",
            f"{_BASE}/{dataset_name}.json.gz",
            f"{_BASE}/{dataset_name}.json",
        ),
        "binary": (
            f"{_BASE}/{dataset_name}/{dataset_name}.hgx.gz",
            f"{_BASE}/{dataset_name}/{dataset_name}.hgx",
            f"{_BASE}/{dataset_name}.hgx.gz",
            f"{_BASE}/{dataset_name}.hgx",
            f"{_BASE}/{dataset_name}.pkl",
        ),
    }
    if fmt is None:
        return urls["json"] + urls["binary"]
    if fmt in {"json"}:
        return urls["json"]
    if fmt in {"binary", "pickle", "pkl", "hgx"}:
        return urls["binary"]
    raise InvalidFormatError("fmt must be one of {'json', 'binary', 'hgx'}")


def _remote_payload_format(url: str):
    path = urlparse(url).path
    if path.endswith(".gz"):
        path = path[:-3]
    if path.endswith(".json"):
        return "json"
    if path.endswith((".hgx", ".pkl", ".pickle")):
        return "binary"
    raise InvalidFormatError(f"Cannot infer remote payload format from URL: {url}")


def _parse_remote_dataset_catalog(payload: bytes):
    text = payload.decode("utf-8")
    text = text.strip()

    if text.startswith("window.RELATED_DATASETS"):
        match = re.match(r"window\.RELATED_DATASETS\s*=\s*(.*?);?\s*$", text, re.S)
        if not match:
            raise InvalidFormatError("Could not parse remote dataset catalog.")
        text = match.group(1)

    try:
        items = json.loads(text)
    except Exception as exc:
        raise InvalidFormatError("Remote dataset catalog is not valid JSON.") from exc

    if not isinstance(items, list):
        raise InvalidFormatError("Remote dataset catalog must be a list.")

    datasets = []
    for item in items:
        if not isinstance(item, dict) or "name" not in item:
            raise InvalidFormatError(
                "Remote dataset catalog entries must contain names."
            )
        tags = list(item.get("tags") or item.get("categories") or [])
        datasets.append(
            {
                "name": item["name"],
                "tags": tags,
                "categories": tags,
                "vertices": item.get("vertices"),
                "edges": item.get("edges"),
            }
        )
    return datasets


def list_remote_datasets(
    *,
    allow_network: bool = False,
    timeout: int = 30,
    verify_ssl: bool = True,
    catalog_url: str | None = None,
):
    """
    List datasets advertised by the remote Hypergraphx-data catalog.

    Returns a list of dictionaries with at least:
    - ``name``
    - ``tags`` / ``categories``
    - ``vertices``
    - ``edges``

    Network access is opt-in, matching ``load_hypergraph_from_server``.
    ``catalog_url`` can point either to a JSON list or to the generated
    ``related-data.js`` file used by the Hypergraphx-data website.
    """
    if not _network_opt_in_allowed(allow_network):
        raise PermissionError(
            "Network loading is disabled by default. "
            "Pass allow_network=True to list remote datasets, "
            "or set HYPERGRAPHX_ALLOW_NETWORK=1 to enable it for this process."
        )

    url = catalog_url or os.environ.get("HYPERGRAPHX_DATA_CATALOG_URL") or _CATALOG_URL
    payload = _download(url, timeout=timeout, verify_ssl=verify_ssl)
    return _parse_remote_dataset_catalog(payload)


def iter_remote_hypergraphs(
    attributes,
    *,
    match_all: bool = True,
    fmt: str = "hgx",
    allow_network: bool = False,
    timeout: int = 30,
    verify_ssl: bool = True,
    catalog_url: str | None = None,
    include_metadata: bool = False,
):
    """
    Yield remote hypergraphs whose catalog tags/categories match ``attributes``.

    Parameters
    ----------
    attributes : str | Iterable[str]
        Tag/category names to match, such as ``"Undirected"`` or
        ``["Undirected", "Temporal"]``. Matching is case-insensitive.
    match_all : bool, default=True
        If True, a dataset must contain all requested attributes. If False,
        any requested attribute is enough.
    fmt : {"hgx", "binary", "json"}, default="hgx"
        Remote format to load for each matching dataset.
    include_metadata : bool, default=False
        If True, yield ``(hypergraph, dataset_info)`` pairs. Otherwise yield
        only the hypergraph object.

    Notes
    -----
    This is a generator: datasets are downloaded and loaded lazily as the
    iterator advances.
    """
    if isinstance(attributes, str):
        requested = {attributes.casefold()}
    else:
        requested = {str(attr).casefold() for attr in attributes}
    if not requested:
        raise ValueError("At least one attribute must be provided.")

    datasets = list_remote_datasets(
        allow_network=allow_network,
        timeout=timeout,
        verify_ssl=verify_ssl,
        catalog_url=catalog_url,
    )

    for dataset in datasets:
        tags = {str(tag).casefold() for tag in dataset.get("tags", [])}
        matched = requested.issubset(tags) if match_all else bool(requested & tags)
        if not matched:
            continue

        hypergraph = load_hypergraph_from_server(
            dataset["name"],
            fmt=fmt,
            allow_network=allow_network,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        if include_metadata:
            yield hypergraph, dataset
        else:
            yield hypergraph


def _load_hgr_file(file_name: str):
    with open(file_name) as file:
        edges = 0
        nodes = 0
        mode = 0
        w_l: List[int] = []
        edge_l: List[Tuple[int, ...]] = []
        read_count = 0
        read_node = 0
        for line in file:
            this_l = line.strip()
            if len(this_l) == 0 or this_l[0] == "%":
                pass
            elif nodes == 0:
                head = this_l.split(" ")
                edges = int(head[0])
                nodes = int(head[1])
                if len(head) == 3:
                    mode = int(head[2])
            elif read_count < edges:
                read_count += 1
                entries = [int(r) for r in this_l.split(" ") if r != ""]
                if mode % 10 == 1 and len(entries) > 1:
                    w_l += [int(entries[0])]
                    edge_l += [tuple(entries[1:])]
                elif mode % 10 != 1 and len(entries) > 0:
                    edge_l += [tuple(entries)]
                else:
                    raise ValueError(f"Empty edge in file. {read_count} edges read.")
            elif read_node < nodes:
                read_node += 1
            else:
                raise ValueError("File read to the end unexpectedly.")
        h = Hypergraph(
            edge_list=edge_l,
            weighted=(mode % 10) == 1,
            weights=w_l if mode % 10 == 1 else None,
        )
        return h


def load_hypergraph(file_name: str, *, fmt: str | None = None):
    """
    Load a hypergraph from disk.

    Parameters
    ----------
    file_name : str
        Input file path.
    fmt : {"json", "pickle", "hgr"} | None
        Optional override for the input format. If None (default), infer format
        from the file extension.
    """
    if fmt is not None:
        fmt = fmt.lower()
        if fmt in {"pickle", "pkl", "binary", "hgx"}:
            return load_pickle(file_name)
        if fmt in {"json"}:
            return load_json_file(file_name)
        if fmt in {"hgr"}:
            return _load_hgr_file(file_name)
        raise InvalidFormatError("fmt must be one of {'json', 'pickle', 'hgr'}")

    ext = os.path.splitext(file_name)[1].lower()
    if ext in {".pkl", ".pickle", ".hgx"}:
        return load_pickle(file_name)
    if ext == ".json":
        return load_json_file(file_name)
    if ext == ".hgr":
        return _load_hgr_file(file_name)
    raise InvalidFileTypeError("Invalid file type")


def load_hypergraph_from_server(
    dataset_name: str,
    *,
    fmt: str | None = None,
    as_dict: bool = False,
    allow_network: bool = False,
    timeout: int = 30,
    verify_ssl: bool = True,
):
    if not _network_opt_in_allowed(allow_network):
        raise PermissionError(
            "Network loading is disabled by default. "
            "Pass allow_network=True to load datasets from the network, "
            "or set HYPERGRAPHX_ALLOW_NETWORK=1 to enable it for this process."
        )

    last_error = None
    url_list = _server_urls(dataset_name, fmt)

    for url in url_list:
        try:
            payload = _decompress_gzip_if_needed(
                _download(url, timeout=timeout, verify_ssl=verify_ssl)
            )
            if _remote_payload_format(url) == "json":
                obj = _parse_json_bytes_to_hypergraph(payload)
            else:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(payload)
                    tmp.flush()
                    obj = load_pickle(tmp.name)
            _ensure_hypergraph_obj(obj)
            return obj if not as_dict else obj.expose_data_structures()
        except Exception as exc:
            last_error = exc
            continue
        finally:
            if "tmp" in locals():
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

    urls = ", ".join(url_list)
    if isinstance(last_error, (ConnectionError, URLError)):
        raise ConnectionError(
            f"Failed to load '{dataset_name}' from server (network error). "
            f"Tried: {urls}. Last error: {last_error}. "
            "Are you offline? For offline use, download the dataset and use load_hypergraph(...) on a local file."
        ) from last_error
    raise FileNotFoundError(
        f"Failed to load '{dataset_name}' from server. Tried: {urls}. Last error: {last_error}"
    ) from last_error


def load(obj_or_path: str | Iterable):
    if isinstance(obj_or_path, str):
        return load_hypergraph(obj_or_path)

    if isinstance(
        obj_or_path,
        (Hypergraph, DirectedHypergraph, MultiplexHypergraph, TemporalHypergraph, dict),
    ):
        return obj_or_path

    if isinstance(obj_or_path, Iterable):
        hgs = []
        for item in obj_or_path:
            if isinstance(item, str):
                hgs.append(load_hypergraph(item))
            else:
                _ensure_hypergraph_obj(item)
                hgs.append(item)
        return hgs

    _ensure_hypergraph_obj(obj_or_path)
    return obj_or_path
