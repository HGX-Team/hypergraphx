from __future__ import annotations

import gzip
import json
import os
import re
import ssl
import tempfile
from pathlib import Path

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
    "search_remote_datasets",
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


def _default_dataset_cache_dir():
    return Path(
        os.environ.get(
            "HYPERGRAPHX_DATA_CACHE",
            os.path.join("~", ".cache", "hypergraphx", "datasets"),
        )
    ).expanduser()


def _remote_cache_path(dataset_name: str, url: str, cache_dir=None):
    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else _default_dataset_cache_dir()
    )
    path = urlparse(url).path
    filename = os.path.basename(path)
    if filename.endswith(".gz"):
        filename = filename[:-3]
    return root / dataset_name / filename


def _load_remote_payload_from_path(path: Path, payload_format: str):
    if payload_format == "json":
        return load_json_file(str(path))
    return load_pickle(str(path))


def _infer_local_payload_format(path: str):
    lower = path.lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    if lower.endswith((".pkl", ".pickle", ".hgx")):
        return "pickle"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".hgr"):
        return "hgr"
    raise InvalidFileTypeError("Invalid file type")


def _load_hypergraph_from_decompressed_bytes(payload: bytes, payload_format: str):
    if payload_format == "json":
        return _parse_json_bytes_to_hypergraph(payload)
    with tempfile.NamedTemporaryFile(suffix=f".{payload_format}") as tmp:
        tmp.write(payload)
        tmp.flush()
        if payload_format == "hgr":
            return _load_hgr_file(tmp.name)
        return load_pickle(tmp.name)


def _load_gzipped_hypergraph(file_name: str, payload_format: str):
    with gzip.open(file_name, "rb") as infile:
        return _load_hypergraph_from_decompressed_bytes(infile.read(), payload_format)


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
    timeout: int = 30,
    verify_ssl: bool = False,
    catalog_url: str | None = None,
):
    """
    List datasets advertised by the remote Hypergraphx-data catalog.

    Returns a list of dictionaries with at least:
    - ``name``
    - ``tags`` / ``categories``
    - ``vertices``
    - ``edges``

    Parameters
    ----------
    timeout : int, default=30
        Download timeout in seconds.
    verify_ssl : bool, default=False
        Whether to verify TLS certificates when downloading the catalog.
        Defaults to False for compatibility with the current dataset server.
    catalog_url : str, optional
        Catalog metadata URL. Defaults to the Hypergraphx-data GitHub raw URL,
        or ``HYPERGRAPHX_DATA_CATALOG_URL`` if set.

    Notes
    -----
    ``catalog_url`` can point either to a JSON list or to the generated
    ``related-data.js`` file used by the Hypergraphx-data website.
    """
    url = catalog_url or os.environ.get("HYPERGRAPHX_DATA_CATALOG_URL") or _CATALOG_URL
    payload = _download(url, timeout=timeout, verify_ssl=verify_ssl)
    return _parse_remote_dataset_catalog(payload)


def iter_remote_hypergraphs(
    attributes,
    *,
    match_all: bool = True,
    fmt: str = "hgx",
    timeout: int = 30,
    verify_ssl: bool = False,
    catalog_url: str | None = None,
    include_metadata: bool = False,
    store: bool = True,
    cache_dir=None,
    overwrite: bool = False,
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
    verify_ssl : bool, default=False
        Whether to verify TLS certificates for remote requests.
    catalog_url : str, optional
        Catalog metadata URL used for filtering.
    include_metadata : bool, default=False
        If True, yield ``(hypergraph, dataset_info)`` pairs. Otherwise yield
        only the hypergraph object.
    store : bool, default=True
        Store downloaded datasets locally before loading them.
    cache_dir : path-like, optional
        Cache directory. Defaults to ``~/.cache/hypergraphx/datasets`` or the
        ``HYPERGRAPHX_DATA_CACHE`` environment variable.
    overwrite : bool, default=False
        If True, re-download matching datasets even when cached files exist.

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
            timeout=timeout,
            verify_ssl=verify_ssl,
            store=store,
            cache_dir=cache_dir,
            overwrite=overwrite,
        )
        if include_metadata:
            yield hypergraph, dataset
        else:
            yield hypergraph


def _matches_range(value, minimum, maximum):
    if value is None:
        return minimum is None and maximum is None
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def search_remote_datasets(
    query: str | None = None,
    *,
    tags=None,
    match_all_tags: bool = True,
    min_nodes: int | None = None,
    max_nodes: int | None = None,
    min_edges: int | None = None,
    max_edges: int | None = None,
    timeout: int = 30,
    verify_ssl: bool = False,
    catalog_url: str | None = None,
):
    """
    Search the remote Hypergraphx-data catalog.

    Parameters
    ----------
    query : str, optional
        Case-insensitive substring matched against dataset names and tags.
    tags : str | Iterable[str], optional
        Tags/categories to require. Matching is case-insensitive.
    match_all_tags : bool, default=True
        If True, all requested tags must be present. If False, any requested
        tag is enough.
    min_nodes, max_nodes, min_edges, max_edges : int, optional
        Inclusive size filters using catalog ``vertices`` and ``edges``.

    Returns
    -------
    list[dict]
        Matching catalog entries in catalog order.

    See Also
    --------
    list_remote_datasets : Return the full remote catalog.
    iter_remote_hypergraphs : Lazily load matching remote hypergraphs.
    """
    datasets = list_remote_datasets(
        timeout=timeout,
        verify_ssl=verify_ssl,
        catalog_url=catalog_url,
    )

    query_cf = query.casefold() if query else None
    if isinstance(tags, str):
        requested_tags = {tags.casefold()}
    elif tags is None:
        requested_tags = set()
    else:
        requested_tags = {str(tag).casefold() for tag in tags}

    results = []
    for dataset in datasets:
        dataset_tags = {str(tag).casefold() for tag in dataset.get("tags", [])}

        if query_cf:
            haystack = " ".join(
                [str(dataset.get("name", ""))]
                + [str(tag) for tag in dataset.get("tags", [])]
            ).casefold()
            if query_cf not in haystack:
                continue

        if requested_tags:
            if match_all_tags:
                if not requested_tags.issubset(dataset_tags):
                    continue
            elif not (requested_tags & dataset_tags):
                continue

        if not _matches_range(dataset.get("vertices"), min_nodes, max_nodes):
            continue
        if not _matches_range(dataset.get("edges"), min_edges, max_edges):
            continue

        results.append(dataset)

    return results


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
        from the file extension. Gzipped files with ``.gz`` suffix are
        supported for each local format, such as ``.json.gz`` and ``.hgx.gz``.
    """
    if fmt is not None:
        fmt = fmt.lower()
        if file_name.lower().endswith(".gz"):
            if fmt in {"pickle", "pkl", "binary", "hgx"}:
                return _load_gzipped_hypergraph(file_name, "pickle")
            if fmt in {"json"}:
                return _load_gzipped_hypergraph(file_name, "json")
            if fmt in {"hgr"}:
                return _load_gzipped_hypergraph(file_name, "hgr")
            raise InvalidFormatError("fmt must be one of {'json', 'pickle', 'hgr'}")
        if fmt in {"pickle", "pkl", "binary", "hgx"}:
            return load_pickle(file_name)
        if fmt in {"json"}:
            return load_json_file(file_name)
        if fmt in {"hgr"}:
            return _load_hgr_file(file_name)
        raise InvalidFormatError("fmt must be one of {'json', 'pickle', 'hgr'}")

    payload_format = _infer_local_payload_format(file_name)
    if file_name.lower().endswith(".gz"):
        return _load_gzipped_hypergraph(file_name, payload_format)
    if payload_format == "pickle":
        return load_pickle(file_name)
    if payload_format == "json":
        return load_json_file(file_name)
    if payload_format == "hgr":
        return _load_hgr_file(file_name)
    raise InvalidFileTypeError("Invalid file type")


def load_hypergraph_from_server(
    dataset_name: str,
    *,
    fmt: str | None = "hgx",
    as_dict: bool = False,
    timeout: int = 30,
    verify_ssl: bool = False,
    store: bool = True,
    cache_dir=None,
    overwrite: bool = False,
):
    """
    Load a dataset by name from the remote Hypergraphx-data server.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier, such as ``"zoo"`` or ``"contacts-hospital"``.
    fmt : {"hgx", "binary", "json"} or None, default="hgx"
        Remote format to load. ``"hgx"`` and ``"binary"`` load the compact
        binary Hypergraphx format; ``"json"`` loads the JSON format. If
        explicitly set to None, JSON URLs are tried first, then binary URLs.
    as_dict : bool, default=False
        If True, return the exposed internal data-structure dictionary instead
        of a hypergraph object.
    timeout : int, default=30
        Download timeout in seconds.
    verify_ssl : bool, default=False
        Whether to verify TLS certificates. Defaults to False for compatibility
        with the current dataset server certificate chain.
    store : bool, default=True
        Store the decompressed remote dataset locally before loading it. Cached
        files are reused on later calls.
    cache_dir : path-like, optional
        Cache directory. Defaults to ``~/.cache/hypergraphx/datasets`` or the
        ``HYPERGRAPHX_DATA_CACHE`` environment variable.
    overwrite : bool, default=False
        If True, re-download even when a matching cached file exists.

    Returns
    -------
    Hypergraph | DirectedHypergraph | TemporalHypergraph | MultiplexHypergraph | dict
        Loaded hypergraph object, or its exposed dictionary if ``as_dict=True``.

    Notes
    -----
    The loader tries current per-dataset ``.json.gz`` / ``.hgx.gz`` URLs first
    and keeps older flat URLs as fallbacks. When ``store=True``, compressed
    downloads are decompressed before being written to the cache.
    """
    last_error = None
    url_list = _server_urls(dataset_name, fmt)

    for url in url_list:
        tmp = None
        try:
            payload_format = _remote_payload_format(url)
            cache_path = _remote_cache_path(dataset_name, url, cache_dir)

            if store and cache_path.exists() and not overwrite:
                obj = _load_remote_payload_from_path(cache_path, payload_format)
            else:
                payload = _decompress_gzip_if_needed(
                    _download(url, timeout=timeout, verify_ssl=verify_ssl)
                )
                if store:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(payload)
                    obj = _load_remote_payload_from_path(cache_path, payload_format)
                elif payload_format == "json":
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
            if tmp is not None:
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
