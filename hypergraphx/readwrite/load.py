from __future__ import annotations

import gzip
import os
import tempfile

from typing import Any, Iterable, List, Tuple
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

__all__ = ["load", "load_hypergraph", "load_hypergraph_from_server"]


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


def _download(url: str, *, timeout: int = 30) -> bytes:
    try:
        req = Request(url, headers={"User-Agent": "hypergraphx-loader/1.0"})
        with urlopen(req, timeout=timeout) as resp:
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
):
    if not _network_opt_in_allowed(allow_network):
        raise PermissionError(
            "Network loading is disabled by default. "
            "Pass allow_network=True to load datasets from the network, "
            "or set HYPERGRAPHX_ALLOW_NETWORK=1 to enable it for this process."
        )

    url_json = f"{_BASE}/{dataset_name}.json"
    url_pkl = f"{_BASE}/{dataset_name}.pkl"

    last_error = None
    if fmt is None:
        url_list = (url_json, url_pkl)
    elif fmt == "json":
        url_list = (url_json,)
    elif fmt == "binary":
        url_list = (url_pkl,)
    else:
        raise InvalidFormatError("fmt must be one of {'json', 'binary'}")

    for url in url_list:
        try:
            payload = _decompress_gzip_if_needed(_download(url, timeout=timeout))
            if url.endswith(".json"):
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
