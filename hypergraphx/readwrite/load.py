from __future__ import annotations
import json
import gzip
import pickle
import os
import tempfile

from typing import Any, Dict, Iterable, List, Tuple, Union
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from hypergraphx.core.hypergraph import Hypergraph
from hypergraphx.core.directed_hypergraph import DirectedHypergraph
from hypergraphx.core.multiplex_hypergraph import MultiplexHypergraph
from hypergraphx.core.temporal_hypergraph import TemporalHypergraph

# --- Fixed base URL for the Trento server ---
_BASE = "https://cricca.disi.unitn.it/datasets/hypergraphx-data"

__all__ = ["load_hypergraph", "load_hypergraph_from_server"]

# ---------------------------------------------------------------------------
# Helpers: validation, decoding, and shared constructors
# ---------------------------------------------------------------------------

def _decompress_gzip_if_needed(raw: bytes) -> bytes:
    """Return gzip-decompressed bytes if possible; else return the original bytes."""
    try:
        return gzip.decompress(raw)
    except OSError:
        return raw

def _ensure_hypergraph_obj(obj: Any):
    """Raise if obj is not a recognized Hypergraph type."""
    allowed = (Hypergraph, DirectedHypergraph, MultiplexHypergraph, TemporalHypergraph, dict)
    if not isinstance(obj, allowed):
        raise TypeError(f"Object has type {type(obj)!r}, expected one of {allowed}.")

def _download(url: str, *, timeout: int = 30) -> bytes:
    """Download bytes from URL with a friendly UA."""
    try:
        req = Request(url, headers={"User-Agent": "hypergraphx-loader/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as e:
        raise FileNotFoundError(f"Not found at {url} (HTTP {e.code}).") from e
    except URLError as e:
        raise ConnectionError(f"Network error reaching {url}: {e.reason}") from e

# ---------------------------------------------------------------------------
# JSON -> Hypergraph (shared by local JSON loader and remote JSON loader)
# ---------------------------------------------------------------------------

def _split_json_records(data_list: List[dict]):
    """Split JSON 'array of objects' into metadata, type, nodes, edges."""
    hypergraph_metadata: Dict[str, Any] = {}
    hypergraph_type: Union[str, None] = None
    nodes: List[dict] = []
    edges: List[dict] = []

    for obj in data_list:
        if "hypergraph_metadata" in obj:
            hypergraph_metadata = obj["hypergraph_metadata"]
        if "hypergraph_type" in obj:
            hypergraph_type = obj["hypergraph_type"]
        t = obj.get("type")
        if t == "node":
            nodes.append(obj)
        elif t == "edge":
            edges.append(obj)

    return hypergraph_type, hypergraph_metadata, nodes, edges

def _build_hypergraph_from_json_objects(data_list: List[dict]):
    """
    Construct the correct Hypergraph* from the JSON 'array of objects' format.
    This is the single source of truth used by *both* local and remote JSON loaders.
    """
    htype, meta, nodes, edges = _split_json_records(data_list)
    if htype not in {"Hypergraph", "DirectedHypergraph", "MultiplexHypergraph", "TemporalHypergraph"}:
        raise ValueError(f"Unsupported or missing 'hypergraph_type': {htype!r}")

    weighted = bool(meta.get("weighted", False))

    # Instantiate
    if htype == "Hypergraph":
        H = Hypergraph(hypergraph_metadata=meta, weighted=weighted)
    elif htype == "DirectedHypergraph":
        H = DirectedHypergraph(hypergraph_metadata=meta, weighted=weighted)
    elif htype == "MultiplexHypergraph":
        H = MultiplexHypergraph(hypergraph_metadata=meta, weighted=weighted)
    else:  # htype == "TemporalHypergraph"
        H = TemporalHypergraph(hypergraph_metadata=meta, weighted=weighted)

    # Add nodes
    for n in nodes:
        H.add_node(n["idx"], n["metadata"])

    # Add edges depending on type
    if htype in {"Hypergraph", "DirectedHypergraph"}:
        for e in edges:
            interaction = e["interaction"]
            weight = e["metadata"].get("weight", None) if weighted else None
            H.add_edge(interaction, weight, metadata=e["metadata"])
    elif htype == "MultiplexHypergraph":
        for e in edges:
            interaction = e["interaction"]
            layer = e["metadata"].get("layer")
            weight = e["metadata"].get("weight", None) if weighted else None
            H.add_edge(interaction, layer, weight=weight, metadata=e["metadata"])
    else:  # TemporalHypergraph
        for e in edges:
            interaction = e["interaction"]
            time = e["metadata"].get("time")
            weight = e["metadata"].get("weight", None) if weighted else None
            H.add_edge(interaction, time, weight=weight, metadata=e["metadata"])

    return H

def _parse_json_bytes_to_hypergraph(data: bytes):
    """Decode UTF-8 JSON bytes (array of objects) and build the hypergraph."""
    try:
        data_list = json.loads(data.decode("utf-8"))
    except Exception as e:
        raise ValueError("Failed to parse JSON payload.") from e
    return _build_hypergraph_from_json_objects(data_list)

# ---------------------------------------------------------------------------
# Local loaders
# ---------------------------------------------------------------------------

def _load_pickle(file_name: str):
    """
    Load a pickle file and reconstruct the appropriate hypergraph.

    Parameters
    ----------
    file_name: str
        Name of the file to load.

    Returns
    -------
    object
        An instance of the appropriate hypergraph type.

    Raises
    ------
    RuntimeError
        If the file cannot be loaded or the data is invalid.
    """
    try:
        with open(file_name, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise ValueError("Pickle data is not a dictionary.")
        if "type" not in data:
            raise KeyError("The data is missing require key: 'type'.")

        h_type = data["type"]
        if h_type == "Hypergraph":
            H = Hypergraph(weighted=data["_weighted"])
        elif h_type == "TemporalHypergraph":
            H = TemporalHypergraph(weighted=data["_weighted"])
        elif h_type == "DirectedHypergraph":
            H = DirectedHypergraph(weighted=data["_weighted"])
        elif h_type == "MultiplexHypergraph":
            H = MultiplexHypergraph(weighted=data["_weighted"])
        else:
            raise ValueError(f"Unknown hypergraph type: {h_type}")

        H.populate_from_dict(data)
        return H
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle '{file_name}': {e}") from e

def _load_json_file(file_name: str):
    """Load a local .json file (array of objects) using the shared JSON builder."""
    with open(file_name, "r", encoding="utf-8") as infile:
        data_list = json.load(infile)
    return _build_hypergraph_from_json_objects(data_list)

def _load_hgr_file(file_name: str):
    """Load a local .hgr (hmetis) file. Logic preserved from your original implementation."""
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
                pass  # comment/empty
            elif nodes == 0:
                head = this_l.split(" ")
                edges = int(head[0])
                nodes = int(head[1])
                if len(head) == 3:
                    mode = int(head[2])
            elif read_count < edges:
                read_count += 1
                entries = [int(r) for r in this_l.split(" ") if r != ""]
                if mode % 10 == 1 and len(entries) > 1:  # weighted
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
        H = Hypergraph(
            edge_list=edge_l,
            weighted=(mode % 10) == 1,
            weights=w_l if mode % 10 == 1 else None,
        )
        return H

# ---------------------------------------------------------------------------
# Public: local loader by file extension
# ---------------------------------------------------------------------------

def load_hypergraph(file_name: str):
    """
    Load a hypergraph from a local file.

    Parameters
    ----------
    file_name : str
        The path to the file. Type inferred from extension.

    Returns
    -------
    Hypergraph
        The loaded hypergraph.

    Raises
    ------
    ValueError
        If the file extension is not one of {"hgx","json","hgr"}.

    Notes
    -----
    Supported local types:
      - "hgx"   : pickled Hypergraph object
      - "json"  : array-of-objects JSON (server/local format)
      - "hgr"   : hMetis format
    """
    file_type = file_name.split(".")[-1].lower()
    if file_type == "hgx":
        return _load_pickle(file_name)
    elif file_type == "json":
        return _load_json_file(file_name)
    elif file_type == "hgr":
        return _load_hgr_file(file_name)
    else:
        raise ValueError("Invalid file type. Expected one of: 'hgx', 'json', 'hgr'.")

# ---------------------------------------------------------------------------
# Public: remote loader (fixed base URL)
# ---------------------------------------------------------------------------

def load_hypergraph_from_server(name: str, fmt: str = "binary", *, timeout: int = 30):
    """
    Load a hypergraph directly from the server
    """
    kind = fmt.lower()
    if kind not in {"binary", "json"}:
        raise ValueError("fmt must be 'binary' or 'json'.")

    suffix = "hgx.gz" if kind == "binary" else "json.gz"
    url = f"{_BASE}/{name}/{name}.{suffix}"

    raw = _download(url, timeout=timeout)
    payload = _decompress_gzip_if_needed(raw)

    if kind == "json":
        return _parse_json_bytes_to_hypergraph(payload)

    # --- binary path: mirror _load_pickle(file_name) exactly ---
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"{name}_", suffix=".hgx", delete=False) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        # Call your existing pickle loader to ensure identical logic/validations
        return _load_pickle(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

