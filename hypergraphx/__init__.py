from hypergraphx.core.directed import DirectedHypergraph
from hypergraphx.core.undirected import Hypergraph
from hypergraphx.core.multiplex import MultiplexHypergraph
from hypergraphx.core.temporal import TemporalHypergraph
from hypergraphx.exceptions import (
    HypergraphxError,
    InvalidFileTypeError,
    InvalidFormatError,
    InvalidParameterError,
    MissingEdgeError,
    MissingNodeError,
    ReadwriteError,
)
from hypergraphx.readwrite import (
    download_remote_dataset,
    download_remote_datasets,
    get_remote_dataset_info,
    iter_remote_hypergraphs,
    list_remote_datasets,
    load_hypergraph,
    load_hypergraph_from_server,
    save_hypergraph,
    search_remote_datasets,
)
from . import readwrite

import logging
import sys
from importlib.metadata import PackageNotFoundError, version

MIN_PYTHON_VERSION = (3, 10)
if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError(
        f"requires Python {'.'.join([str(n) for n in MIN_PYTHON_VERSION])} or newer"
    )

try:
    __version__ = version("hypergraphx")
except PackageNotFoundError:
    __version__ = "0+unknown"

logging.getLogger("hypergraphx").addHandler(logging.NullHandler())

__all__ = [
    "DirectedHypergraph",
    "Hypergraph",
    "MultiplexHypergraph",
    "TemporalHypergraph",
    "HypergraphxError",
    "InvalidFileTypeError",
    "InvalidFormatError",
    "InvalidParameterError",
    "MissingEdgeError",
    "MissingNodeError",
    "ReadwriteError",
    "download_remote_dataset",
    "download_remote_datasets",
    "get_remote_dataset_info",
    "iter_remote_hypergraphs",
    "list_remote_datasets",
    "load_hypergraph",
    "load_hypergraph_from_server",
    "save_hypergraph",
    "search_remote_datasets",
    "readwrite",
]
