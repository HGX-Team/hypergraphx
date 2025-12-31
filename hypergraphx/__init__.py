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
    load_hypergraph,
    load_hypergraph_from_server,
    save_hypergraph,
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
    "load_hypergraph",
    "load_hypergraph_from_server",
    "save_hypergraph",
    "readwrite",
]
