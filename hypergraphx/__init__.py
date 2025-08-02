from hypergraphx.core.IHypergraph import IHypergraph
from hypergraphx.core.IUndirectedHypergraph import IUndirectedHypergraph

from hypergraphx.core.Hypergraph import Hypergraph
from hypergraphx.core.DirectedHypergraph import DirectedHypergraph
from hypergraphx.core.MultiplexHypergraph import MultiplexHypergraph
from hypergraphx.core.TemporalHypergraph import TemporalHypergraph
from . import readwrite

import sys

MIN_PYTHON_VERSION = (3, 10)
assert (
    sys.version_info >= MIN_PYTHON_VERSION
), f"requires Python {'.'.join([str(n) for n in MIN_PYTHON_VERSION])} or newer"

__version__ = "1.7.8"
