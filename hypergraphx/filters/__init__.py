"""
Selection / validation utilities.

Public API:
- `filter_hypergraph`: metadata-based node/edge filtering (supports `inplace=` and callable criteria).
- `filter_by_weight`: keep hyperedges with top percentile weights.
- `get_svh`, `get_svc`: statistically validated hyperlinks/cores (safe defaults: `mp=False`).
"""

from hypergraphx.filters.statistical_filters import get_svc, get_svh
from hypergraphx.filters.metadata_filters import filter_hypergraph
from hypergraphx.filters.weight_filters import filter_by_weight
