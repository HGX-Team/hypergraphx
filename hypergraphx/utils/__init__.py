"""
Utility helpers.

This module is intentionally import-light: most implementations are imported
lazily on first use to avoid pulling heavy dependencies at import time.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def normalize_array(*args, **kwargs):
    from hypergraphx.utils.community import normalize_array as _impl

    return _impl(*args, **kwargs)


def calculate_permutation_matrix(*args, **kwargs):
    from hypergraphx.utils.community import calculate_permutation_matrix as _impl

    return _impl(*args, **kwargs)


def connected_components(*args, **kwargs):
    from hypergraphx.utils.components import connected_components as _impl

    return _impl(*args, **kwargs)


def is_connected(*args, **kwargs):
    from hypergraphx.utils.components import is_connected as _impl

    return _impl(*args, **kwargs)


def isolated_nodes(*args, **kwargs):
    from hypergraphx.utils.components import isolated_nodes as _impl

    return _impl(*args, **kwargs)


def is_isolated(*args, **kwargs):
    from hypergraphx.utils.components import is_isolated as _impl

    return _impl(*args, **kwargs)


def largest_component(*args, **kwargs):
    from hypergraphx.utils.components import largest_component as _impl

    return _impl(*args, **kwargs)


def largest_component_size(*args, **kwargs):
    from hypergraphx.utils.components import largest_component_size as _impl

    return _impl(*args, **kwargs)


def node_connected_component(*args, **kwargs):
    from hypergraphx.utils.components import node_connected_component as _impl

    return _impl(*args, **kwargs)


def num_connected_components(*args, **kwargs):
    from hypergraphx.utils.components import num_connected_components as _impl

    return _impl(*args, **kwargs)


def _bfs(*args, **kwargs):
    from hypergraphx.utils.traversal import _bfs as _impl

    return _impl(*args, **kwargs)


def _dfs(*args, **kwargs):
    from hypergraphx.utils.traversal import _dfs as _impl

    return _impl(*args, **kwargs)


def canon_edge(*args, **kwargs):
    from hypergraphx.utils.edges import canon_edge as _impl

    return _impl(*args, **kwargs)


def merge_metadata(*args, **kwargs):
    from hypergraphx.utils.metadata import merge_metadata as _impl

    return _impl(*args, **kwargs)


def relabel_edge(*args, **kwargs):
    from hypergraphx.utils.labeling import relabel_edge as _impl

    return _impl(*args, **kwargs)


def relabel_edges(*args, **kwargs):
    from hypergraphx.utils.labeling import relabel_edges as _impl

    return _impl(*args, **kwargs)


def inverse_relabel_edge(*args, **kwargs):
    from hypergraphx.utils.labeling import inverse_relabel_edge as _impl

    return _impl(*args, **kwargs)


def inverse_relabel_edges(*args, **kwargs):
    from hypergraphx.utils.labeling import inverse_relabel_edges as _impl

    return _impl(*args, **kwargs)


def map_node(*args, **kwargs):
    from hypergraphx.utils.labeling import map_node as _impl

    return _impl(*args, **kwargs)


def map_nodes(*args, **kwargs):
    from hypergraphx.utils.labeling import map_nodes as _impl

    return _impl(*args, **kwargs)


def inverse_map_nodes(*args, **kwargs):
    from hypergraphx.utils.labeling import inverse_map_nodes as _impl

    return _impl(*args, **kwargs)


def get_inverse_mapping(*args, **kwargs):
    from hypergraphx.utils.labeling import get_inverse_mapping as _impl

    return _impl(*args, **kwargs)


def __getattr__(name: str):
    """
    Lazy access to helper submodules.

    Examples
    --------
    >>> import hypergraphx.utils as u
    >>> u.metadata.merge_metadata({"a": 1}, {"a": 2})
    {'a': [1, 2]}
    """
    if name in {"edges", "metadata", "labeling"}:
        return importlib.import_module(f"{__name__}.{name}")
    if name == "LabelEncoder":
        from hypergraphx.utils.labeling import LabelEncoder as _LabelEncoder

        return _LabelEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["edges", "metadata", "labeling"])


__all__ = [
    # public helpers
    "canon_edge",
    "merge_metadata",
    "LabelEncoder",
    "relabel_edge",
    "relabel_edges",
    "inverse_relabel_edge",
    "inverse_relabel_edges",
    "map_node",
    "map_nodes",
    "inverse_map_nodes",
    "get_inverse_mapping",
    # components/community helpers
    "calculate_permutation_matrix",
    "normalize_array",
    "connected_components",
    "is_connected",
    "isolated_nodes",
    "is_isolated",
    "largest_component",
    "largest_component_size",
    "node_connected_component",
    "num_connected_components",
]
