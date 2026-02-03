"""
Graph/hypergraph representations and projections.

Curated entrypoints are exposed here for discoverability; implementations are
imported lazily on first use.
"""

from __future__ import annotations


def bipartite_projection(*args, **kwargs):
    from hypergraphx.representations.projections import bipartite_projection as _impl

    return _impl(*args, **kwargs)


def clique_projection(*args, **kwargs):
    from hypergraphx.representations.projections import clique_projection as _impl

    return _impl(*args, **kwargs)


def line_graph(*args, **kwargs):
    from hypergraphx.representations.projections import line_graph as _impl

    return _impl(*args, **kwargs)


def directed_line_graph(*args, **kwargs):
    from hypergraphx.representations.projections import directed_line_graph as _impl

    return _impl(*args, **kwargs)


def simplicial_complex(*args, **kwargs):
    from hypergraphx.representations.simplicial_complex import (
        simplicial_complex as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = [
    "bipartite_projection",
    "clique_projection",
    "line_graph",
    "directed_line_graph",
    "simplicial_complex",
]
