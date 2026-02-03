"""
Visualization helpers.

This module is intentionally import-light: it should be safe to import even if optional
plotting dependencies (e.g., matplotlib) are not installed. The actual implementations
are imported lazily when the functions are called.
"""

from __future__ import annotations


def _missing_viz_dep(exc: Exception) -> ImportError:
    err = ImportError(
        "hypergraphx.viz requires optional visualization dependencies. "
        "Install the project with the viz/docs extras (e.g. matplotlib) to use these helpers."
    )
    err.__cause__ = exc
    return err


def draw_communities(*args, **kwargs):
    try:
        from hypergraphx.viz.draw_communities import draw_communities as _impl
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise _missing_viz_dep(exc)
    return _impl(*args, **kwargs)


def draw_hypergraph(*args, **kwargs):
    try:
        from hypergraphx.viz.draw_hypergraph import draw_hypergraph as _impl
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise _missing_viz_dep(exc)
    return _impl(*args, **kwargs)


def draw_motifs(*args, **kwargs):
    try:
        from hypergraphx.viz.draw_motifs import draw_motifs as _impl
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise _missing_viz_dep(exc)
    return _impl(*args, **kwargs)


def draw_bipartite(*args, **kwargs):
    try:
        from hypergraphx.viz.draw_projections import draw_bipartite as _impl
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise _missing_viz_dep(exc)
    return _impl(*args, **kwargs)


def draw_clique(*args, **kwargs):
    try:
        from hypergraphx.viz.draw_projections import draw_clique as _impl
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise _missing_viz_dep(exc)
    return _impl(*args, **kwargs)


def draw_simplicial(*args, **kwargs):
    try:
        from hypergraphx.viz.draw_simplicial import draw_simplicial as _impl
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise _missing_viz_dep(exc)
    return _impl(*args, **kwargs)


def plot_motifs(*args, **kwargs):
    try:
        from hypergraphx.viz.plot_motifs import plot_motifs as _impl
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise _missing_viz_dep(exc)
    return _impl(*args, **kwargs)


__all__ = [
    "draw_communities",
    "draw_hypergraph",
    "draw_motifs",
    "draw_bipartite",
    "draw_clique",
    "draw_simplicial",
    "plot_motifs",
]
