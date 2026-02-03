"""
Motif computations.

Expose a small, stable API surface at package level for discoverability.
Implementations are imported lazily on first use.
"""

from __future__ import annotations


def compute_motifs(*args, **kwargs):
    from hypergraphx.motifs.motifs import compute_motifs as _impl

    return _impl(*args, **kwargs)


def compute_directed_motifs(*args, **kwargs):
    from hypergraphx.motifs.directed_motifs import compute_directed_motifs as _impl

    return _impl(*args, **kwargs)


__all__ = ["compute_motifs", "compute_directed_motifs"]
