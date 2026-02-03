"""
Community detection and inference models.

This package contains multiple community-detection approaches. Only a small,
curated subset is exposed at the package level for discoverability.

Wrappers import lazily on first use.

Stable output conventions for curated entrypoints:
- `core_periphery(...)` returns a dict mapping node -> coreness score.
- `hyperlink_communities(...)` returns a SciPy dendrogram array (edge clustering).

Optional standardized wrappers (fit/predict style) are provided in `hypergraphx.communities.api`:
- `run_core_periphery(...)` -> `CorePeripheryResult(scores=...)`
- `run_hyperlink_communities(...)` -> `HyperlinkCommunitiesResult(dendrogram=...)`
- `fit_hysc(...)` -> `HySCResult(memberships, labels, model)`
- `fit_hypergraph_mt(...)` -> `HypergraphMTResult(...)`
- `fit_hy_mmsbm(...)` -> `HyMMSBMResult(...)`
"""

from __future__ import annotations


def core_periphery(*args, **kwargs):
    from hypergraphx.communities.core_periphery.model import core_periphery as _impl

    return _impl(*args, **kwargs)


def hyperlink_communities(*args, **kwargs):
    from hypergraphx.communities.hyperlink_comm.hyperlink_communities import (
        hyperlink_communities as _impl,
    )

    return _impl(*args, **kwargs)


def get_num_hyperlink_communties(*args, **kwargs):
    from hypergraphx.communities.hyperlink_comm.hyperlink_communities import (
        get_num_hyperlink_communties as _impl,
    )

    return _impl(*args, **kwargs)


def overlapping_communities(*args, **kwargs):
    from hypergraphx.communities.hyperlink_comm.hyperlink_communities import (
        overlapping_communities as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = [
    "core_periphery",
    "hyperlink_communities",
    "get_num_hyperlink_communties",
    "overlapping_communities",
    # Stable wrapper API
    "run_core_periphery",
    "run_hyperlink_communities",
    "fit_hysc",
    "fit_hypergraph_mt",
    "fit_hy_mmsbm",
]


def run_core_periphery(*args, **kwargs):
    from hypergraphx.communities.api import run_core_periphery as _impl

    return _impl(*args, **kwargs)


def run_hyperlink_communities(*args, **kwargs):
    from hypergraphx.communities.api import run_hyperlink_communities as _impl

    return _impl(*args, **kwargs)


def fit_hysc(*args, **kwargs):
    from hypergraphx.communities.api import fit_hysc as _impl

    return _impl(*args, **kwargs)


def fit_hypergraph_mt(*args, **kwargs):
    from hypergraphx.communities.api import fit_hypergraph_mt as _impl

    return _impl(*args, **kwargs)


def fit_hy_mmsbm(*args, **kwargs):
    from hypergraphx.communities.api import fit_hy_mmsbm as _impl

    return _impl(*args, **kwargs)
