"""
Dynamical processes on hypergraphs.

This package intentionally provides lightweight, lazy entrypoints for the most
used processes to improve discoverability without imposing import-time cost.
"""

from __future__ import annotations


def simplicial_contagion(*args, **kwargs):
    from hypergraphx.dynamics.contagion import simplicial_contagion as _impl

    return _impl(*args, **kwargs)


def transition_matrix(*args, **kwargs):
    from hypergraphx.dynamics.randwalk import transition_matrix as _impl

    return _impl(*args, **kwargs)


def random_walk(*args, **kwargs):
    from hypergraphx.dynamics.randwalk import random_walk as _impl

    return _impl(*args, **kwargs)


def RW_stationary_state(*args, **kwargs):
    from hypergraphx.dynamics.randwalk import RW_stationary_state as _impl

    return _impl(*args, **kwargs)


def random_walk_density(*args, **kwargs):
    from hypergraphx.dynamics.randwalk import random_walk_density as _impl

    return _impl(*args, **kwargs)


def MSF(*args, **kwargs):
    from hypergraphx.dynamics.synch import MSF as _impl

    return _impl(*args, **kwargs)


def higher_order_MSF(*args, **kwargs):
    from hypergraphx.dynamics.synch import higher_order_MSF as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "simplicial_contagion",
    "transition_matrix",
    "random_walk",
    "RW_stationary_state",
    "random_walk_density",
    "MSF",
    "higher_order_MSF",
]
