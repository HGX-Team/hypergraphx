"""
Synthetic hypergraph generators.

This package exposes a curated set of generators at the package level for
discoverability (e.g. `hypergraphx.generation.random_hypergraph(...)`).

Implementations are imported lazily on first use.
"""

from __future__ import annotations


def random_hypergraph(*args, **kwargs):
    from hypergraphx.generation.random import random_hypergraph as _impl

    return _impl(*args, **kwargs)


def random_uniform_hypergraph(*args, **kwargs):
    from hypergraphx.generation.random import random_uniform_hypergraph as _impl

    return _impl(*args, **kwargs)


def random_shuffle(*args, **kwargs):
    from hypergraphx.generation.random import random_shuffle as _impl

    return _impl(*args, **kwargs)


def random_shuffle_all_orders(*args, **kwargs):
    from hypergraphx.generation.random import random_shuffle_all_orders as _impl

    return _impl(*args, **kwargs)


def add_random_edge(*args, **kwargs):
    from hypergraphx.generation.random import add_random_edge as _impl

    return _impl(*args, **kwargs)


def add_random_edges(*args, **kwargs):
    from hypergraphx.generation.random import add_random_edges as _impl

    return _impl(*args, **kwargs)


def configuration_model(*args, **kwargs):
    from hypergraphx.generation.configuration_model import configuration_model as _impl

    return _impl(*args, **kwargs)


def directed_configuration_model(*args, **kwargs):
    from hypergraphx.generation.directed_configuration_model import (
        directed_configuration_model as _impl,
    )

    return _impl(*args, **kwargs)


def scale_free_hypergraph(*args, **kwargs):
    from hypergraphx.generation.scale_free import scale_free_hypergraph as _impl

    return _impl(*args, **kwargs)


def HOADmodel(*args, **kwargs):
    from hypergraphx.generation.activity_driven import HOADmodel as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "random_hypergraph",
    "random_uniform_hypergraph",
    "random_shuffle",
    "random_shuffle_all_orders",
    "add_random_edge",
    "add_random_edges",
    "configuration_model",
    "directed_configuration_model",
    "scale_free_hypergraph",
    "HOADmodel",
]
