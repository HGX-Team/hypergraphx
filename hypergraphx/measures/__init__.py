"""
Measures and statistics.

This package exposes a curated set of commonly used measures at the package level
for discoverability (e.g. `hypergraphx.measures.degree(...)`).

Implementation modules can be import-heavy; wrappers import lazily on first use.
"""

from __future__ import annotations


def degree(hypergraph, node, *, order=None, size=None):
    from hypergraphx.measures.degree import degree as _impl

    return _impl(hypergraph, node, order=order, size=size)


def degree_sequence(hypergraph, *, order=None, size=None):
    from hypergraphx.measures.degree import degree_sequence as _impl

    return _impl(hypergraph, order=order, size=size)


def degree_distribution(hypergraph, *, order=None, size=None):
    from hypergraphx.measures.degree import degree_distribution as _impl

    return _impl(hypergraph, order=order, size=size)


def degree_correlation(hypergraph):
    from hypergraphx.measures.degree import degree_correlation as _impl

    return _impl(hypergraph)


def intersection(a, b):
    from hypergraphx.measures.edge_similarity import intersection as _impl

    return _impl(a, b)


def jaccard_similarity(a, b):
    from hypergraphx.measures.edge_similarity import jaccard_similarity as _impl

    return _impl(a, b)


def jaccard_distance(a, b):
    from hypergraphx.measures.edge_similarity import jaccard_distance as _impl

    return _impl(a, b)


def subhypergraph_centrality(hypergraph):
    from hypergraphx.measures.sub_hypergraph_centrality import (
        subhypergraph_centrality as _impl,
    )

    return _impl(hypergraph)


def CEC_centrality(hypergraph, *, tol=1e-7, max_iter=1000, seed=None, rng=None):
    from hypergraphx.measures.eigen_centralities import CEC_centrality as _impl

    return _impl(hypergraph, tol=tol, max_iter=max_iter, seed=seed, rng=rng)


def ZEC_centrality(hypergraph, *, tol=1e-7, max_iter=1000, seed=None, rng=None):
    from hypergraphx.measures.eigen_centralities import ZEC_centrality as _impl

    return _impl(hypergraph, tol=tol, max_iter=max_iter, seed=seed, rng=rng)


def HEC_centrality(hypergraph, *, tol=1e-6, max_iter=100, seed=None, rng=None):
    from hypergraphx.measures.eigen_centralities import HEC_centrality as _impl

    return _impl(hypergraph, tol=tol, max_iter=max_iter, seed=seed, rng=rng)


__all__ = [
    "degree",
    "degree_sequence",
    "degree_distribution",
    "degree_correlation",
    "intersection",
    "jaccard_similarity",
    "jaccard_distance",
    "subhypergraph_centrality",
    "CEC_centrality",
    "ZEC_centrality",
    "HEC_centrality",
]
