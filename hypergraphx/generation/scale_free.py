import numpy as np
from scipy.stats import spearmanr

from hypergraphx import Hypergraph


def scale_free_hypergraph(num_nodes: int, edges_by_size: dict, scale_by_size: dict, correlated: bool = True,
                          corr_target: float = None, num_shuffles: int = 0):
    """
    Generate a scale-free hypergraph.

    Parameters
    ----------
    num_nodes : int
        The number of nodes in the hypergraph.
    edges_by_size : dict
        A dictionary mapping the size of the hyperedges to the number of hyperedges of that size.
    scale_by_size : dict
        A dictionary mapping the size of the hyperedges to the scale parameter of the exponential distribution.
    correlated : bool
        Whether the exponential distributions of the different sizes should be correlated.
    corr_target : float
        The target correlation between the exponential distributions of the different sizes.
    num_shuffles : int

    Returns
    -------
    Hypergraph
        A scale-free hypergraph with the given number of nodes and hyperedges for each size.
    """
    if num_shuffles != 0 and not correlated:
        raise ValueError("Cannot shuffle if correlated == False")
    if num_shuffles < 0:
        raise ValueError("Cannot shuffle negative number of times")
    if corr_target < 0 or corr_target > 1:
        raise ValueError("Correlation must be between 0 and 1")
    if corr_target is not None and not correlated:
        raise ValueError("Cannot provide correlation value if correlated == False")
    if corr_target is not None and num_shuffles != 0:
        raise ValueError("Cannot provide both correlation value and number of shuffles")
    for k in edges_by_size:
        if k not in scale_by_size:
            raise ValueError("Must provide scale for each edge size")
    for k in scale_by_size:
        if k not in edges_by_size:
            raise ValueError("Must provide number of edges for each edge size")
    for k in edges_by_size:
        try:
            edges_by_size[k] = int(edges_by_size[k])
        except ValueError:
            raise ValueError("Number of edges must be an integer")
        if edges_by_size[k] < 0:
            raise ValueError("Number of edges must be non-negative")
    h = Hypergraph()
    nodes = list(range(num_nodes))
    h.add_nodes(nodes)
    old_dist = None
    for size in edges_by_size:
        num_edges = edges_by_size[size]
        scale = scale_by_size[size]
        exp_dist = np.random.exponential(scale, num_nodes)
        if correlated:
            exp_dist = list(sorted(exp_dist, reverse=True))
            for _ in range(num_shuffles):
                a, b = np.random.choice(num_nodes, size=2, replace=False)
                exp_dist[a], exp_dist[b] = exp_dist[b], exp_dist[a]
            if corr_target != 1 and old_dist is not None:
                corr, _ = spearmanr(exp_dist, old_dist)
                while corr > corr_target:
                    a, b = np.random.choice(num_nodes, size=2, replace=False)
                    exp_dist[a], exp_dist[b] = exp_dist[b], exp_dist[a]
                    corr, _ = spearmanr(exp_dist, old_dist)
        edges = set()
        while len(edges) < num_edges:
            edge = np.random.choice(nodes, size=size, replace=False, p=exp_dist / np.sum(exp_dist))
            edge = tuple(sorted(edge))
            edges.add(edge)

        h.add_edges(edges)
        if old_dist is None:
            old_dist = exp_dist
    return h
