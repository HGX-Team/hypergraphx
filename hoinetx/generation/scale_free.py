import numpy as np
from hoinetx import Hypergraph


def scale_free(num_nodes: int, edges_by_size: dict, scale_by_size: dict, correlated: bool = True, num_shuffles: int = 0):
    if num_shuffles != 0 and not correlated:
        raise ValueError("Cannot shuffle if not correlated")
    if num_shuffles < 0:
        raise ValueError("Cannot shuffle negative number of times")
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
    for size in edges_by_size:
        num_edges = edges_by_size[size]
        scale = scale_by_size[size]
        exp_dist = np.random.exponential(scale, num_nodes)
        if correlated:
            exp_dist = list(sorted(exp_dist, reverse=True))
            for _ in range(num_shuffles):
                a, b = np.random.choice(num_nodes, size=2, replace=False)
                exp_dist[a], exp_dist[b] = exp_dist[b], exp_dist[a]
        edges = set()
        while len(edges) < num_edges:
            edge = np.random.choice(nodes, size=size, replace=False, p=exp_dist / np.sum(exp_dist))
            edge = tuple(sorted(edge))
            edges.add(edge)

        h.add_edges(edges)
    return h
