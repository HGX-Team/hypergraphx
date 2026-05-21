import numpy as np


def filter_by_weight(
    hypergraph,
    *,
    top_percent: float,
    inplace: bool = True,
    drop_isolated_nodes_after_filter: bool = False,
):
    """
    Keep only hyperedges whose weight is in the top ``top_percent`` percent.

    Ties at the percentile threshold are kept, so the returned hypergraph can
    contain slightly more than ``top_percent`` percent of edges.
    """
    if not hypergraph.is_weighted():
        raise ValueError("Weight filtering requires a weighted hypergraph.")
    if top_percent <= 0 or top_percent > 100:
        raise ValueError("top_percent must be in the interval (0, 100].")

    if not inplace:
        hypergraph = hypergraph.copy()

    edges = list(hypergraph.get_edges())
    if not edges:
        return None if inplace else hypergraph

    weights = np.array([hypergraph.get_weight(edge) for edge in edges], dtype=float)
    threshold = np.percentile(weights, 100 - top_percent)

    for edge, weight in zip(edges, weights):
        if weight < threshold:
            hypergraph.remove_edge(edge)

    if drop_isolated_nodes_after_filter:
        hypergraph.remove_nodes(hypergraph.isolates())

    if not inplace:
        return hypergraph
