import networkx as nx

from hypergraphx import Hypergraph
from hypergraphx.representations.projections import line_graph


def s_betweenness(H: Hypergraph, s=1):
    """
    Computes the betweenness centrality for each edge in the hypergraph.

    Parameters
    ----------
    H : Hypergraph to compute the betweenness centrality for.
    s

    Returns
    -------
    dict. The betweenness centrality for each edge in the hypergraph. The keys are the edges and the values are the betweenness centrality.
    """

    lg, id_to_edge = line_graph(H, s=s)
    b = nx.betweenness_centrality(lg)
    return {id_to_edge[k]: v for k, v in b.items()}


def s_closeness(H: Hypergraph, s=1):
    """
    Compute the closeness centrality for each edge in the hypergraph.

    Parameters
    ----------
    H : Hypergraph to compute the closeness centrality for.
    s

    Returns
    -------
    dict. The closeness centrality for each edge in the hypergraph. The keys are the edges and the values are the closeness centrality.
    """
    lg, id_to_edge = line_graph(H, s=s)
    c = nx.closeness_centrality(lg)
    return {id_to_edge[k]: v for k, v in c.items()}