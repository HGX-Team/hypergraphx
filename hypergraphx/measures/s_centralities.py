import networkx as nx
from hypergraphx import Hypergraph, TemporalHypergraph, DirectedHypergraph
from hypergraphx.representations.projections import line_graph, bipartite_projection


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


def s_betweenness_averaged(H: TemporalHypergraph, s=1):
    """
    Computes the betweenness centrality for each edge in the temporal hypergraph.
    The function calculates the betweenness centrality during each time of the temporal hypergraph and then
    the result is the average betweenness centrality in each time.
    Parameters
    ----------
    H : TemporalHypergraph
        The temporal hypergraph to compute the betweenness centrality for.
    s : int, optional
    Returns
    -------
    dict.
        The betweenness centrality for each edge in the temporal hypergraph.
        The keys are the edges and the values are the betweenness centrality.
    """
    subhypergraphs = H.subhypergraph()
    T = len(subhypergraphs)
    res = dict()
    for hypergraph in subhypergraphs.values():
        lg, id_to_edge = line_graph(hypergraph, s=s)
        b = nx.betweenness_centrality(lg)
        for k, v in b.items():
            k = id_to_edge[k]
            if k not in res.keys():
                res[k] = 0
            res[k] += v
    return {k: v / T for k, v in res.items()}


def s_closeness_averaged(H: TemporalHypergraph, s=1):
    """
    Computes the closeness centrality for each edge in the temporal hypergraph.
    The function calculates the closeness centrality during each time of the temporal hypergraph and then
    the result is the average closeness centrality in each time.
    Parameters
    ----------
    H : TemporalHypergraph
        The temporal hypergraph to compute the closeness centrality for.
    s : int, optional
    Returns
    -------
    dict.
        The closeness centrality for each edge in the hypergraph.
        The keys are the edges and the values are the closeness centrality.
    """
    subhypergraphs = H.subhypergraph()
    T = len(subhypergraphs)
    res = dict()
    for hypergraph in subhypergraphs.values():
        lg, id_to_edge = line_graph(hypergraph, s=s)
        b = nx.closeness_centrality(lg)
        for k, v in b.items():
            k = id_to_edge[k]
            if k not in res.keys():
                res[k] = 0
            res[k] += v
    return {k: v / T for k, v in res.items()}


def s_betweenness_nodes(H: Hypergraph | DirectedHypergraph):
    """
    Computes the betweenness centrality for each node in the hypergraph.
    Parameters
    ----------
    H : Hypergraph
        The hypergraph to compute the betweenness centrality for.
    Returns
    -------
    dict.
        The betweenness centrality for each node in the hypergraph.
        The keys are the nodes and the values are the betweenness centrality.
    """

    lg, id_to_edge = bipartite_projection(H)
    b = nx.betweenness_centrality(lg)
    return {id_to_edge[k]: v for k, v in b.items() if "E" not in k}


def s_closeness_nodes(H: Hypergraph | DirectedHypergraph):
    """
    Computes the closeness centrality for each node in the hypergraph.
    Parameters
    ----------
    H : Hypergraph to compute the closeness centrality for.
    Returns
    -------
    dict.
        The closeness centrality for each node in the hypergraph.
        The keys are the nodes and the values are the betweenness centrality.
    """

    lg, id_to_edge = bipartite_projection(H)
    b = nx.closeness_centrality(lg)
    return {id_to_edge[k]: v for k, v in b.items() if "E" not in k}


def s_betweenness_nodes_averaged(H: TemporalHypergraph):
    """
    Computes the betweenness centrality for each node in the temporal hypergraph.
    The function calculates the betweenness centrality during each time of the temporal hypergraph and then
    the result is the average betweenness centrality in each time.
    Parameters
    ----------
    H : TemporalHypergraph
        The temporal hypergraph to compute the betweenness centrality for.
    Returns
    -------
    dict.
        The betweenness centrality for each node in the temporal hypergraph.
        The keys are the nodes and the values are the betweenness centrality.
    """

    subhypergraphs = H.subhypergraph()
    T = len(subhypergraphs)
    res = dict()
    for hypergraph in subhypergraphs.values():
        lg, id_to_edge = bipartite_projection(hypergraph)
        b = nx.betweenness_centrality(lg)
        for k, v in b.items():
            if "E" not in k:
                k = id_to_edge[k]
                if k not in res.keys():
                    res[k] = 0
                res[k] += v
    return {k: v / T for k, v in res.items() if "E" not in k}


def s_closenness_nodes_averaged(H: TemporalHypergraph):
    """
    Computes the closeness centrality for each node in the temporal hypergraph.
    The function calculates the closeness centrality during each time of the temporal hypergraph and then
    the result is the average closeness centrality in each time.
    Parameters
    ----------
    H : TemporalHypergraph
        The temporal hypergraph to compute the closeness centrality for.
    Returns
    -------
    dict.
        The closeness centrality for each node in the hypergraph.
        The keys are the nodes and the values are the closeness centrality.
    """
    subhypergraphs = H.subhypergraph()
    T = len(subhypergraphs)
    res = dict()
    for hypergraph in subhypergraphs.values():
        lg, id_to_edge = bipartite_projection(hypergraph)
        b = nx.closeness_centrality(lg)
        for k, v in b.items():
            if "E" not in k:
                k = id_to_edge[k]
                if k not in res.keys():
                    res[k] = 0
                res[k] += v
    return {k: v / T for k, v in res.items() if "E" not in k}
