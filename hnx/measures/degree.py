from hnx.core import Hypergraph


def degree(hg: Hypergraph, node, order=None, size=None):
    """
    Computes the degree of a node in the hypergraph.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph of interest.
    node : Node
        The node to check.
    order : int
        The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int
        The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    int
        The degree of the node.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        return len(hg.get_incident_edges(node))
    elif size is not None:
        return len(hg.get_incident_edges(node, size=size))
    elif order is not None:
        return len(hg.get_incident_edges(node, order=order))


def degree_sequence(hg: Hypergraph, order=None, size=None):
    """
    Computes the degree sequence of the hypergraph.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph ofinterest.
    order : int
        The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int
        The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    dict
        The degree sequence of the hypergraph. The keys are the nodes and the values are the degrees.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if size is not None:
        order = size - 1
    if order is None:
        return {node: hg.degree(node) for node in hg.get_nodes()}
    else:
        return {node: hg.degree(node, order=order) for node in hg.get_nodes()}


def degree_distribution(hg: Hypergraph, order=None, size=None):
    """
    Computes the degree distribution of the hypergraph.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph of interest.
    order : int
        The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int
        The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    dict
        The degree distribution of the hypergraph. The keys are the degrees and the values are the number of nodes with that degree.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if size is not None:
        order = size - 1
    if order is None:
        degree_seq = hg.degree_sequence()
    else:
        degree_seq = hg.degree_sequence(order=order)

    degree_dist = {}
    for node, deg in degree_seq.items():
        if deg not in degree_dist:
            degree_dist[deg] = 0
        degree_dist[deg] += 1

    return degree_dist
