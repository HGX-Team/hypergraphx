import numpy as np

from hypergraphx import (
    Hypergraph,
    DirectedHypergraph,
    TemporalHypergraph,
    MultiplexHypergraph,
)


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


def degree_correlation(hg: "Hypergraph") -> np.ndarray:
    """
    Computes the degree sequence correlation matrix of the hypergraph.

    Parameters
    ----------
    hg: Hypergraph
        The hypergraph of interest.

    Returns
    -------
    np.ndarray
        The degree sequence correlation matrix of the hypergraph.
        The (i, j) entry is the Pearson correlation coefficient between the degree sequence at size i + 2
        and the degree sequence at size j + 2.
    """
    from scipy.stats import pearsonr

    seqs = [hg.degree_sequence(size=size) for size in range(2, hg.max_size() + 1)]
    matrix_degree_corr = np.zeros((len(seqs), len(seqs)))
    for i in range(len(seqs)):
        for j in range(len(seqs)):
            matrix_degree_corr[i, j] = pearsonr(
                list(seqs[i].values()), list(seqs[j].values())
            )[0]

    return matrix_degree_corr


def degree_distribution(hg: "Hypergraph", order=None, size=None):
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
