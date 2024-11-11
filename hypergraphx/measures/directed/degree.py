from hypergraphx import DirectedHypergraph

def in_degree(hypergraph: DirectedHypergraph, node: int, order=None, size=None) -> int:
    return len(list(hypergraph.get_incident_in_edges(node, order=order, size=size)))

def out_degree(hypergraph: DirectedHypergraph, node: int, order=None, size=None) -> int:
    return len(list(hypergraph.get_incident_out_edges(node, order=order, size=size)))

def in_degree_sequence(hg: DirectedHypergraph, order=None, size=None):
    """
    Computes the in-degree sequence of the hypergraph.

    Parameters
    ----------
    hg : DirectedHypergraph
        The hypergraph of interest.
    order : int
        The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int
        The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    dict
        The in-degree sequence of the hypergraph. The keys are the nodes and the values are the degrees.
    """
    return {node: in_degree(hg, node, order=order, size=size) for node in hg.get_nodes()}

def out_degree_sequence(hg: DirectedHypergraph, order=None, size=None):
    """
    Computes the out-degree sequence of the hypergraph.

    Parameters
    ----------
    hg : DirectedHypergraph
        The hypergraph of interest.
    order : int
        The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int
        The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    dict
        The out-degree sequence of the hypergraph. The keys are the nodes and the values are the degrees.
    """
    return {node: out_degree(hg, node, order=order, size=size) for node in hg.get_nodes()}

