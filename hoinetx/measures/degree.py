from hoinetx.core import Hypergraph


def degree(hg: Hypergraph, node, order=None, size=None):
    """
    Compute the degree of a node in the hypergraph.
    Parameters
    ----------
    hg
    node
    order
    size

    Returns
    -------

    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        return sum([1 for edge in hg.get_edges() if node in edge])
    elif size is not None:
        return sum([1 for edge in hg.get_edges(size=size) if node in edge])
    elif order is not None:
        return sum([1 for edge in hg.get_edges(order=order) if node in edge])


def degree_sequence(hg: Hypergraph, order=None, size=None):
    """
    Compute the degree sequence of the hypergraph.
    Parameters
    ----------
    hg
    order
    size

    Returns
    -------

    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if size is not None:
        order = size - 1
    if order is None:
        return {node: hg.degree(node) for node in hg.get_nodes()}
    else:
        return {node: hg.degree(node, order=order) for node in hg.get_nodes()}
