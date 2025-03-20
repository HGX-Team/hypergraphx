from hypergraphx import Hypergraph, DirectedHypergraph, TemporalHypergraph
from hypergraphx.utils.visits import _bfs


def connected_components(hg: Hypergraph, order=None, size=None):
    """
    Return the connected components of the hypergraph.
    Parameters
    ----------
    hg : Hypergraph. The hypergraph to check.
    order : int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    list. The connected components of the hypergraph.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    visited = []
    components = []
    for node in hg.get_nodes():
        if node not in visited:
            component = _bfs(hg, node, size=order, order=size)
            visited += component
            components.append(component)
    return components


def node_connected_component(hg: Hypergraph, node, order=None, size=None):
    """
    Return the connected component of the hypergraph containing the given node.
    Parameters
    ----------
    hg : Hypergraph. The hypergraph to check.
    node : Node. The node to check.
    order : int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    list. The nodes in the connected component of the input node.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return _bfs(hg, node, size=None, order=None)


def num_connected_components(hg: Hypergraph, order=None, size=None):
    """
    Return the number of connected components of the hypergraph.
    Parameters
    ----------
    hg : Hypergraph. The hypergraph to check.
    order : int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    int. The number of connected components.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return len(hg.connected_components(size=None, order=None))


def largest_component(hg: Hypergraph, order=None, size=None):
    """
    Return the largest connected component of the hypergraph.
    Parameters
    ----------
    hg : Hypergraph. The hypergraph to check.
    order : int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    list. The nodes in the largest connected component.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    components = hg.connected_components(size=None, order=None)
    return max(components, key=len)


def largest_component_size(hg: Hypergraph, order=None, size=None):
    """
    Return the size of the largest connected component of the hypergraph.
    Parameters
    ----------
    hg : Hypergraph. The hypergraph to check.
    order : int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    int. The size of the largest connected component.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return len(hg.largest_component(size=None, order=None))


def isolated_nodes(hg: Hypergraph|DirectedHypergraph|TemporalHypergraph, order=None, size=None):
    """
    Return the isolated nodes of the hypergraph.
    Parameters
    ----------
    hg: Hypergraph. The hypergraph to check.
    order: int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size: int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    list. The isolated nodes.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return [
        node
        for node in hg.get_nodes()
        if len(hg.get_neighbors(node, order=order, size=size)) == 0
    ]


def is_isolated(hg: Hypergraph|DirectedHypergraph|TemporalHypergraph, node, order=None, size=None):
    """
    Return True if the given node is isolated.
    Parameters
    ----------
    hg : Hypergraph. The hypergraph to check.
    node : Node. The node to check.
    order : int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    bool. True if the node is isolated, False otherwise.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return len(list(hg.get_neighbors(node, order=order, size=size))) == 0


def is_connected(hg: Hypergraph, order=None, size=None):
    """
    Return True if the hypergraph is connected.
    Parameters
    ----------
    hg : Hypergraph. The hypergraph to check.
    order : int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    bool. True if the hypergraph is connected, False otherwise.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return len(hg.connected_components(order=order, size=size)) == 1
