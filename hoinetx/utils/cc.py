from hoinetx.core import Hypergraph
from hoinetx.utils.visits import _bfs


def connected_components(hg: Hypergraph, order=None, size=None):
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
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return _bfs(hg, node, size=None, order=None)


def num_connected_components(hg: Hypergraph, order=None, size=None):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return len(hg.connected_components(size=None, order=None))


def largest_component(hg: Hypergraph, order=None, size=None):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    components = hg.connected_components(size=None, order=None)
    return max(components, key=len)


def largest_component_size(hg: Hypergraph, order=None, size=None):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return len(hg.largest_component(size=None, order=None))


def isolated_nodes(hg: Hypergraph, order=None, size=None):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return [node for node in hg.get_nodes() if len(hg.get_neighbors(node)) == 0]


def is_isolated(hg: Hypergraph, node, order=None, size=None):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return len(hg.get_neighbors(node)) == 0


def is_connected(hg: Hypergraph, order=None, size=None):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    return len(hg.connected_components(order=order, size=size)) == 1
