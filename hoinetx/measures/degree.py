from hoinetx.core import Hypergraph


def degree(hg: Hypergraph, node, order=None, size=None):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        return sum([1 for edge in hg.edge_list if node in edge])
    elif size is not None:
        return sum([1 for edge in hg.edge_list if node in edge and len(edge) == size])
    elif order is not None:
        return sum([1 for edge in hg.edge_list if node in edge and len(edge) == order + 1])


def degree_sequence(hg: Hypergraph, order=None, size=None):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if size is not None:
        order = size - 1
    if order is None:
        return {node: hg.degree(node) for node in hg.get_nodes()}
    else:
        return {node: hg.degree(node, order=order) for node in hg.get_nodes()}
