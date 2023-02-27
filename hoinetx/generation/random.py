import random
from hoinetx.core.hypergraph import Hypergraph


def random_hypergraph(num_nodes, num_edges_by_size={}):
    h = Hypergraph()
    nodes = list(range(num_nodes))
    h.add_nodes(nodes)
    for size in num_edges_by_size:
        while h.num_edges(size=size) < num_edges_by_size[size]:
            h.add_edge(random.sample(nodes, size))
    return h


def uniform_hypergraph(num_nodes, size, num_edges):
    return random_hypergraph(num_nodes, {size: num_edges})


def random_shuffle_by_order(hg: Hypergraph, order=None, size=None, inplace=True):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1

    num_edges = hg.num_edges(size=size)
    nodes = list(hg.get_nodes())

    edges = set()
    while len(edges) < num_edges:
        edges.add(tuple(sorted(random.sample(nodes, size))))

    if inplace:
        hg.del_edges(hg.get_edges(size=size))
        hg.add_edges(list(edges))
    else:
        h = hg.copy()
        h.del_edges(h.get_edges(size=size))
        h.add_edges(list(edges))
        return h




