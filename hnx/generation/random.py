import random
from hnx.core.hypergraph import Hypergraph


def random_hypergraph(num_nodes, num_edges_by_size={}):
    h = Hypergraph()
    nodes = list(range(num_nodes))
    h.add_nodes(nodes)
    for size in num_edges_by_size:
        edges = set()
        while len(edges) < num_edges_by_size[size]:
            edges.add(tuple(sorted(random.sample(nodes, size))))
        h.add_edges(list(edges))
    return h


def random_uniform_hypergraph(num_nodes, size, num_edges):
    return random_hypergraph(num_nodes, {size: num_edges})


def random_shuffle(hg: Hypergraph, order=None, size=None, inplace=True):
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
        hg.remove_edges(hg.get_edges(size=size))
        hg.add_edges(list(edges))
    else:
        h = hg.copy()
        h.remove_edges(h.get_edges(size=size))
        h.add_edges(list(edges))
        return h


def add_random_edge(hg: Hypergraph, order=None, size=None, inplace=True):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1

    nodes = list(hg.get_nodes())
    edge = tuple(sorted(random.sample(nodes, size)))

    if inplace:
        hg.add_edge(edge)
    else:
        h = hg.copy()
        h.add_edge(edge)
        return h


def add_random_edges(hg: Hypergraph, num_edges, order=None, size=None, inplace=True):
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1

    nodes = list(hg.get_nodes())
    edges = set()
    while len(edges) < num_edges:
        edges.add(tuple(sorted(random.sample(nodes, size))))

    if inplace:
        hg.add_edges(list(edges))
    else:
        h = hg.copy()
        h.add_edges(list(edges))
        return h



