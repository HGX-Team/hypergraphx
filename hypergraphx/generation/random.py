"""
Generate random hypergraphs
"""

import random

from hypergraphx import Hypergraph


def random_hypergraph(num_nodes: int, num_edges_by_size: dict):
    """
    Generate a random hypergraph with a given number of nodes and hyperedges for each size.
    If a hyperedge is sampled multiple times, it will be added to the hypergraph only once.

    Parameters
    ----------
    num_nodes : int
    num_edges_by_size : dict
        A dictionary mapping the size of the hyperedges to the number of hyperedges of that size.

    Returns
    -------
    Hypergraph
        A random hypergraph with the given number of nodes and hyperedges for each size.

    Examples
    --------
    >>> from hypergraphx.generation import random_hypergraph
    >>> random_hypergraph(10, {2: 5, 3: 3})
    Hypergraph with 10 nodes and 8 edges.
    Edge list: [(3, 4), (4, 9), (7, 9), (8, 9), (3, 6), (0, 6, 9), (3, 6, 8), (1, 3, 4)]
    """
    h = Hypergraph()
    nodes = list(range(num_nodes))
    h.add_nodes(nodes)
    for size in num_edges_by_size:
        edges = list()
        while len(edges) < num_edges_by_size[size]:
            edges.append(tuple(sorted(random.sample(nodes, size))))
        edges = set(edges)
        h.add_edges(list(edges))
    return h


def random_uniform_hypergraph(num_nodes: int, size: int, num_edges: int):
    """
    Generate a random hypergraph with a given number of nodes and hyperedges of a given size.
    If a hyperedge is sampled multiple times, it will be added to the hypergraph only once.

    Parameters
    ----------
    num_nodes : int
        The number of nodes in the hypergraph.
    size : int
        The size of the hyperedges.
    num_edges : int
        The number of hyperedges of the given size.

    Returns
    -------
    Hypergraph
        A random hypergraph with the given number of nodes and hyperedges of the given size.

    Examples
    --------
    >>> from hypergraphx.generation import random_uniform_hypergraph
    >>> random_uniform_hypergraph(10, 3, 5)
    Hypergraph with 10 nodes and 5 edges.
    Edge list: [(5, 7, 9), (0, 2, 3), (0, 2, 9), (7, 8, 9), (1, 8, 9)]
    """
    return random_hypergraph(num_nodes, {size: num_edges})


def random_shuffle(hg: Hypergraph, order=None, size=None, inplace=True):
    """
    Shuffle the nodes of a hypergraph's hyperedges of a given order / size.

    Parameters
    ----------
    hg : hypergraph
        The Hypergraph of interest.
    order : int
        The order of the hyperedges to shuffle.
    size : int
        The size of the hyperedges to shuffle.
    inplace : bool
        Whether to modify the hypergraph in place or return a copy.

    Returns
    -------
    Hypergraph or None
        The hypergraph with the shuffled hyperedges or None if inplace is True.

    Raises
    ------
    ValueError
        If order and size are both specified or neither are specified.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1

    num_edges = hg.num_edges(size=size)
    nodes = list(hg.get_nodes())

    edges = list()
    while len(edges) < num_edges:
        edges.append(tuple(sorted(random.sample(nodes, size))))
    edges = set(edges)

    if inplace:
        hg.remove_edges(hg.get_edges(size=size))
        hg.add_edges(list(edges))
    else:
        h = hg.copy()
        h.remove_edges(h.get_edges(size=size))
        h.add_edges(list(edges))
        return h


def add_random_edge(hg: Hypergraph, order=None, size=None, inplace=True):
    """
    Add a random hyperedge of a given order / size to a hypergraph.

    Parameters
    ----------
    hg : hypergraph
        The Hypergraph of interest.
    order : int
        The order of the edge to add.
    size : int
        The size of the edge to add.
    inplace : bool
        Whether to modify the hypergraph in place or return a new one.

    Returns
    -------
    Hypergraph or None
        The hypergraph with the added edge or None if inplace is True.

    Raises
    ------
    ValueError
        If order and size are both specified or neither are specified.
    """
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
    """
    Add random hyperedges of a given order / size to a hypergraph.

    Parameters
    ----------
    hg : hypergraph
        The Hypergraph of interest.
    num_edges : int
        The number of edges to add.
    order : int
        The order of the edges to add.
    size : int
        The size of the edges to add.
    inplace : bool
        Whether to modify the hypergraph in place or return a copy.

    Returns
    -------
    Hypergraph or None
        The hypergraph with the added edges or None if inplace is True.

    Raises
    ------
    ValueError
        If order and size are both specified or neither are specified.
    """
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



