"""
Generate random hypergraphs
"""

import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.exceptions import InvalidParameterError
from hypergraphx.generation._rng import np_rng, py_rng, split_seed


def random_hypergraph(num_nodes: int, num_edges_by_size: dict, seed: int | None = None):
    """
    Generate a random hypergraph with a given number of nodes and hyperedges for each size.
    If a hyperedge is sampled multiple times, it will be added to the hypergraph only once.

    Parameters
    ----------
    num_nodes : int
    num_edges_by_size : dict
        A dictionary mapping the size of the hyperedges to the number of hyperedges of that size.
    seed : int, optional
        Seed for the random number generator.

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
    rng = py_rng(seed)
    h = Hypergraph()
    nodes = list(range(num_nodes))
    h.add_nodes(nodes)
    for size in num_edges_by_size:
        edges = list()
        while len(edges) < num_edges_by_size[size]:
            edges.append(tuple(sorted(rng.sample(nodes, size))))
        edges = set(edges)
        h.add_edges(list(edges))
    return h


def random_uniform_hypergraph(
    num_nodes: int, size: int, num_edges: int, seed: int | None = None
):
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
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    Hypergraph
        A random hypergraph with the given number of nodes and hyperedges of the given size.
    """
    return random_hypergraph(num_nodes, {size: num_edges}, seed)


def random_shuffle(
    hg: Hypergraph,
    order=None,
    size=None,
    inplace: bool = False,
    p=1.0,
    preserve_degree=False,
    seed: int | None = None,
):
    """
    Shuffle the nodes of a hypergraph's hyperedges of a given order/size,
    replacing a fraction p of them.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to process.
    p : float, optional
        The fraction of hyperedges to randomize (0 <= p <= 1). Default is 1.0.
    order : int
        The order of the hyperedges to shuffle.
    size : int
        The size of the hyperedges to shuffle.
    inplace : bool, optional
        If True, modify the given hypergraph directly; if False, operate on a copy and return it.
    preserve_degree : bool, optional
        If True, attempt to preserve the degree distribution of the nodes during shuffling.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    Hypergraph
        The shuffled hypergraph. If `inplace=True`, this is the input object.

    Raises
    ------
    ValueError
        If order and size are both specified or neither are specified,
        or if p is not between 0 and 1.

    Examples
    --------
    >>> from hypergraphx import Hypergraph
    >>> from hypergraphx.generation import random_shuffle
    >>> H = Hypergraph(edge_list=[(0, 1, 2), (2, 3, 4)], weighted=False)
    >>> H2 = random_shuffle(H, size=3, p=1.0, inplace=False, seed=0)
    >>> H is H2
    False
    """
    if order is not None and size is not None:
        raise InvalidParameterError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")

    npgen = np_rng(seed)
    pyrand = py_rng(seed)

    # Retrieve current hyperedges of the specified size.
    current_edges = list(hg.get_edges(size=size))
    num_edges = len(current_edges)
    num_to_randomize = int(p * num_edges)

    # Randomly choose indices of hyperedges to replace.
    indices_to_replace = set(pyrand.sample(range(num_edges), num_to_randomize))

    # Build a pool of nodes only from the hyperedges being randomized.
    pool_nodes = {}
    for i in indices_to_replace:
        for node in current_edges[i]:
            if node not in pool_nodes:
                pool_nodes[node] = 1
            elif preserve_degree:
                pool_nodes[node] += 1

    weights = np.array(list(pool_nodes.values()))
    weights = weights / np.sum(weights)
    pool_nodes = np.array(list(pool_nodes.keys()))

    new_edges = []
    for i, edge in enumerate(current_edges):
        if i in indices_to_replace:
            # Build a new hyperedge using nodes only from the pool.
            new_edge = tuple(
                sorted(npgen.choice(pool_nodes, size, replace=False, p=weights))
            )

            new_edges.append(new_edge)
        else:
            new_edges.append(edge)

    if inplace:
        hg.remove_edges(current_edges)
        hg.add_edges(new_edges)
        return hg
    else:
        h = hg.copy()
        h.remove_edges(current_edges)
        h.add_edges(new_edges)
        return h


def random_shuffle_all_orders(
    hg: Hypergraph,
    p: float = 1.0,
    inplace: bool = False,
    preserve_degree: bool = False,
    seed: int | None = None,
) -> Hypergraph:
    """
    Shuffle the nodes of a hypergraph's hyperedges of a given order/size,
    replacing a fraction p of them. The process is repeated for every order of interaction.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to process.
    p : float, optional
        The fraction of hyperedges to randomize (0 <= p <= 1). Default is 1.0.
    inplace : bool, optional
        If True, modify the given hypergraph directly; if False, operate on a copy and return it.
    preserve_degree : bool, optional
        If True, attempt to preserve the degree distribution of the nodes during shuffling.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    Hypergraph
        The hypergraph with shuffled hyperedges. This is either the original hypergraph (if inplace is True)
        or a new, modified copy (if inplace is False).

    Raises
    ------
    ValueError
        If `p` is not between 0 and 1.

    Examples
    --------
    >>> from hypergraphx import Hypergraph
    >>> from hypergraphx.generation import random_shuffle_all_orders
    >>> H = Hypergraph(edge_list=[(0, 1), (0, 1, 2)], weighted=False)
    >>> H2 = random_shuffle_all_orders(H, p=1.0, inplace=False, seed=0)
    >>> H.num_edges() == H2.num_edges()
    True
    """
    if not (0 <= p <= 1):
        raise ValueError("Parameter 'p' must be between 0 and 1.")

    target_hg = hg if inplace else hg.copy()
    # If a seed is provided, derive per-size seeds deterministically so each size gets
    # a distinct stream of randomness.
    seed_rng = np_rng(seed) if seed is not None else None

    for size in set(hg.get_sizes()):
        per_size_seed = split_seed(seed_rng) if seed_rng is not None else None
        if inplace:
            random_shuffle(
                target_hg,
                size=size,
                p=p,
                inplace=True,
                preserve_degree=preserve_degree,
                seed=per_size_seed,
            )
        else:
            target_hg = random_shuffle(
                target_hg,
                size=size,
                p=p,
                inplace=False,
                preserve_degree=preserve_degree,
                seed=per_size_seed,
            )

    return target_hg


def add_random_edge(
    hg: Hypergraph,
    order=None,
    size=None,
    inplace: bool = False,
    seed: int | None = None,
):
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
    seed : int, optional
        Seed for the random number generator

    Returns
    -------
    Hypergraph or None
        The hypergraph with the added edge or None if inplace is True.

    Raises
    ------
    ValueError
        If order and size are both specified or neither are specified.

    Examples
    --------
    >>> from hypergraphx import Hypergraph
    >>> from hypergraphx.generation import add_random_edge
    >>> H = Hypergraph(edge_list=[(0, 1), (1, 2)], weighted=False)
    >>> H2 = add_random_edge(H, size=3, inplace=False, seed=0)
    >>> H.num_edges(), H2.num_edges()
    (2, 3)
    """
    if order is not None and size is not None:
        raise InvalidParameterError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1

    rng = py_rng(seed)

    nodes = list(hg.get_nodes())
    edge = tuple(sorted(rng.sample(nodes, size)))

    if inplace:
        hg.add_edge(edge)
        return hg
    else:
        h = hg.copy()
        h.add_edge(edge)
        return h


def add_random_edges(
    hg: Hypergraph,
    num_edges,
    order=None,
    size=None,
    inplace: bool = False,
    seed: int | None = None,
):
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
    seed : int, optional
        Seed for the random number generator

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
        raise InvalidParameterError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1
    rng = py_rng(seed)

    nodes = list(hg.get_nodes())
    edges = set()
    while len(edges) < num_edges:
        edges.add(tuple(sorted(rng.sample(nodes, size))))

    if inplace:
        hg.add_edges(list(edges))
        return hg
    else:
        h = hg.copy()
        h.add_edges(list(edges))
        return h
