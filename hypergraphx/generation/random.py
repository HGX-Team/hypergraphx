"""
Generate random hypergraphs
"""

import random, numpy as np
from hypergraphx import Hypergraph


def random_hypergraph(num_nodes: int, num_edges_by_size: dict, seed=None):
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
    if seed is not None:
        random.seed(seed)
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


def random_uniform_hypergraph(num_nodes: int, size: int, num_edges: int, seed=None):
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
    hg: Hypergraph, order=None, size=None, inplace=True, p=1.0, preserve_degree=False, seed=None
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
    Hypergraph or None
        The hypergraph with the shuffled hyperedges or None if inplace is True.

    Raises
    ------
    ValueError
        If order and size are both specified or neither are specified,
        or if p is not between 0 and 1.
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")

    if seed is not None:
        np.random.seed(seed)

    # Retrieve current hyperedges of the specified size.
    current_edges = list(hg.get_edges(size=size))
    num_edges = len(current_edges)
    num_to_randomize = int(p * num_edges)

    # Randomly choose indices of hyperedges to replace.
    indices_to_replace = set(random.sample(range(num_edges), num_to_randomize))

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
                sorted(np.random.choice(pool_nodes, size, replace=False, p=weights))
            )

            new_edges.append(new_edge)
        else:
            new_edges.append(edge)

    if inplace:
        hg.remove_edges(current_edges)
        hg.add_edges(new_edges)
    else:
        h = hg.copy()
        h.remove_edges(current_edges)
        h.add_edges(new_edges)
        return h


def random_shuffle_all_orders(
    hg: Hypergraph, p: float = 1.0, inplace: bool = True, preserve_degree: bool = False, seed=None
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
    """
    if not (0 <= p <= 1):
        raise ValueError("Parameter 'p' must be between 0 and 1.")

    target_hg = hg if inplace else hg.copy()

    for size in set(hg.get_sizes()):
        if inplace:
            random_shuffle(
                target_hg, size=size, p=p, inplace=True, preserve_degree=preserve_degree, seed=seed
            )
        else:
            target_hg = random_shuffle(
                target_hg,
                size=size,
                p=p,
                inplace=False,
                preserve_degree=preserve_degree,
                seed=seed
            )

    return target_hg


def add_random_edge(hg: Hypergraph, order=None, size=None, inplace=True, seed=None):
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
    """
    if order is not None and size is not None:
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1

    if seed is not None:
        random.seed(seed)

    nodes = list(hg.get_nodes())
    edge = tuple(sorted(random.sample(nodes, size)))

    if inplace:
        hg.add_edge(edge)
    else:
        h = hg.copy()
        h.add_edge(edge)
        return h


def add_random_edges(hg: Hypergraph, num_edges, order=None, size=None, inplace=True, seed=None):
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
        raise ValueError("Order and size cannot be both specified.")
    if order is None and size is None:
        raise ValueError("Order or size must be specified.")
    if size is None:
        size = order + 1
    if seed is not None:
        random.seed(seed)

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