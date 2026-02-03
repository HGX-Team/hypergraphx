from itertools import chain, combinations

from hypergraphx import Hypergraph


def get_all_subsets(s):
    """
    Returns all subsets of a set.
    Parameters
    ----------
    s : set. The set to get all subsets of.

    Returns
    -------
    subsets : list. All subsets of the set.
    """
    return chain(*map(lambda x: combinations(s, x), range(0, len(s) + 1)))


def simplicial_complex(h: Hypergraph):
    """
    Returns a simplicial complex representation of the hypergraph.

    Parameters
    ----------
    h : Hypergraph. The hypergraph to be projected.

    Returns
    -------
    S : Hypergraph. The simplicial complex representation of the hypergraph.
    """
    s_edges = set()

    for edge in h.get_edges():
        subsets = get_all_subsets(edge)
        for subset in subsets:
            subset_key = h._normalize_edge(subset)
            s_edges.add(subset_key)
    S = Hypergraph(s_edges)
    return S
