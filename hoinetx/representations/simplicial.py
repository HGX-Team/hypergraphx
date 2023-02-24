from itertools import chain, combinations
from hoinetx.core.hypergraph import Hypergraph


def get_all_subsets(s):
    return chain(*map(lambda x: combinations(s, x), range(0, len(s)+1)))


def simplicial(h: Hypergraph):
    s_edges = set()

    for edge in h.get_edges():
        subsets = get_all_subsets(edge)
        for subset in subsets:
            subset = tuple(sorted(subset))
            s_edges.add(subset)

    return Hypergraph(s_edges)

