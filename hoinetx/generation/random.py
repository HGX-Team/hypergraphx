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


