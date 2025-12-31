from hypergraphx import MultiplexHypergraph
from hypergraphx.exceptions import MissingEdgeError


def edge_overlap(h: MultiplexHypergraph, edge):
    edge = tuple(sorted(edge))
    overlap = 0

    for layer in h.get_existing_layers():
        try:
            w = h.get_weight(edge, layer)
            overlap += w
        except MissingEdgeError:
            pass
    return overlap
