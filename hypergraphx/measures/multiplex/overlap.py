from hypergraphx import MultiplexHypergraph

def edge_overlap(h: MultiplexHypergraph, edge):
    edge = tuple(sorted(edge))
    overlap = 0

    for layer in h.existing_layers:
        k = (edge, layer)
        if k in h._edge_list:
            overlap += h._edge_list[k]
    return overlap