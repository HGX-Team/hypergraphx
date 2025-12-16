"""
def degree(h: MultiplexHypergraph, node, size=None):
    degree = {}
    for layer_name in h.existing_layers:
        degree[layer_name] = layer.degree(node, size=size)
    return degree

def degree_sequence(h: MultiplexHypergraph, size=None):
    degree = {}
    for node in h.get_nodes():
        degree[node] = h.degree(node, size=size)
    return degree
"""
