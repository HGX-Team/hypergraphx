from hypergraphx import DirectedHypergraph


def exact_reciprocity(hypergraph: DirectedHypergraph, max_hyperedge_size: int) -> dict:
    """
    Calculate the exact reciprocity ratio for each hyperedge size in the hypergraph.

    Parameters
    ----------
    hypergraph: DirectedHypergraph
        The hypergraph to calculate the exact reciprocity ratio for.
    max_hyperedge_size: int
        The maximum hyperedge size to consider

    Returns
    -------
    dict
        A dictionary containing the exact reciprocity ratio for each hyperedge size.
    """

    edges = hypergraph.get_edges()
    rec = {i: 0 for i in range(2, max_hyperedge_size + 1)}
    tot = {i: 0 for i in range(2, max_hyperedge_size + 1)}
    edge_set = {}

    # Count total edges of each size and populate edge_set
    for edge in edges:
        size = len(edge[0]) + len(edge[1])
        if 2 <= size <= max_hyperedge_size:
            tot[size] += 1
            edge_tuple = (tuple(edge[0]), tuple(edge[1]))
            edge_set[edge_tuple] = 1

    # Count reciprocated edges
    for edge in edge_set:
        reciprocated_edge = (edge[1], edge[0])
        if reciprocated_edge in edge_set:
            size = len(edge[0]) + len(edge[1])
            rec[size] += 1

    # Calculate reciprocity ratios
    for size in range(2, max_hyperedge_size + 1):
        if tot[size] != 0:
            rec[size] = rec[size] / tot[size]
        else:
            rec[size] = 0

    return rec