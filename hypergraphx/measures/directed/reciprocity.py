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


def strong_reciprocity(hypergraph: DirectedHypergraph, max_hyperedge_size: int) -> dict:
    """
    Calculate the strong reciprocity ratio for each hyperedge size in the hypergraph.

    Parameters
    ----------
    hypergraph: DirectedHypergraph
        The hypergraph to calculate the strong reciprocity ratio for.
    max_hyperedge_size: int
        The maximum hyperedge size to consider.

    Returns
    -------
    dict
        A dictionary containing the strong reciprocity ratio for each hyperedge size.
    """

    edges = hypergraph.get_edges()
    rec = {i: 0 for i in range(2, max_hyperedge_size + 1)}
    tot = {i: 0 for i in range(2, max_hyperedge_size + 1)}
    edge_set = {}
    node_reach = {}

    # Count total edges of each size, populate edge_set, and track reachability
    for edge in edges:
        size = len(edge[0]) + len(edge[1])
        if 2 <= size <= max_hyperedge_size:
            tot[size] += 1
            edge_tuple = (tuple(edge[0]), tuple(edge[1]))
            edge_set[edge_tuple] = 1

            # Track reachable nodes for each head node
            for node in edge[0]:
                if node not in node_reach:
                    node_reach[node] = set(edge[1])
                else:
                    node_reach[node] = node_reach[node].union(set(edge[1]))

    # Count reciprocated edges based on reachability
    for edge in edge_set:
        source, target = edge
        covered = set()

        # Accumulate nodes reachable from each node in the tail
        for node in target:
            if node in node_reach:
                covered = covered.union(node_reach[node])

        # Check if all head nodes are reachable from the tail
        if set(source).issubset(covered):
            size = len(source) + len(target)
            rec[size] += 1

    # Calculate reciprocity ratios
    for size in range(2, max_hyperedge_size + 1):
        if tot[size] != 0:
            rec[size] = rec[size] / tot[size]
        else:
            rec[size] = 0

    return rec


def weak_reciprocity(hypergraph: DirectedHypergraph, max_hyperedge_size: int) -> dict:
    """
    Calculate the weak reciprocity ratio for each hyperedge size in the hypergraph.

    Parameters
    ----------
    hypergraph: DirectedHypergraph
        The hypergraph to calculate the weak reciprocity ratio for.
    max_hyperedge_size: int
        The maximum hyperedge size to consider.

    Returns
    -------
    dict
        A dictionary containing the weak reciprocity ratio for each hyperedge size.
    """

    edges = hypergraph.get_edges()
    rec = {i: 0 for i in range(2, max_hyperedge_size + 1)}
    tot = {i: 0 for i in range(2, max_hyperedge_size + 1)}
    edge_set = {}
    bin_edges = {}

    # Count total edges of each size, populate edge_set, and track individual node connections
    for edge in edges:
        size = len(edge[0]) + len(edge[1])
        if 2 <= size <= max_hyperedge_size:
            tot[size] += 1
            edge_tuple = (tuple(edge[0]), tuple(edge[1]))
            edge_set[edge_tuple] = 1
            source, target = edge_tuple

            # Record direct connections between source and target nodes
            for i in source:
                for j in target:
                    bin_edges[(i, j)] = 1

    # Count reciprocated edges based on any pairwise reciprocity
    for edge in edge_set:
        source, target = edge
        is_reciprocated = False

        # Check if there exists a reverse connection for any source-target node pair
        for i in source:
            for j in target:
                if (j, i) in bin_edges:
                    is_reciprocated = True
                    break
            if is_reciprocated:
                break

        if is_reciprocated:
            size = len(source) + len(target)
            rec[size] += 1

    # Calculate reciprocity ratios
    for size in range(2, max_hyperedge_size + 1):
        if tot[size] != 0:
            rec[size] = rec[size] / tot[size]
        else:
            rec[size] = 0

    return rec
