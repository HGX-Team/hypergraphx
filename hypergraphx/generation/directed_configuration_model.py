import random

from hypergraphx import DirectedHypergraph


def directed_configuration_model(hypergraph: DirectedHypergraph) -> DirectedHypergraph:
    """
    This function implements the directed configuration model to generate
    random hyperedges based on the given hypergraph. It ensures that each node's
    in-degree and out-degree are preserved in the generated hypergraph. 
    The number of hyperedges is preserved.
    Head and tail size is preserved for each hyperedge.

    Parameters
    ----------
    hypergraph : DirectedHypergraph
        The input hypergraph.

    Returns
    -------
    DirectedHypergraph
        The generated hypergraph.
    """

    hypergraph = list(hypergraph.get_edges())
    num_steps_sources = len(hypergraph)*10
    num_steps_targets = len(hypergraph)*10
    
    new_hyperedges = []

    for hyperedge in hypergraph:
        new_hyperedges.append((list(hyperedge[0]), list(hyperedge[1])))
    for _ in range(num_steps_sources):
        id1 = random.randint(0, len(new_hyperedges) - 1)
        id2 = random.randint(0, len(new_hyperedges) - 1)

        if id1 == id2:
            continue

        source1 = new_hyperedges[id1][0]
        source2 = new_hyperedges[id2][0]

        # select random node from source1
        node1 = random.choice(source1)
        # select random node from source2
        node2 = random.choice(source2)

        if node2 in source1 or node1 in source2:
            continue

        # swap node1 and node2
        source1[source1.index(node1)] = node2
        source2[source2.index(node2)] = node1

    for _ in range(num_steps_targets):
        id1 = random.randint(0, len(new_hyperedges) - 1)
        id2 = random.randint(0, len(new_hyperedges) - 1)

        if id1 == id2:
            continue

        target1 = new_hyperedges[id1][1]
        target2 = new_hyperedges[id2][1]

        # select random node from target1
        node1 = random.choice(target1)
        # select random node from target2
        node2 = random.choice(target2)

        # swap node1 and node2
        if node2 in target1 or node1 in target2:
            continue
        target1[target1.index(node1)] = node2
        target2[target2.index(node2)] = node1

    final_hyperedges = []
    for edge in new_hyperedges:
        final_hyperedges.append(tuple((tuple(sorted(edge[0])), tuple(sorted(edge[1])))))
    
    new_hypergraph = DirectedHypergraph(edge_list=final_hyperedges)
    return new_hypergraph

