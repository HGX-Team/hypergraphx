import random




def directed_configuration_model(hypergraph):
    """
    This function implements the directed configuration model to generate
    random hyperedges based on the given hypergraph. It ensures that each node's
    in-degree and out-degree are preserved in the generated hypergraph. 
    The number of hyperedges is preserved.
    Head and tail size is preserved for each hyperedge.
    hypergraph: list of hyperedges, where each hyperedge is a tuple containing two tuples: head and tail.
    """
    num_steps_heads = len(hypergraph)*10
    num_steps_tails = len(hypergraph)*10
    
    new_hyperedges = []
    for hyperedge in hypergraph:
        new_hyperedges.append((list(hyperedge[0]), list(hyperedge[1])))
    for _ in range(num_steps_heads):
        id1 = random.randint(0, len(new_hyperedges) - 1)
        id2 = random.randint(0, len(new_hyperedges) - 1)

        if id1 == id2:
            continue

        head1 = new_hyperedges[id1][0]
        head2 = new_hyperedges[id2][0]

        # select random node from head1
        node1 = random.choice(head1)
        # select random node from head2
        node2 = random.choice(head2)

        if node2 in head1 or node1 in head2:
            continue

        # swap node1 and node2
        head1[head1.index(node1)] = node2
        head2[head2.index(node2)] = node1

    for _ in range(num_steps_tails):
        id1 = random.randint(0, len(new_hyperedges) - 1)
        id2 = random.randint(0, len(new_hyperedges) - 1)

        if id1 == id2:
            continue

        tail1 = new_hyperedges[id1][1]
        tail2 = new_hyperedges[id2][1]

        # select random node from tail1
        node1 = random.choice(tail1)
        # select random node from tail2
        node2 = random.choice(tail2)

        # swap node1 and node2
        if node2 in tail1 or node1 in tail2:
            continue
        tail1[tail1.index(node1)] = node2
        tail2[tail2.index(node2)] = node1

    final_hyperedges = []
    for edge in new_hyperedges:
        final_hyperedges.append(tuple((tuple(sorted(edge[0])), tuple(sorted(edge[1])))))
    
    
    return final_hyperedges

