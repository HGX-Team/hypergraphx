import random

import numpy as np

from hypergraphx import TemporalHypergraph
from hypergraphx.linalg.linalg import adjacency_matrix, annealed_adjacency_matrix


def _snapshot_mean(temporal):
    """The mean of the adjacency matrices of the snapshots, on all nodes."""
    _, inverse = annealed_adjacency_matrix(temporal, return_mapping=True)
    index = {node: i for i, node in inverse.items()}
    snapshots = temporal.subhypergraph()
    mean = np.zeros((len(index), len(index)))
    for hypergraph in snapshots.values():
        adj, mapping = adjacency_matrix(hypergraph, return_mapping=True)
        adj = adj.toarray()
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                mean[index[mapping[i]], index[mapping[j]]] += adj[i, j]
    return mean / len(snapshots)


def test_annealed_adjacency_is_the_mean_of_the_snapshots_with_gaps_in_time():
    rng = random.Random(1)
    for _ in range(10):
        edges = [tuple(rng.sample(range(12), rng.randint(2, 4))) for _ in range(40)]
        times = [rng.choice([0, 3, 4, 9, 15]) for _ in edges]  # not consecutive
        temporal = TemporalHypergraph(edge_list=edges, time_list=times, weighted=False)
        adj = annealed_adjacency_matrix(temporal)
        n = temporal.num_nodes()
        assert adj.shape == (n, n)
        assert not (adj != adj.transpose()).data.any()
        assert np.all(adj.diagonal() == 0)
        assert np.allclose(adj.toarray(), _snapshot_mean(temporal))


def test_annealed_adjacency_of_two_snapshots():
    temporal = TemporalHypergraph(
        edge_list=[(1, 2), (2, 3), (1, 2)], time_list=[5, 7, 7], weighted=False
    )
    adj, mapping = annealed_adjacency_matrix(temporal, return_mapping=True)
    index = {node: i for i, node in mapping.items()}
    assert adj[index[1], index[2]] == 1.0  # in both snapshots: (1 + 1) / 2
    assert adj[index[2], index[3]] == 0.5
    assert adj[index[1], index[3]] == 0.0
