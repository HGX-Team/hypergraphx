import numpy as np
from scipy import sparse

from hoinetx import Hypergraph
from hoinetx.linalg.linalg import adjacency_matrix


# Fixture loaded_hypergraph defined inside the package-level conftest.py


def test_adjacency_type(loaded_hypergraph: Hypergraph):
    adj = adjacency_matrix(loaded_hypergraph)
    assert isinstance(adj, sparse.csr_array)


def test_adjacency_shape(loaded_hypergraph: Hypergraph):
    N = loaded_hypergraph.num_nodes()
    adj = adjacency_matrix(loaded_hypergraph)
    assert adj.shape == (N, N)


def test_adjacency_is_symmetric(loaded_hypergraph: Hypergraph):
    adj = adjacency_matrix(loaded_hypergraph)
    assert not (adj != adj.transpose()).data.any()


def test_adjacency_from_dense_incidence(loaded_hypergraph: Hypergraph):
    N = loaded_hypergraph.num_nodes()
    dense_incidence = loaded_hypergraph.binary_incidence_matrix().todense()
    dense_adj = dense_incidence @ dense_incidence.transpose()
    dense_adj[np.arange(N), np.arange(N)] = 0  # set diagonal to 0

    sparse_adj = loaded_hypergraph.adjacency_matrix()

    assert np.all(dense_adj == sparse_adj.todense())
