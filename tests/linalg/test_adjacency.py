import numpy as np
from scipy import sparse

from hypergraphx import Hypergraph
from hypergraphx.linalg.linalg import adjacency_matrix


# Fixture loaded_hypergraph defined inside the package-level conftest.py


########################################################################################
# method Hypergraph.adjacency_matrix
def test_adjacency_type(loaded_hypergraph: Hypergraph):
    adj = loaded_hypergraph.adjacency_matrix()
    assert isinstance(adj, sparse.csr_array)


def test_adjacency_shape(loaded_hypergraph: Hypergraph):
    N = loaded_hypergraph.num_nodes()
    adj = loaded_hypergraph.adjacency_matrix()
    assert adj.shape == (N, N)


def test_adjacency_is_symmetric(loaded_hypergraph: Hypergraph):
    adj = adjacency_matrix(loaded_hypergraph)
    assert not (adj != adj.transpose()).data.any()


def test_adjacency_diagonal_is_zero(loaded_hypergraph: Hypergraph):
    adj = adjacency_matrix(loaded_hypergraph)
    assert np.all(adj.diagonal() == 0)


def test_adjacency_from_dense_incidence(loaded_hypergraph: Hypergraph):
    N = loaded_hypergraph.num_nodes()
    dense_incidence = loaded_hypergraph.binary_incidence_matrix().todense()
    dense_adj = dense_incidence @ dense_incidence.transpose()
    dense_adj[np.arange(N), np.arange(N)] = 0  # set diagonal to 0

    sparse_adj = loaded_hypergraph.adjacency_matrix()

    assert np.all(dense_adj == sparse_adj.todense())


def test_adj_mapping_type(loaded_hypergraph: Hypergraph):
    _, mapping = loaded_hypergraph.adjacency_matrix(return_mapping=True)
    assert isinstance(mapping, dict)


def test_adj_mapping_has_values_only_nodes_in_hypergraph(loaded_hypergraph: Hypergraph):
    _, mapping = loaded_hypergraph.adjacency_matrix(return_mapping=True)
    assert set(mapping.values()).issubset(set(loaded_hypergraph.get_nodes()))


def test_adj_mapping_maps_back_all_and_only_nodes_in_hypergraph(
    loaded_hypergraph: Hypergraph,
):
    _, mapping = loaded_hypergraph.adjacency_matrix(return_mapping=True)
    back_mapped_nodes = {mapping[i] for i in range(loaded_hypergraph.num_nodes())}
    hypergraph_nodes = set(loaded_hypergraph.get_nodes())
    assert back_mapped_nodes == hypergraph_nodes


########################################################################################
# Method Hypergraph.dual_random_walk_adjacency
def test_rw_walk_adjacency_type(loaded_hypergraph: Hypergraph):
    adj = loaded_hypergraph.dual_random_walk_adjacency()
    assert isinstance(adj, sparse.csr_array)


def test_rw_walk_adjacency_shape(loaded_hypergraph: Hypergraph):
    adj = loaded_hypergraph.dual_random_walk_adjacency()
    E = loaded_hypergraph.num_edges()
    assert adj.shape == (E, E)


def test_rw_walk_adjacency_only_contains_ones(loaded_hypergraph: Hypergraph):
    adj = loaded_hypergraph.dual_random_walk_adjacency()
    assert np.all(adj.data == 1)


def test_rw_walk_adjacency_diagonal_is_one(loaded_hypergraph: Hypergraph):
    adj = loaded_hypergraph.dual_random_walk_adjacency()
    assert np.all(adj.diagonal() == 1)


def test_rw_walk_comparing_with_explicit_construction(loaded_hypergraph: Hypergraph):
    adj = loaded_hypergraph.dual_random_walk_adjacency()

    E = loaded_hypergraph.num_edges()
    explicit_adj = np.zeros((E, E))
    edge_sets = [set(edge) for edge in loaded_hypergraph.get_edges()]
    for i, e in enumerate(edge_sets):
        for j, f in enumerate(edge_sets):
            explicit_adj[i, j] = bool(e & f)

    assert np.all(adj.todense() == explicit_adj)


def test_rw_adj_mapping_type(loaded_hypergraph: Hypergraph):
    _, mapping = loaded_hypergraph.dual_random_walk_adjacency(return_mapping=True)
    assert isinstance(mapping, dict)


def test_rw_adj_mapping_has_values_only_edges_in_hypergraph(
    loaded_hypergraph: Hypergraph,
):
    _, mapping = loaded_hypergraph.dual_random_walk_adjacency(return_mapping=True)
    assert set(mapping.values()).issubset(set(loaded_hypergraph.get_edges()))


def test_rw_adj_mapping_maps_back_all_and_only_edges_in_hypergraph(
    loaded_hypergraph: Hypergraph,
):
    _, mapping = loaded_hypergraph.dual_random_walk_adjacency(return_mapping=True)
    back_mapped_edges = {mapping[i] for i in range(loaded_hypergraph.num_edges())}
    hypergraph_edges = set(loaded_hypergraph.get_edges())
    assert back_mapped_edges == hypergraph_edges
