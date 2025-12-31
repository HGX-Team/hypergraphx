import numpy as np
from scipy import sparse

import hypergraphx.linalg as hl
from hypergraphx import Hypergraph


# Fixture loaded_hypergraph defined inside the package-level conftest.py


def test_binary_incidence_type(loaded_hypergraph: Hypergraph):
    inc = loaded_hypergraph.binary_incidence_matrix(return_mapping=False)
    assert isinstance(inc, sparse.csr_array)


def test_binary_incidence_only_contains_ones(loaded_hypergraph: Hypergraph):
    inc = loaded_hypergraph.binary_incidence_matrix(return_mapping=False)
    assert np.all(inc.data == 1)


def test_binary_incidence_shape(loaded_hypergraph: Hypergraph):
    inc = loaded_hypergraph.binary_incidence_matrix(return_mapping=False)
    assert inc.shape == (loaded_hypergraph.num_nodes(), loaded_hypergraph.num_edges())


def test_mapping_type(loaded_hypergraph: Hypergraph):
    _, mapping = loaded_hypergraph.binary_incidence_matrix(return_mapping=True)
    assert isinstance(mapping, dict)


def test_mapping_has_values_only_nodes_in_hypergraph(loaded_hypergraph: Hypergraph):
    _, mapping = loaded_hypergraph.binary_incidence_matrix(return_mapping=True)
    assert set(mapping.values()).issubset(set(loaded_hypergraph.get_nodes()))


def test_mapping_maps_back_all_and_only_nodes_in_hypergraph(
    loaded_hypergraph: Hypergraph,
):
    _, mapping = loaded_hypergraph.binary_incidence_matrix(return_mapping=True)
    back_mapped_nodes = {mapping[i] for i in range(loaded_hypergraph.num_nodes())}
    hypergraph_nodes = set(loaded_hypergraph.get_nodes())
    assert back_mapped_nodes == hypergraph_nodes


def test_binary_incidence_from_dense(loaded_hypergraph: Hypergraph):
    def transform_hye_to_new_indices(hye, mapping_):
        return tuple(mapping_[node] for node in hye)

    inc, mapping = loaded_hypergraph.binary_incidence_matrix(return_mapping=True)
    inverse_mapping = dict(zip(mapping.values(), mapping.keys()))

    hye_list = [
        transform_hye_to_new_indices(hye, inverse_mapping)
        for hye in loaded_hypergraph.get_edges()
    ]
    dense_incidence = np.zeros(
        (loaded_hypergraph.num_nodes(), loaded_hypergraph.num_edges())
    )
    for i, hye in enumerate(hye_list):
        dense_incidence[hye, i] = 1

    assert np.all(inc.todense() == dense_incidence)


def test_incidence(loaded_hypergraph: Hypergraph):
    def transform_hye_to_new_indices(hye, mapping_):
        return tuple(mapping_[node] for node in hye)

    inc, mapping = loaded_hypergraph.incidence_matrix(return_mapping=True)
    inverse_mapping = dict(zip(mapping.values(), mapping.keys()))

    hye_list = [
        transform_hye_to_new_indices(hye, inverse_mapping)
        for hye in loaded_hypergraph.get_edges()
    ]
    incidence = np.zeros((loaded_hypergraph.num_nodes(), loaded_hypergraph.num_edges()))
    weights_arr = loaded_hypergraph.get_weights()
    for i, hye in enumerate(hye_list):
        incidence[hye, i] = weights_arr[i]

    assert np.all(inc.todense() == incidence)


def test_incidence_by_order(loaded_hypergraph: Hypergraph):
    def transform_hye_to_new_indices(hye, mapping_):
        return tuple(mapping_[node] for node in hye)

    order_to_test = 2

    inc, mapping = hl.incidence_matrix_by_order(
        loaded_hypergraph, order=order_to_test, return_mapping=True
    )

    inverse_mapping = dict(zip(mapping.values(), mapping.keys()))

    subhypergraph_by_order = loaded_hypergraph.get_edges(
        order=order_to_test, subhypergraph=True, keep_isolated_nodes=False
    )
    hye_list = [
        transform_hye_to_new_indices(hye, inverse_mapping)
        for hye in subhypergraph_by_order.get_edges()
    ]
    incidence = np.zeros(
        (subhypergraph_by_order.num_nodes(), subhypergraph_by_order.num_edges())
    )
    weights_arr = subhypergraph_by_order.get_weights()
    for i, hye in enumerate(hye_list):
        incidence[hye, i] = weights_arr[i]

    assert np.all(inc.todense() == incidence)


def test_incidence_all_orders(loaded_hypergraph: Hypergraph):
    def transform_hye_to_new_indices(hye, mapping_):
        return tuple(mapping_[node] for node in hye)

    inc_arr = hl.incidence_matrices_all_orders(loaded_hypergraph)

    for order_to_test in range(1, loaded_hypergraph.max_order() + 1):
        inc, mapping = hl.incidence_matrix_by_order(
            loaded_hypergraph, order=order_to_test, return_mapping=True
        )
        inverse_mapping = dict(zip(mapping.values(), mapping.keys()))

        subhypergraph_by_order = loaded_hypergraph.get_edges(
            order=order_to_test, subhypergraph=True, keep_isolated_nodes=False
        )
        hye_list = [
            transform_hye_to_new_indices(hye, inverse_mapping)
            for hye in subhypergraph_by_order.get_edges()
        ]
        incidence = np.zeros(
            (subhypergraph_by_order.num_nodes(), subhypergraph_by_order.num_edges())
        )
        weights_arr = subhypergraph_by_order.get_weights()
        for i, hye in enumerate(hye_list):
            incidence[hye, i] = weights_arr[i]

        assert np.all(inc_arr[order_to_test].todense() == incidence)
