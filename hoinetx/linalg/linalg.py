from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.special import factorial
from sklearn.preprocessing import LabelEncoder

from hoinetx.core.hypergraph import Hypergraph


def hye_list_to_binary_incidence(
    hye_list: List[Tuple[int]], shape: Optional[Tuple[int]] = None
) -> sparse.coo_array:
    """Convert a list of hyperedges into a scipy sparse COO array.
    The hyperedges need to be list of integers, representing nodes, starting from 0.
    If no shape is provided, this is inferred from the hyperedge list as (N, E).
    N is the number of nodes, given by the maximum integer observed in the hyperedge
    list plus one (since the node index starts from 0).
    E is the number of hyperedges in the list.
    If not None, the shape can only specify a tuple (N', E') where N' is greater or
    equal than the N inferred from the hyperedge list, and E' is greater or equal than
    the number of hyperedges in the list.

    Parameters
    ----------
    hye_list: the list of hyperedges.
        Every hyperedge is represented as a tuple of integer nodes.
    shape: the shape of the adjacency matrix, passed to the array constructor.
        If None, it is inferred.

    Returns
    -------
    The binary adjacency matrix representing the hyperedges.
    """
    rows = []
    columns = []
    for j, hye in enumerate(hye_list):
        # If there are repeated nodes in the hyperedge, count them once
        set_hye = set(hye)
        rows.extend(list(set_hye))
        columns.extend([j] * len(set_hye))

    #inferred_N = max(rows) + 1
    #inferred_E = len(hye_list)
    inferred_N = shape[0]
    inferred_E = shape[1]

    if shape is not None:
        if shape[0] < inferred_N or shape[1] < inferred_E:
            raise ValueError("Provided shape incompatible with input hyperedge list.")
    else:
        shape = (inferred_N, inferred_E)

    data = np.ones_like(rows)

    return sparse.coo_array((data, (rows, columns)), shape=shape, dtype=np.uint8)


def binary_incidence_matrix(
    hypergraph: Hypergraph,
    shape: Optional[Tuple[int]] = None,
    return_mapping: bool = False,
) -> sparse.csr_array | Tuple[sparse.csr_array, Dict[int, Any]]:
    """Produce the binary incidence matrix representing a hypergraph.
    For any node i and hyperedge e, the entry (i, e) of the binary incidence matrix is 1
    if the node belongs to the hyperedge, 0 otherwise.

    Parameters
    ----------
    hypergraph: instance of the class Hypergraph.
        Every hyperedge is represented as either a tuple or list of nodes.
    shape: the shape of the adjacency matrix, passed to the array constructor.
        If None, it is inferred.
    return_mapping: return the dictionary mapping the new node indices to the hypergraph
        nodes.
        The node indices in the incidence matrix vary from 0 to N-1, where N is the
        total number of distinct nodes.

    Returns
    -------
    The binary adjacency matrix representing the hyperedges.
    If return_mapping is True, return the dictionary of node mappings.
    """
    encoder = LabelEncoder()
    encoder.fit(hypergraph.get_nodes())
    hye_list = [tuple(encoder.transform(hye)) for hye in hypergraph.get_edges()]

    shape = (hypergraph.num_nodes(), hypergraph.num_edges())
    incidence = hye_list_to_binary_incidence(hye_list, shape).tocsr()
    if return_mapping:
        mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
        return incidence, mapping
    return incidence


def incidence_matrix(
    hypergraph: Hypergraph,
    shape: Optional[Tuple[int]] = None,
    return_mapping: bool = False,
) -> sparse.csr_array | Tuple[sparse.csr_array, Dict[int, Any]]:
    """Produce the binary incidence matrix representing a hypergraph.
    For any node i and hyperedge e, the entry (i, e) of the binary incidence matrix is
    the weight of the hyperedge if the node belongs to it, 0 otherwise.

    Parameters
    ----------
    hypergraph: instance of the class Hypergraph.
        Every hyperedge is represented as either a tuple or list of nodes.
    shape: the shape of the adjacency matrix, passed to the array constructor.
        If None, it is inferred.
    return_mapping: return the dictionary mapping the new node indices to the hypergraph
        nodes.
        The node indices in the incidence matrix vary from 0 to N-1, where N is the
        total number of distinct nodes.

    Returns
    -------
    The binary adjacency matrix representing the hyperedges.
    If return_mapping is True, return the dictionary of node mappings.
    """
    binary_incidence, mapping = binary_incidence_matrix(
        hypergraph, shape, return_mapping=True
    )
    print(binary_incidence.shape)
    incidence = binary_incidence.multiply(hypergraph.get_weights()).tocsr()
    if return_mapping:
        return incidence, mapping
    return incidence


def incidence_matrix_by_order(
    hypergraph: Hypergraph, order: int, shape: Optional[Tuple[int]] = None
) -> sparse.spmatrix:
    binary_incidence = binary_incidence_matrix(
        hypergraph.get_edges(order=order, subhypergraph=True), shape
    )
    incidence = binary_incidence.multiply(hypergraph.get_weights(order=order)).tocsr()
    return incidence


def incidence_matrices_all_orders(
    hypergraph: Hypergraph, shape: Optional[Tuple[int]] = None
) -> List[sparse.spmatrix]:
    incidence_matrices = {}
    for order in range(1, hypergraph.max_order() + 1):
        incidence_matrices[order] = incidence_matrix_by_order(hypergraph, order, shape)
    return incidence_matrices


def adjacency_matrix(
    hypergraph: Hypergraph, return_mapping: bool = False
) -> sparse.csc_array | Tuple[sparse.csc_array, Dict[int, Any]]:
    """Compute the adjacency matrix of the hypergraph.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix
    counts the number of hyperedges where both i and j are contained.

    Parameters
    ----------
    hypergraph: the hypergraph.
    return_mapping: return the dictionary mapping the new node indices to the hypergraph
        nodes.
        The node indices in the adjacency matrix vary from 0 to N-1, where N is the
        total number of distinct nodes.

    Returns
    -------
    The adjacency matrix of the hypergraph.
    If return_mapping is True, return the dictionary of node mappings.
    """
    incidence, mapping = hypergraph.binary_incidence_matrix(return_mapping=True)
    adj = incidence @ incidence.transpose()
    adj.setdiag(0)
    if return_mapping:
        return adj, mapping
    return adj


def dual_random_walk_adjacency(
    hypergraph: Hypergraph, return_mapping: bool = False
) -> sparse.csr_array | Tuple[sparse.csr_array, Dict[int, Any]]:
    """Compute the adjacency matrix matrix associated to the dual hypergraph random
    walk. For any two hyperedges e, f in the hypergraph, the entry (e, f) of the random
    walk adjacency has value 1 if their intersection is non-null, else 0. This is the
    matrix of adjacency between hyperedges in the dual hypergraph.

    Parameters
    ----------
    hypergraph: the hypergraph.
    return_mapping: return the dictionary mapping the new node indices to the hypergraph
        nodes.
        The node indices in the adjacency matrix vary from 0 to N-1, where N is the
        total number of distinct nodes.

    Returns
    -------
    The random walk adjacency matrix of the hypergraph.
    If return_mapping is True, return the dictionary of node mappings.
    """
    incidence, mapping = hypergraph.binary_incidence_matrix(return_mapping=True)
    adj = incidence.transpose() @ incidence
    adj.data = np.ones_like(adj.data)
    if return_mapping:
        return adj, mapping
    return adj


def laplacian_matrix_by_order(
    hypergraph: Hypergraph,
    order: int,
    weighted=False,
    shape: Optional[Tuple[int]] = None,
) -> sparse.spmatrix:
    binary_incidence = binary_incidence_matrix(
        hypergraph.get_edges(order=order, subhypergraph=True, keep_nodes=True), shape
    )
    incidence = binary_incidence.multiply(hypergraph.get_weights(order=order)).tocsr()

    degree_dct = hypergraph.degree_sequence(order)
    degree_lst = [degree_dct[key] for key in sorted(degree_dct.keys(), reverse=False)]

    degree_matrix = sparse.diags(degree_lst)
    laplacian = degree_matrix.multiply(order + 1) - incidence.dot(incidence.transpose())

    if weighted:
        scale_factor = factorial(order - 1)
        laplacian = laplacian.multiply(scale_factor)

    return laplacian


def laplacian_matrices_all_orders(
    hypergraph: Hypergraph, weighted=False, shape: Optional[Tuple[int]] = None
) -> List[sparse.spmatrix]:
    laplacian_matrices = {}
    for order in range(1, hypergraph.max_order() + 1):
        laplacian_matrices[order] = laplacian_matrix_by_order(
            hypergraph, order, weighted, shape
        )
    return laplacian_matrices


def compute_multiorder_laplacian(
    hypergraph: Hypergraph, sigmas, order_weighted=False, degree_weighted=True
) -> sparse.spmatrix:
    if not type(sigmas) == np.ndarray:
        sigmas = np.array(sigmas)

    laplacians = laplacian_matrices_all_orders(hypergraph, order_weighted)
    laplacians = [laplacians[key] for key in sorted(laplacians.keys(), reverse=False)]
    weighted_laplacians = [
        laplacian.multiply(sigma) for laplacian, sigma in zip(laplacians, sigmas)
    ]

    if degree_weighted:
        average_degrees = [
            np.average(list(hypergraph.degree_sequence(order).values()))
            for order in range(1, hypergraph.max_order() + 1)
        ]
        weighted_laplacians = [
            laplacian.multiply(1.0 / degree)
            for laplacian, degree in zip(weighted_laplacians, average_degrees)
        ]

    multiorder_laplacian = sum(weighted_laplacians)

    return multiorder_laplacian


def are_commuting(laplacian_matrices: List[sparse.spmatrix], verbose=True) -> bool:
    orders = len(laplacian_matrices)

    for d1 in range(orders - 1):
        laplacian_d1 = laplacian_matrices[d1]
        for d2 in range(d1 + 1, orders):
            laplacian_d2 = laplacian_matrices[d2]

            d1d2_product = laplacian_d1.dot(laplacian_d2)
            d2d1_product = laplacian_d2.dot(laplacian_d1)

            commutator = d1d2_product - d2d1_product

            if not commutator.any():
                if verbose:
                    print("The Laplacian matrices do not commute")
                return False

    if verbose:
        print("The Laplacian matrices commute")
    return True
