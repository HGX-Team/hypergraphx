from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.special import factorial

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
        rows.extend(list(hye))
        columns.extend([j] * len(hye))

    inferred_N = max(map(max, (hye for hye in hye_list if hye))) + 1
    inferred_E = len(hye_list)
    if shape is not None:
        if shape[0] < inferred_N or shape[1] < inferred_E:
            raise ValueError("Provided shape incompatible with input hyperedge list.")
    else:
        shape = (inferred_N, inferred_E)

    data = np.ones_like(rows)

    return sparse.coo_array((data, (rows, columns)), shape=shape, dtype=np.uint8)


def binary_incidence_matrix(
    hypergraph: Hypergraph, shape: Optional[Tuple[int]] = None
) -> sparse.spmatrix:
    """Produce the binary incidence matrix representing a hypergraph.
    For any node i and hyperedge e, the entry (i, e) of the binary incidence matrix is 1
    if the node belongs to the hyperedge, 0 otherwise.

    Parameters
    ----------
    hypergraph: instance of the class Hypergraph.
        Every hyperedge is represented as either a tuple or list of nodes.
    shape: the shape of the adjacency matrix, passed to the array constructor.
        If None, it is inferred.

    Returns
    -------
    The binary adjacency matrix representing the hyperedges.
    """
    hye_list = list(hypergraph.edge_list.keys())
    return hye_list_to_binary_incidence(hye_list, shape).tocsr()


def incidence_matrix(
    hypergraph: Hypergraph, shape: Optional[Tuple[int]] = None
) -> sparse.spmatrix:
    """Produce the binary incidence matrix representing a hypergraph.
    For any node i and hyperedge e, the entry (i, e) of the binary incidence matrix is
    the weight of the hyperedge if the node belongs to it, 0 otherwise.

    Parameters
    ----------
    hypergraph: instance of the class Hypergraph.
        Every hyperedge is represented as either a tuple or list of nodes.
    shape: the shape of the adjacency matrix, passed to the array constructor.
        If None, it is inferred.

    Returns
    -------
    The binary adjacency matrix representing the hyperedges.
    """
    binary_incidence = binary_incidence_matrix(hypergraph, shape)
    incidence = binary_incidence.multiply(hypergraph.get_weights()).tocsr()
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


def adjacency_matrix(hypergraph: Hypergraph) -> sparse.spmatrix:
    """Compute the adjacency matrix of the hypergraph.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix
    counts the number of hyperedges where both i and j are contained.

    Parameters
    ----------
    hypergraph: the hypergraph.

    Returns
    -------
    The adjacency matrix of the hypergraph.
    """
    incidence = hypergraph.binary_incidence_matrix()
    adj = incidence @ incidence.transpose()
    adj.setdiag(0)
    return adj


def random_walk_adjacency(hypergraph: Hypergraph) -> sparse.spmatrix:
    """Compute the adjacency matrix matrix associated to the hypergraph random walk.
    For any two hyperedges e, f in the hypergraph, the entry (e, f) of the random walk
    adjacency has value 1 if their intersection is non-null, else 0.

    Parameters
    ----------
    hypergraph: the hypergraph.

    Returns
    -------
    The random walk adjacency matrix of the hypergraph.
    """
    incidence = hypergraph.binary_incidence_matrix()
    adj = incidence.transpose() @ incidence
    return adj


def laplacian_matrix_by_order(
    hypergraph: Hypergraph,
    order: int,
    weighted=False,
    shape: Optional[Tuple[int]] = None,
) -> sparse.spmatrix:
    binary_incidence = binary_incidence_matrix(
        hypergraph.get_edges(order=order, subhypergraph=True), shape
    )
    incidence = binary_incidence.multiply(hypergraph.get_weights(order=order)).tocsr()

    degree_dct = hypergraph.degree_sequence(order)
    degree_lst = [degree_dct[key] for key in sorted(degree_dct.keys(), reverse=False)]

    degree_matrix = sparse.diags(degree_lst)
    laplacian = degree_matrix.multiply(order+1) - incidence.dot(incidence.transpose())

    if weighted:
        scale_factor = factorial(order-1)
        laplacian = laplacian.multiply(scale_factor)

    return laplacian


def laplacian_matrices_all_orders(
    hypergraph: Hypergraph, weighted=False, shape: Optional[Tuple[int]] = None
) -> List[sparse.spmatrix]:
    laplacian_matrices = {}
    for order in range(1, hypergraph.max_order()+1):
        laplacian_matrices[order] = laplacian_matrix_by_order(hypergraph, order, weighted, shape)
    return laplacian_matrices


def compute_multiorder_laplacian(hypergraph: Hypergraph, sigmas, order_weighted = False, degree_weighted = True) -> sparse.spmatrix:
    if not type(sigmas) == np.ndarray: sigmas = np.array(sigmas)

    laplacians = laplacian_matrices_all_orders(hypergraph,order_weighted)
    laplacians = [laplacians[key] for key in sorted(laplacians.keys(), reverse=False)]
    weighted_laplacians = [laplacian.multiply(sigma) for laplacian,sigma in zip(laplacians,sigmas)]

    if degree_weighted: 
        average_degrees = [np.average(list(hypergraph.degree_sequence(order).values())) for order in range(1,hypergraph.max_order()+1)]
        weighted_laplacians = [laplacian.multiply(1.0/degree) for laplacian,degree in zip(weighted_laplacians,average_degrees)]
    
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
