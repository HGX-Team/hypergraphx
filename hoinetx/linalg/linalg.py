from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse

from hoinetx.core.hypergraph import Hypergraph


def hye_list_to_binary_incidence(
    hye_list: List[Tuple[int]], shape: Optional[Tuple[int]]
) -> sparse.spmatrix:
    """Convert a list of hyperedges into a scipy sparse csc array.

    Parameters
    ----------
    hye_list: the list of hyperedges.
        Every hyperedge is represented as a tuple of nodes.
    shape: the shape of the adjacency matrix, passed to the array constructor.
        If None, it is inferred.

    Returns
    -------
    The binary adjacency matrix representing the hyperedges.
    """
    # See docs:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html
    len_list = [0] + [len(hye) for hye in hye_list]
    indptr = np.cumsum(len_list)

    type_ = type(hye_list[0])
    indices = type_(chain(*hye_list))

    data = np.ones_like(indices)

    return sparse.csc_array((data, indices, indptr), shape=shape).tocsr()


def binary_incidence_matrix(
    hypergraph: Hypergraph, shape: Optional[Tuple[int]] = None
) -> sparse.spmatrix:
    """Produce the binary incidence matrix representing a hypergraph.

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
    # See docs:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html
    hye_list = list(hypergraph.edge_list.keys())
    return hye_list_to_binary_incidence(hye_list, shape)


def incidence_matrix(
    hypergraph: Hypergraph, shape: Optional[Tuple[int]] = None
) -> sparse.spmatrix:
    binary_incidence = binary_incidence_matrix(hypergraph, shape)
    incidence = binary_incidence.multiply(hypergraph.get_weights()).tocsr()
    return incidence


def incidence_matrix_by_order(
    hypergraph: Hypergraph, order: int, shape: Optional[Tuple[int]] = None
) -> sparse.spmatrix:
    binary_incidence = binary_incidence_matrix(hypergraph.filter_by_order(order), shape)
    incidence = binary_incidence.multiply(hypergraph.get_weights()).tocsr()
    return incidence


def incidence_matrices_all_orders(hypergraph: Hypergraph, shape: Optional[Tuple[int]] = None) -> List[sparse.spmatrix]:
    incidence_matrices = {}
    for order in range(2, hypergraph.max_order() + 1):
        incidence_matrices[order] = incidence_matrix_by_order(hypergraph, order, shape)
    return incidence_matrices


def laplacian_matrix_by_order(
    hypergraph: Hypergraph, order: int, shape: Optional[Tuple[int]] = None
) -> sparse.spmatrix:
    binary_incidence = binary_incidence_matrix(hypergraph.filter_by_order(order), shape)
    incidence = binary_incidence.multiply(hypergraph.get_weights()).tocsr()
    
    degree_dct = hypergraph.degree_sequence(order)
    degree_lst = [degree_dct[key] for key in sorted(degree_dct.keys(), reverse=True)]

    degree_matrix = sparse.diags(degree_lst)
    laplacian = degree_matrix.multiply(order) - incidence.multiply(incidence.transpose())
    return laplacian


def laplacian_matrices_all_orders(hypergraph: Hypergraph, shape: Optional[Tuple[int]] = None) -> List[sparse.spmatrix]:
    laplacian_matrices = {}
    for order in range(2, hypergraph.max_order() + 1):
        laplacian_matrices[order] = laplacian_matrix_by_order(hypergraph, order, shape)
    return laplacian_matrices

def compute_effect_laplacian(laplacians: List[sparse.spmatrix], sigmas) -> sparse.spmatrix:
    if not type(sigmas) == np.ndarray: sigmas = np.array(sigmas)

    weighted_laplacians = [laplacian.multiply(sigma) for laplacian,sigma in zip(laplacians,sigmas)]
    effective_laplacian = sum(weighted_laplacians)

    return effective_laplacian

def are_commuting(laplacian_matrices: List[sparse.spmatrix], verbose = True):

    orders = len(laplacian_matrices)

    for d1 in range(orders-1):
        laplacian_d1 = laplacian_matrices[d1]
        for d2 in range(d1+1,orders):
            laplacian_d2 = laplacian_matrices[d2]

            d1d2_product = laplacian_d1.dot(laplacian_d2)
            d2d1_product = laplacian_d2.dot(laplacian_d1)

            commutator = d1d2_product - d2d1_product

            if not commutator.any():
                if verbose: print("The Laplacian matrices do not commute")
                return False

    if verbose: print("The Laplacian matrices commute")
    return True