from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.special import factorial

from hypergraphx import Hypergraph
from hypergraphx.utils.labeling import get_inverse_mapping


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
    return_mapping: bool = False,
) -> sparse.csr_array | Tuple[sparse.csr_array, Dict[int, Any]]:
    """Produce the binary incidence matrix representing a hypergraph.
    For any node i and hyperedge e, the entry (i, e) of the binary incidence matrix is 1
    if the node belongs to the hyperedge, 0 otherwise.

    Parameters
    ----------
    hypergraph: instance of the class Hypergraph.
        Every hyperedge is represented as either a tuple or list of nodes.
    return_mapping: return the dictionary mapping the new node indices to the hypergraph
        nodes.
        The node indices in the incidence matrix vary from 0 to N-1, where N is the
        total number of distinct nodes.

    Returns
    -------
    The binary adjacency matrix representing the hyperedges.
    If return_mapping is True, return the dictionary of node mappings.
    """
    encoder = hypergraph.get_mapping()
    hye_list = [tuple(encoder.transform(hye)) for hye in hypergraph.get_edges()]

    shape = (hypergraph.num_nodes(), hypergraph.num_edges())
    incidence = hye_list_to_binary_incidence(hye_list, shape).tocsr()
    if return_mapping:
        mapping = get_inverse_mapping(encoder)
        return incidence, mapping
    return incidence


def incidence_matrix(
    hypergraph: Hypergraph,
    return_mapping: bool = False,
) -> sparse.csr_array | Tuple[sparse.csr_array, Dict[int, Any]]:
    """Produce the binary incidence matrix representing a hypergraph.
    For any node i and hyperedge e, the entry (i, e) of the binary incidence matrix is
    the weight of the hyperedge if the node belongs to it, 0 otherwise.

    Parameters
    ----------
    hypergraph: instance of the class Hypergraph.
        Every hyperedge is represented as either a tuple or list of nodes.
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
        hypergraph, return_mapping=True
    )
    incidence = binary_incidence.multiply(hypergraph.get_weights()).tocsr()
    if return_mapping:
        return incidence, mapping
    return incidence


def incidence_matrix_by_order(
    hypergraph: Hypergraph, order: int, shape: Optional[Tuple[int]] = None, keep_isolated_nodes:bool=False, return_mapping: bool = False,
) -> sparse.spmatrix:
    """ Produce the incidence matrix of a hypergraph at a given order.
    For any node i and hyperedge e, the entry (i, e) of the incidence matrix is the
    weight of the hyperedge if the node belongs to it, 0 otherwise.

    Parameters
    ----------
    hypergraph
        The hypergraph.
    order
        The order of the hyperedges to consider.
    shape
        The shape of the incidence matrix.
    keep_isolated_nodes
        If True, keep the isolated nodes in the incidence matrix.
    return_mapping
        If True, return the dictionary mapping the node indices in the matrix to the hypergraph nodes.

    Returns
    -------
    The incidence matrix.
    If return_mapping is True, return the dictionary of node mappings.
    """
    binary_incidence, mapping = binary_incidence_matrix(
        hypergraph.get_edges(order=order, subhypergraph=True, keep_isolated_nodes=keep_isolated_nodes), shape, return_mapping
    )
    incidence = binary_incidence.multiply(hypergraph.get_weights(order=order)).tocsr()
    return incidence, mapping


def incidence_matrices_all_orders(
    hypergraph: Hypergraph, shape: Optional[Tuple[int]] = None, keep_isolated_nodes:bool=False, return_mapping: bool = False,
) -> List[sparse.spmatrix]:
    """ Produce the incidence matrices of a hypergraph at all orders.
    For any node i and hyperedge e, the entry (i, e) of the incidence matrix is the
    weight of the hyperedge if the node belongs to it, 0 otherwise.

    Parameters
    ----------
    hypergraph
        The hypergraph.
    shape
        The shape of the incidence matrix.
    keep_isolated_nodes
        If True, keep the isolated nodes in the incidence matrix.
    return_mapping
        If True, return the dictionary mapping the node indices in the matrix to the hypergraph nodes.

    Returns
    -------
    The incidence matrix.
    If return_mapping is True, return the dictionary of node mappings.
    """
    incidence_matrices = {}
    for order in range(1, hypergraph.max_order() + 1):
        # fix lista di mappings ad ogni ordine
        incidence_matrices[order], _ = incidence_matrix_by_order(hypergraph, order, shape, keep_isolated_nodes, return_mapping)
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


def adjacency_matrix_by_order(
    hypergraph: Hypergraph, order: int
) -> sparse.csc_array | Tuple[sparse.csc_array, Dict[int, Any]]:
    """Compute the adjacency matrix of the hypergraph by order.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix
    counts the number of hyperedges of a given order where both i and j are contained.

    Parameters
    ----------
    hypergraph: the hypergraph.
    order: the order.

    Returns
    -------
    The adjacency matrix of the hypergraph for a given order and the dictionary of node mappings.
    """
    incidence, mapping = incidence_matrix_by_order(hypergraph,order,keep_isolated_nodes=True,return_mapping=True)
    adj = incidence @ incidence.transpose()
    adj.setdiag(0)
    return adj, mapping

def temporal_adjacency_matrix_by_order(
    temporal_hypergraph: Dict[int, Hypergraph], order: int
) -> Dict[int, sparse.csc_array]:
    """Compute the temporal adjacency matrix of the temporal hypergraph by order.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix at time t
    counts the number of hyperedges of a given order, existing at time t, where both i and j are contained.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    order: the order.

    Returns
    -------
    A dictionary encoding the temporal adjacency matrix, i.e., {time : adjacency matrix}, and the dictionary of node mappings.
    """
    temporal_adjacency = {}
    for t in temporal_hypergraph.keys():
        hypergraph_t = temporal_hypergraph[t]
        adj_t, mapping = adjacency_matrix_by_order(hypergraph_t, order)
        temporal_adjacency[t] = adj_t   
    return temporal_adjacency, mapping

def temporal_adjacency_matrices_all_orders(
    temporal_hypergraph: Dict[int, Hypergraph], max_order: int
): # -> Dict[int, Tuple[sparse.csc_array]] ### Fra, I am not sure how to declare the output type
    """Compute the temporal adjacency matrices of the temporal hypergraph for all orders.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix of order d at time t
    counts the number of hyperedges of order d, existing at time t, where both i and j are contained.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    max_order: the maximum order of the hypergraph.

    Returns
    -------
    A dictionary encoding the temporal adjacency matrices, i.e., {time : (adjacency matrices)}, and the dictionary of node mappings.
    """
    temporal_adjacencies = {}
    for t in temporal_hypergraph.keys():
        hypergraph_t = temporal_hypergraph[t]
        adjacency_list_t = []
        for order in range(1,max_order + 1):
            adj_t_order_d, mapping = adjacency_matrix_by_order(hypergraph_t, order)
            adjacency_list_t.append(adj_t_order_d)
        temporal_adjacencies[t] = tuple(adjacency_list_t)   
    return temporal_adjacencies, mapping


def annealed_adjacency_matrix(
    temporal_adjacency_matrix: Dict[int, sparse.csc_array]
) -> sparse.csc_array:
    """Compute the annealed adjacency matrix of the temporal hypergraph by order.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix
    counts the average number of hyperedges of a given order where both i and j are contained over time.

    Parameters
    ----------
    temporal_adjacency_matrix: the temporal adjacency matrix for a given order, as a dictionary {time : adjacency matrix}.

    Returns
    -------
    The annealed adjacency matrix for a given order.
    """
    T = max(temporal_adjacency_matrix.keys())
    temporal_adjacency_matrix_lst = temporal_adjacency_matrix.values()
    annealed_adjacency_matrix = sum(temporal_adjacency_matrix_lst)/T
    return annealed_adjacency_matrix

def annealed_adjacency_matrices_all_orders(
    temporal_adjacency_matrices
): # -> Tuple[sparse.csc_array]: ###Fra, here as well I am not sure!
    """Compute the annealed adjacency matrices of the temporal hypergraph for all orders.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix of order d
    counts the average number of hyperedges of order d where both i and j are contained over time.

    Parameters
    ----------
    temporal_adjacency_matrix: a dictionary {time : adjacency matrix}.
    order: the order.

    Returns
    -------
    The annealed adjacency matrix for all orders.
    """
    temporal_adjacency_matrices_vals = temporal_adjacency_matrices.values()
    max_order = len(temporal_adjacency_matrices_vals[0])
    annealed_adjacency_matrices = []
    for order in range(max_order):
        temporal_adjacency_matrix_lst = [adjacencies_matrices_t[order] for adjacencies_matrices_t in temporal_adjacency_matrices_vals]
        temporal_adjacency_matrix_dct = dict(zip(temporal_adjacency_matrices.keys(), temporal_adjacency_matrix_lst))
        annealed_adjacency_matrix = annealed_adjacency_matrix(temporal_adjacency_matrix_dct)
        annealed_adjacency_matrices.append(annealed_adjacency_matrix)
    return tuple(annealed_adjacency_matrices)


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


def degree_matrix(hypergraph, order, mapping = None):
    """
    Compute the degree matrix of the hypergraph for a given order.
    For any node i in the hypergraph, the entry (i, i) of the degree matrix of order d is the degree of i in the hypergraph of order d.

    Parameters
    ----------
    hypergraph
        the hypergraph.
    order
        the order.
    mapping
        the dictionary mapping the new node indices to the hypergraph nodes.

    Returns
    -------
    The degree matrix of the hypergraph for a given order.
    """
    degree_dct = hypergraph.degree_sequence(order)
    inverse_mapping = {}
    if not mapping==None:
        for name in mapping.keys():
            inverse_mapping[mapping[name]] = name
    else:
        # calcolare il mapping dall'ipergrafo
        pass
    degree_lst = [degree_dct[inverse_mapping[n]] for n in sorted(inverse_mapping.keys())]

    return sparse.diags(degree_lst)


def laplacian_matrix_by_order(
    hypergraph: Hypergraph,
    order: int,
    weighted=False,
    shape: Optional[Tuple[int]] = None,
) -> sparse.spmatrix:
    incidence, mapping = incidence_matrix_by_order(hypergraph,order,shape,keep_isolated_nodes=True,return_mapping=True)

    #maybe wrong mapping of nodes? binary incidence returns the mapping of the nodes in the hypergraph
    degree_mtx = degree_matrix(hypergraph, order, mapping)
    laplacian = degree_mtx.multiply(order + 1) - incidence.dot(incidence.transpose())

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


def adjacency_tensor(hypergraph: Hypergraph) -> np.ndarray:
    """
    Compute the tensor of a uniform hypergraph.
    For a hypergraph of order m, create a tensor of order m+1 where i,j,k... is 1 if the hyperedge (i,j,k...) is in the
    hypergraph and 0 otherwise

    Parameters
    ----------
    
    hypergraph : Hypergraph
        The uniform hypergraph on which the tensor is computed.
    
    Returns
    -------
    T : np.ndarray
        The tensor of the hypergraph.
    """

    if not hypergraph.is_uniform():
        raise Exception("The hypergraph is not uniform.")
    
    order = len(hypergraph.get_edges()[0]) - 1
    T = np.zeros((hypergraph.num_nodes(),) * (order + 1))
    for edge in hypergraph.get_edges():
        from itertools import permutations
        for perm in permutations(edge):
            T[tuple(perm)] = 1
            
    return T

