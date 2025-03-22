from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse import csc_array
from scipy.special import factorial
from hypergraphx import Hypergraph, TemporalHypergraph
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

    if len(rows) > 0:
        inferred_N = max(rows) + 1
    else:
        inferred_N = 0
    inferred_E = len(hye_list)
    # inferred_N = shape[0]
    # inferred_E = shape[1]

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
    """Produce the incidence matrix representing a hypergraph.
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
    binary_incidence, mapping = binary_incidence_matrix(hypergraph, return_mapping=True)
    incidence = binary_incidence.multiply(hypergraph.get_weights()).tocsr()
    if return_mapping:
        return incidence, mapping
    return incidence

def incidence_matrix_by_order(
    hypergraph: Hypergraph,
    order: int,
    shape: Optional[Tuple[int]] = None,
    keep_isolated_nodes: bool = False,
    return_mapping: bool = False,
) -> sparse.spmatrix:
    """Produce the incidence matrix of a hypergraph at a given order.
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
        hypergraph.get_edges(
            order=order, subhypergraph=True, keep_isolated_nodes=keep_isolated_nodes
        ),
        return_mapping=True,
    )

    incidence = binary_incidence.multiply(hypergraph.get_weights(order=order)).tocsr()
    if return_mapping:
        return incidence, mapping
    return incidence

def incidence_matrices_all_orders(
    hypergraph: Hypergraph,
    shape: Optional[Tuple[int]] = None,
    keep_isolated_nodes: bool = False,
    return_mapping: bool = False,
) -> List[sparse.spmatrix]:
    """Produce the incidence matrices of a hypergraph at all orders.
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
        if return_mapping:
            incidence_matrices[order], _ = incidence_matrix_by_order(
                hypergraph, order, shape, keep_isolated_nodes, return_mapping
            )
        else:
            incidence_matrices[order] = incidence_matrix_by_order(
                hypergraph, order, shape, keep_isolated_nodes, return_mapping
            )
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
    hypergraph: Hypergraph, order: int, return_mapping: bool = False
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
    incidence, mapping = incidence_matrix_by_order(
        hypergraph, order, keep_isolated_nodes=True, return_mapping=True
    )
    adj = incidence @ incidence.transpose()
    diagonal = adj.diagonal()
    diagonal_matrix = sparse.diags(diagonal)
    adj = adj - diagonal_matrix
    if return_mapping:
        return adj, mapping
    else:
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

def degree_matrix(hypergraph, order, mapping=None):
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
    if not mapping == None:
        for name in mapping.keys():
            inverse_mapping[mapping[name]] = name
    else:
        # calcolare il mapping dall'ipergrafo
        pass
    degree_lst = [
        degree_dct[inverse_mapping[n]] for n in sorted(inverse_mapping.keys())
    ]

    return sparse.diags(degree_lst)

def laplacian_matrix_by_order(
    hypergraph: Hypergraph,
    order: int,
    weighted=False,
    shape: Optional[Tuple[int]] = None,
) -> sparse.spmatrix:
    incidence, mapping = incidence_matrix_by_order(
        hypergraph, order, shape, keep_isolated_nodes=True, return_mapping=True
    )

    # maybe wrong mapping of nodes? binary incidence returns the mapping of the nodes in the hypergraph
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

def adjacency_factor(hypergraph: Hypergraph|TemporalHypergraph, t: int = 0):
    if isinstance(hypergraph, Hypergraph):
        matrix, mapping = hypergraph.adjacency_matrix(return_mapping=True)
    elif isinstance(hypergraph, TemporalHypergraph):
        matrix, mapping = hypergraph.annealed_adjacency_matrix(return_mapping=True)
    else:
        raise ValueError("An Hypergraph or Temporal Hypergraph must be provided.")
    new_mapping = dict()
    for k, v in mapping.items():
        new_mapping[v] = k
    mapping = new_mapping
    res = dict()
    for node1 in hypergraph.get_nodes():
        res[node1] = 0
        for node2 in hypergraph.get_nodes():
            if node1 != node2:
                try:
                    val = round(float(matrix[mapping[node1], mapping[node2]]),3)
                except IndexError:
                    val = round(float(matrix[mapping[node2], mapping[node1]]),3)
                if val != 0:
                    res[node1] += val ** t
    return res

#Temporal Hypergraph Adjacency Matrix
def temporal_adjacency_matrix(
    temporal_hypergraph: TemporalHypergraph,
    return_mapping: bool = False,
) -> Dict[int, sparse.csc_array] | Tuple[Dict[int, sparse.csc_array], dict[int,dict[int,int]]]:
    """
    Compute the temporal adjacency matrix of the temporal hypergraph by order.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix at time t
    counts the number of hyperedges of a given order, existing at time t, where both i and j are contained.
    Parameters
    ----------
    temporal_hypergraph: TemporalHypergraph.
    return_mapping: bool,optional
        Return the dictionary mapping the new node indices to the Temporal Hypergraph
    Returns
    -------
    temporal_adjacency_matrixes: Dict[int, sparse.csc_array]
        A dictionary encoding the temporal adjacency matrixes, i.e., {time : adjacency matrix}
    mapping: Dict[int, Dict[int,int]]
        The dictionary of node mappings for each adjacency matrix.
    """
    temporal_adjacency_matrixes = {}
    subhypergraphs = temporal_hypergraph.subhypergraph()
    mapping = dict()
    for t in subhypergraphs.keys():
        hypergraph_t = subhypergraphs[t]
        adj_t, matrix_map = adjacency_matrix(hypergraph_t, return_mapping=True)
        if return_mapping:
            mapping[t] = matrix_map
        temporal_adjacency_matrixes[t] = adj_t
    if return_mapping:
        return temporal_adjacency_matrixes, mapping
    else:
        return temporal_adjacency_matrixes

def temporal_adjacency_matrix_by_order(
    temporal_hypergraph: TemporalHypergraph, order: int, return_mapping: bool = False
) -> Dict[int, sparse.csc_array] | Tuple[Dict[int, sparse.csc_array], dict[int,dict[int,int]]]:
    """Compute the temporal adjacency matrix of the temporal hypergraph by order.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix at time t
    counts the number of hyperedges of a given order, existing at time t, where both i and j are contained.

    Parameters
    ----------
    temporal_hypergraph: TemporalHypergraph.
    order: int
        The specific order to evaluate
    return_mapping: bool,optional
        Return the dictionary mapping the new node indices to the Temporal Hypergraph
    Returns
    -------
    temporal_adjacency_matrixes: Dict[int, sparse.csc_array]
        A dictionary encoding the temporal adjacency matrixes, i.e., {time : adjacency matrix}
    mapping: Dict[int, Dict[int,int]]
        The dictionary of node mappings for each adjacency matrix.
    """
    temporal_adjacency = {}
    subhypergraphs = temporal_hypergraph.subhypergraph()
    mapping = dict()
    for t in subhypergraphs.keys():
        hypergraph_t = subhypergraphs[t]
        adj_t, matrix_map = adjacency_matrix_by_order(hypergraph_t, order, return_mapping = return_mapping)
        temporal_adjacency[t] = adj_t
        if return_mapping:
            mapping[t] = matrix_map
    if return_mapping:
        return temporal_adjacency, mapping
    else:
        return temporal_adjacency

def temporal_adjacency_matrices_all_orders(
    temporal_hypergraph: TemporalHypergraph,
    max_order: int = None,
    return_mapping: bool = False
)-> dict[int, Tuple[sparse.csc_array]] | Tuple[Dict[int, Tuple[sparse.csc_array]], dict[int,dict[int,int]]] :
    """Compute the temporal adjacency matrices of the temporal hypergraph for all orders.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix of order d at time t
    counts the number of hyperedges of order d, existing at time t, where both i and j are contained.

    Parameters
    ----------
    temporal_hypergraph: TemporalHypergraph
    max_order: int | None, optional
        The maximum order of the hypergraph. If not specified will be automatically selected
    return_mapping: bool,optional
        Return the dictionary mapping the new node indices to the Temporal Hypergraph
    Returns
    -------
    temporal_adjacencies: Dict[int, sparse.csc_array]
        A dictionary encoding the temporal adjacency matrixes, i.e., {order : {time : adjacency matrix}}
    mapping: Dict[int, Dict[int,int]]
        The dictionary of node mappings for each adjacency matrix.
    """
    temporal_adjacencies = {}
    if max_order is None:
        max_order = temporal_hypergraph.max_order()
    mapping_dict = dict()
    for order in range(1, max_order + 1):
        adj_order_d, mapping = temporal_adjacency_matrix_by_order(
            temporal_hypergraph, order, return_mapping= True
        )
        temporal_adjacencies[order] = adj_order_d
        mapping_dict[order] = mapping
    if return_mapping:
        return temporal_adjacencies, mapping_dict
    else:
        return temporal_adjacencies

def annealed_adjacency_matrix(
    temporal_hypergraph: TemporalHypergraph,
    return_mapping: bool = False,
) -> sparse.csc_array | Tuple[sparse.csc_array, Dict[int, Any]]:
    """Compute the annealed adjacency matrix of the temporal hypergraph by order.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix
    counts the average number of hyperedges of a given order where both i and j are contained over time.

    Parameters
    ----------
    temporal_hypergraph: TemporalHypergraph
    return_mapping: bool,optional
        Return the dictionary mapping the new node indices to the Temporal Hypergraph
    Returns
    -------
    matrix: sparse.csc_array
        The annealed adjacency matrix for a given order.
    return_mapping: bool,optional
        Return the dictionary mapping the new node indices to the Temporal Hypergraph
    """
    encoder = temporal_hypergraph.get_mapping()
    temporal_adjacency_matrix, mapping = temporal_hypergraph.temporal_adjacency_matrix(return_mapping=True)
    T = len(temporal_adjacency_matrix.keys())
    temporal_adjacency_matrix_lst = temporal_adjacency_matrix.values()
    res = dict()
    t = min(mapping.keys())
    for matrix in temporal_adjacency_matrix_lst:
        for j in range(matrix.shape[1]):
            for i in range(matrix.indptr[j], matrix.indptr[j + 1]):
                row = mapping[t][matrix.indices[i]]
                column = mapping[t][j]
                if row != column:
                    cell = (row, column)
                    cell = tuple(sorted(cell))
                    if cell not in res.keys():
                        res[cell] = 0
                    res[cell] += matrix.data[i]
        t+=1
    res = {k: v / T for k,v in res.items()}
    matrix_row = []
    matrix_col = []
    matrix_val = []
    for k,v in res.items():
        matrix_row.append(encoder.transform([k[0]])[0])
        matrix_col.append(encoder.transform([k[1]])[0])
        matrix_val.append(v)
    matrix = csc_array((matrix_val, (matrix_row, matrix_col)))
    if return_mapping:
        return matrix, get_inverse_mapping(encoder)
    else:
        return matrix

def annealed_adjacency_matrices_all_orders(
    temporal_hypergraph: TemporalHypergraph
)->dict[int, sparse.csc_array]:
    """Compute the annealed adjacency matrices of the temporal hypergraph for all orders.
    For any two nodes i, j in the hypergraph, the entry (i, j) of the adjacency matrix of order d
    counts the average number of hyperedges of order d where both i and j are contained over time.

    Parameters
    ----------
    temporal_hypergraph: TemporalHypergraph
    Returns
    -------
    annealed_adjacency_matrices_order: dict[int, sparse.csc_array]
        The annealed adjacency matrix for all orders, i.e., {order : annealed adjacency matrix}.
    """
    temporal_adjacency_matrices = temporal_adjacency_matrices_all_orders(temporal_hypergraph)
    annealed_adjacency_matrices_order = dict()
    for order, temporal_adjacency_matrix in temporal_adjacency_matrices.items():
        T = len(temporal_adjacency_matrix.keys())
        temporal_adjacency_matrix_lst = temporal_adjacency_matrix.values()
        annealed_adjacency_matrix = sum(temporal_adjacency_matrix_lst) / T
        annealed_adjacency_matrices_order[order] = annealed_adjacency_matrix
    return annealed_adjacency_matrices_order