from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.special import factorial

from hnx.core import Hypergraph
from hnx.linalg import temporal_adjacency_matrix_by_order, annealed_adjacency_matrix_by_order

def intra_order_correlation_matrix_by_order(
    temporal_hypergraph: Dict[int, Hypergraph], order: int, tau: int
) -> sparse.csc_array:
    """ Compute the intra-order correlation matrix for hyperedges of order d and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    order: the order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation matrix of order d at time lag tau, as a sparse matrix.
    """
    T = max(temporal_hypergraph.keys())
    temporal_adjacency_matrix = temporal_adjacency_matrix_by_order(temporal_hypergraph, order)
    annealed_adjacency_matrix = annealed_adjacency_matrix_by_order(temporal_adjacency_matrix, order)

    correlation_matrix = sparse.csc_array(annealed_adjacency_matrix.shape, dtype=np.int8)
    for t in range(T-tau):
        adjacency_matrix_t = temporal_adjacency_matrix[t]
        adjacency_matrix_t_lagged = temporal_adjacency_matrix[t+tau]   

        centered_adjacency_matrix_t = adjacency_matrix_t - annealed_adjacency_matrix
        centered_adjacency_matrix_t_lagged = adjacency_matrix_t_lagged - annealed_adjacency_matrix

        correlation_matrix = correlation_matrix + centered_adjacency_matrix_t.dot(centered_adjacency_matrix_t_lagged.tranpose())
    
    correlation_matrix = correlation_matrix/(factorial(order)**2)/(T-tau)

    return correlation_matrix

def intra_order_correlation_function_by_order(
    temporal_hypergraph: Dict[int, Hypergraph], order: int, tau: int
) -> float:
    """ Compute the intra-order correlation function for hyperedges of order d and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    order: the order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation function of order d at time lag tau.
    """
    correlation_matrix = intra_order_correlation_matrix_by_order(temporal_hypergraph, order, tau)
    correlation_function = correlation_matrix.trace()

    return correlation_function

def intra_order_correlation_matrices_all_orders(
    temporal_hypergraph: Dict[int, Hypergraph], max_order: int, tau: int
): # -> Tuple[sparse.csc_array]: ### to check
    """ Compute the intra-order correlation matrices for hyperedges of all orders and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation matrices for all orders at time lag tau, as sparse matrices.
    """

    correlation_matrices = []
    for order in range(max_order):
        correlation_matrix = intra_order_correlation_function_by_order(temporal_hypergraph, order, tau)
        correlation_matrices.append(correlation_matrix)
        
    return tuple(correlation_matrices)

def intra_order_correlation_functions_all_orders(
    temporal_hypergraph: Dict[int, Hypergraph], max_order: int, tau: int
): # -> Tuple[float]: ### to check
    """ Compute the intra-order correlation function for hyperedges of every order and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation functions for all orders at time lag tau.
    """
    correlation_functions = []
    for order in range(max_order):
        correlation_function = intra_order_correlation_function_by_order(temporal_hypergraph, order, tau)
        correlation_functions.append(correlation_function)

    return tuple(correlation_functions)
